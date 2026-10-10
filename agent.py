"""
ReAct Financial KPI Extraction Agent
=====================================
Implements the Thought → Action → Observation loop for extracting
KPIs (Revenue, Net Income, Debt-to-Equity Ratio) from SEC 10-K filings.

Framework:  LangChain AgentExecutor with ReAct prompt (matches methodology paper exactly)
Model:      Gemini 2.0 Flash via langchain-google-genai (free tier)
Tools:      DocumentSearchTool (FAISS vector search) + PythonCalculatorTool (sandboxed eval)

Based on:   Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
Research:   Evaluating ReAct's impact on hallucination rates in financial IE.

SETUP:
    pip install langchain langchain-google-genai langchain-community faiss-cpu
    Get free Gemini API key at: aistudio.google.com
    export GEMINI_API_KEY="your_key_here"
"""

import os
import re
import json
import math
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ── LangChain imports (installed via: pip install langchain langchain-google-genai langchain-community faiss-cpu)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain.prompts import PromptTemplate
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import AIMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Stubs so the file parses without langchain installed
    AgentExecutor = create_react_agent = Tool = PromptTemplate = None
    BaseCallbackHandler = object
    ChatGoogleGenerativeAI = GoogleGenerativeAIEmbeddings = None
    RecursiveCharacterTextSplitter = FAISS = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Config — mirrors research methodology (Section 6.3)
# ============================================================================

MODEL         = "gemini-2.0-flash"
TEMPERATURE   = 0           # Deterministic — required by methodology
MAX_ITER      = 10          # Set empirically via pilot (90th-percentile rule)
CHUNK_SIZE    = 500         # Token window for document chunking
CHUNK_OVERLAP = 50          # Overlap to avoid splitting tables mid-row
TOP_K_CHUNKS  = 5           # Number of chunks returned per search

TOLERANCE_PCT = {
    "revenue":        2.0,  # Section 6.4 — directly stated values
    "net_income":     2.0,  # Section 6.4 — directly stated values
    "debt_to_equity": 1.0,  # Section 6.4 — derived metric, tighter threshold
}


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Step:
    """One complete Thought → Action → Observation cycle."""
    iteration:    int
    thought:      str
    action:       str   # "document_search" | "python_calculator" | (final answer)
    action_input: str
    observation:  str

@dataclass
class AgentResult:
    """Full result for one company × KPI extraction run."""
    company:    str
    kpi:        str
    answer:     Optional[float]
    steps:      list = field(default_factory=list)
    status:     str  = "success"   # success | agent_failure | parse_error
    iterations: int  = 0
    raw_output: str  = ""


# ============================================================================
# Step 1 — Build the FAISS vector store from 10-K text
# (Semantic search: addresses contextual displacement — Shi et al., 2023)
# ============================================================================

def build_vector_store(document_text: str, api_key: str) -> FAISS:
    """
    Chunks the 10-K text and embeds it into a FAISS vector store.
    Semantic search means 'shareholders equity' matches 'stockholders equity'
    even if the exact wording differs across companies — critical for 40 filings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],   # Prefer splitting on blank lines (section breaks)
    )
    chunks    = splitter.create_documents([document_text])
    embedder  = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    log.info(f"[VectorStore] Embedding {len(chunks)} chunks from 10-K filing...")
    store = FAISS.from_documents(chunks, embedder)
    log.info("[VectorStore] FAISS index built successfully.")
    return store


# ============================================================================
# Step 2 — Tool definitions
# ============================================================================

def make_document_search_tool(vector_store: FAISS) -> Tool:
    """
    Tool 1: Semantic document search over the 10-K filing.
    Uses FAISS similarity search rather than keyword matching —
    handles terminology variation across companies and sectors.
    Returns TOP_K_CHUNKS most relevant passages with their content.
    """
    def search(query: str) -> str:
        docs = vector_store.similarity_search(query, k=TOP_K_CHUNKS)
        if not docs:
            return "No relevant passages found. Try a different search query."
        passages = []
        for i, doc in enumerate(docs, 1):
            passages.append(f"[Passage {i}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(passages)

    return Tool(
        name="document_search",
        func=search,
        description=(
            "Search the 10-K filing for relevant financial information. "
            "Input should be a natural language query describing what you are looking for. "
            "Examples: 'total net sales 2023', 'total liabilities balance sheet', "
            "'total stockholders equity'. "
            "Returns the most relevant passages from the document. "
            "Always search before citing any number."
        ),
    )


def make_calculator_tool() -> Tool:
    """
    Tool 2: Sandboxed Python calculator.
    Executes arithmetic expressions using only the math module —
    no arbitrary code execution, no imports, no builtins.
    Used for derived KPIs like Debt-to-Equity (Total Liabilities / Equity).
    """
    safe_namespace = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}

    def calculate(expression: str) -> str:
        # Strip any accidental markdown formatting the LLM might add
        expression = expression.strip().strip("`").strip()
        try:
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            return f"Result: {round(float(result), 6)}"
        except ZeroDivisionError:
            return "Error: Division by zero. Check that equity value is non-zero."
        except SyntaxError:
            return f"Error: Invalid expression syntax. Input a plain arithmetic expression like '290437 / 62146'."
        except Exception as e:
            return f"Error: {e}. Input must be a plain arithmetic expression."

    return Tool(
        name="python_calculator",
        func=calculate,
        description=(
            "Evaluate arithmetic expressions for financial calculations. "
            "Input must be a plain arithmetic expression — no variables, no code. "
            "Examples: '290437 / 62146', '96995 / 383285 * 100', '145308 + 145129'. "
            "Use this for any ratio or multi-step calculation. "
            "Always use numbers you retrieved from document_search, never invented values."
        ),
    )


# ============================================================================
# Step 3 — ReAct prompt (matches LangChain's AgentExecutor expected format)
# ============================================================================

REACT_PROMPT_TEMPLATE = """You are a precise financial analyst extracting KPIs from SEC 10-K annual filings.
Your task is to extract one specific KPI accurately and faithfully from the document.

EXTRACTION RULES:
- Use ONLY numbers retrieved from the document via document_search. Never use memorized figures.
- Always search before citing any number. Ground every figure in an observation.
- For fiscal year disambiguation: prefer the most recent fiscal year unless instructed otherwise.
- For Debt-to-Equity Ratio: value = Total Liabilities / Total Stockholders Equity.
- Units matter: note whether values are in millions, billions, or as a plain ratio.
- If the document does not contain the information, report null — do not fabricate.
- Think step by step. Show your reasoning before acting.

You have access to the following tools:
{tools}

Use the following format EXACTLY for every step:

Question: the input question you must answer
Thought: your reasoning about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to give the final answer
Final Answer: a JSON object like this (no markdown, no backticks):
{{"value": <number or null>, "unit": "<millions|billions|ratio>", "confidence": "<high|medium|low>", "reasoning": "<one sentence explaining how you found this>"}}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE) if PromptTemplate else None


# ============================================================================
# Step 4 — Trace callback (captures every Thought/Action/Observation)
# ============================================================================

class ReActTraceCallback(BaseCallbackHandler):
    """
    LangChain callback that intercepts every step of the AgentExecutor loop
    and stores it as a structured Step object for faithfulness scoring.
    This is what generates the audit trail for the 2x2 Safety Matrix.
    """

    def __init__(self):
        self.steps: list[Step] = []
        self._iteration        = 0
        self._current_thought  = ""
        self._current_action   = ""
        self._current_input    = ""

    def on_agent_action(self, action, **kwargs):
        self._iteration      += 1
        self._current_action  = action.tool
        self._current_input   = str(action.tool_input)
        # Thought is embedded in the log field of the action
        self._current_thought = getattr(action, "log", "").strip()
        # Strip "Thought:" prefix if present
        if self._current_thought.lower().startswith("thought:"):
            self._current_thought = self._current_thought[8:].strip()

        log.info(
            f"\n{'='*60}\n"
            f"[Iter {self._iteration}] THOUGHT:      {self._current_thought}\n"
            f"[Iter {self._iteration}] ACTION:       {self._current_action}\n"
            f"[Iter {self._iteration}] ACTION INPUT: {self._current_input}"
        )

    def on_tool_end(self, output: str, **kwargs):
        observation = str(output)
        log.info(
            f"[Iter {self._iteration}] OBSERVATION:  "
            f"{observation[:400]}{'...' if len(observation) > 400 else ''}"
        )
        self.steps.append(Step(
            iteration=self._iteration,
            thought=self._current_thought,
            action=self._current_action,
            action_input=self._current_input,
            observation=observation,
        ))

    def on_agent_finish(self, finish, **kwargs):
        log.info(
            f"\n{'='*60}\n"
            f"[Final] THOUGHT: {getattr(finish, 'log', '').strip()}\n"
            f"[Final] OUTPUT:  {finish.return_values.get('output', '')[:300]}"
        )


# ============================================================================
# Step 5 — Main ReAct Agent class
# ============================================================================

class ReActFinancialAgent:
    """
    LangChain AgentExecutor implementing the ReAct loop for financial KPI extraction.
    Matches Section 6.3 (Condition B) of the research methodology exactly.
    """

    def __init__(self, document_text: str, api_key: str):
        self.api_key       = api_key
        self.document_text = document_text

        # LLM — Gemini 2.0 Flash, temperature=0 per methodology
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=TEMPERATURE,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

        # Build FAISS vector store from the filing text
        self.vector_store = build_vector_store(document_text, api_key)

        # Initialise tools
        self.tools = [
            make_document_search_tool(self.vector_store),
            make_calculator_tool(),
        ]

        # Build the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT,
        )

    def run(self, company: str, kpi: str) -> AgentResult:
        """
        Runs one extraction: company × KPI.
        Returns a fully populated AgentResult with all steps logged.
        """
        result   = AgentResult(company=company, kpi=kpi, answer=None)
        callback = ReActTraceCallback()

        # AgentExecutor wraps the agent with iteration limits, error handling,
        # and callback support — this is the LangChain equivalent of our manual loop
        executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=MAX_ITER,
            verbose=False,              # We handle logging via callback
            handle_parsing_errors=True, # Retry on malformed LLM output
            callbacks=[callback],
            return_intermediate_steps=True,
        )

        question = (
            f"Extract the {kpi} for {company} from the 10-K filing. "
            f"Report the numeric value, its unit (millions, billions, or ratio), "
            f"and your confidence level. "
            f"For Debt-to-Equity: divide Total Liabilities by Total Stockholders Equity. "
            f"Ground every number in a document_search observation before using it."
        )

        try:
            output = executor.invoke({"input": question})

            result.raw_output = output.get("output", "")
            result.iterations = len(callback.steps)
            result.steps      = callback.steps

            # Determine status
            if output.get("output", "").strip().lower() in ["agent stopped due to iteration limit or time limit.", ""]:
                result.status = "agent_failure"
            else:
                # Parse the JSON answer from the final output
                result.answer, result.status = _parse_final_answer(result.raw_output)

        except Exception as e:
            log.error(f"[AgentExecutor] Exception during run: {e}")
            result.status     = "agent_failure"
            result.steps      = callback.steps
            result.iterations = len(callback.steps)

        return result


def _parse_final_answer(raw_output: str) -> tuple:
    """
    Extract the numeric value from the agent's final JSON answer.
    Returns (value, status).
    """
    # Try to find a JSON object in the output
    json_match = re.search(r'\{[^{}]+\}', raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            value  = parsed.get("value")
            if value is not None:
                return float(value), "success"
            return None, "success"   # Null is a valid answer (doc didn't contain data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract a bare number
    number_match = re.search(r'\b(\d[\d,]*\.?\d*)\b', raw_output)
    if number_match:
        try:
            value = float(number_match.group(1).replace(",", ""))
            return value, "success"
        except ValueError:
            pass

    return None, "parse_error"


# ============================================================================
# Step 6 — Exact Match Scorer (Section 6.4 — Metric 1)
# ============================================================================

def exact_match(predicted: Optional[float], ground_truth: float, kpi: str) -> bool:
    """
    Returns True if predicted is within the KPI-specific tolerance of ground truth.
    Tolerance: 2% for Revenue/Net Income, 1% for Debt-to-Equity (Section 6.4).
    """
    if predicted is None or ground_truth == 0:
        return False
    key       = kpi.lower().replace(" ", "_").replace("-to-", "_to_").replace("-", "_")
    tolerance = TOLERANCE_PCT.get(key, 2.0) / 100
    pct_diff  = abs(predicted - ground_truth) / abs(ground_truth)
    return pct_diff <= tolerance


# ============================================================================
# Step 7 — Trace Logger (for faithfulness scoring — Section 6.4 Metric 2)
# ============================================================================

def save_trace(result: AgentResult, output_dir: str = "traces") -> str:
    """
    Saves the full reasoning trace as structured JSON.
    Each trace is scored by human rater using the 5-point faithfulness rubric.
    File naming: {company}_{kpi}.json (spaces replaced with underscores).
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = f"{result.company}_{result.kpi}".replace(" ", "_").replace("/", "_")
    filename  = os.path.join(output_dir, f"{safe_name}.json")

    trace = {
        "company":    result.company,
        "kpi":        result.kpi,
        "answer":     result.answer,
        "status":     result.status,
        "iterations": result.iterations,
        "raw_output": result.raw_output,
        "steps": [
            {
                "iteration":    s.iteration,
                "thought":      s.thought,
                "action":       s.action,
                "action_input": s.action_input,
                "observation":  s.observation,
            }
            for s in result.steps
        ],
        # Placeholder for human faithfulness scorer (Section 7.4)
        "faithfulness_score": None,   # Fill in: 1-5 per rubric
        "safety_matrix_cell": None,   # Fill in: TRUE_POSITIVE | LUCKY_HALLUCINATION | FAITHFUL_FAILURE | TRUE_HALLUCINATION
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    log.info(f"Trace saved → {filename}")
    return filename


# ============================================================================
# Mock agent — runs without any API key for testing and demo
# ============================================================================

class MockReActAgent:
    """
    Simulates the full ReAct loop without making any API calls.
    Use this to verify the pipeline works before spending API quota.
    Swap with ReActFinancialAgent once GEMINI_API_KEY is set.
    """

    MOCK_DIALOGUE = {
        "Revenue": [
            (
                "I need to find total net sales for fiscal year 2023 from the income statement.",
                "document_search",
                "total net sales revenue fiscal year 2023",
                "[Passage 1]\nTotal net sales\n  Products     $ 298,085\n  Services       85,200\nTotal net sales  383,285\n\n[Passage 2]\nNet income $ 96,995",
            ),
            (
                "The document shows Total net sales = $383,285 million for FY2023. "
                "This is the Revenue figure. No calculation needed.",
                "final_answer_internal",
                '{"value": 383285, "unit": "millions", "confidence": "high", "reasoning": "Total net sales for year ended September 30 2023 found directly in Consolidated Statements of Operations."}',
                "[Final answer produced]",
            ),
        ],
        "Net Income": [
            (
                "I need to find Net Income for fiscal year 2023.",
                "document_search",
                "net income 2023 consolidated statements operations",
                "[Passage 1]\nNet income $ 96,995   $ 99,803\n(FY2023 and FY2022 respectively)",
            ),
            (
                "Net income for FY2023 is $96,995 million. Clearly stated in the income statement.",
                "final_answer_internal",
                '{"value": 96995, "unit": "millions", "confidence": "high", "reasoning": "Net income found directly in Consolidated Statements of Operations for year ended September 30 2023."}',
                "[Final answer produced]",
            ),
        ],
        "Debt-to-Equity": [
            (
                "Debt-to-Equity = Total Liabilities / Total Stockholders Equity. "
                "I need to find Total Liabilities first.",
                "document_search",
                "total liabilities consolidated balance sheet 2023",
                "[Passage 1]\nTotal current liabilities    $ 145,308\nTotal non-current liabilities  145,129\nTotal liabilities              290,437",
            ),
            (
                "Total Liabilities = $290,437 million. Now I need Total Stockholders Equity.",
                "document_search",
                "total shareholders equity stockholders equity 2023",
                "[Passage 1]\nTotal shareholders equity   62,146\nTotal liabilities and shareholders equity  352,583",
            ),
            (
                "Total Liabilities = $290,437M, Total Equity = $62,146M. "
                "Now I calculate D/E = 290437 / 62146.",
                "python_calculator",
                "290437 / 62146",
                "Result: 4.673038",
            ),
            (
                "Debt-to-Equity ratio = 4.673. Both inputs grounded in document observations.",
                "final_answer_internal",
                '{"value": 4.673, "unit": "ratio", "confidence": "high", "reasoning": "D/E = Total Liabilities 290437 / Total Shareholders Equity 62146, both from Consolidated Balance Sheet September 30 2023."}',
                "[Final answer produced]",
            ),
        ],
    }

    def __init__(self, document_text: str, api_key: str = "mock"):
        log.info("[MockAgent] Running in mock mode — no API calls will be made.")

    def run(self, company: str, kpi: str) -> AgentResult:
        result   = AgentResult(company=company, kpi=kpi, answer=None)
        dialogue = self.MOCK_DIALOGUE.get(kpi, [])

        for i, (thought, action, action_input, observation) in enumerate(dialogue, 1):
            result.iterations = i

            if action == "final_answer_internal":
                # Parse the mock final answer
                parsed        = json.loads(action_input)
                result.answer = parsed.get("value")
                result.status = "success"
                result.raw_output = action_input
                log.info(
                    f"\n{'='*60}\n"
                    f"[Iter {i}] THOUGHT:      {thought}\n"
                    f"[Iter {i}] FINAL ANSWER: {action_input}"
                )
                # Add a pseudo-step for the final answer
                result.steps.append(Step(i, thought, "final_answer", action_input, observation))
                return result

            log.info(
                f"\n{'='*60}\n"
                f"[Iter {i}] THOUGHT:      {thought}\n"
                f"[Iter {i}] ACTION:       {action}\n"
                f"[Iter {i}] ACTION INPUT: {action_input}\n"
                f"[Iter {i}] OBSERVATION:  {observation}"
            )
            result.steps.append(Step(i, thought, action, action_input, observation))

        result.status = "agent_failure"
        return result


# ============================================================================
# Demo runner
# ============================================================================

MOCK_10K_TEXT = """
APPLE INC.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares and per share amounts)

                                    Years ended
                        September 30, 2023    September 24, 2022
Net sales:
  Products                   $  298,085            $  316,199
  Services                      85,200                78,129
Total net sales               383,285               394,328

Cost of sales:
  Products                      189,282               201,471
  Services                       24,855                22,075
Total cost of sales             214,137               223,546

Gross margin                    169,148               170,782

Operating expenses:
  Research and development        29,915                26,251
  Selling, general and administrative  24,932            25,094
Total operating expenses          54,847                51,345

Operating income                114,301               119,437
Net income                      $ 96,995              $ 99,803

CONSOLIDATED BALANCE SHEETS
(In millions)
                                   September 30, 2023   September 24, 2022
ASSETS
  Total current assets              $ 143,566             $ 135,405
  Total non-current assets            209,822               217,350
  Total assets                        352,583               352,755

LIABILITIES AND SHAREHOLDERS EQUITY
  Total current liabilities           $ 145,308             $ 153,982
  Total non-current liabilities         145,129               148,101
  Total liabilities                     290,437               302,083

  Total shareholders equity              62,146                50,672

Total liabilities and shareholders equity  352,583             352,755
"""

GROUND_TRUTH = {
    ("Apple Inc.", "Revenue"):        383285.0,
    ("Apple Inc.", "Net Income"):      96995.0,
    ("Apple Inc.", "Debt-to-Equity"):     4.673,
}


if __name__ == "__main__":
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ReAct Financial KPI Extraction Agent                       ║
    ║  Framework: LangChain AgentExecutor                         ║
    ║  Model:     Gemini 2.0 Flash (Free Tier)                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """))

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        print("✓ GEMINI_API_KEY detected — running with real Gemini 2.0 Flash.\n")
        AgentClass = ReActFinancialAgent
    else:
        print("ℹ No GEMINI_API_KEY found — running in mock mode (no API calls).")
        print("  To use real Gemini:")
        print("    1. pip install langchain langchain-google-genai langchain-community faiss-cpu")
        print("    2. Get free key at aistudio.google.com")
        print("    3. export GEMINI_API_KEY='your_key_here'\n")
        AgentClass = MockReActAgent
        api_key    = "mock"

    agent       = AgentClass(MOCK_10K_TEXT, api_key=api_key)
    all_results = []

    for kpi in ["Revenue", "Net Income", "Debt-to-Equity"]:
        print(f"\n{'#'*62}")
        print(f"  Extracting: {kpi} for Apple Inc.")
        print(f"{'#'*62}")

        result     = agent.run("Apple Inc.", kpi)
        trace_file = save_trace(result)

        gt  = GROUND_TRUTH[("Apple Inc.", kpi)]
        em  = exact_match(result.answer, gt, kpi)
        tol = TOLERANCE_PCT.get(
            kpi.lower().replace(" ", "_").replace("-to-", "_to_"), 2.0
        )

        print(f"\n  ── RESULT SUMMARY ──────────────────────────────────")
        print(f"  KPI:          {kpi}")
        print(f"  Predicted:    {result.answer}")
        print(f"  Ground Truth: {gt}")
        print(f"  Tolerance:    {tol}%")
        print(f"  Exact Match:  {'✓ PASS' if em else '✗ FAIL'}")
        print(f"  Status:       {result.status}")
        print(f"  Iterations:   {result.iterations}")
        print(f"  Trace:        {trace_file}")

        all_results.append({
            "kpi": kpi, "predicted": result.answer, "ground_truth": gt,
            "exact_match": em, "status": result.status, "iterations": result.iterations,
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL RESULTS — Apple Inc.")
    print(f"{'='*62}")
    print(f"  {'KPI':<22} {'Predicted':>12} {'GT':>10}  {'EM':<5} Status")
    print(f"  {'-'*58}")
    for r in all_results:
        em_str = "PASS" if r["exact_match"] else "FAIL"
        print(
            f"  {r['kpi']:<22} {str(r['predicted']):>12} "
            f"{str(r['ground_truth']):>10}  {em_str:<5} {r['status']}"
        )

    total_em = sum(1 for r in all_results if r["exact_match"])
    print(f"\n  Exact Match Accuracy: {total_em}/{len(all_results)} ({100*total_em/len(all_results):.0f}%)")
    print(f"\n  Traces written to ./traces/")
    print(f"  Open each JSON to score faithfulness using the 5-point rubric.\n")
