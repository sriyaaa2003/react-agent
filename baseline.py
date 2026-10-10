"""
Zero-Shot Baseline with Chain-of-Thought
=========================================
Condition A from the research methodology (Section 6.3).

Uses the strongest reasonable baseline: Zero-Shot + CoT (Wei et al., 2022).
This means prompting the model to show its reasoning before answering —
making it comparable to the ReAct trace for faithfulness scoring.

NOTE: This is NOT an agentic loop. The model gets the full document section
in one prompt and must extract the KPI in a single forward pass.
No tools. No document search. No calculator.
That is the entire point — it is the control condition.

Framework:  LangChain (direct LLM call, no agent)
Model:      Gemini 2.0 Flash via langchain-google-genai (free tier)

SETUP:
    pip install langchain langchain-google-genai
    export GEMINI_API_KEY="your_key_here"
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGoogleGenerativeAI = ChatPromptTemplate = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL       = "gemini-2.0-flash"
TEMPERATURE = 0   # Deterministic, matches methodology


# ============================================================================
# Data structure
# ============================================================================

@dataclass
class BaselineResult:
    company:   str
    kpi:       str
    answer:    Optional[float]
    reasoning: str = ""    # The CoT trace — scored for faithfulness
    status:    str = "success"   # success | parse_error


# ============================================================================
# Prompt — Zero-Shot + Chain-of-Thought (Wei et al., 2022)
# ============================================================================
# We explicitly request chain-of-thought because:
# (a) It is the strongest reasonable baseline (not just plain Zero-Shot)
# (b) It generates a reasoning trace we can score for faithfulness
# (c) It makes the comparison to ReAct more conservative — less likely
#     to overstate ReAct's advantage (Section 6.3, Condition A note)

if ChatPromptTemplate:
    ZERO_SHOT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise financial analyst. Your job is to extract a specific "
            "KPI from the 10-K annual report text provided. "
            "Think step by step before answering. "
            "Cite the exact line or section where you found each number. "
            "If the value is not present in the text, report null — do not guess."
        )),
        ("human", (
            "Extract the following KPI from the 10-K filing text below.\n\n"
            "Company: {company}\n"
            "KPI to extract: {kpi}\n\n"
            "10-K Filing Text:\n"
            "{document_text}\n\n"
            "Instructions:\n"
            "1. Write your step-by-step reasoning, citing specific lines from the text.\n"
            "2. For Debt-to-Equity Ratio: compute Total Liabilities / Total Stockholders Equity.\n"
            "3. After your reasoning, output ONLY this JSON on the final line "
            "(no markdown, no backticks, no extra text after it):\n"
            '{{"kpi": "{kpi}", "value": <number or null>, '
            '"unit": "<millions|billions|ratio>", "confidence": "<high|medium|low>"}}'
        )),
    ])
else:
    ZERO_SHOT_PROMPT = None


# ============================================================================
# Zero-Shot Baseline class
# ============================================================================

class ZeroShotBaseline:

    def __init__(self, document_text: str, api_key: str):
        self.document_text = document_text
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=TEMPERATURE,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
        # LangChain chain: prompt → LLM
        self.chain = ZERO_SHOT_PROMPT | self.llm

    def run(self, company: str, kpi: str) -> BaselineResult:
        """
        Single forward pass — no tools, no loop.
        Passes the extracted financial sections (up to ~12k chars / ~3k tokens)
        to Gemini and parses the JSON answer from the response.
        """
        result = BaselineResult(company=company, kpi=kpi, answer=None)

        log.info(f"\n{'='*60}")
        log.info(f"[Baseline] {company} — {kpi}")
        log.info(f"[Baseline] Sending single Zero-Shot + CoT prompt to {MODEL}...")

        response = self.chain.invoke({
            "company":       company,
            "kpi":           kpi,
            "document_text": self.document_text[:12000],  # ~15k token cap per methodology
        })

        raw              = response.content.strip()
        result.reasoning = raw

        log.info(f"[Baseline] Response received ({len(raw)} chars)")

        # Extract the JSON from the last line of the response
        result.answer, result.status = _parse_baseline_answer(raw)

        log.info(f"[Baseline] Parsed answer: {result.answer}  Status: {result.status}")
        return result


def _parse_baseline_answer(raw: str) -> tuple:
    """
    Parses the JSON answer from the end of the Zero-Shot CoT response.
    Tries last line first, then searches the full text as fallback.
    Returns (value, status).
    """
    # Try last non-empty line (where we instructed the model to put the JSON)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        json_match = re.search(r'\{[^{}]+\}', line)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                value  = parsed.get("value")
                if value is not None:
                    return float(value), "success"
                return None, "success"
            except (json.JSONDecodeError, ValueError):
                continue

    # Fallback: search entire response
    for match in re.finditer(r'\{[^{}]+\}', raw):
        try:
            parsed = json.loads(match.group())
            if "value" in parsed:
                value = parsed["value"]
                if value is not None:
                    return float(value), "success"
                return None, "success"
        except (json.JSONDecodeError, ValueError):
            continue

    return None, "parse_error"


# ============================================================================
# Save baseline reasoning trace (for faithfulness scoring)
# ============================================================================

def save_baseline_trace(result: BaselineResult, output_dir: str = "traces_baseline") -> str:
    """
    Saves the Zero-Shot CoT reasoning as a trace file.
    Used for faithfulness scoring with the adapted 3-level rubric
    (Section 6.4 — RUBRIC ADAPTATION FOR ZERO-SHOT CoT).
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = f"{result.company}_{result.kpi}".replace(" ", "_").replace("/", "_")
    filename  = os.path.join(output_dir, f"{safe_name}.json")

    trace = {
        "company":   result.company,
        "kpi":       result.kpi,
        "answer":    result.answer,
        "status":    result.status,
        "condition": "zero_shot_cot",
        "reasoning": result.reasoning,
        # Faithfulness scored on adapted 3-level scale (not 5-point) per methodology
        # Score 3 = reasoning logic present and all numbers traceable to document
        # Score 2 = numbers not traceable to document
        # Score 1 = circular or absent reasoning
        "faithfulness_score": None,   # Fill in after human scoring
        "safety_matrix_cell": None,   # TRUE_POSITIVE | LUCKY_HALLUCINATION | FAITHFUL_FAILURE | TRUE_HALLUCINATION
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    log.info(f"Baseline trace saved → {filename}")
    return filename


# ============================================================================
# Mock baseline — no API key needed
# ============================================================================

class MockZeroShotBaseline:
    """Mock baseline for testing the pipeline without API calls."""

    MOCK_ANSWERS = {
        # Intentionally imperfect to simulate realistic baseline behaviour:
        # Revenue: correct (model likely saw this in training)
        # Net Income: slightly wrong (parametric memory conflict)
        # D/E: None (model fails on multi-step derived KPI without tools)
        "Revenue":        (383285, "The income statement shows Total net sales of $383,285 million for FY2023."),
        "Net Income":     (99803,  "Net income appears to be $99,803 million based on my knowledge of Apple FY2023."),
        "Debt-to-Equity": (None,   "I cannot reliably compute the Debt-to-Equity ratio from the provided text."),
    }

    def __init__(self, document_text: str, api_key: str = "mock"):
        log.info("[MockBaseline] Running in mock mode — no API calls.")

    def run(self, company: str, kpi: str) -> BaselineResult:
        answer, reasoning = self.MOCK_ANSWERS.get(kpi, (None, "No reasoning available."))
        status = "success" if answer is not None else "parse_error"
        log.info(f"\n[Baseline] {company} — {kpi} → {answer} [{status}]")
        return BaselineResult(
            company=company, kpi=kpi,
            answer=answer, reasoning=reasoning, status=status,
        )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from agent import MOCK_10K_TEXT, GROUND_TRUTH, exact_match

    print("\n" + "="*62)
    print("  Zero-Shot + CoT Baseline — Gemini 2.0 Flash")
    print("="*62 + "\n")

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        print("✓ GEMINI_API_KEY detected — running with real Gemini.\n")
        BaselineClass = ZeroShotBaseline
    else:
        print("ℹ No GEMINI_API_KEY — running in mock mode.\n")
        BaselineClass = MockZeroShotBaseline
        api_key       = "mock"

    baseline    = BaselineClass(MOCK_10K_TEXT, api_key=api_key)
    all_results = []

    for kpi in ["Revenue", "Net Income", "Debt-to-Equity"]:
        result     = baseline.run("Apple Inc.", kpi)
        trace_file = save_baseline_trace(result)
        gt         = GROUND_TRUTH[("Apple Inc.", kpi)]
        em         = exact_match(result.answer, gt, kpi)

        print(f"  {kpi:<22} Predicted: {str(result.answer):<12} GT: {gt}  EM: {'PASS' if em else 'FAIL'}  [{result.status}]")
        all_results.append({"kpi": kpi, "exact_match": em, "status": result.status})

    total_em = sum(1 for r in all_results if r["exact_match"])
    print(f"\n  Baseline Exact Match Accuracy: {total_em}/{len(all_results)} ({100*total_em/len(all_results):.0f}%)")
    print(f"  Baseline traces saved to ./traces_baseline/\n")
    print("  NOTE: In the mock, Net Income is intentionally wrong to simulate")
    print("  parametric memory conflict (Mallen et al., 2023) — the key failure")
    print("  mode that ReAct is designed to fix.\n")
