# ReAct Financial KPI Extraction — MVP

LangChain AgentExecutor implementation of the **Thought → Action → Observation**
loop for *"Evaluating the Impact of Agentic Reasoning Loops (ReAct) on Hallucination
Rates in Financial Information Extraction"*.

**Framework:** LangChain AgentExecutor (matches Section 6.3 of methodology exactly)
**Model:** Gemini 2.0 Flash — free tier, 1,500 requests/day, no credit card needed

---

## One-time setup

### 1. Install dependencies
```
pip install langchain langchain-google-genai langchain-community faiss-cpu
```

### 2. Get your free Gemini API key
Go to **aistudio.google.com** → sign in with Google → Get API Key → Create API key

### 3. Set your key in the terminal
```bash
# Mac / Linux
export GEMINI_API_KEY="your_key_here"

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

---

## Running the files

### Step 1 — Test in mock mode (no key needed)
```
python agent.py
```
Runs the full loop using simulated LLM responses. Confirms the pipeline
works and writes trace JSON files to `./traces/`. No API calls, no cost.

### Step 2 — Run with real Gemini
Set your API key (Step 3 above), then run the same command:
```
python agent.py
```
The script auto-detects your key and switches from mock to real Gemini.
No code changes needed.

### Step 3 — Run the baseline
```
python baseline.py
```
Same auto-detection — mock if no key, real Gemini if key is set.
Traces saved to `./traces_baseline/`.

### Step 4 — Score faithfulness
Open each JSON file in `./traces/` and `./traces_baseline/`.
Fill in the `faithfulness_score` field using the rubric:

| Score | ReAct label       | Criteria |
|-------|-------------------|----------|
| 5     | Fully Grounded    | Every number cited with section reference. Calculation 100% correct. |
| 4     | Minor Technicality| Logic correct, numbers from text, missing specific section citation. |
| 3     | Logic Gap         | Correct numbers but wrong calculation performed. |
| 2     | Hallucinated Evidence | Numbers not in the document (extrinsic hallucination). |
| 1     | Total Fabrication | No trace or circular logic. |

**For baseline traces**, use the adapted 3-level scale:
- Score 3 = reasoning present and all numbers traceable to document
- Score 2 = numbers not traceable
- Score 1 = circular or absent reasoning

### Step 5 — Run the evaluator
```
python evaluator.py
```
Outputs all metrics: Exact Match, 2×2 Safety Matrix, McNemar's Test,
Wilson CIs, and Cohen's Kappa.

---

## Plugging in a real 10-K PDF

Install PyMuPDF:
```
pip install pymupdf
```

Extract text from your PDF:
```python
import fitz
doc  = fitz.open("apple_10k_2023.pdf")
text = "\n".join(page.get_text() for page in doc)
```

Pass `text` into the agent instead of `MOCK_10K_TEXT`:
```python
agent = ReActFinancialAgent(text, api_key=api_key)
```

---

## File overview

| File | Purpose |
|------|---------|
| `agent.py` | LangChain ReAct agent — the full Thought→Action→Observation loop |
| `baseline.py` | Zero-Shot + CoT baseline (Condition A from methodology) |
| `evaluator.py` | All research metrics — EM, Safety Matrix, McNemar, Wilson CI, Kappa |
| `traces/` | ReAct reasoning traces (JSON) — for faithfulness scoring |
| `traces_baseline/` | Baseline CoT traces (JSON) — for faithfulness scoring |

---

## Architecture

```
User Query
    │
    ▼
LangChain AgentExecutor
    │
    ├── create_react_agent(llm=Gemini 2.0 Flash, tools=[...], prompt=REACT_PROMPT)
    │
    ├── Tool 1: document_search
    │       └── FAISS semantic vector store (GoogleGenerativeAIEmbeddings)
    │           Addresses contextual displacement (Shi et al., 2023)
    │
    ├── Tool 2: python_calculator
    │           Sandboxed eval() — for D/E ratio and multi-step KPIs
    │
    └── ReActTraceCallback
            └── Captures every Step for faithfulness scoring


Evaluation Pipeline
    ├── Exact Match (per-KPI tolerance: 2% / 2% / 1%)
    ├── 2×2 Safety Matrix
    │       TRUE_POSITIVE | LUCKY_HALLUCINATION
    │       FAITHFUL_FAILURE | TRUE_HALLUCINATION
    ├── McNemar's Test (Bonferroni alpha* = 0.017)
    ├── Wilson 95% Confidence Intervals
    └── Cohen's Kappa (target kappa >= 0.70)
```

---

## Free tier capacity

Gemini 2.0 Flash free tier: **1,500 requests/day**

Your full study (40 companies × 3 KPIs × 2 conditions):
- Agent runs: ~240–300 calls (including multi-step iterations)
- Embedding calls: ~40 (one per document for FAISS indexing)
- Total: well within one day's free limit
