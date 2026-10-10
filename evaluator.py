"""
Evaluator
=========
Computes all metrics from the research methodology:

  Metric 1 — Exact Match accuracy with per-KPI tolerance (Section 6.4)
  Metric 2 — 2x2 Safety Matrix classification (Section 6.4)
  Metric 3 — McNemar's Test with Bonferroni correction (Section 7.1)
  Metric 4 — 95% Wilson confidence intervals (Section 7.3)
  Metric 5 — Cohen's Kappa for faithfulness inter-rater reliability (Section 7.4)

Usage (after running agent.py and baseline.py on all 40 companies):

    python evaluator.py

Or import and call directly:

    from evaluator import run_evaluation
    report = run_evaluation(baseline_results, react_results, faithfulness_scores, ground_truth)
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Tolerance thresholds — Section 6.4
# ============================================================================

TOLERANCE = {
    "revenue":        0.02,   # 2% — directly stated value
    "net_income":     0.02,   # 2% — directly stated value
    "debt_to_equity": 0.01,   # 1% — derived metric, tighter per methodology
}

# Bonferroni-corrected alpha — Section 7.1
# Three simultaneous KPI-level tests → alpha* = 0.05 / 3
BONFERRONI_ALPHA = round(0.05 / 3, 4)   # 0.0167


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ExtractionRecord:
    """One extraction result for one company × KPI × condition."""
    company:           str
    kpi:               str
    condition:         str             # "baseline" or "react"
    predicted:         Optional[float]
    ground_truth:      float
    status:            str             # success | agent_failure | parse_error
    faithfulness_score: Optional[int]  # 1-5 (react) or 1-3 (baseline adapted rubric)

@dataclass
class KPIReport:
    kpi:                  str
    n:                    int
    baseline_correct:     int
    react_correct:        int
    baseline_accuracy:    float
    react_accuracy:       float
    baseline_95ci:        tuple
    react_95ci:           tuple
    mcnemar_b:            int     # Zero-Shot correct, ReAct wrong
    mcnemar_c:            int     # Zero-Shot wrong, ReAct correct
    mcnemar_chi2:         float
    mcnemar_p:            float
    significant:          bool    # p < BONFERRONI_ALPHA
    safety_counts:        dict

@dataclass
class FullReport:
    by_kpi:           dict = field(default_factory=dict)
    overall:          dict = field(default_factory=dict)
    kappa:            dict = field(default_factory=dict)
    bonferroni_alpha: float = BONFERRONI_ALPHA


# ============================================================================
# Metric 1 — Exact Match
# ============================================================================

def exact_match(predicted: Optional[float], ground_truth: float, kpi: str) -> bool:
    """
    Returns True if |predicted - ground_truth| / |ground_truth| <= tolerance.
    Uses per-KPI thresholds from Section 6.4.
    """
    if predicted is None or ground_truth == 0:
        return False
    key       = _normalise_kpi(kpi)
    tolerance = TOLERANCE.get(key, 0.02)
    return abs(predicted - ground_truth) / abs(ground_truth) <= tolerance


def exact_match_at_threshold(predicted, ground_truth, threshold_pct):
    """Robustness check — reports EM at a specific threshold (Section 6.4)."""
    if predicted is None or ground_truth == 0:
        return False
    return abs(predicted - ground_truth) / abs(ground_truth) <= threshold_pct / 100


# ============================================================================
# Metric 2 — 2x2 Safety Matrix
# ============================================================================

# Faithfulness thresholds:
#   ReAct:    Score 4-5 = Faithful (5-point rubric)
#   Baseline: Score 3   = Faithful (adapted 3-point rubric per Section 6.4)
FAITHFUL_THRESHOLD = {"react": 4, "baseline": 3}

def classify_safety_cell(em: bool, faithfulness_score: Optional[int], condition: str) -> str:
    """
    Classifies one extraction into the 2x2 Safety Matrix.

    Returns one of:
      TRUE_POSITIVE       — correct answer, faithful reasoning  → ideal
      LUCKY_HALLUCINATION — correct answer, unfaithful reasoning → dangerous
      FAITHFUL_FAILURE    — wrong answer,   faithful reasoning   → safe (catchable)
      TRUE_HALLUCINATION  — wrong answer,   unfaithful reasoning → worst case
      UNSCORED            — faithfulness_score is None (not yet rated)
    """
    if faithfulness_score is None:
        return "UNSCORED"

    threshold = FAITHFUL_THRESHOLD.get(condition, 4)
    faithful  = faithfulness_score >= threshold

    if em     and faithful:     return "TRUE_POSITIVE"
    elif em   and not faithful: return "LUCKY_HALLUCINATION"
    elif not em and faithful:   return "FAITHFUL_FAILURE"
    else:                       return "TRUE_HALLUCINATION"


# ============================================================================
# Metric 3 — McNemar's Test (Section 7.1)
# ============================================================================

def mcnemar_test(b: int, c: int) -> dict:
    """
    McNemar's Test for paired binary data.
    b = Zero-Shot correct and ReAct wrong (regression)
    c = Zero-Shot wrong and ReAct correct (improvement)

    Uses continuity correction: chi2 = (|b-c| - 1)^2 / (b+c)
    One-tailed test (H1 predicts c > b, i.e., ReAct improves over baseline).
    """
    if b + c == 0:
        return {
            "b": b, "c": c, "chi2": 0.0, "p": 1.0,
            "note": "No discordant pairs — methods agreed on all cases. Cannot test.",
        }

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p    = _chi2_sf_df1(chi2) / 2   # Divide by 2 for one-tailed test

    return {
        "b":    b,
        "c":    c,
        "chi2": round(chi2, 4),
        "p":    round(p, 4),
        "one_tailed": True,
        "note": (
            f"ReAct improved on {c} cases where baseline failed. "
            f"Baseline was better on {b} cases (regression)."
        ),
    }


def _chi2_sf_df1(x: float) -> float:
    """Survival function (1 - CDF) of chi-squared distribution with df=1."""
    if x <= 0:
        return 1.0
    try:
        return 1.0 - _regularised_gamma(0.5, x / 2)
    except Exception:
        return float("nan")


def _regularised_gamma(a: float, x: float, terms: int = 150) -> float:
    """Regularised lower incomplete gamma function P(a, x) via series expansion."""
    if x <= 0:
        return 0.0
    s = term = 1.0 / a
    for n in range(1, terms):
        term *= x / (a + n)
        s    += term
        if abs(term) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * s)


# ============================================================================
# Metric 4 — Wilson 95% Confidence Intervals (Section 7.3)
# ============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson score interval for proportion k/n at confidence level z.
    Preferred over normal approximation for small samples (n=40).
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat  = k / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


# ============================================================================
# Metric 5 — Cohen's Kappa for faithfulness reliability (Section 7.4)
# ============================================================================

def cohens_kappa(ratings_session_1: list, ratings_session_2: list) -> dict:
    """
    Computes Cohen's Kappa between two scoring sessions (intra-rater reliability).
    Target: kappa >= 0.70 (substantial agreement) before reporting conclusions.

    ratings_session_1, ratings_session_2: lists of integer scores (same order).
    """
    assert len(ratings_session_1) == len(ratings_session_2), "Rating lists must be same length."
    n = len(ratings_session_1)
    if n == 0:
        return {"kappa": None, "note": "No ratings provided."}

    # Observed agreement
    p_o = sum(1 for a, b in zip(ratings_session_1, ratings_session_2) if a == b) / n

    # Expected agreement by chance
    categories  = sorted(set(ratings_session_1) | set(ratings_session_2))
    p_e = sum(
        (ratings_session_1.count(cat) / n) * (ratings_session_2.count(cat) / n)
        for cat in categories
    )

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    interpretation = (
        "Poor"        if kappa < 0.20 else
        "Fair"        if kappa < 0.40 else
        "Moderate"    if kappa < 0.60 else
        "Substantial" if kappa < 0.80 else
        "Almost perfect"
    )

    meets_threshold = kappa >= 0.70

    return {
        "kappa":             round(kappa, 4),
        "p_observed":        round(p_o, 4),
        "p_expected_chance": round(p_e, 4),
        "n_traces":          n,
        "interpretation":    interpretation,
        "meets_threshold":   meets_threshold,
        "note": (
            "Kappa >= 0.70 achieved. Faithfulness conclusions are valid."
            if meets_threshold else
            "Kappa < 0.70. Revise rubric anchor definitions and re-score. "
            "Do not report faithfulness-based conclusions until threshold is met."
        ),
    }


# ============================================================================
# Robustness check — Section 6.4
# ============================================================================

def robustness_check(records: list, kpi: str) -> dict:
    """
    Reports EM accuracy at ±0.5% around the primary threshold.
    Pre-specified to prevent post-hoc threshold selection (Section 6.4).
    """
    primary = TOLERANCE[_normalise_kpi(kpi)] * 100
    thresholds = [primary - 0.5, primary, primary + 0.5]
    results    = {}

    react_records    = [r for r in records if r.condition == "react"]
    baseline_records = [r for r in records if r.condition == "baseline"]

    for t in thresholds:
        react_correct    = sum(1 for r in react_records
                               if exact_match_at_threshold(r.predicted, r.ground_truth, t))
        baseline_correct = sum(1 for r in baseline_records
                               if exact_match_at_threshold(r.predicted, r.ground_truth, t))
        n                = len(react_records)
        results[f"{t}%"] = {
            "react_accuracy":    round(react_correct / n, 3) if n else 0,
            "baseline_accuracy": round(baseline_correct / n, 3) if n else 0,
        }

    return results


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def run_evaluation(
    records: list[ExtractionRecord],
    ground_truth: dict,
    kappa_session_1: Optional[list] = None,
    kappa_session_2: Optional[list] = None,
) -> FullReport:
    """
    Runs the complete evaluation pipeline over all records.

    records:      List of ExtractionRecord (one per company × KPI × condition)
    ground_truth: {"Apple Inc._revenue": 383285.0, ...}
    kappa_session_1/2: lists of faithfulness scores from two rating sessions
    """
    report = FullReport()
    kpis   = ["revenue", "net_income", "debt_to_equity"]

    for kpi in kpis:
        kpi_records  = [r for r in records if _normalise_kpi(r.kpi) == kpi]
        bl_records   = [r for r in kpi_records if r.condition == "baseline"]
        rc_records   = [r for r in kpi_records if r.condition == "react"]

        n             = len(bl_records)
        bl_correct    = 0
        rc_correct    = 0
        b             = 0   # McNemar: baseline right, react wrong
        c             = 0   # McNemar: react right, baseline wrong
        safety_counts = {
            "TRUE_POSITIVE":       0,
            "LUCKY_HALLUCINATION": 0,
            "FAITHFUL_FAILURE":    0,
            "TRUE_HALLUCINATION":  0,
            "UNSCORED":            0,
        }

        for bl, rc in zip(bl_records, rc_records):
            gt    = ground_truth.get(f"{bl.company}_{_normalise_kpi(kpi)}")
            if gt is None:
                continue

            bl_em = exact_match(bl.predicted, gt, kpi)
            rc_em = exact_match(rc.predicted, gt, kpi)

            if bl_em: bl_correct += 1
            if rc_em: rc_correct += 1

            if bl_em and not rc_em: b += 1
            if rc_em and not bl_em: c += 1

            cell = classify_safety_cell(rc_em, rc.faithfulness_score, "react")
            safety_counts[cell] += 1

        mcnemar   = mcnemar_test(b, c)
        bl_ci     = wilson_ci(bl_correct, n)
        rc_ci     = wilson_ci(rc_correct, n)
        rob_check = robustness_check(kpi_records, kpi)

        report.by_kpi[kpi] = {
            "n":                  n,
            "baseline_correct":   bl_correct,
            "react_correct":      rc_correct,
            "baseline_accuracy":  round(bl_correct / n, 3) if n else 0,
            "react_accuracy":     round(rc_correct / n, 3) if n else 0,
            "baseline_95ci":      bl_ci,
            "react_95ci":         rc_ci,
            "mcnemar":            mcnemar,
            "significant_at_bonferroni": mcnemar["p"] < BONFERRONI_ALPHA if mcnemar["p"] < 1 else False,
            "safety_matrix":      safety_counts,
            "robustness_check":   rob_check,
        }

    # Overall summary across all 3 KPIs
    total_n         = sum(v["n"] for v in report.by_kpi.values())
    total_bl_correct = sum(v["baseline_correct"] for v in report.by_kpi.values())
    total_rc_correct = sum(v["react_correct"] for v in report.by_kpi.values())

    report.overall = {
        "total_pairs":        total_n,
        "baseline_accuracy":  round(total_bl_correct / total_n, 3) if total_n else 0,
        "react_accuracy":     round(total_rc_correct / total_n, 3) if total_n else 0,
        "baseline_95ci":      wilson_ci(total_bl_correct, total_n),
        "react_95ci":         wilson_ci(total_rc_correct, total_n),
        "bonferroni_alpha":   BONFERRONI_ALPHA,
    }

    # Cohen's Kappa (only if rating sessions provided)
    if kappa_session_1 and kappa_session_2:
        report.kappa = cohens_kappa(kappa_session_1, kappa_session_2)
    else:
        report.kappa = {
            "kappa": None,
            "note": "No kappa data provided. Run calibration phase (Section 7.4) and re-run."
        }

    return report


# ============================================================================
# Helpers
# ============================================================================

def _normalise_kpi(kpi: str) -> str:
    return kpi.lower().replace(" ", "_").replace("-to-", "_to_").replace("-", "_")


def load_traces_as_records(
    react_dir:    str = "traces",
    baseline_dir: str = "traces_baseline",
) -> list[ExtractionRecord]:
    """
    Loads all saved trace JSON files and converts them to ExtractionRecord objects.
    Call this after running agent.py and baseline.py on all 40 companies.
    """
    records = []

    for condition, directory in [("react", react_dir), ("baseline", baseline_dir)]:
        if not os.path.exists(directory):
            continue
        for fname in os.listdir(directory):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(directory, fname), encoding="utf-8") as f:
                data = json.load(f)
            records.append(ExtractionRecord(
                company=data["company"],
                kpi=data["kpi"],
                condition=condition,
                predicted=data.get("answer"),
                ground_truth=0.0,   # Will be filled from ground_truth.csv
                status=data.get("status", "unknown"),
                faithfulness_score=data.get("faithfulness_score"),
            ))

    return records


# ============================================================================
# Demo — runs with synthetic data to show the full output
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  Evaluator — Full Metrics Demo")
    print("="*62 + "\n")

    # Synthetic records simulating 3 companies × 3 KPIs × 2 conditions
    # In the real study: load from trace JSON files via load_traces_as_records()
    DEMO_RECORDS = [
        # Apple Inc.
        ExtractionRecord("Apple Inc.", "Revenue",        "baseline", 383285,  383285.0, "success", 3),
        ExtractionRecord("Apple Inc.", "Revenue",        "react",    383285,  383285.0, "success", 5),
        ExtractionRecord("Apple Inc.", "Net Income",     "baseline", 99803,    96995.0, "success", 2),  # wrong — parametric memory
        ExtractionRecord("Apple Inc.", "Net Income",     "react",    96995,    96995.0, "success", 5),
        ExtractionRecord("Apple Inc.", "Debt-to-Equity", "baseline", None,      4.673,  "parse_error", 1),
        ExtractionRecord("Apple Inc.", "Debt-to-Equity", "react",    4.673,     4.673,  "success", 5),
        # Microsoft (synthetic)
        ExtractionRecord("Microsoft",  "Revenue",        "baseline", 211915,  211915.0, "success", 3),
        ExtractionRecord("Microsoft",  "Revenue",        "react",    211915,  211915.0, "success", 4),
        ExtractionRecord("Microsoft",  "Net Income",     "baseline", 72361,    72361.0, "success", 3),
        ExtractionRecord("Microsoft",  "Net Income",     "react",    72361,    72361.0, "success", 5),
        ExtractionRecord("Microsoft",  "Debt-to-Equity", "baseline", 2.1,       2.08,   "success", 2),
        ExtractionRecord("Microsoft",  "Debt-to-Equity", "react",    2.08,      2.08,   "success", 4),
        # Amazon (synthetic)
        ExtractionRecord("Amazon",     "Revenue",        "baseline", 554028,  554028.0, "success", 3),
        ExtractionRecord("Amazon",     "Revenue",        "react",    554028,  554028.0, "success", 5),
        ExtractionRecord("Amazon",     "Net Income",     "baseline", 29134,    30425.0, "success", 2),  # wrong
        ExtractionRecord("Amazon",     "Net Income",     "react",    30425,    30425.0, "success", 5),
        ExtractionRecord("Amazon",     "Debt-to-Equity", "baseline", None,       1.85,  "parse_error", 1),
        ExtractionRecord("Amazon",     "Debt-to-Equity", "react",    1.85,       1.85,  "success", 5),
    ]

    DEMO_GROUND_TRUTH = {
        "Apple Inc._revenue":        383285.0,
        "Apple Inc._net_income":      96995.0,
        "Apple Inc._debt_to_equity":     4.673,
        "Microsoft_revenue":         211915.0,
        "Microsoft_net_income":       72361.0,
        "Microsoft_debt_to_equity":      2.08,
        "Amazon_revenue":            554028.0,
        "Amazon_net_income":          30425.0,
        "Amazon_debt_to_equity":         1.85,
    }

    # Update ground truths into records
    for r in DEMO_RECORDS:
        key = f"{r.company}_{_normalise_kpi(r.kpi)}"
        r.ground_truth = DEMO_GROUND_TRUTH.get(key, 0.0)

    # Demo kappa — simulating two scoring sessions with near-perfect agreement
    KAPPA_S1 = [5, 2, 5, 1, 5, 3, 4, 5, 2, 4, 5, 1]
    KAPPA_S2 = [5, 2, 5, 1, 5, 3, 4, 4, 2, 4, 5, 2]

    report = run_evaluation(DEMO_RECORDS, DEMO_GROUND_TRUTH, KAPPA_S1, KAPPA_S2)

    # ── Print results ───────────────────────────────────────────────────────
    print("PER-KPI RESULTS")
    print("-"*62)
    for kpi, data in report.by_kpi.items():
        print(f"\n  KPI: {kpi.upper()}")
        print(f"    N pairs:           {data['n']}")
        print(f"    Baseline accuracy: {data['baseline_accuracy']:.1%}  95CI: {data['baseline_95ci']}")
        print(f"    ReAct accuracy:    {data['react_accuracy']:.1%}  95CI: {data['react_95ci']}")
        mn = data["mcnemar"]
        print(f"    McNemar's Test:    chi2={mn['chi2']}, p={mn['p']}  "
              f"({'SIGNIFICANT' if data['significant_at_bonferroni'] else 'not significant'} at alpha*={BONFERRONI_ALPHA})")
        print(f"    {mn['note']}")
        print(f"    Safety Matrix:     {data['safety_matrix']}")
        print(f"    Robustness Check:  {data['robustness_check']}")

    print(f"\n{'='*62}")
    print("OVERALL SUMMARY")
    print(f"{'='*62}")
    ov = report.overall
    print(f"  Total pairs:        {ov['total_pairs']}")
    print(f"  Baseline accuracy:  {ov['baseline_accuracy']:.1%}  95CI: {ov['baseline_95ci']}")
    print(f"  ReAct accuracy:     {ov['react_accuracy']:.1%}  95CI: {ov['react_95ci']}")
    print(f"  Bonferroni alpha*:  {ov['bonferroni_alpha']}")

    print(f"\n{'='*62}")
    print("COHEN'S KAPPA (Faithfulness reliability)")
    print(f"{'='*62}")
    k = report.kappa
    print(f"  Kappa:          {k['kappa']}")
    print(f"  Interpretation: {k.get('interpretation', 'N/A')}")
    print(f"  Meets threshold: {k.get('meets_threshold', 'N/A')}")
    print(f"  {k['note']}\n")
