"""
Hallucination Analysis of LLM-Generated Explanations in BARO+

Analyzes the faithfulness of LLM reasoning by:
1. Extracting factual claims (percentages, values, timestamps, log references) from reasoning text
2. Reconstructing the exact context the LLM saw (deterministic via random_state=42)
3. Verifying extracted claims against the reconstructed context
4. Categorizing wrong predictions by error pattern

Taxonomy:
  A1: Metric fabrication - LLM cites a % or value not in context
  A2: Log fabrication - LLM references a log message not in context
  A3: Temporal fabrication - LLM cites a timestamp not in context
  B1: Causal confusion - LLM cites real data but wrong causal direction
  B2: Evidence misweighting - LLM overweights misleading signal (e.g., Redis diskio)
  C1: GT not in candidates - BARO pre-filtering failure, not LLM error
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from baro.context_builder import ContextBuilder
from baro.root_cause_analysis import robust_scorer_dict
from data_loader import DataLoader


# ──────────────────────────────────────────────────────────────────
# Claim Extractor
# ──────────────────────────────────────────────────────────────────

class ClaimExtractor:
    """Extract factual claims from LLM reasoning text."""

    # Percentage patterns: +518.7%, -12.3%, 769.5%
    PCT_RE = re.compile(r'([+-]?\d+\.?\d*)%')

    # Infinity patterns: +inf%, inf%
    INF_RE = re.compile(r'\+?inf%?', re.IGNORECASE)

    # Value-pair patterns: 15.2 → 78.3  or  15.2 to 78.3
    VALUEPAIR_RE = re.compile(r'(\d+\.?\d*)\s*(?:→|->|to)\s*(\d+\.?\d*)')

    # Timestamp patterns: 10-digit unix timestamps starting with 17
    TIMESTAMP_RE = re.compile(r'\b(17\d{8})\b')

    # Exception/Error class names: NullPointerException, ConnectionError
    EXCEPTION_RE = re.compile(r'\b(\w+(?:Exception|Error|Fault|Timeout))\b')

    # Quoted error phrases: 'no healthy upstream', "upstream connect error"
    QUOTED_RE = re.compile(r"""['"]([^'"]{5,80})['"]""")

    @staticmethod
    def extract(reasoning: str) -> Dict[str, list]:
        """Extract all factual claims from reasoning text.

        Returns:
            {
                "percentages": [518.7, -12.3, ...],
                "infinity": True/False,
                "value_pairs": [(15.2, 78.3), ...],
                "timestamps": [1732365557, ...],
                "exceptions": ["NullPointerException", ...],
                "quoted_phrases": ["no healthy upstream", ...],
            }
        """
        claims = {
            "percentages": [],
            "infinity": False,
            "value_pairs": [],
            "timestamps": [],
            "exceptions": [],
            "quoted_phrases": [],
        }

        # Percentages (exclude infinity matches)
        for m in ClaimExtractor.PCT_RE.finditer(reasoning):
            try:
                val = float(m.group(1))
                claims["percentages"].append(val)
            except ValueError:
                pass

        # Infinity
        if ClaimExtractor.INF_RE.search(reasoning):
            claims["infinity"] = True

        # Value pairs
        for m in ClaimExtractor.VALUEPAIR_RE.finditer(reasoning):
            try:
                v1, v2 = float(m.group(1)), float(m.group(2))
                claims["value_pairs"].append((v1, v2))
            except ValueError:
                pass

        # Timestamps
        for m in ClaimExtractor.TIMESTAMP_RE.finditer(reasoning):
            claims["timestamps"].append(int(m.group(1)))

        # Exception names
        for m in ClaimExtractor.EXCEPTION_RE.finditer(reasoning):
            claims["exceptions"].append(m.group(1))

        # Quoted phrases
        for m in ClaimExtractor.QUOTED_RE.finditer(reasoning):
            claims["quoted_phrases"].append(m.group(1))

        return claims


# ──────────────────────────────────────────────────────────────────
# Context Reconstruction
# ──────────────────────────────────────────────────────────────────

def reconstruct_context(
    dataset: dict,
    k: int = 5,
) -> Dict[str, any]:
    """Reconstruct the exact context the LLM saw for a given case.

    Uses the same pipeline as run_bench.py: robust_scorer_dict → aggregate_by_service
    → build_metrics/logs/traces_context.

    Returns dict with keys: metrics_ctx, logs_ctx, traces_ctx, candidates, inject_time
    """
    metrics = dataset["metrics"]
    inject_time = dataset["inject_time"]
    logs = dataset.get("logs")
    traces = dataset.get("traces")

    # Step 1: BARO scoring
    ranked = robust_scorer_dict(metrics, inject_time)

    # Step 2: Aggregate to service level
    services = ContextBuilder.aggregate_by_service(ranked)
    top_k = services[:k]
    top_k_names = [s for s, _ in top_k]

    # Step 3: Build context (same parameters as run_bench.py)
    metrics_ctx = ContextBuilder.build_metrics_context(
        metrics, top_k_names, inject_time
    )
    logs_ctx = ContextBuilder.build_logs_context(
        logs, top_k_names, inject_time
    )
    traces_ctx = ContextBuilder.build_traces_context(
        traces, top_k_names, inject_time
    )

    return {
        "metrics_ctx": metrics_ctx,
        "logs_ctx": logs_ctx,
        "traces_ctx": traces_ctx,
        "candidates": top_k,
        "inject_time": inject_time,
    }


# ──────────────────────────────────────────────────────────────────
# Claim Verifier
# ──────────────────────────────────────────────────────────────────

class ClaimVerifier:
    """Verify extracted claims against reconstructed context."""

    @staticmethod
    def verify(
        claims: Dict[str, list],
        ctx: Dict[str, any],
    ) -> Dict[str, dict]:
        """Verify each claim type against the context.

        Returns:
            {
                "percentages": {"total": N, "verified": M},
                "value_pairs": {"total": N, "verified": M},
                "timestamps": {"total": N, "verified": M},
                "log_refs":   {"total": N, "verified": M},
                "infinity":   {"total": N, "verified": M},
            }
        """
        metrics_ctx = ctx["metrics_ctx"]
        logs_ctx = ctx["logs_ctx"]
        traces_ctx = ctx["traces_ctx"]
        inject_time = ctx["inject_time"]

        results = {}

        # ── Percentage claims ──
        context_pcts = ClaimVerifier._collect_context_percentages(
            metrics_ctx, traces_ctx
        )
        pct_total = len(claims["percentages"])
        pct_verified = 0
        for pct in claims["percentages"]:
            if ClaimVerifier._match_percentage(pct, context_pcts):
                pct_verified += 1
        results["percentages"] = {"total": pct_total, "verified": pct_verified}

        # ── Value-pair claims ──
        context_pairs = ClaimVerifier._collect_context_value_pairs(
            metrics_ctx, traces_ctx
        )
        vp_total = len(claims["value_pairs"])
        vp_verified = 0
        for v1, v2 in claims["value_pairs"]:
            if ClaimVerifier._match_value_pair(v1, v2, context_pairs):
                vp_verified += 1
        results["value_pairs"] = {"total": vp_total, "verified": vp_verified}

        # ── Timestamp claims ──
        context_timestamps = ClaimVerifier._collect_context_timestamps(
            logs_ctx, traces_ctx, inject_time
        )
        ts_total = len(claims["timestamps"])
        ts_verified = 0
        for ts in claims["timestamps"]:
            if ts in context_timestamps:
                ts_verified += 1
        results["timestamps"] = {"total": ts_total, "verified": ts_verified}

        # ── Log reference claims (exceptions + quoted phrases) ──
        context_log_text = ClaimVerifier._collect_context_log_text(logs_ctx)
        log_ref_total = len(claims["exceptions"]) + len(claims["quoted_phrases"])
        log_ref_verified = 0
        for exc in claims["exceptions"]:
            if ClaimVerifier._match_log_ref(exc, context_log_text):
                log_ref_verified += 1
        for phrase in claims["quoted_phrases"]:
            if ClaimVerifier._match_log_ref(phrase, context_log_text):
                log_ref_verified += 1
        results["log_refs"] = {"total": log_ref_total, "verified": log_ref_verified}

        # ── Infinity claims ──
        has_inf_in_context = ClaimVerifier._context_has_infinity(metrics_ctx, traces_ctx)
        if claims["infinity"]:
            results["infinity"] = {
                "total": 1,
                "verified": 1 if has_inf_in_context else 0,
            }
        else:
            results["infinity"] = {"total": 0, "verified": 0}

        return results

    # ── Helpers ──

    @staticmethod
    def _collect_context_percentages(
        metrics_ctx: Dict, traces_ctx: Optional[Dict]
    ) -> Set[float]:
        """Collect all percentage values from context."""
        pcts = set()
        if metrics_ctx:
            for service, mets in metrics_ctx.items():
                for mtype, stats in mets.items():
                    cp = stats.get("change_pct")
                    if cp is not None and not math.isinf(cp):
                        pcts.add(float(cp))

        # Trace latency change percentages (computed in _format_traces_context)
        if traces_ctx:
            for service, stats in traces_ctx.items():
                pre = stats.get("avg_duration_pre", 0)
                post = stats.get("avg_duration_post", 0)
                if pre > 0 and post > 0:
                    change = ((post - pre) / pre) * 100
                    pcts.add(round(change, 1))

        return pcts

    @staticmethod
    def _match_percentage(claimed: float, context_pcts: Set[float]) -> bool:
        """Match if within 1.0 absolute difference of any context percentage."""
        for cp in context_pcts:
            if abs(claimed - cp) <= 1.0:
                return True
        return False

    @staticmethod
    def _collect_context_value_pairs(
        metrics_ctx: Dict, traces_ctx: Optional[Dict]
    ) -> List[Tuple[float, float]]:
        """Collect all (pre, post) value pairs from context."""
        pairs = []
        if metrics_ctx:
            for service, mets in metrics_ctx.items():
                for mtype, stats in mets.items():
                    pre = stats.get("pre_mean")
                    post = stats.get("post_mean")
                    if pre is not None and post is not None:
                        pairs.append((float(pre), float(post)))
                    # Also add post_max as a potential "post" value
                    post_max = stats.get("post_max")
                    if pre is not None and post_max is not None:
                        pairs.append((float(pre), float(post_max)))

        if traces_ctx:
            for service, stats in traces_ctx.items():
                pre_dur = stats.get("avg_duration_pre", 0)
                post_dur = stats.get("avg_duration_post", 0)
                if pre_dur > 0 or post_dur > 0:
                    pairs.append((float(pre_dur), float(post_dur)))
                # error rates
                pre_err = stats.get("error_rate_pre", 0)
                post_err = stats.get("error_rate_post", 0)
                pairs.append((float(pre_err), float(post_err)))

        return pairs

    @staticmethod
    def _match_value_pair(
        v1: float, v2: float, context_pairs: List[Tuple[float, float]]
    ) -> bool:
        """Match if within 5% relative of any context value pair."""
        for cp1, cp2 in context_pairs:
            if ClaimVerifier._close_enough(v1, cp1) and ClaimVerifier._close_enough(v2, cp2):
                return True
        return False

    @staticmethod
    def _close_enough(claimed: float, actual: float, rel_tol: float = 0.05) -> bool:
        """Check if claimed value is within rel_tol of actual value."""
        if actual == 0:
            return abs(claimed) < 0.01
        return abs(claimed - actual) / max(abs(actual), 1e-9) <= rel_tol

    @staticmethod
    def _collect_context_timestamps(
        logs_ctx: Optional[Dict],
        traces_ctx: Optional[Dict],
        inject_time: int,
    ) -> Set[int]:
        """Collect all timestamps appearing in context."""
        timestamps = {inject_time}
        if logs_ctx:
            ts_re = re.compile(r'\[(\d{10})\]')
            for service, messages in logs_ctx.items():
                for msg in messages:
                    for m in ts_re.finditer(msg):
                        timestamps.add(int(m.group(1)))
        return timestamps

    @staticmethod
    def _collect_context_log_text(logs_ctx: Optional[Dict]) -> str:
        """Concatenate all log messages into one searchable string."""
        if not logs_ctx:
            return ""
        parts = []
        for service, messages in logs_ctx.items():
            for msg in messages:
                parts.append(msg)
        return "\n".join(parts)

    @staticmethod
    def _match_log_ref(ref: str, context_log_text: str) -> bool:
        """Check if a log reference (exception name or phrase) appears in context."""
        return ref.lower() in context_log_text.lower()

    @staticmethod
    def _context_has_infinity(
        metrics_ctx: Dict, traces_ctx: Optional[Dict]
    ) -> bool:
        """Check if any metric has change_pct == inf."""
        if metrics_ctx:
            for service, mets in metrics_ctx.items():
                for mtype, stats in mets.items():
                    cp = stats.get("change_pct")
                    if cp is not None and math.isinf(cp):
                        return True
        return False


# ──────────────────────────────────────────────────────────────────
# Error Categorization (for wrong predictions)
# ──────────────────────────────────────────────────────────────────

def categorize_wrong_case(
    dataset_id: str,
    ground_truth: str,
    predicted: str,
    reasoning: str,
    candidates: List[Tuple[str, float]],
) -> str:
    """Categorize why a prediction was wrong.

    Returns one of: "C1", "B2", "B1"
    """
    candidate_names = [s for s, _ in candidates]

    # C1: Ground truth not in top-5 candidates (BARO pre-filtering failure)
    if ground_truth not in candidate_names:
        return "C1"

    # B2: Redis distraction - predicted redis and reasoning mentions diskio
    reasoning_lower = reasoning.lower()
    if predicted == "redis" and ("diskio" in reasoning_lower or "disk" in reasoning_lower):
        return "B2"

    # B1: Causal confusion (general case)
    return "B1"


# ──────────────────────────────────────────────────────────────────
# Main Analysis
# ──────────────────────────────────────────────────────────────────

# Map output directory names to display names
MODEL_DIR_MAP = {
    "baro+opus": "Opus 4.5",
    "baro+opus4.6": "Opus 4.6",
    "baro+sonnet": "Sonnet 4.5",
    "baro+sonnet4.6": "Sonnet 4.6",
    "baro+sonnet-m": "Sonnet 4.5 (M)",
    "baro+sonnet-ml": "Sonnet 4.5 (ML)",
    "baro+sonnet4.6-m": "Sonnet 4.6 (M)",
    "baro+sonnet4.6-ml": "Sonnet 4.6 (ML)",
    "baro+haiku": "Haiku 4.5",
    "baro+gpt4o": "GPT-4o",
    "baro+gpt4.1": "GPT-4.1",
    "baro+gpt5.2": "GPT-5.2",
    "baro+o4-mini": "o4-mini",
    "baro+gemini-pro": "Gemini Pro",
    "baro+gemini-flash": "Gemini Flash",
    "baro+gemini-flash-lite": "Gemini Flash Lite",
}

DATASET_PATTERN = re.compile(
    r"^re(\d)(ob|ss|tt)_(.+?)_(cpu|mem|delay|loss|disk|socket|f\d)_(\d+)$"
)


def discover_models(output_dir: str) -> List[str]:
    """Find all baro+ model output directories with 90 files."""
    models = []
    for name in sorted(os.listdir(output_dir)):
        dirpath = os.path.join(output_dir, name)
        if not os.path.isdir(dirpath) or not name.startswith("baro+"):
            continue
        # Count JSON files
        n = sum(1 for f in os.listdir(dirpath) if f.endswith(".json"))
        if n >= 90:
            models.append(name)
    return models


def load_model_outputs(output_dir: str, model_dir: str) -> List[dict]:
    """Load all JSON output files for a model."""
    dirpath = os.path.join(output_dir, model_dir)
    results = []
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(dirpath, fname)) as f:
            results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hallucinations in BARO+ LLM explanations"
    )
    parser.add_argument(
        "--data-path", default="rcaeval-data",
        help="Path to RCAEval dataset directory"
    )
    parser.add_argument(
        "--output", default="output/hallucination_analysis.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory containing model output JSONs"
    )
    args = parser.parse_args()

    # Load dataset (RE3 = 90 code-level fault cases)
    print("Loading RCAEval RE3 datasets...")
    loader = DataLoader(args.data_path, filter_patterns=["re3"])
    datasets = {}
    for ds in loader:
        datasets[ds["dataset_id"]] = ds
    print(f"  Loaded {len(datasets)} datasets")

    # Discover models
    models = discover_models(args.output_dir)
    print(f"  Found {len(models)} models with 90+ outputs: {models}")

    # ── Per-case analysis ──
    all_rows = []           # CSV rows
    model_claim_stats = {}  # model → aggregated stats
    model_error_cats = {}   # model → {B1: n, B2: n, C1: n}

    for model_dir in models:
        display_name = MODEL_DIR_MAP.get(model_dir, model_dir)
        print(f"\nAnalyzing {display_name} ({model_dir})...")

        outputs = load_model_outputs(args.output_dir, model_dir)

        # Aggregate counters
        total_claims = defaultdict(lambda: {"total": 0, "verified": 0})
        error_cats = defaultdict(int)
        n_correct = 0
        n_wrong = 0

        for out in outputs:
            dataset_id = out["dataset"]
            reasoning = out.get("llm_metadata", {}).get("reasoning", "")
            error = out.get("llm_metadata", {}).get("error")

            if not reasoning or error:
                continue

            # Parse ground truth
            m = DATASET_PATTERN.match(dataset_id)
            if not m:
                continue
            gt_service = m.group(3)

            # Get prediction
            top1 = out["results"][0]["candidate"] if out["results"] else None

            # Look up dataset
            if dataset_id not in datasets:
                continue
            ds = datasets[dataset_id]

            # Reconstruct context
            ctx = reconstruct_context(ds)

            # Extract claims
            claims = ClaimExtractor.extract(reasoning)

            # Verify claims
            verification = ClaimVerifier.verify(claims, ctx)

            # Aggregate
            for claim_type, stats in verification.items():
                total_claims[claim_type]["total"] += stats["total"]
                total_claims[claim_type]["verified"] += stats["verified"]

            # Correct/wrong + categorization
            is_correct = (top1 == gt_service)
            if is_correct:
                n_correct += 1
                error_cat = ""
            else:
                n_wrong += 1
                error_cat = categorize_wrong_case(
                    dataset_id, gt_service, top1, reasoning, ctx["candidates"]
                )
                error_cats[error_cat] += 1

            # Compute per-case totals
            case_total = sum(v["total"] for v in verification.values())
            case_verified = sum(v["verified"] for v in verification.values())

            all_rows.append({
                "model": display_name,
                "model_dir": model_dir,
                "dataset_id": dataset_id,
                "ground_truth": gt_service,
                "predicted": top1,
                "correct": is_correct,
                "error_category": error_cat,
                "total_claims": case_total,
                "verified_claims": case_verified,
                "pct_total": verification["percentages"]["total"],
                "pct_verified": verification["percentages"]["verified"],
                "vp_total": verification["value_pairs"]["total"],
                "vp_verified": verification["value_pairs"]["verified"],
                "ts_total": verification["timestamps"]["total"],
                "ts_verified": verification["timestamps"]["verified"],
                "log_total": verification["log_refs"]["total"],
                "log_verified": verification["log_refs"]["verified"],
                "inf_total": verification["infinity"]["total"],
                "inf_verified": verification["infinity"]["verified"],
                "reasoning": reasoning[:500],
            })

        model_claim_stats[model_dir] = dict(total_claims)
        model_error_cats[model_dir] = dict(error_cats)
        print(f"  Correct: {n_correct}, Wrong: {n_wrong}, Error cats: {dict(error_cats)}")

    # ── Write CSV ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = [
        "model", "model_dir", "dataset_id", "ground_truth", "predicted",
        "correct", "error_category", "total_claims", "verified_claims",
        "pct_total", "pct_verified", "vp_total", "vp_verified",
        "ts_total", "ts_verified", "log_total", "log_verified",
        "inf_total", "inf_verified", "reasoning",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written to {args.output} ({len(all_rows)} rows)")

    # ── Print summary tables ──
    print("\n" + "=" * 80)
    print("TABLE 1: Factual Claim Verification Summary")
    print("=" * 80)

    header = f"{'Model':<22} {'Total':>6} {'Metric%':>9} {'ValPair':>9} {'Tstamp':>9} {'LogRef':>9} {'Inf':>6} {'Overall':>9}"
    print(header)
    print("-" * len(header))

    for model_dir in models:
        display = MODEL_DIR_MAP.get(model_dir, model_dir)
        stats = model_claim_stats[model_dir]

        grand_total = sum(v["total"] for v in stats.values())
        grand_verified = sum(v["verified"] for v in stats.values())

        def pct_str(key):
            t = stats.get(key, {}).get("total", 0)
            v = stats.get(key, {}).get("verified", 0)
            if t == 0:
                return "  -  "
            return f"{v}/{t}={v/t*100:.0f}%"

        overall = f"{grand_verified}/{grand_total}={grand_verified/grand_total*100:.1f}%" if grand_total > 0 else "-"

        print(f"{display:<22} {grand_total:>6} {pct_str('percentages'):>9} {pct_str('value_pairs'):>9} "
              f"{pct_str('timestamps'):>9} {pct_str('log_refs'):>9} {pct_str('infinity'):>6} {overall:>9}")

    print("\n" + "=" * 80)
    print("TABLE 2: Error Pattern Distribution (wrong cases only)")
    print("=" * 80)

    header2 = f"{'Model':<22} {'B1:Causal':>10} {'B2:Redis':>10} {'C1:Filter':>10} {'Total':>6}"
    print(header2)
    print("-" * len(header2))

    for model_dir in models:
        display = MODEL_DIR_MAP.get(model_dir, model_dir)
        cats = model_error_cats.get(model_dir, {})
        b1 = cats.get("B1", 0)
        b2 = cats.get("B2", 0)
        c1 = cats.get("C1", 0)
        total = b1 + b2 + c1
        print(f"{display:<22} {b1:>10} {b2:>10} {c1:>10} {total:>6}")

    # ── Print LaTeX tables ──
    print("\n" + "=" * 80)
    print("LATEX: Table 1 (Factual Claim Verification)")
    print("=" * 80)

    # Select representative models for the paper table
    representative = [
        "baro+opus", "baro+sonnet4.6-ml", "baro+gemini-flash",
    ]
    # Fall back to available models
    representative = [m for m in representative if m in model_claim_stats]
    if not representative:
        representative = models[:3]

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Factual claim verification in LLM explanations. Each claim (percentage, value, timestamp, log reference) extracted from the reasoning is checked against the reconstructed context.}")
    print(r"\label{tab:claim-verification}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Model & Claims & Metric\% & Value & Log Ref & Overall \\")
    print(r"\midrule")

    for model_dir in representative:
        display = MODEL_DIR_MAP.get(model_dir, model_dir)
        stats = model_claim_stats[model_dir]

        grand_total = sum(v["total"] for v in stats.values())
        grand_verified = sum(v["verified"] for v in stats.values())

        def latex_frac(key):
            t = stats.get(key, {}).get("total", 0)
            v = stats.get(key, {}).get("verified", 0)
            if t == 0:
                return "--"
            return f"{v/t*100:.1f}\\%"

        overall = f"{grand_verified/grand_total*100:.1f}\\%" if grand_total > 0 else "--"

        print(f"{display} & {grand_total} & {latex_frac('percentages')} & {latex_frac('value_pairs')} "
              f"& {latex_frac('log_refs')} & {overall} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("LATEX: Table 2 (Error Pattern Distribution)")
    print("=" * 80)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Distribution of error patterns in incorrect predictions (RE3, 90 cases per model).}")
    print(r"\label{tab:error-patterns}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Model & B1 & B2 & C1 & Total \\")
    print(r"\midrule")

    for model_dir in representative:
        display = MODEL_DIR_MAP.get(model_dir, model_dir)
        cats = model_error_cats.get(model_dir, {})
        b1 = cats.get("B1", 0)
        b2 = cats.get("B2", 0)
        c1 = cats.get("C1", 0)
        total = b1 + b2 + c1
        print(f"{display} & {b1} & {b2} & {c1} & {total} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── Print example explanations ──
    print("\n" + "=" * 80)
    print("EXAMPLE EXPLANATIONS")
    print("=" * 80)

    # Find good examples from representative models
    for model_dir in representative[:1]:  # Just use first representative model
        display = MODEL_DIR_MAP.get(model_dir, model_dir)
        model_rows = [r for r in all_rows if r["model_dir"] == model_dir]

        # Example 1: Correct & faithful
        correct_rows = [r for r in model_rows if r["correct"] and r["total_claims"] > 3]
        if correct_rows:
            ex = correct_rows[0]
            print(f"\n--- Example 1: Correct & Faithful ({display}) ---")
            print(f"Case: {ex['dataset_id']}, GT={ex['ground_truth']}, Pred={ex['predicted']}")
            print(f"Claims: {ex['verified_claims']}/{ex['total_claims']} verified")
            print(f"Reasoning: {ex['reasoning']}")

        # Example 2: Wrong but faithful (B2 - Redis distraction)
        b2_rows = [r for r in model_rows if r["error_category"] == "B2"]
        if b2_rows:
            ex = b2_rows[0]
            print(f"\n--- Example 2: Wrong but Faithful - B2 Redis ({display}) ---")
            print(f"Case: {ex['dataset_id']}, GT={ex['ground_truth']}, Pred={ex['predicted']}")
            print(f"Claims: {ex['verified_claims']}/{ex['total_claims']} verified")
            print(f"Reasoning: {ex['reasoning']}")

        # Example 3: Wrong - Causal confusion (B1)
        b1_rows = [r for r in model_rows if r["error_category"] == "B1"]
        if b1_rows:
            ex = b1_rows[0]
            print(f"\n--- Example 3: Wrong - Causal Confusion B1 ({display}) ---")
            print(f"Case: {ex['dataset_id']}, GT={ex['ground_truth']}, Pred={ex['predicted']}")
            print(f"Claims: {ex['verified_claims']}/{ex['total_claims']} verified")
            print(f"Reasoning: {ex['reasoning']}")

        # Example 4: C1 - GT not in candidates
        c1_rows = [r for r in model_rows if r["error_category"] == "C1"]
        if c1_rows:
            ex = c1_rows[0]
            print(f"\n--- Example 4: C1 - GT not in candidates ({display}) ---")
            print(f"Case: {ex['dataset_id']}, GT={ex['ground_truth']}, Pred={ex['predicted']}")
            print(f"Claims: {ex['verified_claims']}/{ex['total_claims']} verified")
            print(f"Reasoning: {ex['reasoning']}")


if __name__ == "__main__":
    main()
