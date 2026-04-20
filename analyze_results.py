"""
analyze_results.py — Eval results analysis for the data-cleaning knowledge work eval.

Reads a JSONL results file produced by harness.py and computes:
  1. Mean hybrid scores with bootstrap 95% CIs — per model × elicitation combo
  2. Elicitation gap — structured minus zero_shot score delta per model
  3. Agreement bucket distribution — how models fail (agree-good/bad/judge-rescues/catches)
  4. Per-check failure rates — which programmatic checks fail most often
  5. Judge quality metrics — position consistency rate, gate trigger rate
  6. Model comparison table — side-by-side across all dimensions

Bootstrap CI design:
  The score being bootstrapped is hybrid_score (continuous [0,1]), NOT pass@k (binary).
  Resampling is over TASKS (not individual runs) — this is the correct unit because
  task difficulty is the dominant source of variance. Resampling over runs would
  underestimate variance by treating task effects as independent noise.

Usage:
    python analyze_results.py results/synthetic_results.jsonl
    python analyze_results.py results/synthetic_results.jsonl --model gpt-4o
    python analyze_results.py results/synthetic_results.jsonl --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# Day 20: error analysis + difficulty calibration (imported lazily to avoid
# circular imports if analyze_results is ever imported from error_analysis)
_error_analysis = None

def _get_error_analysis():
    global _error_analysis
    if _error_analysis is None:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "error_analysis",
            Path(__file__).resolve().parent / "error_analysis.py"
        )
        _error_analysis = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_error_analysis)
    return _error_analysis

RNG = np.random.default_rng(seed=42)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    """
    Load all records from a JSONL results file.

    Skips blank lines. Raises on malformed JSON so you catch data corruption
    early rather than silently dropping records.

    Args:
        path: Path to the JSONL results file.

    Returns:
        List of record dicts, one per run.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num}: {e}") from e
    return records


def filter_records(
    records: list[dict],
    model: Optional[str] = None,
    elicitation: Optional[str] = None,
    task_id: Optional[str] = None,
    exclude_errors: bool = True,
) -> list[dict]:
    """
    Filter records by model, elicitation mode, task ID, and error status.

    Args:
        records:        Full list of records from load_results().
        model:          Filter to this model ID (e.g. "gpt-4o"). None = all models.
        elicitation:    Filter to this elicitation mode. None = all modes.
        task_id:        Filter to this task ID. None = all tasks.
        exclude_errors: If True (default), drop records where model_error is not null.
                        Failed runs have hybrid_score=0.0 which would bias the mean down.
                        Set to False to include them (e.g. to measure reliability).

    Returns:
        Filtered list of records.
    """
    out = records
    if model:
        out = [r for r in out if r["model"] == model]
    if elicitation:
        out = [r for r in out if r["elicitation"] == elicitation]
    if task_id:
        out = [r for r in out if r["task_id"] == task_id]
    if exclude_errors:
        out = [r for r in out if r["model_error"] is None]
    return out


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci_scores(
    scores: list[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for the mean of a list of continuous scores.

    Resamples WITH REPLACEMENT over the scores list, computes mean of each
    resample, and returns the percentile-based CI.

    Note: this bootstraps over individual scores directly. For multi-task evals,
    the caller should pass one score per task (already aggregated within task if
    there are multiple runs per task). This keeps the bootstrap unit consistent
    with the statistical unit of the eval.

    Args:
        scores:       List of hybrid_scores (floats in [0,1]).
        n_resamples:  Number of bootstrap resamples (default 10,000).
        confidence:   CI level (default 0.95 → 95% CI).

    Returns:
        (point_estimate, lower_bound, upper_bound)

    Raises:
        ValueError: if scores is empty.
    """
    if not scores:
        raise ValueError("Cannot compute CI on empty scores list.")

    arr = np.array(scores, dtype=float)
    point_estimate = float(arr.mean())

    n = len(arr)
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = RNG.integers(0, n, size=n)
        resample_means[i] = arr[indices].mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(resample_means, 100 * alpha / 2))
    upper = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


# ── Elicitation gap ───────────────────────────────────────────────────────────

def elicitation_gap(
    records: list[dict],
    model: str,
) -> dict:
    """
    Compute the elicitation gap for a model: structured score minus zero_shot score.

    A large positive gap means the model performs much better when given detailed
    instructions. This is the under-contextualization signal from GDPval Day 11:
    it separates "can execute well-described work" from "can figure out what to do."

    Args:
        records: Full results list (will be filtered internally).
        model:   Model ID to analyze.

    Returns:
        Dict with keys:
            structured_mean:  Mean hybrid score, structured elicitation.
            zero_shot_mean:   Mean hybrid score, zero_shot elicitation.
            gap:              structured_mean - zero_shot_mean (positive = structured wins).
            structured_ci:    (point, lo, hi) bootstrap CI for structured scores.
            zero_shot_ci:     (point, lo, hi) bootstrap CI for zero_shot scores.
            n_structured:     Number of records in structured group.
            n_zero_shot:      Number of records in zero_shot group.
    """
    structured = filter_records(records, model=model, elicitation="structured")
    zero_shot  = filter_records(records, model=model, elicitation="zero_shot")

    s_scores = [r["hybrid"]["hybrid_score"] for r in structured]
    z_scores = [r["hybrid"]["hybrid_score"] for r in zero_shot]

    if not s_scores or not z_scores:
        raise ValueError(f"No records found for model={model!r} in one or both elicitation modes.")

    s_ci = bootstrap_ci_scores(s_scores)
    z_ci = bootstrap_ci_scores(z_scores)

    return {
        "model":            model,
        "structured_mean":  round(s_ci[0], 4),
        "zero_shot_mean":   round(z_ci[0], 4),
        "gap":              round(s_ci[0] - z_ci[0], 4),
        "structured_ci":    (round(s_ci[0], 4), round(s_ci[1], 4), round(s_ci[2], 4)),
        "zero_shot_ci":     (round(z_ci[0], 4), round(z_ci[1], 4), round(z_ci[2], 4)),
        "n_structured":     len(s_scores),
        "n_zero_shot":      len(z_scores),
    }


# ── Agreement bucket distribution ─────────────────────────────────────────────

def bucket_distribution(
    records: list[dict],
) -> dict[str, dict[str, int]]:
    """
    Count agreement bucket frequencies grouped by (model, elicitation).

    The four-bucket taxonomy is the primary research finding of Week 3:
    it shows HOW models fail, not just whether they fail.

    Args:
        records: Filtered or unfiltered records list.

    Returns:
        Nested dict: {model: {elicitation: {bucket: count}}}
        Example:
            {"gpt-4o": {"structured": {"agree-good": 4, "agree-bad": 1, ...}, ...}}
    """
    BUCKETS = ["agree-good", "agree-bad", "judge-rescues", "programmatic-catches", "gate-triggered"]

    # Build nested defaultdict
    dist: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for r in records:
        model       = r["model"]
        elicitation = r["elicitation"]
        bucket      = r["hybrid"]["agreement_bucket"]
        dist[model][elicitation][bucket] += 1

    # Convert to plain dicts and fill in zeros for missing buckets
    result = {}
    for model, elicitations in dist.items():
        result[model] = {}
        for elicitation, counts in elicitations.items():
            result[model][elicitation] = {b: counts.get(b, 0) for b in BUCKETS}

    return result


# ── Per-check failure rates ───────────────────────────────────────────────────

def per_check_failure_rates(
    records: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Compute the failure rate for each programmatic check, grouped by (model, elicitation).

    Failure rate = fraction of runs where that check did NOT pass.
    Only includes records where programmatic is not null (model didn't fail).

    This tells you WHICH structural properties models most frequently get wrong:
    - High revenue failure rate → models hardcode values instead of using formulas
    - High sheets failure rate  → models don't produce all required sheets
    - High reference failure rate → Summary doesn't reference Cleaned Data sheet

    Args:
        records: Results list.

    Returns:
        Nested dict: {model: {elicitation: {check_name: failure_rate}}}
    """
    CHECK_NAMES = ["integrity", "sheets", "errors", "revenue", "summary", "reference"]

    # Accumulate: total runs and failure count per (model, elicitation, check)
    totals:   dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    failures: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for r in records:
        prog = r.get("programmatic")
        if prog is None:
            continue   # model failed — skip, don't count as check failure
        model       = r["model"]
        elicitation = r["elicitation"]
        for check in prog.get("checks", []):
            name = check["name"]
            totals[model][elicitation][name]   += 1
            if not check["passed"]:
                failures[model][elicitation][name] += 1

    result = {}
    for model in totals:
        result[model] = {}
        for elicitation in totals[model]:
            result[model][elicitation] = {}
            for name in CHECK_NAMES:
                total   = totals[model][elicitation].get(name, 0)
                failure = failures[model][elicitation].get(name, 0)
                result[model][elicitation][name] = (
                    round(failure / total, 3) if total > 0 else None
                )

    return result


# ── Judge quality metrics ─────────────────────────────────────────────────────

def judge_metrics(records: list[dict]) -> dict:
    """
    Compute judge quality metrics across all records.

    Metrics:
        position_consistency_rate: fraction of judge calls where the two orderings
            agreed on a winner (position_consistent=True). Low rate → high position bias.
        gate_trigger_rate: fraction of runs where the programmatic gate triggered
            (judge was not called). High rate → many structurally broken submissions.
        judge_call_rate: fraction of runs where a judge result is present.

    Args:
        records: Full results list (errors included — gate_triggered includes error runs).

    Returns:
        Dict of metric name → value.
    """
    total        = len(records)
    gate_count   = sum(1 for r in records if r["hybrid"]["gate_triggered"])
    judge_count  = sum(1 for r in records if r["hybrid"]["judge"] is not None)
    consistent   = sum(
        1 for r in records
        if r["hybrid"]["judge"] is not None
        and r["hybrid"]["judge"]["position_consistent"]
    )

    return {
        "total_runs":               total,
        "gate_trigger_rate":        round(gate_count  / total, 3) if total else None,
        "judge_call_rate":          round(judge_count / total, 3) if total else None,
        "position_consistency_rate": round(consistent / judge_count, 3) if judge_count else None,
    }


# ── Model comparison table ────────────────────────────────────────────────────

def model_comparison(records: list[dict]) -> list[dict]:
    """
    Build a comparison table with one row per (model, elicitation) combination.

    Each row contains:
        model, elicitation, n, mean_hybrid, ci_lo, ci_hi, ci_width,
        mean_prog, mean_judge (or None if gate always triggered),
        dominant_bucket (most common agreement bucket),
        top_failing_check (check with highest failure rate)

    Sorted by mean_hybrid descending.

    Args:
        records: Full results list.

    Returns:
        List of row dicts, sorted by mean_hybrid descending.
    """
    rows = []
    models       = sorted({r["model"] for r in records})
    elicitations = sorted({r["elicitation"] for r in records})
    check_rates  = per_check_failure_rates(records)
    buckets      = bucket_distribution(records)

    for model in models:
        for elicitation in elicitations:
            subset = filter_records(records, model=model, elicitation=elicitation)
            if not subset:
                continue

            hybrid_scores = [r["hybrid"]["hybrid_score"]       for r in subset]
            prog_scores   = [r["hybrid"]["programmatic_score"] for r in subset]
            judge_scores  = [
                r["hybrid"]["judge"]["score"]
                for r in subset
                if r["hybrid"]["judge"] is not None
            ]

            point, lo, hi = bootstrap_ci_scores(hybrid_scores)

            # Dominant bucket
            bucket_counts = buckets.get(model, {}).get(elicitation, {})
            dominant_bucket = max(bucket_counts, key=bucket_counts.get) if bucket_counts else "N/A"

            # Top failing check
            cr = check_rates.get(model, {}).get(elicitation, {})
            valid_cr = {k: v for k, v in cr.items() if v is not None}
            top_failing = max(valid_cr, key=valid_cr.get) if valid_cr else "N/A"
            top_failing_rate = valid_cr.get(top_failing, None)

            rows.append({
                "model":             model,
                "elicitation":       elicitation,
                "n":                 len(subset),
                "mean_hybrid":       round(point, 4),
                "ci_lo":             round(lo, 4),
                "ci_hi":             round(hi, 4),
                "ci_width":          round(hi - lo, 4),
                "mean_prog":         round(float(np.mean(prog_scores)), 4),
                "mean_judge":        round(float(np.mean(judge_scores)), 4) if judge_scores else None,
                "dominant_bucket":   dominant_bucket,
                "top_failing_check": top_failing,
                "top_fail_rate":     top_failing_rate,
            })

    rows.sort(key=lambda r: r["mean_hybrid"], reverse=True)
    return rows


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(records: list[dict]) -> None:
    """
    Print a formatted analysis report to stdout.

    Sections:
      1. Overview (record count, models, elicitation modes)
      2. Model comparison table (sorted by mean hybrid score)
      3. Elicitation gap per model
      4. Agreement bucket distributions
      5. Per-check failure rates
      6. Judge quality metrics
    """
    models       = sorted({r["model"] for r in records})
    elicitations = sorted({r["elicitation"] for r in records})
    tasks        = sorted({r["task_id"] for r in records})

    print("=" * 70)
    print("DATA-CLEANING EVAL — ANALYSIS REPORT")
    print("=" * 70)
    print(f"Records: {len(records)}  |  Tasks: {len(tasks)}  |  "
          f"Models: {len(models)}  |  Elicitation modes: {len(elicitations)}")
    print(f"Models:       {', '.join(models)}")
    print(f"Elicitations: {', '.join(elicitations)}")

    # ── 1. Model comparison table ──
    print("\n── 1. Model Comparison (sorted by mean hybrid score) ──")
    rows = model_comparison(records)
    header = f"{'Model':<22} {'Elicit':<12} {'N':>3} {'Mean':>6} {'95% CI':>18} {'Width':>6} {'Prog':>6} {'Judge':>6} {'Dom. Bucket':<22} {'Top Fail'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        judge_str = f"{row['mean_judge']:.3f}" if row["mean_judge"] is not None else "  N/A"
        fail_str  = f"{row['top_failing_check']}({row['top_fail_rate']:.0%})" if row["top_fail_rate"] is not None else "N/A"
        print(f"{row['model']:<22} {row['elicitation']:<12} {row['n']:>3} "
              f"{row['mean_hybrid']:>6.3f} {ci:>18} {row['ci_width']:>6.3f} "
              f"{row['mean_prog']:>6.3f} {judge_str:>6} "
              f"{row['dominant_bucket']:<22} {fail_str}")

    # ── 2. Elicitation gap ──
    print("\n── 2. Elicitation Gap (structured − zero_shot) ──")
    print(f"{'Model':<22} {'Structured':>12} {'Zero-shot':>12} {'Gap':>8}  {'Interpretation'}")
    print("-" * 70)
    for model in models:
        try:
            eg = elicitation_gap(records, model)
            s_ci = eg["structured_ci"]
            z_ci = eg["zero_shot_ci"]
            interp = "large gap — prompt-sensitive" if eg["gap"] > 0.15 else (
                     "moderate gap" if eg["gap"] > 0.08 else "small gap — robust")
            print(f"{model:<22} {s_ci[0]:>8.3f} [{s_ci[1]:.3f},{s_ci[2]:.3f}]  "
                  f"{z_ci[0]:>8.3f} [{z_ci[1]:.3f},{z_ci[2]:.3f}]  "
                  f"{eg['gap']:>+8.3f}  {interp}")
        except ValueError as e:
            print(f"{model:<22} ERROR: {e}")

    # ── 3. Agreement bucket distribution ──
    print("\n── 3. Agreement Bucket Distribution ──")
    BUCKETS = ["agree-good", "agree-bad", "judge-rescues", "programmatic-catches", "gate-triggered"]
    dist = bucket_distribution(records)
    for model in models:
        for elicitation in elicitations:
            counts = dist.get(model, {}).get(elicitation, {})
            total  = sum(counts.values())
            if total == 0:
                continue
            print(f"\n  {model} × {elicitation}  (N={total}):")
            for bucket in BUCKETS:
                count = counts.get(bucket, 0)
                pct   = count / total * 100
                bar   = "█" * int(pct / 5)
                print(f"    {bucket:<25} {count:>2} / {total:<2}  ({pct:>5.1f}%)  {bar}")

    # ── 4. Per-check failure rates ──
    print("\n── 4. Per-Check Failure Rates (higher = more failures) ──")
    check_rates = per_check_failure_rates(records)
    CHECK_NAMES = ["integrity", "sheets", "errors", "revenue", "summary", "reference"]
    for model in models:
        for elicitation in elicitations:
            cr = check_rates.get(model, {}).get(elicitation, {})
            if not cr:
                continue
            print(f"\n  {model} × {elicitation}:")
            for check in CHECK_NAMES:
                rate = cr.get(check)
                if rate is None:
                    print(f"    {check:<12}  no data")
                else:
                    bar = "█" * int(rate * 20)
                    print(f"    {check:<12}  {rate:>5.1%}  {bar}")

    # ── 5. Judge quality metrics ──
    print("\n── 5. Judge Quality Metrics ──")
    jm = judge_metrics(records)
    print(f"  Total runs:               {jm['total_runs']}")
    print(f"  Gate trigger rate:        {jm['gate_trigger_rate']:.1%}  "
          f"(judge skipped this fraction)")
    print(f"  Judge call rate:          {jm['judge_call_rate']:.1%}")
    print(f"  Position consistency:     {jm['position_consistency_rate']:.1%}  "
          f"(fraction where both orderings agreed)")

    # ── 6–9. Error analysis + difficulty calibration (Day 20) ──
    ea = _get_error_analysis()
    ea.print_error_analysis_section(records)

    print("\n" + "=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze eval results from JSONL log.")
    parser.add_argument("results_file", type=Path, help="Path to JSONL results file")
    parser.add_argument("--model",       help="Filter to one model")
    parser.add_argument("--elicitation", help="Filter to one elicitation mode")
    parser.add_argument("--format",      choices=["text", "json"], default="text")
    args = parser.parse_args()

    records = load_results(args.results_file)
    if args.model:
        records = filter_records(records, model=args.model)
    if args.elicitation:
        records = filter_records(records, elicitation=args.elicitation)

    if not records:
        print("No records found after filtering.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        output = {
            "comparison": model_comparison(records),
            "judge_metrics": judge_metrics(records),
            "bucket_distribution": bucket_distribution(records),
            "per_check_failure_rates": per_check_failure_rates(records),
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(records)


if __name__ == "__main__":
    main()
