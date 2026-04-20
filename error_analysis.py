"""
error_analysis.py — Error analysis and difficulty calibration for the data-cleaning eval.

Day 20: After running the eval and computing per-model scores (Day 19), this module
digs one level deeper to answer two questions:

  1. ERROR ANALYSIS: HOW are models failing?
       - Which programmatic checks fail most? (per-task, not just aggregate)
       - Are failures structural (format), formula (hardcoding), or content (analysis)?
       - Are certain tasks systematically harder for specific models?

  2. DIFFICULTY CALIBRATION: Is the eval well-designed?
       - Do tasks span an appropriate difficulty range (not all easy, not all hard)?
       - Do models agree on which tasks are hard? (Spearman correlation as IRR proxy)
       - Is the eval discriminating (std of task difficulties is large enough to detect
         meaningful model differences)?

Why this matters for eval design:
  - An eval where all tasks have mean hybrid score > 0.85 can't discriminate models.
  - An eval where one model finds task_003 hard but another doesn't suggests a
    task-design problem (the task is actually testing model-specific quirks, not
    the target capability).
  - Failure mode taxonomy tells you WHERE to improve: structural failures → fix prompt
    format instructions; formula failures → add explicit formula requirements to elicitation;
    content failures → harder to fix (genuine capability gap).

Functions:
  per_task_scores()          → difficulty profile per task (mean, spread, dominant failures)
  failure_mode_taxonomy()    → 3-category breakdown: structural / formula / content
  model_task_agreement()     → Spearman rank correlation between models' task rankings
  task_difficulty_ranking()  → tasks sorted hard→easy with tier labels
  difficulty_calibration_report() → summary stats: spread, discrimination, tier counts

All functions accept the same list[dict] format produced by harness.py / load_results().
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

# ── Failure mode taxonomy ─────────────────────────────────────────────────────
#
# The 6 programmatic checks group into 3 failure mode categories:
#
#   STRUCTURAL: integrity, sheets
#     → Model didn't produce the required file structure. Hard failure.
#     → Fix: tighten format requirements in elicitation prompt.
#
#   FORMULA: revenue, reference
#     → Model computed correct numbers but hardcoded them instead of using formulas.
#     → Fix: add explicit formula instructions ("use =G2*H2, not the number 239.97").
#     → This is the most interesting failure mode — it looks right to a human reviewer
#       but fails programmatic checks. Exactly what programmatic-catches catches.
#
#   CONTENT: errors, summary
#     → Model left formula errors in cells, or summary analysis is weak.
#     → Fix: harder. Requires better model capability, not just better prompting.

FAILURE_MODE_CATEGORIES = {
    "structural": ["integrity", "sheets"],
    "formula":    ["revenue", "reference"],
    "content":    ["errors", "summary"],
}

# Tier thresholds for task difficulty (based on mean hybrid score)
DIFFICULTY_TIERS = {
    "easy":   (0.70, 1.01),   # mean hybrid ≥ 0.70
    "medium": (0.40, 0.70),   # 0.40 ≤ mean hybrid < 0.70
    "hard":   (0.00, 0.40),   # mean hybrid < 0.40
}


def _check_failure_rates_for_subset(records: list[dict]) -> dict[str, Optional[float]]:
    """
    Compute per-check failure rate for a subset of records.

    Returns {check_name: failure_rate} where failure_rate is None if no data.
    Skips records with programmatic=None (model API failures).
    """
    CHECK_NAMES = ["integrity", "sheets", "errors", "revenue", "summary", "reference"]
    totals: dict[str, int]   = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)

    for r in records:
        prog = r.get("programmatic")
        if prog is None:
            continue
        for check in prog.get("checks", []):
            name = check["name"]
            totals[name]   += 1
            if not check["passed"]:
                failures[name] += 1

    return {
        name: round(failures[name] / totals[name], 4) if totals[name] > 0 else None
        for name in CHECK_NAMES
    }


def _spearman_correlation(ranks_a: list[float], ranks_b: list[float]) -> float:
    """
    Compute Spearman rank correlation between two ranked lists.

    Spearman's ρ = 1 - (6 * Σd²) / (n * (n² - 1))
    where d = difference in ranks for each item.

    This is the standard IRR measure for ordinal data — it answers:
    "Do these two judges agree on the ordering?" (not just the absolute values).

    Args:
        ranks_a: List of rank values (e.g. [1, 3, 2, 4] means item 0 is rank 1).
        ranks_b: Paired list of rank values for the same items.

    Returns:
        Spearman ρ in [-1, 1]. 1.0 = perfect agreement, -1.0 = perfect disagreement.

    Raises:
        ValueError: if lists are different lengths or fewer than 2 items.
    """
    n = len(ranks_a)
    if n != len(ranks_b):
        raise ValueError(f"Rank lists must have equal length: {n} vs {len(ranks_b)}")
    if n < 2:
        raise ValueError("Need at least 2 items to compute Spearman correlation.")

    d_sq_sum = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    denom    = n * (n ** 2 - 1)
    if denom == 0:
        return 1.0   # degenerate case: n=1 (handled above, but safety)
    return round(1.0 - (6.0 * d_sq_sum) / denom, 4)


def _scores_to_ranks(scores: list[float]) -> list[float]:
    """
    Convert scores to ranks (1 = highest score, n = lowest score).

    Handles ties with average rank (standard Spearman convention).

    Example:
        [0.8, 0.5, 0.8, 0.3] → [1.5, 3.0, 1.5, 4.0]
        (the two 0.8 items share ranks 1 and 2, average = 1.5)
    """
    # Pair each score with its original index
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])  # descending
    ranks   = [0.0] * len(scores)
    i       = 0
    while i < len(indexed):
        # Find all items with the same score (ties)
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        # Average rank for tied items: ranks are (i+1) through j (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


# ── Core analysis functions ───────────────────────────────────────────────────

def per_task_scores(records: list[dict]) -> dict[str, dict]:
    """
    Compute a difficulty profile for each task.

    Aggregates scores across all models and elicitation modes to produce a
    task-level view: how hard is this task, and HOW do models fail on it?

    Args:
        records: Full results list from load_results(). Errors included (contributes
                 a 0.0 hybrid_score, which is correct — API failure = 0 capability).

    Returns:
        Dict {task_id: profile} where each profile contains:
            task_id:           str
            mean_hybrid:       float — mean score across all models/elicitations
            std_hybrid:        float — score spread (high std = model-specific difficulty)
            n_runs:            int   — total runs for this task
            difficulty_tier:   str   — "easy" | "medium" | "hard"
            model_scores:      dict  — {model: {elicitation: hybrid_score}} (first run only)
            check_failure_rates: dict — per-check failure rate across all runs on this task
            dominant_failure:  str   — check with highest failure rate (or "none")
    """
    task_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        task_records[r["task_id"]].append(r)

    profiles = {}
    for task_id, recs in task_records.items():
        scores = [r["hybrid"]["hybrid_score"] for r in recs]
        mean_h = round(sum(scores) / len(scores), 4)
        n      = len(scores)

        # Standard deviation
        if n > 1:
            variance = sum((s - mean_h) ** 2 for s in scores) / (n - 1)
            std_h    = round(math.sqrt(variance), 4)
        else:
            std_h = 0.0

        # Difficulty tier
        tier = next(
            name for name, (lo, hi) in DIFFICULTY_TIERS.items()
            if lo <= mean_h < hi
        )

        # Per-model, per-elicitation breakdown (first run per combo, excluding errors)
        model_scores: dict = defaultdict(dict)
        for r in recs:
            if r["model_error"] is None:
                model_scores[r["model"]][r["elicitation"]] = r["hybrid"]["hybrid_score"]

        # Per-check failure rates across all runs on this task
        cfr = _check_failure_rates_for_subset(recs)
        valid_cfr = {k: v for k, v in cfr.items() if v is not None}
        dominant  = max(valid_cfr, key=valid_cfr.__getitem__) if valid_cfr else "none"

        profiles[task_id] = {
            "task_id":             task_id,
            "mean_hybrid":         mean_h,
            "std_hybrid":          std_h,
            "n_runs":              n,
            "difficulty_tier":     tier,
            "model_scores":        dict(model_scores),
            "check_failure_rates": cfr,
            "dominant_failure":    dominant,
        }

    return profiles


def failure_mode_taxonomy(records: list[dict]) -> dict:
    """
    Categorize failure modes into structural / formula / content groups.

    The 6 programmatic checks map to 3 failure mode categories (see module header).
    This function computes per-category failure rates and identifies which category
    is the dominant source of failures overall.

    Interpretation guide:
        structural_rate high → Models aren't following format instructions.
                                Fix: tighten elicitation prompt structure requirements.
        formula_rate high    → Models compute correctly but hardcode values.
                                Fix: add explicit formula requirements to elicitation.
        content_rate high    → Models produce wrong analysis. Harder to fix —
                                this is a genuine capability limitation, not a prompting gap.

    Args:
        records: Full results list.

    Returns:
        Dict with keys:
            per_category: {category_name: {
                checks:        list of check names in this category
                failure_rate:  mean failure rate across checks in category (None if no data)
                check_rates:   {check_name: failure_rate}
            }}
            dominant_category: name of the category with highest failure rate
            total_valid_runs:  runs with programmatic data (excludes model errors)
    """
    # Get overall per-check failure rates
    cfr = _check_failure_rates_for_subset(records)
    total_valid = sum(1 for r in records if r.get("programmatic") is not None)

    per_category = {}
    category_means = {}

    for category, check_names in FAILURE_MODE_CATEGORIES.items():
        check_rates = {name: cfr.get(name) for name in check_names}
        valid_rates = [v for v in check_rates.values() if v is not None]
        cat_mean    = round(sum(valid_rates) / len(valid_rates), 4) if valid_rates else None

        per_category[category] = {
            "checks":       check_names,
            "failure_rate": cat_mean,
            "check_rates":  check_rates,
        }
        if cat_mean is not None:
            category_means[category] = cat_mean

    dominant = max(category_means, key=category_means.__getitem__) if category_means else "none"

    return {
        "per_category":       per_category,
        "dominant_category":  dominant,
        "total_valid_runs":   total_valid,
    }


def model_task_agreement(records: list[dict]) -> dict:
    """
    Measure whether models agree on which tasks are hard.

    Approach: for each model, compute the mean hybrid score per task (across
    elicitations). Then compute Spearman rank correlation between every pair of
    models' per-task difficulty rankings.

    High correlation (ρ > 0.7) → tasks have intrinsic difficulty that's model-agnostic.
      This is GOOD for eval design: tasks measure the capability, not model-specific quirks.

    Low correlation (ρ < 0.3) → models disagree on task difficulty.
      This is a WARNING: some tasks may be testing model-specific behaviors
      (e.g. quirks in how Claude formats formulas vs how GPT-4o does it).
      These tasks should be inspected and possibly redesigned.

    Args:
        records: Full results list (errors excluded internally — 0.0 scores
                 would distort per-task difficulty estimates).

    Returns:
        Dict with keys:
            task_ids:          list of task IDs (shared basis for all rankings)
            model_mean_scores: {model: {task_id: mean_hybrid}} — raw data
            model_rankings:    {model: {task_id: rank}} — 1=hardest, n=easiest
            pairwise_spearman: {(model_a, model_b): ρ} — all model pairs
            mean_spearman:     float — mean ρ across all pairs (summary IRR)
            interpretation:    str — human-readable agreement label
    """
    # Collect per-model, per-task hybrid scores (exclude errors)
    model_task_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["model_error"] is None:
            model_task_scores[r["model"]][r["task_id"]].append(r["hybrid"]["hybrid_score"])

    models   = sorted(model_task_scores.keys())
    task_ids = sorted({r["task_id"] for r in records})

    # Compute per-task mean for each model
    model_means: dict[str, dict[str, float]] = {}
    for model in models:
        model_means[model] = {}
        for task_id in task_ids:
            task_scores = model_task_scores[model].get(task_id, [])
            if task_scores:
                model_means[model][task_id] = round(
                    sum(task_scores) / len(task_scores), 4
                )
            else:
                # Model has no data for this task — treat as 0.0 (conservative)
                model_means[model][task_id] = 0.0

    # Compute ranks per model (higher score = easier = higher rank number when inverted)
    # We rank by DIFFICULTY: rank 1 = hardest (lowest score), rank n = easiest (highest score)
    model_rankings: dict[str, dict[str, float]] = {}
    for model in models:
        scores_ordered = [model_means[model][t] for t in task_ids]
        # Ranks where 1 = lowest score (hardest)
        asc_ranks = _scores_to_ranks([-s for s in scores_ordered])  # negate → ascending = hardest first
        model_rankings[model] = {t: asc_ranks[i] for i, t in enumerate(task_ids)}

    # Pairwise Spearman correlations
    pairwise: dict[tuple[str, str], float] = {}
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            ranks_a = [model_rankings[model_a][t] for t in task_ids]
            ranks_b = [model_rankings[model_b][t] for t in task_ids]
            rho     = _spearman_correlation(ranks_a, ranks_b)
            pairwise[(model_a, model_b)] = rho

    # Mean Spearman
    rho_values   = list(pairwise.values())
    mean_spearman = round(sum(rho_values) / len(rho_values), 4) if rho_values else None

    if mean_spearman is None:
        interp = "N/A — fewer than 2 models"
    elif mean_spearman >= 0.70:
        interp = "high agreement — tasks have model-agnostic difficulty (good eval design)"
    elif mean_spearman >= 0.30:
        interp = "moderate agreement — some model-specific task effects present"
    else:
        interp = "low agreement — tasks may be testing model-specific behaviors (review task design)"

    # Serialize pairwise dict with string keys for JSON compatibility
    pairwise_str = {f"{a} vs {b}": rho for (a, b), rho in pairwise.items()}

    return {
        "task_ids":          task_ids,
        "model_mean_scores": model_means,
        "model_rankings":    model_rankings,
        "pairwise_spearman": pairwise_str,
        "mean_spearman":     mean_spearman,
        "interpretation":    interp,
    }


def task_difficulty_ranking(records: list[dict]) -> list[dict]:
    """
    Rank tasks from hardest to easiest, with tier labels and spread metrics.

    Uses per_task_scores() internally. Each row in the output represents one task
    with its mean difficulty, spread (std), tier, and dominant failure mode.

    Args:
        records: Full results list.

    Returns:
        List of task profile dicts, sorted by mean_hybrid ascending (hardest first).
        Each dict contains: task_id, rank, mean_hybrid, std_hybrid, difficulty_tier,
        dominant_failure, n_runs.
    """
    profiles = per_task_scores(records)
    rows = [
        {
            "task_id":          task_id,
            "mean_hybrid":      p["mean_hybrid"],
            "std_hybrid":       p["std_hybrid"],
            "difficulty_tier":  p["difficulty_tier"],
            "dominant_failure": p["dominant_failure"],
            "n_runs":           p["n_runs"],
        }
        for task_id, p in profiles.items()
    ]
    rows.sort(key=lambda r: r["mean_hybrid"])  # ascending = hardest first
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def difficulty_calibration_report(records: list[dict]) -> dict:
    """
    Assess whether the eval's task set is well-calibrated for discriminating models.

    A well-calibrated eval has:
      1. Diversity of difficulty — not all easy or all hard tasks.
      2. Sufficient spread — std(task difficulties) > 0.15 so models separate.
      3. No extreme floor/ceiling — no tasks with mean > 0.95 (everyone passes) or
         mean < 0.05 (everyone fails, even the best model).

    Calibration verdict:
      "well_calibrated"    → meets all three criteria
      "ceiling_risk"       → too many easy tasks (easy_fraction > 0.6)
      "floor_risk"         → too many hard tasks (hard_fraction > 0.6)
      "low_discrimination" → spread too low (std < 0.10) — tasks don't separate models
      "small_n"            → fewer than 5 tasks — insufficient to assess calibration

    Args:
        records: Full results list.

    Returns:
        Dict with keys:
            n_tasks:           total tasks evaluated
            mean_difficulty:   mean hybrid score across all tasks (lower = harder)
            std_difficulty:    std of per-task means (higher = better discrimination)
            tier_counts:       {"easy": n, "medium": n, "hard": n}
            tier_fractions:    {"easy": f, "medium": f, "hard": f}
            floor_tasks:       list of task_ids with mean < 0.10 (degenerate — too hard)
            ceiling_tasks:     list of task_ids with mean > 0.90 (degenerate — too easy)
            discrimination_ok: bool — True if std_difficulty ≥ 0.10
            calibration_verdict: str — summary assessment
    """
    profiles = per_task_scores(records)
    n        = len(profiles)

    if n == 0:
        return {
            "n_tasks":           0,
            "calibration_verdict": "no_data",
        }

    means = [p["mean_hybrid"] for p in profiles.values()]
    mean_d = round(sum(means) / n, 4)

    if n > 1:
        variance = sum((m - mean_d) ** 2 for m in means) / (n - 1)
        std_d    = round(math.sqrt(variance), 4)
    else:
        std_d = 0.0

    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    floor_tasks    = []
    ceiling_tasks  = []

    for task_id, p in profiles.items():
        tier_counts[p["difficulty_tier"]] += 1
        if p["mean_hybrid"] < 0.10:
            floor_tasks.append(task_id)
        if p["mean_hybrid"] > 0.90:
            ceiling_tasks.append(task_id)

    tier_fractions = {
        tier: round(count / n, 3) for tier, count in tier_counts.items()
    }
    discrimination_ok = std_d >= 0.10

    # Calibration verdict
    if n < 5:
        verdict = "small_n"
    elif not discrimination_ok:
        verdict = "low_discrimination"
    elif tier_fractions["easy"] > 0.60:
        verdict = "ceiling_risk"
    elif tier_fractions["hard"] > 0.60:
        verdict = "floor_risk"
    else:
        verdict = "well_calibrated"

    return {
        "n_tasks":             n,
        "mean_difficulty":     mean_d,
        "std_difficulty":      std_d,
        "tier_counts":         tier_counts,
        "tier_fractions":      tier_fractions,
        "floor_tasks":         sorted(floor_tasks),
        "ceiling_tasks":       sorted(ceiling_tasks),
        "discrimination_ok":   discrimination_ok,
        "calibration_verdict": verdict,
    }


# ── Report section ────────────────────────────────────────────────────────────

def print_error_analysis_section(records: list[dict]) -> None:
    """
    Print the Day 20 error analysis and difficulty calibration sections.

    Called from analyze_results.print_report() to extend the standard report.
    Prints 4 new sections: failure mode taxonomy, per-task difficulty, model
    agreement, and calibration verdict.
    """
    print("\n── 6. Failure Mode Taxonomy ──")
    fmt = failure_mode_taxonomy(records)
    print(f"  Total valid runs (with programmatic data): {fmt['total_valid_runs']}")
    print(f"  Dominant failure category: {fmt['dominant_category'].upper()}")
    print()
    for cat, info in fmt["per_category"].items():
        rate_str = f"{info['failure_rate']:.1%}" if info["failure_rate"] is not None else "no data"
        print(f"  {cat.upper():<12} overall failure rate: {rate_str}")
        for check, rate in info["check_rates"].items():
            r_str = f"{rate:.1%}" if rate is not None else "no data"
            bar   = "█" * int((rate or 0) * 20)
            print(f"    {check:<12} {r_str:>6}  {bar}")

    print("\n── 7. Task Difficulty Ranking (hardest → easiest) ──")
    ranking = task_difficulty_ranking(records)
    TIER_EMOJI = {"hard": "🔴", "medium": "🟡", "easy": "🟢"}
    header = f"  {'Rank':>4}  {'Task':<12} {'Mean':>6} {'Std':>6} {'Tier':<8} {'Dom. Failure'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in ranking:
        tier_label = TIER_EMOJI.get(row["difficulty_tier"], "") + " " + row["difficulty_tier"]
        print(f"  {row['rank']:>4}  {row['task_id']:<12} {row['mean_hybrid']:>6.3f} "
              f"{row['std_hybrid']:>6.3f}  {tier_label:<10} {row['dominant_failure']}")

    print("\n── 8. Model-Task Agreement (Spearman ρ) ──")
    mta = model_task_agreement(records)
    if mta["pairwise_spearman"]:
        for pair, rho in mta["pairwise_spearman"].items():
            print(f"  {pair}: ρ = {rho:+.3f}")
        print(f"  Mean ρ: {mta['mean_spearman']:+.3f}")
        print(f"  {mta['interpretation']}")
    else:
        print("  Fewer than 2 models — cannot compute agreement.")

    print("\n── 9. Difficulty Calibration Report ──")
    cal = difficulty_calibration_report(records)
    print(f"  Tasks: {cal['n_tasks']}  |  "
          f"Mean difficulty: {cal['mean_difficulty']:.3f}  |  "
          f"Std: {cal['std_difficulty']:.3f}")
    print(f"  Tiers:  easy={cal['tier_counts']['easy']} "
          f"({cal['tier_fractions']['easy']:.0%})  "
          f"medium={cal['tier_counts']['medium']} "
          f"({cal['tier_fractions']['medium']:.0%})  "
          f"hard={cal['tier_counts']['hard']} "
          f"({cal['tier_fractions']['hard']:.0%})")
    if cal["floor_tasks"]:
        print(f"  ⚠ Floor tasks (mean < 0.10): {cal['floor_tasks']}")
    if cal["ceiling_tasks"]:
        print(f"  ⚠ Ceiling tasks (mean > 0.90): {cal['ceiling_tasks']}")
    disc_str = "✓ adequate" if cal["discrimination_ok"] else "✗ too low (std < 0.10)"
    print(f"  Discrimination (std ≥ 0.10): {disc_str}")
    print(f"  Verdict: {cal['calibration_verdict'].upper()}")
