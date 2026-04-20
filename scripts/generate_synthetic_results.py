"""
generate_synthetic_results.py — Synthetic eval run data for Day 19 analysis.

Since we can't call real model APIs, this script generates a realistic JSONL
results file that mirrors what harness.py would produce from a real grid run.

Design:
  - 5 tasks (task_001 through task_005) × 2 models × 2 elicitation modes = 20 records
  - Score distributions calibrated to realistic model behavior:
      GPT-4o structured:   strong (mean hybrid ~0.78)
      GPT-4o zero_shot:    weaker (mean hybrid ~0.55) — elicitation gap
      Claude structured:   competitive (mean hybrid ~0.72)
      Claude zero_shot:    weaker (mean hybrid ~0.48) — larger elicitation gap
  - Agreement bucket distributions reflect realistic failure mode mix
  - Per-check scores reflect realistic programmatic check outcomes
  - One simulated model failure per elicitation mode (model_error not null)
  - Seed 42 for reproducibility

Usage:
    cd Code/data-cleaning-eval
    python scripts/generate_synthetic_results.py
    # → writes results/synthetic_results.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# Add repo root so imports work
_EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EVAL_ROOT.parent))

RNG = np.random.default_rng(seed=42)

OUTPUT_PATH = _EVAL_ROOT / "results" / "synthetic_results.jsonl"

# ── Score profiles per (model, elicitation) ───────────────────────────────────
# Each profile controls the synthetic score generation.
# prog_mean/judge_mean: center of normal distribution for that layer's score
# prog_std/judge_std:   spread (clipped to [0,1])
# failure_rate:         probability that a given check fails (per check)
# gate_rate:            probability the gate triggers (prog < 0.3)
# error_rate:           probability of model API failure for this combo

PROFILES = {
    ("gpt-4o", "structured"): {
        "prog_mean": 0.82, "prog_std": 0.12,
        "judge_mean": 0.78, "judge_std": 0.15,
        "gate_rate": 0.05, "error_rate": 0.0,
        "check_pass_rates": {
            "integrity": 1.00, "sheets": 0.95, "errors": 0.90,
            "revenue": 0.85, "summary": 0.88, "reference": 0.90,
        },
    },
    ("gpt-4o", "zero_shot"): {
        "prog_mean": 0.58, "prog_std": 0.18,
        "judge_mean": 0.52, "judge_std": 0.20,
        "gate_rate": 0.20, "error_rate": 0.0,
        "check_pass_rates": {
            "integrity": 1.00, "sheets": 0.80, "errors": 0.75,
            "revenue": 0.55, "summary": 0.60, "reference": 0.65,
        },
    },
    ("claude-3-5-sonnet", "structured"): {
        "prog_mean": 0.76, "prog_std": 0.14,
        "judge_mean": 0.70, "judge_std": 0.18,
        "gate_rate": 0.10, "error_rate": 0.0,
        "check_pass_rates": {
            "integrity": 1.00, "sheets": 0.92, "errors": 0.88,
            "revenue": 0.78, "summary": 0.82, "reference": 0.85,
        },
    },
    ("claude-3-5-sonnet", "zero_shot"): {
        "prog_mean": 0.50, "prog_std": 0.20,
        "judge_mean": 0.44, "judge_std": 0.22,
        "gate_rate": 0.25, "error_rate": 0.0,
        "check_pass_rates": {
            "integrity": 1.00, "sheets": 0.75, "errors": 0.70,
            "revenue": 0.48, "summary": 0.55, "reference": 0.60,
        },
    },
}

PROG_WEIGHT  = 0.4
JUDGE_WEIGHT = 0.6
PROG_GATE    = 0.3
PROG_PASS_T  = 0.7
JUDGE_PASS_T = 0.5

CHECK_WEIGHTS = {
    "integrity": 0.10, "sheets": 0.20, "errors": 0.25,
    "revenue":   0.20, "summary": 0.15, "reference": 0.10,
}


def _clamp(v: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, v))


def _bucket(prog: float, judge: float, gate: bool) -> str:
    if gate:
        return "gate-triggered"
    prog_pass  = prog  >= PROG_PASS_T
    judge_pass = judge >= JUDGE_PASS_T
    if prog_pass and judge_pass:
        return "agree-good"
    elif not prog_pass and not judge_pass:
        return "agree-bad"
    elif not prog_pass and judge_pass:
        return "judge-rescues"
    else:
        return "programmatic-catches"


def _make_checks(profile: dict) -> tuple[list[dict], float]:
    """Generate per-check results and compute aggregate prog score."""
    checks = []
    prog_score = 0.0
    for name, weight in CHECK_WEIGHTS.items():
        passed = RNG.random() < profile["check_pass_rates"][name]
        score  = _clamp(RNG.normal(0.85, 0.10)) if passed else _clamp(RNG.normal(0.20, 0.15))
        prog_score += weight * score
        checks.append({
            "name": name,
            "passed": bool(passed),
            "score": round(float(score), 4),
            "detail": f"{'OK' if passed else 'FAIL'}: {name} check",
        })
    return checks, _clamp(prog_score)


def _make_judge(judge_mean: float, judge_std: float) -> dict:
    """Generate a synthetic judge result."""
    score = _clamp(RNG.normal(judge_mean, judge_std))
    # Simulate position consistency ~80% of the time
    consistent = bool(RNG.random() < 0.80)
    # Back out raw scores consistent with the reconciled score
    if consistent:
        # Both calls agree: model_wins_1 = score, model_wins_2 = score
        score_ab = round(float(_clamp(1.0 - score + RNG.normal(0, 0.02))), 1)
        score_ba = round(float(_clamp(score     + RNG.normal(0, 0.02))), 1)
        # Snap to valid values
        score_ab = min([0.0, 0.5, 1.0], key=lambda x: abs(x - score_ab))
        score_ba = min([0.0, 0.5, 1.0], key=lambda x: abs(x - score_ba))
    else:
        score_ab = 0.5
        score_ba = 0.5
        score    = 0.5
    return {
        "score": round(float(score), 4),
        "score_ab": score_ab,
        "score_ba": score_ba,
        "position_consistent": consistent,
        "reasoning_ab": f"Synthetic reasoning AB. SCORE: {score_ab}",
        "reasoning_ba": f"Synthetic reasoning BA. SCORE: {score_ba}",
    }


def generate_record(
    task_id: str,
    model: str,
    elicitation: str,
    base_time: datetime,
    run_index: int,
) -> dict:
    """Generate one synthetic run record matching the harness.py JSONL schema."""
    profile   = PROFILES[(model, elicitation)]
    started   = (base_time + timedelta(minutes=run_index * 3)).isoformat()
    ts        = (base_time + timedelta(minutes=run_index * 3)).strftime("%Y%m%dT%H%M%S")
    run_id    = f"{task_id}__{model}__{elicitation}__{ts}"

    # Simulate model API failure
    if RNG.random() < profile["error_rate"]:
        return {
            "run_id": run_id, "task_id": task_id, "model": model,
            "elicitation": elicitation, "started_at": started,
            "model_error": "Simulated API timeout after 30s",
            "programmatic": None,
            "hybrid": {
                "hybrid_score": 0.0, "programmatic_score": 0.0,
                "judge": None, "agreement_bucket": "gate-triggered",
                "gate_triggered": True,
            },
        }

    # Generate programmatic result
    checks, prog_score = _make_checks(profile)

    # Simulate gate
    gate = bool(RNG.random() < profile["gate_rate"]) or prog_score < PROG_GATE

    # Generate judge result (only if gate not triggered)
    judge_dict = None
    if not gate:
        judge_dict = _make_judge(profile["judge_mean"], profile["judge_std"])

    # Compute hybrid
    if gate or judge_dict is None:
        hybrid_score    = round(prog_score * PROG_WEIGHT, 4)
        agreement_bucket = "gate-triggered"
    else:
        hybrid_score    = round(_clamp(PROG_WEIGHT * prog_score + JUDGE_WEIGHT * judge_dict["score"]), 4)
        agreement_bucket = _bucket(prog_score, judge_dict["score"], False)

    prog_result = {
        "score": round(prog_score, 4),
        "file_integrity": True,
        "sheets_present": ["Raw Data", "Cleaned Data", "Summary"],
        "sheets_missing": [],
        "formula_error_count": 0,
        "revenue_formula_fraction": round(float(RNG.uniform(0.6, 1.0)
            if checks[3]["passed"] else RNG.uniform(0.0, 0.3)), 3),
        "summary_formula_fraction": round(float(RNG.uniform(0.7, 1.0)
            if checks[4]["passed"] else RNG.uniform(0.0, 0.4)), 3),
        "summary_references_cleaned": checks[5]["passed"],
        "cleaned_row_count": 70,
        "raw_row_count": 75,
        "checks": checks,
    }

    return {
        "run_id":      run_id,
        "task_id":     task_id,
        "model":       model,
        "elicitation": elicitation,
        "started_at":  started,
        "model_error": None,
        "programmatic": prog_result,
        "hybrid": {
            "hybrid_score":       hybrid_score,
            "programmatic_score": round(prog_score, 4),
            "judge":              judge_dict,
            "agreement_bucket":   agreement_bucket,
            "gate_triggered":     gate,
        },
    }


def main():
    tasks        = [f"task_{i:03d}" for i in range(1, 6)]   # task_001 … task_005
    models       = ["gpt-4o", "claude-3-5-sonnet"]
    elicitations = ["zero_shot", "structured"]
    base_time    = datetime(2026, 4, 12, 10, 0, 0, tzinfo=timezone.utc)

    records = []
    run_idx = 0
    for task in tasks:
        for model in models:
            for elicitation in elicitations:
                records.append(generate_record(task, model, elicitation, base_time, run_idx))
                run_idx += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} synthetic records to {OUTPUT_PATH}")
    print(f"Tasks: {len(tasks)}  |  Models: {models}  |  Elicitations: {elicitations}")

    # Quick sanity check
    from collections import Counter
    buckets = Counter(r["hybrid"]["agreement_bucket"] for r in records)
    print(f"Bucket distribution: {dict(buckets)}")
    models_present = Counter(r["model"] for r in records)
    print(f"Records per model: {dict(models_present)}")


if __name__ == "__main__":
    main()
