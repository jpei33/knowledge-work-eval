"""
test_error_analysis.py — Tests for error_analysis.py.

Tests cover:
  _spearman_correlation        — formula, ties, edge cases
  _scores_to_ranks             — basic, ties, all-equal
  per_task_scores              — difficulty tiers, dominant failure, error handling
  failure_mode_taxonomy        — category rates, dominant category, no-data path
  model_task_agreement         — Spearman computation, interpretation labels
  task_difficulty_ranking      — sort order, rank assignment
  difficulty_calibration_report — tier counts, verdict logic, floor/ceiling tasks

Run:
    cd Code/data-cleaning-eval
    pytest tests/test_error_analysis.py -v
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_EVAL_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EVAL_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

_spec = _ilu.spec_from_file_location("error_analysis", _EVAL_ROOT / "error_analysis.py")
_ea   = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ea)

per_task_scores              = _ea.per_task_scores
failure_mode_taxonomy        = _ea.failure_mode_taxonomy
model_task_agreement         = _ea.model_task_agreement
task_difficulty_ranking      = _ea.task_difficulty_ranking
difficulty_calibration_report = _ea.difficulty_calibration_report
_spearman_correlation        = _ea._spearman_correlation
_scores_to_ranks             = _ea._scores_to_ranks


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHECK_NAMES = ["integrity", "sheets", "errors", "revenue", "summary", "reference"]


def _make_check(name: str, passed: bool, score: float = None) -> dict:
    if score is None:
        score = 0.9 if passed else 0.1
    return {"name": name, "passed": passed, "score": score, "detail": "test"}


def _make_record(
    task_id: str = "task_001",
    model: str   = "gpt-4o",
    elicitation: str = "structured",
    hybrid_score: float = 0.75,
    prog_score: float   = 0.80,
    judge_score: float  = 0.70,
    bucket: str  = "agree-good",
    gate: bool   = False,
    model_error: str = None,
    failed_checks: list[str] = None,
) -> dict:
    """Build a minimal record matching the harness.py JSONL schema."""
    failed_checks = failed_checks or []
    checks = [_make_check(n, n not in failed_checks) for n in CHECK_NAMES]

    if model_error:
        return {
            "run_id": f"{task_id}__{model}__{elicitation}__20260414T120000",
            "task_id": task_id, "model": model, "elicitation": elicitation,
            "started_at": "2026-04-14T12:00:00+00:00",
            "model_error": model_error,
            "programmatic": None,
            "hybrid": {
                "hybrid_score": 0.0, "programmatic_score": 0.0,
                "judge": None, "agreement_bucket": "gate-triggered",
                "gate_triggered": True,
            },
        }

    return {
        "run_id": f"{task_id}__{model}__{elicitation}__20260414T120000",
        "task_id": task_id, "model": model, "elicitation": elicitation,
        "started_at": "2026-04-14T12:00:00+00:00",
        "model_error": None,
        "programmatic": {
            "score": prog_score,
            "file_integrity": True,
            "sheets_present": ["Raw Data", "Cleaned Data", "Summary"],
            "sheets_missing": [],
            "formula_error_count": 0,
            "revenue_formula_fraction": 0.95,
            "summary_formula_fraction": 0.90,
            "summary_references_cleaned": True,
            "cleaned_row_count": 70,
            "raw_row_count": 75,
            "checks": checks,
        },
        "hybrid": {
            "hybrid_score":       hybrid_score,
            "programmatic_score": prog_score,
            "judge": {
                "score": judge_score,
                "score_ab": 0.0, "score_ba": 1.0,
                "position_consistent": True,
                "reasoning_ab": "test", "reasoning_ba": "test",
            } if not gate else None,
            "agreement_bucket":   bucket,
            "gate_triggered":     gate,
        },
    }


# ── _spearman_correlation tests ───────────────────────────────────────────────

class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        # Same ranking → ρ = 1.0
        rho = _spearman_correlation([1, 2, 3, 4], [1, 2, 3, 4])
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        # Reversed ranking → ρ = -1.0
        rho = _spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1])
        assert rho == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self):
        # Known example: d²=[1,1,1,1] → ρ = 1 - 6*4/(4*15) = 1 - 24/60 = 0.6
        # [1,2,3,4] vs [2,1,4,3]: d=[1,1,1,1], Σd²=4
        rho = _spearman_correlation([1, 2, 3, 4], [2, 1, 4, 3])
        assert rho == pytest.approx(0.6, abs=1e-4)

    def test_two_items(self):
        # n=2: ρ = 1 - 6*d²/(2*3)
        # [1,2] vs [1,2]: d=0 → ρ=1.0
        rho = _spearman_correlation([1, 2], [1, 2])
        assert rho == 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            _spearman_correlation([1, 2], [1, 2, 3])

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            _spearman_correlation([1], [1])

    def test_result_in_valid_range(self):
        # Use valid rank sequences (each element is a rank in 1..n, with ties averaged)
        # [1, 2, 3, 4, 5] vs [3, 1, 5, 2, 4] — scrambled ranks, valid Spearman input
        rho = _spearman_correlation([1, 2, 3, 4, 5], [3, 1, 5, 2, 4])
        assert -1.0 <= rho <= 1.0


# ── _scores_to_ranks tests ────────────────────────────────────────────────────

class TestScoresToRanks:
    def test_no_ties(self):
        # [0.8, 0.5, 0.3] → rank 1=0.8, 2=0.5, 3=0.3
        ranks = _scores_to_ranks([0.8, 0.5, 0.3])
        assert ranks == [1.0, 2.0, 3.0]

    def test_tied_top_two(self):
        # [0.8, 0.8, 0.3] → tied ranks 1 and 2 average to 1.5, third gets rank 3
        ranks = _scores_to_ranks([0.8, 0.8, 0.3])
        assert ranks[0] == pytest.approx(1.5)
        assert ranks[1] == pytest.approx(1.5)
        assert ranks[2] == pytest.approx(3.0)

    def test_all_tied(self):
        # All same score → all get average rank (1+2+3)/3 = 2.0
        ranks = _scores_to_ranks([0.5, 0.5, 0.5])
        assert all(r == pytest.approx(2.0) for r in ranks)

    def test_single_item(self):
        ranks = _scores_to_ranks([0.7])
        assert ranks == [1.0]

    def test_preserves_indices(self):
        # Verify original index order is preserved
        ranks = _scores_to_ranks([0.3, 0.9, 0.6])
        # 0.9→rank1, 0.6→rank2, 0.3→rank3 → [3.0, 1.0, 2.0]
        assert ranks == [3.0, 1.0, 2.0]


# ── per_task_scores tests ─────────────────────────────────────────────────────

class TestPerTaskScores:
    def _two_task_records(self):
        return [
            _make_record("task_001", "gpt-4o",   "structured",   hybrid_score=0.80),
            _make_record("task_001", "claude",    "structured",   hybrid_score=0.70),
            _make_record("task_002", "gpt-4o",   "structured",   hybrid_score=0.30),
            _make_record("task_002", "claude",    "structured",   hybrid_score=0.25),
        ]

    def test_returns_dict_keyed_by_task(self):
        records = self._two_task_records()
        profiles = per_task_scores(records)
        assert set(profiles.keys()) == {"task_001", "task_002"}

    def test_mean_hybrid_correct(self):
        records = self._two_task_records()
        profiles = per_task_scores(records)
        assert profiles["task_001"]["mean_hybrid"] == pytest.approx(0.75, abs=1e-4)
        assert profiles["task_002"]["mean_hybrid"] == pytest.approx(0.275, abs=1e-4)

    def test_difficulty_tiers_assigned(self):
        records = self._two_task_records()
        profiles = per_task_scores(records)
        # task_001 mean=0.75 → easy (≥0.70)
        assert profiles["task_001"]["difficulty_tier"] == "easy"
        # task_002 mean=0.275 → hard (<0.40)
        assert profiles["task_002"]["difficulty_tier"] == "hard"

    def test_medium_tier(self):
        records = [_make_record("task_001", hybrid_score=0.55)]
        profiles = per_task_scores(records)
        assert profiles["task_001"]["difficulty_tier"] == "medium"

    def test_n_runs_counts_all(self):
        records = self._two_task_records()
        profiles = per_task_scores(records)
        assert profiles["task_001"]["n_runs"] == 2

    def test_error_runs_counted_in_n_but_not_model_scores(self):
        records = [
            _make_record("task_001", "gpt-4o", "structured", hybrid_score=0.80),
            _make_record("task_001", "gpt-4o", "zero_shot",  model_error="timeout"),
        ]
        profiles = per_task_scores(records)
        assert profiles["task_001"]["n_runs"] == 2
        # model_scores only for non-error runs
        assert "zero_shot" not in profiles["task_001"]["model_scores"].get("gpt-4o", {})

    def test_dominant_failure_check_identified(self):
        # revenue and reference failing → revenue and reference should appear
        records = [
            _make_record("task_001", failed_checks=["revenue", "reference"]),
        ]
        profiles = per_task_scores(records)
        # dominant_failure is whichever has highest rate — both 100% — max by key
        assert profiles["task_001"]["dominant_failure"] in {"revenue", "reference"}

    def test_empty_records_returns_empty(self):
        assert per_task_scores([]) == {}

    def test_std_is_zero_for_single_run(self):
        records = [_make_record("task_001", hybrid_score=0.7)]
        profiles = per_task_scores(records)
        assert profiles["task_001"]["std_hybrid"] == 0.0


# ── failure_mode_taxonomy tests ───────────────────────────────────────────────

class TestFailureModeTaxonomy:
    def test_returns_expected_keys(self):
        records = [_make_record()]
        result = failure_mode_taxonomy(records)
        assert "per_category" in result
        assert "dominant_category" in result
        assert "total_valid_runs" in result

    def test_three_categories_present(self):
        records = [_make_record()]
        result = failure_mode_taxonomy(records)
        assert set(result["per_category"].keys()) == {"structural", "formula", "content"}

    def test_all_checks_passing_zero_failure(self):
        # No failed checks → all failure rates should be 0.0
        records = [_make_record(failed_checks=[])]
        result = failure_mode_taxonomy(records)
        for cat in result["per_category"].values():
            assert cat["failure_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_formula_dominant_when_revenue_reference_fail(self):
        records = [
            _make_record(failed_checks=["revenue", "reference"]),
        ]
        result = failure_mode_taxonomy(records)
        assert result["dominant_category"] == "formula"

    def test_structural_dominant_when_integrity_sheets_fail(self):
        records = [_make_record(failed_checks=["integrity", "sheets"])]
        result = failure_mode_taxonomy(records)
        assert result["dominant_category"] == "structural"

    def test_total_valid_runs_excludes_errors(self):
        records = [
            _make_record("task_001"),
            _make_record("task_001", model_error="timeout"),
        ]
        result = failure_mode_taxonomy(records)
        assert result["total_valid_runs"] == 1

    def test_all_error_records(self):
        records = [_make_record(model_error="timeout")]
        result = failure_mode_taxonomy(records)
        assert result["total_valid_runs"] == 0
        assert result["dominant_category"] == "none"


# ── model_task_agreement tests ────────────────────────────────────────────────

class TestModelTaskAgreement:
    def _two_model_records(self):
        # Both models agree: task_001 is easy, task_002 is hard
        return [
            _make_record("task_001", "gpt-4o",  "structured", hybrid_score=0.85),
            _make_record("task_001", "claude",   "structured", hybrid_score=0.80),
            _make_record("task_002", "gpt-4o",  "structured", hybrid_score=0.30),
            _make_record("task_002", "claude",   "structured", hybrid_score=0.25),
        ]

    def test_returns_expected_keys(self):
        records = self._two_model_records()
        result = model_task_agreement(records)
        assert "task_ids" in result
        assert "pairwise_spearman" in result
        assert "mean_spearman" in result
        assert "interpretation" in result

    def test_perfect_agreement(self):
        # Both models rank tasks the same way → ρ = 1.0
        records = self._two_model_records()
        result = model_task_agreement(records)
        assert result["mean_spearman"] == pytest.approx(1.0, abs=1e-4)

    def test_task_ids_sorted(self):
        records = self._two_model_records()
        result = model_task_agreement(records)
        assert result["task_ids"] == sorted(result["task_ids"])

    def test_high_agreement_interpretation(self):
        records = self._two_model_records()
        result = model_task_agreement(records)
        assert "high agreement" in result["interpretation"]

    def test_single_model_no_pairs(self):
        records = [
            _make_record("task_001", "gpt-4o", "structured", hybrid_score=0.8),
            _make_record("task_002", "gpt-4o", "structured", hybrid_score=0.5),
        ]
        result = model_task_agreement(records)
        # No pairs → mean_spearman is None (no pairs to compute)
        assert result["mean_spearman"] is None
        assert "fewer than 2" in result["interpretation"]

    def test_three_tasks_consistent_ranking(self):
        # 3 tasks, both models agree on ranking
        records = [
            _make_record("task_001", "gpt-4o",  "structured", hybrid_score=0.90),
            _make_record("task_002", "gpt-4o",  "structured", hybrid_score=0.60),
            _make_record("task_003", "gpt-4o",  "structured", hybrid_score=0.30),
            _make_record("task_001", "claude",   "structured", hybrid_score=0.85),
            _make_record("task_002", "claude",   "structured", hybrid_score=0.55),
            _make_record("task_003", "claude",   "structured", hybrid_score=0.25),
        ]
        result = model_task_agreement(records)
        assert result["mean_spearman"] == pytest.approx(1.0, abs=1e-4)

    def test_errors_excluded_from_scores(self):
        # Error record for claude on task_001 should not bias the mean
        records = [
            _make_record("task_001", "gpt-4o",  "structured", hybrid_score=0.80),
            _make_record("task_001", "claude",   "structured", model_error="timeout"),
            _make_record("task_002", "gpt-4o",  "structured", hybrid_score=0.40),
            _make_record("task_002", "claude",   "structured", hybrid_score=0.35),
        ]
        result = model_task_agreement(records)
        # claude has no valid data for task_001 → treated as 0.0
        assert "task_001" in result["task_ids"]
        assert "task_002" in result["task_ids"]


# ── task_difficulty_ranking tests ─────────────────────────────────────────────

class TestTaskDifficultyRanking:
    def test_sorted_hardest_first(self):
        records = [
            _make_record("task_001", hybrid_score=0.80),
            _make_record("task_002", hybrid_score=0.30),
            _make_record("task_003", hybrid_score=0.55),
        ]
        ranking = task_difficulty_ranking(records)
        means = [r["mean_hybrid"] for r in ranking]
        assert means == sorted(means)  # ascending = hardest first

    def test_ranks_assigned_from_one(self):
        records = [
            _make_record("task_001", hybrid_score=0.80),
            _make_record("task_002", hybrid_score=0.30),
        ]
        ranking = task_difficulty_ranking(records)
        ranks = [r["rank"] for r in ranking]
        assert ranks == [1, 2]

    def test_each_row_has_required_keys(self):
        records = [_make_record()]
        ranking = task_difficulty_ranking(records)
        required = {"task_id", "mean_hybrid", "std_hybrid", "difficulty_tier",
                    "dominant_failure", "n_runs", "rank"}
        assert required.issubset(set(ranking[0].keys()))

    def test_empty_input(self):
        assert task_difficulty_ranking([]) == []


# ── difficulty_calibration_report tests ──────────────────────────────────────

class TestDifficultyCalibrationReport:
    def _spread_records(self):
        # 5 tasks spanning hard/medium/easy → well-calibrated (n≥5 needed to avoid small_n)
        return [
            _make_record("task_001", hybrid_score=0.15),  # hard
            _make_record("task_002", hybrid_score=0.30),  # hard
            _make_record("task_003", hybrid_score=0.55),  # medium
            _make_record("task_004", hybrid_score=0.75),  # easy
            _make_record("task_005", hybrid_score=0.85),  # easy
        ]

    def test_empty_records(self):
        result = difficulty_calibration_report([])
        assert result["n_tasks"] == 0
        assert result["calibration_verdict"] == "no_data"

    def test_tier_counts_correct(self):
        records = self._spread_records()
        result = difficulty_calibration_report(records)
        # _spread_records: task_001(0.15)→hard, task_002(0.30)→hard,
        # task_003(0.55)→medium, task_004(0.75)→easy, task_005(0.85)→easy
        assert result["tier_counts"]["hard"]   == 2
        assert result["tier_counts"]["medium"] == 1
        assert result["tier_counts"]["easy"]   == 2

    def test_well_calibrated_verdict(self):
        records = self._spread_records()
        result = difficulty_calibration_report(records)
        # std should be high enough, good tier spread
        assert result["calibration_verdict"] == "well_calibrated"

    def test_ceiling_risk_verdict(self):
        # All easy tasks
        records = [
            _make_record(f"task_{i:03d}", hybrid_score=0.85)
            for i in range(1, 6)
        ]
        result = difficulty_calibration_report(records)
        assert result["calibration_verdict"] in {"ceiling_risk", "low_discrimination"}

    def test_floor_risk_verdict(self):
        # All hard tasks
        records = [
            _make_record(f"task_{i:03d}", hybrid_score=0.25)
            for i in range(1, 6)
        ]
        result = difficulty_calibration_report(records)
        assert result["calibration_verdict"] in {"floor_risk", "low_discrimination"}

    def test_small_n_verdict(self):
        records = [
            _make_record("task_001", hybrid_score=0.5),
            _make_record("task_002", hybrid_score=0.6),
        ]
        result = difficulty_calibration_report(records)
        assert result["calibration_verdict"] == "small_n"

    def test_floor_tasks_detected(self):
        records = [
            _make_record("task_001", hybrid_score=0.05),  # floor
            _make_record("task_002", hybrid_score=0.70),
        ]
        result = difficulty_calibration_report(records)
        assert "task_001" in result["floor_tasks"]

    def test_ceiling_tasks_detected(self):
        records = [
            _make_record("task_001", hybrid_score=0.95),  # ceiling
            _make_record("task_002", hybrid_score=0.50),
        ]
        result = difficulty_calibration_report(records)
        assert "task_001" in result["ceiling_tasks"]

    def test_discrimination_ok_true_when_std_high(self):
        records = self._spread_records()
        result = difficulty_calibration_report(records)
        assert result["std_difficulty"] >= 0.10
        assert result["discrimination_ok"] is True

    def test_discrimination_false_when_std_low(self):
        # All nearly same score
        records = [
            _make_record(f"task_{i:03d}", hybrid_score=0.50 + i * 0.001)
            for i in range(1, 6)
        ]
        result = difficulty_calibration_report(records)
        assert result["discrimination_ok"] is False

    def test_mean_and_std_computed(self):
        records = [
            _make_record("task_001", hybrid_score=0.6),
            _make_record("task_002", hybrid_score=0.8),
        ]
        result = difficulty_calibration_report(records)
        assert result["mean_difficulty"] == pytest.approx(0.7, abs=1e-4)
        assert result["std_difficulty"] == pytest.approx(0.1414, abs=1e-2)
