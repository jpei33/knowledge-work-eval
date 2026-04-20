"""
test_analyze_results.py — Tests for analyze_results.py functions.

Tests cover: load_results, filter_records, bootstrap_ci_scores,
elicitation_gap, bucket_distribution, per_check_failure_rates, judge_metrics,
and model_comparison. All tests use in-memory synthetic records — no file I/O
except for load_results tests which use tmp_path.

Run:
    cd Code/data-cleaning-eval
    pytest tests/test_analyze_results.py -v
"""

from __future__ import annotations

import importlib.util as _ilu
import json
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

_spec = _ilu.spec_from_file_location("analyze_results", _EVAL_ROOT / "analyze_results.py")
_ar   = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

load_results           = _ar.load_results
filter_records         = _ar.filter_records
bootstrap_ci_scores    = _ar.bootstrap_ci_scores
elicitation_gap        = _ar.elicitation_gap
bucket_distribution    = _ar.bucket_distribution
per_check_failure_rates = _ar.per_check_failure_rates
judge_metrics          = _ar.judge_metrics
model_comparison       = _ar.model_comparison


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_check(name: str, passed: bool, score: float = None) -> dict:
    if score is None:
        score = 0.9 if passed else 0.1
    return {"name": name, "passed": passed, "score": score, "detail": "test"}


CHECK_NAMES = ["integrity", "sheets", "errors", "revenue", "summary", "reference"]


def _make_record(
    task_id="task_001",
    model="gpt-4o",
    elicitation="structured",
    hybrid_score=0.8,
    prog_score=0.85,
    judge_score=0.75,
    bucket="agree-good",
    gate_triggered=False,
    model_error=None,
    check_overrides: dict = None,
    position_consistent=True,
) -> dict:
    """Build a minimal but complete synthetic record for testing."""
    checks = []
    for name in CHECK_NAMES:
        passed = True
        if check_overrides and name in check_overrides:
            passed = check_overrides[name]
        checks.append(_make_check(name, passed))

    judge_dict = None
    if not gate_triggered and model_error is None:
        judge_dict = {
            "score": judge_score,
            "score_ab": round(1.0 - judge_score, 1),
            "score_ba": round(judge_score, 1),
            "position_consistent": position_consistent,
            "reasoning_ab": "Test reasoning AB.",
            "reasoning_ba": "Test reasoning BA.",
        }

    return {
        "run_id":      f"{task_id}__{model}__{elicitation}__20260412T100000",
        "task_id":     task_id,
        "model":       model,
        "elicitation": elicitation,
        "started_at":  "2026-04-12T10:00:00+00:00",
        "model_error": model_error,
        "programmatic": None if model_error else {
            "score":                     prog_score,
            "file_integrity":            True,
            "sheets_present":            ["Raw Data", "Cleaned Data", "Summary"],
            "sheets_missing":            [],
            "formula_error_count":       0,
            "revenue_formula_fraction":  0.95 if (check_overrides or {}).get("revenue", True) else 0.0,
            "summary_formula_fraction":  0.90,
            "summary_references_cleaned": True,
            "cleaned_row_count":         70,
            "raw_row_count":             75,
            "checks":                    checks,
        },
        "hybrid": {
            "hybrid_score":       hybrid_score,
            "programmatic_score": prog_score,
            "judge":              judge_dict,
            "agreement_bucket":   bucket,
            "gate_triggered":     gate_triggered,
        },
    }


@pytest.fixture
def simple_records():
    """4 records: 2 models × 2 elicitation modes, all successful."""
    return [
        _make_record("task_001", "gpt-4o",   "structured",  0.82, 0.85, 0.80, "agree-good"),
        _make_record("task_001", "gpt-4o",   "zero_shot",   0.55, 0.60, 0.51, "judge-rescues"),
        _make_record("task_001", "claude",   "structured",  0.75, 0.78, 0.72, "agree-good"),
        _make_record("task_001", "claude",   "zero_shot",   0.42, 0.50, 0.37, "agree-bad"),
    ]


@pytest.fixture
def multi_task_records():
    """10 records: 2 models × 1 elicitation × 5 tasks."""
    records = []
    gpt_scores   = [0.90, 0.75, 0.60, 0.85, 0.70]
    claude_scores = [0.80, 0.65, 0.55, 0.78, 0.62]
    for i, (gs, cs) in enumerate(zip(gpt_scores, claude_scores), 1):
        records.append(_make_record(f"task_{i:03d}", "gpt-4o", "structured", gs, gs, gs))
        records.append(_make_record(f"task_{i:03d}", "claude", "structured", cs, cs, cs))
    return records


# ── load_results ──────────────────────────────────────────────────────────────

class TestLoadResults:

    def test_loads_all_records(self, tmp_path):
        path = tmp_path / "results.jsonl"
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loaded = load_results(path)
        assert len(loaded) == 3

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n\n')
        loaded = load_results(path)
        assert len(loaded) == 2

    def test_raises_on_malformed_json(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text('{"a": 1}\nNOT_JSON\n')
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_results(path)

    def test_roundtrip_complex_record(self, tmp_path, simple_records):
        path = tmp_path / "results.jsonl"
        with open(path, "w") as f:
            for r in simple_records:
                f.write(json.dumps(r) + "\n")
        loaded = load_results(path)
        assert loaded[0]["hybrid"]["hybrid_score"] == simple_records[0]["hybrid"]["hybrid_score"]


# ── filter_records ────────────────────────────────────────────────────────────

class TestFilterRecords:

    def test_filter_by_model(self, simple_records):
        result = filter_records(simple_records, model="gpt-4o")
        assert all(r["model"] == "gpt-4o" for r in result)
        assert len(result) == 2

    def test_filter_by_elicitation(self, simple_records):
        result = filter_records(simple_records, elicitation="zero_shot")
        assert all(r["elicitation"] == "zero_shot" for r in result)
        assert len(result) == 2

    def test_filter_combined(self, simple_records):
        result = filter_records(simple_records, model="gpt-4o", elicitation="structured")
        assert len(result) == 1
        assert result[0]["hybrid"]["hybrid_score"] == pytest.approx(0.82)

    def test_exclude_errors_default(self):
        records = [
            _make_record("task_001", "gpt-4o", "structured", model_error="API timeout"),
            _make_record("task_002", "gpt-4o", "structured"),
        ]
        result = filter_records(records)
        assert len(result) == 1
        assert result[0]["task_id"] == "task_002"

    def test_include_errors_when_flag_false(self):
        records = [
            _make_record("task_001", "gpt-4o", "structured", model_error="API timeout"),
            _make_record("task_002", "gpt-4o", "structured"),
        ]
        result = filter_records(records, exclude_errors=False)
        assert len(result) == 2

    def test_no_filter_returns_all(self, simple_records):
        assert len(filter_records(simple_records)) == 4


# ── bootstrap_ci_scores ───────────────────────────────────────────────────────

class TestBootstrapCIScores:

    def test_point_estimate_is_mean(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        point, _, _ = bootstrap_ci_scores(scores, n_resamples=1000)
        assert point == pytest.approx(sum(scores) / len(scores))

    def test_ci_contains_point_estimate(self):
        scores = [0.6, 0.65, 0.7, 0.75, 0.8]
        point, lo, hi = bootstrap_ci_scores(scores, n_resamples=1000)
        assert lo <= point <= hi

    def test_ci_width_shrinks_with_more_samples(self):
        import numpy as np
        rng = np.random.default_rng(0)
        small = list(rng.normal(0.7, 0.1, 5))
        large = list(rng.normal(0.7, 0.1, 100))
        _, lo_s, hi_s = bootstrap_ci_scores(small, n_resamples=2000)
        _, lo_l, hi_l = bootstrap_ci_scores(large, n_resamples=2000)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_raises_on_empty_scores(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci_scores([])

    def test_all_same_scores_zero_width(self):
        scores = [0.75] * 10
        point, lo, hi = bootstrap_ci_scores(scores, n_resamples=1000)
        assert point == pytest.approx(0.75)
        assert (hi - lo) == pytest.approx(0.0, abs=1e-9)

    def test_ci_bounds_in_valid_range(self):
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        _, lo, hi = bootstrap_ci_scores(scores, n_resamples=1000)
        assert 0.0 <= lo <= hi <= 1.0


# ── elicitation_gap ───────────────────────────────────────────────────────────

class TestElicitationGap:

    def test_gap_sign_and_value(self, simple_records):
        eg = elicitation_gap(simple_records, "gpt-4o")
        assert eg["gap"] == pytest.approx(eg["structured_mean"] - eg["zero_shot_mean"], abs=0.001)
        assert eg["gap"] > 0   # structured should beat zero_shot

    def test_gap_fields_present(self, simple_records):
        eg = elicitation_gap(simple_records, "gpt-4o")
        for field in ("model", "structured_mean", "zero_shot_mean", "gap",
                      "structured_ci", "zero_shot_ci", "n_structured", "n_zero_shot"):
            assert field in eg

    def test_gap_raises_on_missing_elicitation(self):
        records = [_make_record("task_001", "gpt-4o", "structured")]
        with pytest.raises(ValueError):
            elicitation_gap(records, "gpt-4o")

    def test_ci_tuple_has_three_elements(self, simple_records):
        eg = elicitation_gap(simple_records, "gpt-4o")
        assert len(eg["structured_ci"]) == 3
        assert len(eg["zero_shot_ci"])  == 3


# ── bucket_distribution ───────────────────────────────────────────────────────

class TestBucketDistribution:

    def test_counts_correct(self, simple_records):
        dist = bucket_distribution(simple_records)
        # gpt-4o × structured → agree-good
        assert dist["gpt-4o"]["structured"]["agree-good"] == 1
        # gpt-4o × zero_shot → judge-rescues
        assert dist["gpt-4o"]["zero_shot"]["judge-rescues"] == 1

    def test_all_five_buckets_present(self, simple_records):
        dist = bucket_distribution(simple_records)
        BUCKETS = ["agree-good", "agree-bad", "judge-rescues", "programmatic-catches", "gate-triggered"]
        for model in dist:
            for elicitation in dist[model]:
                for bucket in BUCKETS:
                    assert bucket in dist[model][elicitation]

    def test_zero_counts_for_missing_buckets(self, simple_records):
        dist = bucket_distribution(simple_records)
        # claude × structured has no agree-bad
        assert dist["claude"]["structured"]["agree-bad"] == 0

    def test_multiple_tasks_counted(self, multi_task_records):
        dist = bucket_distribution(multi_task_records)
        total = sum(dist["gpt-4o"]["structured"].values())
        assert total == 5   # 5 tasks


# ── per_check_failure_rates ───────────────────────────────────────────────────

class TestPerCheckFailureRates:

    def test_all_passing_gives_zero_failure(self, simple_records):
        rates = per_check_failure_rates(simple_records)
        for model in rates:
            for elicitation in rates[model]:
                for check, rate in rates[model][elicitation].items():
                    if rate is not None:
                        assert rate == pytest.approx(0.0), f"{model}/{elicitation}/{check} should be 0"

    def test_known_failure_rate(self):
        """3 records: revenue fails in 2 of 3 → failure rate = 0.667."""
        records = [
            _make_record("task_001", "gpt-4o", "structured", check_overrides={"revenue": False}),
            _make_record("task_002", "gpt-4o", "structured", check_overrides={"revenue": False}),
            _make_record("task_003", "gpt-4o", "structured"),  # revenue passes
        ]
        rates = per_check_failure_rates(records)
        assert rates["gpt-4o"]["structured"]["revenue"] == pytest.approx(2/3, abs=0.01)

    def test_skips_model_error_records(self):
        records = [
            _make_record("task_001", "gpt-4o", "structured", model_error="timeout"),
            _make_record("task_002", "gpt-4o", "structured"),
        ]
        rates = per_check_failure_rates(records)
        # Only 1 non-error record → integrity passes → 0% failure
        assert rates["gpt-4o"]["structured"]["integrity"] == pytest.approx(0.0)

    def test_none_for_no_data(self):
        """If all records have model_error, no programmatic data → all None."""
        records = [
            _make_record("task_001", "gpt-4o", "structured", model_error="timeout"),
        ]
        rates = per_check_failure_rates(records)
        # No programmatic data at all for this combination
        assert "gpt-4o" not in rates


# ── judge_metrics ─────────────────────────────────────────────────────────────

class TestJudgeMetrics:

    def test_all_judges_called_no_gate(self, simple_records):
        jm = judge_metrics(simple_records)
        assert jm["gate_trigger_rate"] == pytest.approx(0.0)
        assert jm["judge_call_rate"]   == pytest.approx(1.0)

    def test_position_consistency_rate(self):
        """2 consistent, 1 inconsistent → rate = 2/3."""
        records = [
            _make_record("t1", "gpt-4o", "structured", position_consistent=True),
            _make_record("t2", "gpt-4o", "structured", position_consistent=True),
            _make_record("t3", "gpt-4o", "structured", position_consistent=False),
        ]
        jm = judge_metrics(records)
        assert jm["position_consistency_rate"] == pytest.approx(2/3, abs=0.01)

    def test_gate_triggered_reduces_judge_call_rate(self):
        records = [
            _make_record("t1", "gpt-4o", "structured", gate_triggered=True),
            _make_record("t2", "gpt-4o", "structured", gate_triggered=False),
        ]
        jm = judge_metrics(records)
        assert jm["gate_trigger_rate"] == pytest.approx(0.5)
        assert jm["judge_call_rate"]   == pytest.approx(0.5)


# ── model_comparison ──────────────────────────────────────────────────────────

class TestModelComparison:

    def test_returns_one_row_per_combo(self, simple_records):
        rows = model_comparison(simple_records)
        combos = {(r["model"], r["elicitation"]) for r in rows}
        assert len(combos) == 4   # 2 models × 2 elicitations

    def test_sorted_by_hybrid_score_descending(self, simple_records):
        rows = model_comparison(simple_records)
        scores = [r["mean_hybrid"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_required_fields_present(self, simple_records):
        rows = model_comparison(simple_records)
        for row in rows:
            for field in ("model", "elicitation", "n", "mean_hybrid",
                          "ci_lo", "ci_hi", "ci_width", "mean_prog",
                          "dominant_bucket", "top_failing_check"):
                assert field in row, f"Missing field: {field}"

    def test_ci_width_positive(self, multi_task_records):
        rows = model_comparison(multi_task_records)
        for row in rows:
            assert row["ci_width"] >= 0


# ── Synthetic results file integration test ───────────────────────────────────

_SYNTHETIC_PATH = Path(__file__).resolve().parent.parent / "results" / "synthetic_results.jsonl"

@pytest.mark.skipif(
    not _SYNTHETIC_PATH.exists(),
    reason="Synthetic results file not generated — run scripts/generate_synthetic_results.py"
)
class TestSyntheticResults:

    @pytest.fixture
    def records(self):
        return load_results(_SYNTHETIC_PATH)

    def test_expected_record_count(self, records):
        assert len(records) == 20   # 5 tasks × 2 models × 2 elicitations

    def test_gpt4o_structured_beats_zero_shot(self, records):
        eg = elicitation_gap(records, "gpt-4o")
        assert eg["gap"] > 0

    def test_claude_structured_beats_zero_shot(self, records):
        eg = elicitation_gap(records, "claude-3-5-sonnet")
        assert eg["gap"] > 0

    def test_all_five_buckets_seen(self, records):
        dist = bucket_distribution(records)
        all_buckets = set()
        for model_data in dist.values():
            for elicitation_data in model_data.values():
                all_buckets.update(k for k, v in elicitation_data.items() if v > 0)
        expected = {"agree-good", "agree-bad", "judge-rescues", "programmatic-catches", "gate-triggered"}
        assert all_buckets >= expected

    def test_report_runs_without_error(self, records):
        """Smoke test: print_report should complete without raising."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ar.print_report(records)
        output = buf.getvalue()
        assert "ANALYSIS REPORT" in output
        assert "Elicitation Gap" in output
