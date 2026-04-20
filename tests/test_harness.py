"""
test_harness.py — Tests for the eval harness pipeline.

All tests use MockModelClient + MockJudge. No real API calls. No network.
Covers: task loading, prompt selection, output path logic, JSONL logging,
model failure handling, gate behavior, grid runner, and record schema.

Run:
    cd Code/data-cleaning-eval
    pytest tests/test_harness.py -v
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
_EVAL_ROOT  = Path(__file__).resolve().parent.parent
_FIXTURES   = Path(__file__).resolve().parent / "fixtures"
_TASKS_DIR  = _EVAL_ROOT / "tasks"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

# Import harness by file path (same importlib pattern as test_llm_judge.py)
_spec = _ilu.spec_from_file_location("harness", _EVAL_ROOT / "harness.py")
_harness = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_harness)

run_one          = _harness.run_one
run_grid         = _harness.run_grid
JSONLLogger      = _harness.JSONLLogger
MockModelClient  = _harness.MockModelClient

# Import judge mocks
_judge_spec = _ilu.spec_from_file_location("llm_judge", _EVAL_ROOT / "llm_judge.py")
_llm_judge  = _ilu.module_from_spec(_judge_spec)
_judge_spec.loader.exec_module(_llm_judge)
MockJudge = _llm_judge.MockJudge

from common.scoring import HybridOracle  # noqa: E402

# Task path
TASK_PATH = _TASKS_DIR / "task_001.json"

pytestmark = pytest.mark.skipif(
    not TASK_PATH.exists(),
    reason="task_001.json not found — skipping harness tests"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def good_client():
    return MockModelClient(_FIXTURES / "good_minimal.xlsx", model_name="mock-good")

@pytest.fixture
def bad_client():
    return MockModelClient(_FIXTURES / "bad_hardcoded_revenue.xlsx", model_name="mock-bad")

@pytest.fixture
def missing_summary_client():
    return MockModelClient(_FIXTURES / "bad_missing_summary.xlsx", model_name="mock-missing-summary")

@pytest.fixture
def winning_judge():
    """MockJudge that always says model wins."""
    return MockJudge(fixed_score_ab=0.0, fixed_score_ba=1.0)

@pytest.fixture
def losing_judge():
    """MockJudge that always says gold wins."""
    return MockJudge(fixed_score_ab=1.0, fixed_score_ba=0.0)

@pytest.fixture
def oracle():
    return HybridOracle(prog_weight=0.4, judge_weight=0.6, prog_gate=0.3)


# ── Task loading and prompt selection ─────────────────────────────────────────

class TestTaskLoading:

    def test_structured_prompt_selected(self, good_client, winning_judge, oracle, tmp_path):
        result = run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=winning_judge,
            oracle=oracle,
        )
        assert result is not None

    def test_zero_shot_prompt_selected(self, good_client, winning_judge, oracle, tmp_path):
        result = run_one(
            TASK_PATH, good_client, "zero_shot",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=winning_judge,
            oracle=oracle,
        )
        assert result is not None

    def test_invalid_elicitation_raises(self, good_client, oracle, tmp_path):
        with pytest.raises(ValueError, match="Elicitation mode"):
            run_one(
                TASK_PATH, good_client, "chain_of_thought",
                output_dir=tmp_path / "outputs",
                logger=JSONLLogger(tmp_path / "results.jsonl"),
                oracle=oracle,
            )


# ── Output path logic ─────────────────────────────────────────────────────────

class TestOutputPaths:

    def test_output_saved_under_model_elicitation(self, good_client, oracle, tmp_path):
        """Output file goes to outputs/{model_id}/{elicitation}/{task_id}.xlsx."""
        run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            oracle=oracle,
        )
        expected = tmp_path / "outputs" / "mock-good" / "structured" / "task_001.xlsx"
        assert expected.exists(), f"Expected output at {expected}"

    def test_different_elicitations_dont_overwrite(self, good_client, oracle, tmp_path):
        """zero_shot and structured produce separate output files."""
        logger = JSONLLogger(tmp_path / "results.jsonl")
        for elicitation in ("zero_shot", "structured"):
            run_one(TASK_PATH, good_client, elicitation,
                    output_dir=tmp_path / "outputs",
                    logger=logger, oracle=oracle)

        assert (tmp_path / "outputs" / "mock-good" / "zero_shot" / "task_001.xlsx").exists()
        assert (tmp_path / "outputs" / "mock-good" / "structured" / "task_001.xlsx").exists()


# ── JSONL logging ─────────────────────────────────────────────────────────────

class TestJSONLLogging:

    def test_one_record_written_per_run(self, good_client, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_one(TASK_PATH, good_client, "structured",
                output_dir=tmp_path / "outputs",
                logger=logger, oracle=oracle)
        records = logger.read_all()
        assert len(records) == 1

    def test_records_accumulate_across_runs(self, good_client, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        for _ in range(3):
            run_one(TASK_PATH, good_client, "structured",
                    output_dir=tmp_path / "outputs",
                    logger=logger, oracle=oracle)
        assert len(logger.read_all()) == 3

    def test_record_schema_has_required_fields(self, good_client, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_one(TASK_PATH, good_client, "structured",
                output_dir=tmp_path / "outputs",
                logger=logger, oracle=oracle)
        record = logger.read_all()[0]
        for field in ("run_id", "task_id", "model", "elicitation",
                      "started_at", "model_error", "programmatic", "hybrid"):
            assert field in record, f"Missing field: {field}"

    def test_record_run_id_contains_task_model_elicitation(self, good_client, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_one(TASK_PATH, good_client, "structured",
                output_dir=tmp_path / "outputs",
                logger=logger, oracle=oracle)
        run_id = logger.read_all()[0]["run_id"]
        assert "task_001" in run_id
        assert "mock-good" in run_id
        assert "structured" in run_id

    def test_hybrid_dict_in_record(self, good_client, winning_judge, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_one(TASK_PATH, good_client, "structured",
                output_dir=tmp_path / "outputs",
                logger=logger, judge=winning_judge, oracle=oracle)
        hybrid = logger.read_all()[0]["hybrid"]
        assert "hybrid_score" in hybrid
        assert "agreement_bucket" in hybrid
        assert "gate_triggered" in hybrid

    def test_programmatic_dict_in_record(self, good_client, oracle, tmp_path):
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_one(TASK_PATH, good_client, "structured",
                output_dir=tmp_path / "outputs",
                logger=logger, oracle=oracle)
        prog = logger.read_all()[0]["programmatic"]
        assert prog is not None
        assert "score" in prog
        assert "checks" in prog

    def test_jsonl_each_line_valid_json(self, good_client, oracle, tmp_path):
        """Each line in the JSONL file must be independently parseable."""
        log_path = tmp_path / "results.jsonl"
        logger = JSONLLogger(log_path)
        for _ in range(3):
            run_one(TASK_PATH, good_client, "structured",
                    output_dir=tmp_path / "outputs",
                    logger=logger, oracle=oracle)
        with open(log_path) as f:
            for i, line in enumerate(f):
                parsed = json.loads(line)   # raises if invalid
                assert "run_id" in parsed, f"Line {i} missing run_id"


# ── Scoring integration ───────────────────────────────────────────────────────

class TestScoringIntegration:

    def test_good_fixture_scores_above_gate(self, good_client, winning_judge, oracle, tmp_path):
        """good_minimal fixture passes all checks → prog_score well above gate."""
        result = run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=winning_judge, oracle=oracle,
        )
        assert result.programmatic_score > 0.3
        assert result.gate_triggered is False

    def test_good_fixture_with_winning_judge_agree_good(self, good_client, winning_judge, oracle, tmp_path):
        result = run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=winning_judge, oracle=oracle,
        )
        assert result.agreement_bucket == "agree-good"

    def test_good_fixture_with_losing_judge_programmatic_catches(
        self, good_client, losing_judge, oracle, tmp_path
    ):
        result = run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=losing_judge, oracle=oracle,
        )
        assert result.agreement_bucket == "programmatic-catches"

    def test_no_judge_triggers_gate(self, good_client, oracle, tmp_path):
        """When judge=None, gate always triggers regardless of prog score."""
        result = run_one(
            TASK_PATH, good_client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            judge=None, oracle=oracle,
        )
        assert result.gate_triggered is True
        assert result.agreement_bucket == "gate-triggered"


# ── Model failure handling ────────────────────────────────────────────────────

class TestModelFailure:

    def test_model_failure_logs_zero_score(self, oracle, tmp_path):
        """If model client raises, the run is logged with score=0 not skipped."""
        class FailingClient(MockModelClient):
            def run(self, prompt, raw_csv_path, output_path):
                raise RuntimeError("Simulated API timeout")

        client = FailingClient(_FIXTURES / "good_minimal.xlsx", "failing-model")
        logger = JSONLLogger(tmp_path / "results.jsonl")
        result = run_one(
            TASK_PATH, client, "structured",
            output_dir=tmp_path / "outputs",
            logger=logger, oracle=oracle,
        )
        assert result.hybrid_score == 0.0
        records = logger.read_all()
        assert len(records) == 1
        assert records[0]["model_error"] is not None
        assert "API timeout" in records[0]["model_error"]

    def test_model_failure_does_not_raise(self, oracle, tmp_path):
        """A model failure is captured and returned, not propagated as exception."""
        class FailingClient(MockModelClient):
            def run(self, prompt, raw_csv_path, output_path):
                raise RuntimeError("Network error")

        client = FailingClient(_FIXTURES / "good_minimal.xlsx", "failing-model")
        # Should NOT raise
        result = run_one(
            TASK_PATH, client, "structured",
            output_dir=tmp_path / "outputs",
            logger=JSONLLogger(tmp_path / "results.jsonl"),
            oracle=oracle,
        )
        assert result is not None


# ── Grid runner ───────────────────────────────────────────────────────────────

class TestGridRunner:

    def test_grid_produces_n_results(self, oracle, tmp_path):
        """Grid of 1 task × 2 models × 2 elicitation modes = 4 results."""
        clients = [
            MockModelClient(_FIXTURES / "good_minimal.xlsx", "mock-model-a"),
            MockModelClient(_FIXTURES / "bad_hardcoded_revenue.xlsx", "mock-model-b"),
        ]
        logger = JSONLLogger(tmp_path / "results.jsonl")
        results = run_grid(
            task_paths=[TASK_PATH],
            model_clients=clients,
            elicitation_modes=["zero_shot", "structured"],
            output_dir=tmp_path / "outputs",
            logger=logger,
            judge=None,
            oracle=oracle,
        )
        assert len(results) == 4
        assert len(logger.read_all()) == 4

    def test_grid_different_models_logged_separately(self, oracle, tmp_path):
        """Each model's results are logged with the correct model identifier."""
        clients = [
            MockModelClient(_FIXTURES / "good_minimal.xlsx", "model-alpha"),
            MockModelClient(_FIXTURES / "good_minimal.xlsx", "model-beta"),
        ]
        logger = JSONLLogger(tmp_path / "results.jsonl")
        run_grid(
            task_paths=[TASK_PATH],
            model_clients=clients,
            elicitation_modes=["structured"],
            output_dir=tmp_path / "outputs",
            logger=logger, oracle=oracle,
        )
        records = logger.read_all()
        models_logged = {r["model"] for r in records}
        assert "model-alpha" in models_logged
        assert "model-beta" in models_logged


# ── JSONLLogger unit tests ────────────────────────────────────────────────────

class TestJSONLLoggerUnit:

    def test_read_all_empty_on_missing_file(self, tmp_path):
        logger = JSONLLogger(tmp_path / "nonexistent.jsonl")
        assert logger.read_all() == []

    def test_write_and_read_roundtrip(self, tmp_path):
        logger = JSONLLogger(tmp_path / "test.jsonl")
        logger.write({"key": "value", "num": 42})
        records = logger.read_all()
        assert records == [{"key": "value", "num": 42}]

    def test_append_preserves_order(self, tmp_path):
        logger = JSONLLogger(tmp_path / "test.jsonl")
        for i in range(5):
            logger.write({"i": i})
        records = logger.read_all()
        assert [r["i"] for r in records] == list(range(5))

    def test_nested_dicts_preserved(self, tmp_path):
        logger = JSONLLogger(tmp_path / "test.jsonl")
        logger.write({"outer": {"inner": [1, 2, 3]}})
        assert logger.read_all()[0]["outer"]["inner"] == [1, 2, 3]
