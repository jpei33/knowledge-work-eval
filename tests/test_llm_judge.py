"""
test_llm_judge.py — Tests for BlindedPairwiseJudge internals and MockJudge.

Test strategy:
  - Never calls the real Gemini API. All BlindedPairwiseJudge tests are skipped
    unless GEMINI_API_KEY is set. MockJudge tests run unconditionally.
  - Covers: score parsing, position-swap reconciliation, consistency flag,
    agreement bucket classification, gate integration, and mock behavior.

Run:
    cd Code/data-cleaning-eval
    pytest tests/test_llm_judge.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add repo root to path (same pattern as programmatic_checks tests)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.scoring import HybridOracle  # noqa: E402

# ── Import path fix: llm_judge.py lives one level up from tests/ ──────────────
# Import directly by path so we don't need the package to be installed.
import importlib.util as _ilu  # noqa: E402

_JUDGE_MODULE_PATH = Path(__file__).resolve().parent.parent / "llm_judge.py"
_spec = _ilu.spec_from_file_location("llm_judge", _JUDGE_MODULE_PATH)
_llm_judge_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_llm_judge_mod)

MockJudge    = _llm_judge_mod.MockJudge
_parse_score = _llm_judge_mod._parse_score
_reconcile   = _llm_judge_mod._reconcile
BlindedPairwiseJudge = _llm_judge_mod.BlindedPairwiseJudge

# Gold fixture path (may not exist in CI — used only for integration test)
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLD_PATH = Path(__file__).resolve().parent.parent / "gold" / "task_001_gold.xlsx"

TASK_PROMPT = "Clean the CSV and produce a 3-sheet xlsx with Raw Data, Cleaned Data, and Summary."


# ── Score parsing tests ───────────────────────────────────────────────────────

class TestParseScore:
    """Tests for _parse_score() — extracts numeric score from judge response text."""

    def test_parse_zero(self):
        assert _parse_score("Some reasoning here.\nSCORE: 0") == 0.0

    def test_parse_half(self):
        assert _parse_score("It's a close call.\nSCORE: 0.5") == 0.5

    def test_parse_one(self):
        assert _parse_score("B is clearly better.\nSCORE: 1") == 1.0

    def test_parse_with_trailing_whitespace(self):
        assert _parse_score("Good work.\nSCORE: 1  ") == 1.0

    def test_parse_lowercase_score(self):
        # Case-insensitive match
        assert _parse_score("Good work.\nscore: 0.5") == 0.5

    def test_parse_last_score_wins(self):
        # If there are multiple SCORE lines (judge draft reasoning), use the last
        text = "Draft: SCORE: 1\nRevised: SCORE: 0.5"
        assert _parse_score(text) == 0.5

    def test_parse_fallback_on_no_score(self):
        # No SCORE line → return 0.5 (tie) as safe fallback
        result = _parse_score("The spreadsheets are similar.")
        assert result == 0.5

    def test_parse_fallback_on_invalid_value(self):
        # SCORE: 2 is not a valid value — fallback to 0.5
        result = _parse_score("SCORE: 2")
        assert result == 0.5


# ── Reconciliation tests ──────────────────────────────────────────────────────

class TestReconcile:
    """Tests for _reconcile() — combines two position-swapped scores."""

    def test_model_wins_both(self):
        # Call 1: A=model, score=0.0 → model_wins_1 = 1.0
        # Call 2: A=gold,  score=1.0 → model_wins_2 = 1.0
        score, consistent = _reconcile(score_ab=0.0, score_ba=1.0)
        assert score == 1.0
        assert consistent is True

    def test_gold_wins_both(self):
        # Call 1: A=model, score=1.0 → model_wins_1 = 0.0
        # Call 2: A=gold,  score=0.0 → model_wins_2 = 0.0
        score, consistent = _reconcile(score_ab=1.0, score_ba=0.0)
        assert score == 0.0
        assert consistent is True

    def test_inconsistent_position_bias(self):
        # Call 1: A=model wins (score_ab=0.0 → model_wins_1=1.0)
        # Call 2: A=gold wins (score_ba=0.0 → model_wins_2=0.0)
        # Disagreement → position bias → 0.5
        score, consistent = _reconcile(score_ab=0.0, score_ba=0.0)
        assert score == 0.5
        assert consistent is False

    def test_inconsistent_other_direction(self):
        # Call 1: gold wins  (score_ab=1.0 → model_wins_1=0.0)
        # Call 2: model wins (score_ba=1.0 → model_wins_2=1.0)
        score, consistent = _reconcile(score_ab=1.0, score_ba=1.0)
        assert score == 0.5
        assert consistent is False

    def test_both_ties(self):
        score, consistent = _reconcile(score_ab=0.5, score_ba=0.5)
        assert score == 0.5
        assert consistent is True   # ties don't contradict each other

    def test_ab_tie_ba_wins(self):
        # Call 1 is tie → overall tie (conservative)
        score, consistent = _reconcile(score_ab=0.5, score_ba=1.0)
        assert score == 0.5
        assert consistent is True


# ── MockJudge tests ───────────────────────────────────────────────────────────

class TestMockJudge:
    """Tests for MockJudge — deterministic fake judge for pipeline testing."""

    def test_default_mock_model_wins(self):
        """Default config: model wins both orderings."""
        mock = MockJudge()
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", GOLD_PATH if GOLD_PATH.exists() else _FIXTURES / "good_minimal.xlsx")
        assert result.score == 1.0
        assert result.position_consistent is True
        assert result.score_ab == 0.0
        assert result.score_ba == 1.0

    def test_gold_wins_config(self):
        mock = MockJudge(fixed_score_ab=1.0, fixed_score_ba=0.0)
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert result.score == 0.0
        assert result.position_consistent is True

    def test_position_bias_config(self):
        """AB says model wins, BA says gold wins → inconsistent → 0.5."""
        mock = MockJudge(fixed_score_ab=0.0, fixed_score_ba=0.0)
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert result.score == 0.5
        assert result.position_consistent is False

    def test_tie_config(self):
        mock = MockJudge(fixed_score_ab=0.5, fixed_score_ba=0.5)
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert result.score == 0.5
        assert result.position_consistent is True

    def test_call_count_increments(self):
        mock = MockJudge()
        assert mock.call_count == 0
        mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert mock.call_count == 1
        mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert mock.call_count == 2

    def test_reasoning_stored(self):
        mock = MockJudge(
            reasoning_ab="Model is better. SCORE: 0",
            reasoning_ba="Model is better. SCORE: 1",
        )
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        assert "Model is better" in result.reasoning_ab
        assert "Model is better" in result.reasoning_ba

    def test_as_dict_serializable(self):
        """JudgeResult.as_dict() must produce a JSON-compatible dict."""
        import json
        mock = MockJudge()
        result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        d = result.as_dict()
        json_str = json.dumps(d)   # will raise if not serializable
        assert '"score"' in json_str
        assert '"position_consistent"' in json_str


# ── HybridOracle integration tests ───────────────────────────────────────────

class TestHybridOracleWithMockJudge:
    """
    Integration tests: MockJudge → JudgeResult → HybridOracle.score() → HybridResult.

    These tests exercise the full scoring pipeline without touching files or APIs.
    They verify that the four agreement buckets are classified correctly when
    programmatic and judge scores are fed into HybridOracle together.
    """

    @pytest.fixture
    def oracle(self):
        return HybridOracle(prog_weight=0.4, judge_weight=0.6, prog_gate=0.3)

    def _run(self, oracle, prog_score, mock_score_ab, mock_score_ba):
        mock = MockJudge(fixed_score_ab=mock_score_ab, fixed_score_ba=mock_score_ba)
        judge_result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        return oracle.score(prog_score=prog_score, judge=judge_result)

    def test_agree_good_bucket(self, oracle):
        """prog ≥ 0.7, judge ≥ 0.5 → agree-good."""
        result = self._run(oracle, prog_score=0.9, mock_score_ab=0.0, mock_score_ba=1.0)
        assert result.agreement_bucket == "agree-good"
        assert result.gate_triggered is False
        assert result.hybrid_score == pytest.approx(0.4 * 0.9 + 0.6 * 1.0)

    def test_agree_bad_bucket(self, oracle):
        """prog < 0.7, judge < 0.5 → agree-bad."""
        result = self._run(oracle, prog_score=0.5, mock_score_ab=1.0, mock_score_ba=0.0)
        assert result.agreement_bucket == "agree-bad"
        # judge score = 0.0 (gold won both orderings)
        assert result.hybrid_score == pytest.approx(0.4 * 0.5 + 0.6 * 0.0)

    def test_judge_rescues_bucket(self, oracle):
        """prog < 0.7, judge ≥ 0.5 → judge-rescues (good analysis, formula bugs)."""
        result = self._run(oracle, prog_score=0.5, mock_score_ab=0.0, mock_score_ba=1.0)
        assert result.agreement_bucket == "judge-rescues"

    def test_programmatic_catches_bucket(self, oracle):
        """prog ≥ 0.7, judge < 0.5 → programmatic-catches (valid structure, weak analysis)."""
        result = self._run(oracle, prog_score=0.8, mock_score_ab=1.0, mock_score_ba=0.0)
        assert result.agreement_bucket == "programmatic-catches"

    def test_gate_triggered_low_prog(self, oracle):
        """prog < 0.3 → gate triggers, judge not used even if provided."""
        mock = MockJudge()
        judge_result = mock.compare(TASK_PROMPT, _FIXTURES / "good_minimal.xlsx", _FIXTURES / "good_minimal.xlsx")
        result = oracle.score(prog_score=0.2, judge=judge_result)
        assert result.gate_triggered is True
        assert result.agreement_bucket == "gate-triggered"
        assert result.hybrid_score == pytest.approx(0.2 * 0.4)
        assert result.judge is None

    def test_gate_triggered_no_judge(self, oracle):
        """No judge provided → gate triggers regardless of prog score."""
        result = oracle.score(prog_score=0.9, judge=None)
        assert result.gate_triggered is True
        assert result.agreement_bucket == "gate-triggered"

    def test_hybrid_score_clamped_to_one(self, oracle):
        """hybrid_score should never exceed 1.0 even if weights sum > 1."""
        fat_oracle = HybridOracle(prog_weight=0.6, judge_weight=0.8)
        result = self._run(fat_oracle, prog_score=1.0, mock_score_ab=0.0, mock_score_ba=1.0)
        assert result.hybrid_score <= 1.0

    def test_position_inconsistent_gives_tie(self, oracle):
        """When judge is inconsistent, score=0.5 → bucket depends on threshold."""
        result = self._run(oracle, prog_score=0.8, mock_score_ab=0.0, mock_score_ba=0.0)
        # model_wins_1=1.0, model_wins_2=0.0 → inconsistent → judge.score=0.5
        assert result.judge.score == 0.5
        assert result.judge.position_consistent is False
        # prog=0.8 ≥ 0.7 (pass), judge=0.5 ≥ 0.5 (pass) → agree-good (0.5 is the threshold)
        assert result.agreement_bucket == "agree-good"


# ── BlindedPairwiseJudge (real API) — skipped without key ────────────────────

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping real API test"
)
@pytest.mark.skipif(
    not GOLD_PATH.exists(),
    reason="Gold file not found — skipping real API integration test"
)
class TestBlindedPairwiseJudgeReal:
    """
    Integration test against the real Gemini Flash API.
    Runs only when GEMINI_API_KEY is set AND the gold file exists.
    Validates the full pipeline: render → 2 API calls → parse → reconcile.
    """

    @pytest.fixture
    def judge(self):
        return BlindedPairwiseJudge()

    def test_compare_gold_vs_good_minimal(self, judge):
        """Gold vs. itself should return a tie or near-tie."""
        result = judge.compare(TASK_PROMPT, GOLD_PATH, GOLD_PATH)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reasoning_ab, str)
        assert len(result.reasoning_ab) > 10

    def test_compare_bad_vs_good_produces_result(self, judge):
        """Bad fixture vs. gold — should complete without error and return valid result."""
        bad_path = _FIXTURES / "bad_hardcoded_revenue.xlsx"
        result = judge.compare(TASK_PROMPT, bad_path, GOLD_PATH)
        assert 0.0 <= result.score <= 1.0
        assert result.position_consistent in (True, False)
