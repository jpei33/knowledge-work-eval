"""
Unit tests for ProgrammaticChecker.

Test strategy: one fixture per failure mode.
  - good_minimal.xlsx          → all checks pass, score ≥ 0.9
  - bad_missing_summary.xlsx   → sheets check fails
  - bad_hardcoded_revenue.xlsx → revenue formula check fails
  - bad_hardcoded_summary.xlsx → summary formula check fails
  - bad_formula_errors.xlsx    → formula error check fails
  - bad_refs_raw_not_cleaned   → summary reference check fails

We also test the real gold file (task_001_gold.xlsx) as the production good case.

Run from repo root:
    pytest Code/data-cleaning-eval/tests/ -v
"""

import json
from pathlib import Path

import pytest

# Allow imports from parent directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from programmatic_checks import ProgrammaticChecker, ProgrammaticResult

# ── Paths ─────────────────────────────────────────────────────────────────────

FIXTURES  = Path(__file__).parent / "fixtures"
TASKS_DIR = Path(__file__).parents[1] / "tasks"
GOLD_DIR  = Path(__file__).parents[1] / "gold"

GOOD_MINIMAL     = FIXTURES / "good_minimal.xlsx"
BAD_MISSING      = FIXTURES / "bad_missing_summary.xlsx"
BAD_HC_REVENUE   = FIXTURES / "bad_hardcoded_revenue.xlsx"
BAD_HC_SUMMARY   = FIXTURES / "bad_hardcoded_summary.xlsx"
BAD_ERRORS       = FIXTURES / "bad_formula_errors.xlsx"
BAD_REFS_RAW     = FIXTURES / "bad_refs_raw_not_cleaned.xlsx"
GOLD_TASK_001    = GOLD_DIR  / "task_001_gold.xlsx"


# ── Shared task config ────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Minimal task config for fixture-based tests (5 raw rows, 4 cleaned)."""
    return {
        "required_sheets":    ["Raw Data", "Cleaned Data", "Summary"],
        "revenue_col_index":  9,
        "raw_row_count":      5,
        "cleaned_row_count":  4,
    }


@pytest.fixture
def task_001_config():
    """Real task_001 config loaded from JSON."""
    return json.loads((TASKS_DIR / "task_001.json").read_text())


@pytest.fixture
def checker(minimal_config):
    return ProgrammaticChecker(minimal_config)


@pytest.fixture
def checker_001(task_001_config):
    return ProgrammaticChecker(task_001_config)


# ── Helper ────────────────────────────────────────────────────────────────────

def check_name(result: ProgrammaticResult, name: str):
    """Get a named CheckResult from the checks list."""
    return next(c for c in result.checks if c.name == name)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GOOD FILE — all checks should pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoodFile:
    def test_score_is_high(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.score >= 0.9, f"Expected score ≥ 0.9, got {result.score:.2f}"

    def test_file_integrity(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.file_integrity is True

    def test_all_sheets_present(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.sheets_missing == []

    def test_no_formula_errors(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.formula_error_count == 0

    def test_revenue_uses_formulas(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.revenue_formula_fraction == 1.0

    def test_summary_uses_formulas(self, checker):
        result = checker.check(GOOD_MINIMAL)
        # Allow for the 2 intentional annotation hardcodes
        assert result.summary_formula_fraction >= 0.7

    def test_summary_references_cleaned(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.summary_references_cleaned is True

    def test_gate_passes(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.gate_passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MISSING SUMMARY SHEET
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingSummary:
    def test_sheets_check_fails(self, checker):
        result = checker.check(BAD_MISSING)
        sheets = check_name(result, "required_sheets")
        assert sheets.passed is False
        assert "Summary" in result.sheets_missing

    def test_sheets_score_is_partial(self, checker):
        result = checker.check(BAD_MISSING)
        # 2 of 3 sheets present → 0.67
        assert abs(result.checks[0].score - 1.0) < 0.01  # file_integrity
        sheets = check_name(result, "required_sheets")
        assert abs(sheets.score - 2/3) < 0.01

    def test_summary_checks_are_zero(self, checker):
        result = checker.check(BAD_MISSING)
        assert result.summary_formula_fraction == 0.0
        assert result.summary_references_cleaned is False

    def test_overall_score_lower(self, checker):
        good   = checker.check(GOOD_MINIMAL)
        bad    = checker.check(BAD_MISSING)
        assert bad.score < good.score


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HARDCODED REVENUE
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardcodedRevenue:
    def test_revenue_fraction_is_zero(self, checker):
        result = checker.check(BAD_HC_REVENUE)
        assert result.revenue_formula_fraction == 0.0

    def test_revenue_check_fails(self, checker):
        result = checker.check(BAD_HC_REVENUE)
        rev = check_name(result, "revenue_formulas")
        assert rev.passed is False

    def test_other_checks_unaffected(self, checker):
        """Hardcoded revenue should not affect sheet presence or formula error checks."""
        result = checker.check(BAD_HC_REVENUE)
        assert result.sheets_missing == []
        assert result.formula_error_count == 0
        assert result.summary_references_cleaned is True

    def test_score_penalized(self, checker):
        good = checker.check(GOOD_MINIMAL)
        bad  = checker.check(BAD_HC_REVENUE)
        # Revenue weight is 0.20, so score should drop by ~0.20
        assert bad.score < good.score - 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HARDCODED SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardcodedSummary:
    def test_summary_fraction_is_low(self, checker):
        result = checker.check(BAD_HC_SUMMARY)
        # All 7 value cells are hardcoded — only the 2 annotation hardcodes
        # overlap but everything is raw here, so fraction = 0.0
        assert result.summary_formula_fraction < 0.3

    def test_summary_check_fails(self, checker):
        result = checker.check(BAD_HC_SUMMARY)
        s = check_name(result, "summary_formulas")
        assert s.passed is False

    def test_revenue_check_still_passes(self, checker):
        """Hardcoded summary should not affect revenue column check."""
        result = checker.check(BAD_HC_SUMMARY)
        assert result.revenue_formula_fraction == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FORMULA ERRORS (#REF!)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormulaErrors:
    def test_error_count_nonzero(self, checker):
        result = checker.check(BAD_ERRORS)
        assert result.formula_error_count >= 1

    def test_error_check_fails(self, checker):
        result = checker.check(BAD_ERRORS)
        err = check_name(result, "formula_errors")
        assert err.passed is False

    def test_error_score_penalized(self, checker):
        result = checker.check(BAD_ERRORS)
        err = check_name(result, "formula_errors")
        # 1 error → score = 1.0 - 0.25 = 0.75
        assert abs(err.score - 0.75) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY REFERENCES RAW DATA INSTEAD OF CLEANED DATA
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefsRawNotCleaned:
    def test_reference_check_fails(self, checker):
        result = checker.check(BAD_REFS_RAW)
        assert result.summary_references_cleaned is False

    def test_reference_check_detail(self, checker):
        result = checker.check(BAD_REFS_RAW)
        ref = check_name(result, "summary_references_cleaned")
        assert ref.passed is False
        assert "Cleaned Data" in ref.detail or "does not reference" in ref.detail

    def test_revenue_formulas_still_pass(self, checker):
        """Wrong sheet reference in Summary should not affect Cleaned Data checks."""
        result = checker.check(BAD_REFS_RAW)
        assert result.revenue_formula_fraction == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GATE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateLogic:
    def test_good_file_gate_passes(self, checker):
        result = checker.check(GOOD_MINIMAL)
        assert result.gate_passed is True

    def test_missing_sheets_gate_still_passes(self, checker):
        """Missing 1 of 3 sheets → score ~0.67 → gate still passes (>0.3)."""
        result = checker.check(BAD_MISSING)
        assert result.gate_passed is True  # not catastrophic enough to gate

    def test_corrupt_file_gate_fails(self, checker, tmp_path):
        """A non-xlsx file should fail integrity and return score=0 / gate=False."""
        bad_file = tmp_path / "corrupt.xlsx"
        bad_file.write_bytes(b"this is not a valid xlsx file")
        result = checker.check(bad_file)
        assert result.file_integrity is False
        assert result.score == 0.0
        assert result.gate_passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GOLD FILE (task_001_gold.xlsx — the real production oracle)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not GOLD_TASK_001.exists(), reason="gold file not found")
class TestGoldFile:
    def test_gold_score_near_perfect(self, checker_001):
        result = checker_001.check(GOLD_TASK_001)
        assert result.score >= 0.85, f"Gold file scored {result.score:.2f}, expected ≥ 0.85"

    def test_gold_row_counts(self, checker_001):
        result = checker_001.check(GOLD_TASK_001)
        assert result.raw_row_count == 75
        assert result.cleaned_row_count == 70

    def test_gold_revenue_all_formulas(self, checker_001):
        result = checker_001.check(GOLD_TASK_001)
        assert result.revenue_formula_fraction == 1.0

    def test_gold_summary_references_cleaned(self, checker_001):
        result = checker_001.check(GOLD_TASK_001)
        assert result.summary_references_cleaned is True

    def test_gold_gate_passes(self, checker_001):
        result = checker_001.check(GOLD_TASK_001)
        assert result.gate_passed is True
