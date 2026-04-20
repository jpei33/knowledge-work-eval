"""
programmatic_checks.py — ProgrammaticChecker for data-cleaning .xlsx deliverables.

Checks a model's submitted .xlsx against the task spec. Returns a ProgrammaticResult
with a [0,1] aggregate score and per-check breakdowns.

Oracle hierarchy rationale:
  These checks are Goodhart-resistant: they measure objective, structural properties
  that cannot be gamed by stylistic optimization. A model cannot fake a correct
  SUMIF formula by sounding confident — either the formula is there or it isn't.

Usage:
    import json
    from pathlib import Path
    from programmatic_checks import ProgrammaticChecker

    task = json.loads(Path("tasks/task_001.json").read_text())
    checker = ProgrammaticChecker(task)
    result = checker.check(Path("outputs/model_output.xlsx"))
    print(f"Score: {result.score:.2f}  Gate: {result.gate_passed}")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from repo root or from data-cleaning-eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.xlsx_utils import (
    EXCEL_ERRORS,
    count_data_rows,
    get_column_cells,
    load_workbook,
)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of one individual programmatic check."""
    name: str
    passed: bool         # True / False — for quick summary
    score: float         # [0, 1] — for weighted aggregation
    detail: str          # human-readable explanation logged to JSONL


@dataclass
class ProgrammaticResult:
    """
    Aggregate result from all programmatic checks on one deliverable.

    score: [0, 1] weighted aggregate — the number fed into HybridOracle.
    gate_passed: if score < 0.3, HybridOracle skips the expensive judge call.
    checks: list of per-check breakdowns for error analysis (Day 20).
    """
    score: float
    file_integrity: bool
    sheets_present: list[str]
    sheets_missing: list[str]
    formula_error_count: int
    revenue_formula_fraction: float   # fraction of Cleaned Data revenue cells that are formulas
    summary_formula_fraction: float   # fraction of Summary value cells that are formulas
    summary_references_cleaned: bool  # does Summary reference 'Cleaned Data' sheet?
    cleaned_row_count: int
    raw_row_count: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        """True if this deliverable is worth sending to the LLM judge."""
        return self.score >= 0.3

    def as_dict(self) -> dict:
        """Serialize to dict for JSONL logging."""
        return {
            "score": round(self.score, 4),
            "gate_passed": self.gate_passed,
            "file_integrity": self.file_integrity,
            "sheets_present": self.sheets_present,
            "sheets_missing": self.sheets_missing,
            "formula_error_count": self.formula_error_count,
            "revenue_formula_fraction": round(self.revenue_formula_fraction, 4),
            "summary_formula_fraction": round(self.summary_formula_fraction, 4),
            "summary_references_cleaned": self.summary_references_cleaned,
            "cleaned_row_count": self.cleaned_row_count,
            "raw_row_count": self.raw_row_count,
            "checks": [
                {"name": c.name, "passed": c.passed, "score": round(c.score, 4), "detail": c.detail}
                for c in self.checks
            ],
        }


# ── Checker ───────────────────────────────────────────────────────────────────

class ProgrammaticChecker:
    """
    Run all programmatic checks on a .xlsx deliverable for a data-cleaning task.

    Design principles:
      1. Each check is a private method (_check_*) returning a CheckResult.
         This makes the checks independently testable and easy to extend.
      2. The aggregate score is a weighted sum — weights reflect how objectively
         each check measures capability vs. how gameable it is.
      3. Scoring is conservative: formula errors cost 0.25 each (4 errors = 0.0),
         because a model that produces #REF! errors has not cleaned the data.

    Scoring weights (must sum to 1.0):
      file_integrity  0.10  — can it open at all?
      sheets          0.20  — are the 3 required sheets present?
      errors          0.25  — zero formula errors required (heaviest non-structural check)
      revenue         0.20  — Cleaned Data revenue column must use formulas
      summary         0.15  — Summary value cells must use formulas
      reference       0.10  — Summary must reference 'Cleaned Data', not 'Raw Data'
    """

    REQUIRED_SHEETS = ["Raw Data", "Cleaned Data", "Summary"]

    WEIGHTS = {
        "integrity":  0.10,
        "sheets":     0.20,
        "errors":     0.25,
        "revenue":    0.20,
        "summary":    0.15,
        "reference":  0.10,
    }

    def __init__(self, task_config: dict):
        """
        Args:
            task_config: parsed task_NNN.json. Key fields used:
                required_sheets    — list of expected sheet names
                revenue_col_index  — 1-indexed column for revenue in Cleaned Data (default 9)
                raw_row_count      — expected raw row count (for validation, not scoring)
                cleaned_row_count  — expected cleaned row count (for validation, not scoring)
        """
        self.required_sheets: list[str] = task_config.get("required_sheets", self.REQUIRED_SHEETS)
        self.revenue_col: int = task_config.get("revenue_col_index", 9)
        self.expected_raw_rows: int | None = task_config.get("raw_row_count")
        self.expected_cleaned_rows: int | None = task_config.get("cleaned_row_count")

    # ── Public interface ──────────────────────────────────────────────────────

    def check(self, xlsx_path: Path | str) -> ProgrammaticResult:
        """
        Run all checks on a deliverable. Returns ProgrammaticResult.

        Flow:
          1. Attempt to open the file (file_integrity).
             If it fails, return score=0.0 immediately — no further checks possible.
          2. Run all structural checks.
          3. Compute weighted aggregate score.
          4. Return result with per-check breakdown.
        """
        path = Path(xlsx_path)
        checks: list[CheckResult] = []

        # ── 1. File integrity ─────────────────────────────────────────────────
        # Must be first: if the file can't open, all other checks are impossible.
        try:
            # data_only=False: we need formula strings, not cached values.
            # This is the critical flag — see xlsx_utils.load_workbook docstring.
            wb = load_workbook(path, data_only=False)
            checks.append(CheckResult("file_integrity", True, 1.0, "File opened successfully"))
        except Exception as exc:
            # Return early with score=0. Don't attempt further checks.
            return ProgrammaticResult(
                score=0.0,
                file_integrity=False,
                sheets_present=[],
                sheets_missing=list(self.required_sheets),
                formula_error_count=0,
                revenue_formula_fraction=0.0,
                summary_formula_fraction=0.0,
                summary_references_cleaned=False,
                cleaned_row_count=0,
                raw_row_count=0,
                checks=[CheckResult("file_integrity", False, 0.0, f"Cannot open file: {exc}")],
            )

        # ── 2. Required sheets ────────────────────────────────────────────────
        checks.append(self._check_required_sheets(wb))

        # ── 3. Formula errors ─────────────────────────────────────────────────
        # Count across ALL sheets — a #REF! in Raw Data still counts.
        error_check, error_count = self._check_formula_errors(wb)
        checks.append(error_check)

        # ── 4. Revenue formula check ──────────────────────────────────────────
        # The most subtle check: did the model hardcode revenue values instead
        # of writing =G{n}*H{n}? Both look identical visually but the programmatic
        # checker catches it.
        rev_check, rev_fraction = self._check_revenue_formulas(wb)
        checks.append(rev_check)

        # ── 5. Summary formula check ──────────────────────────────────────────
        # Did the model compute summary stats as Excel formulas or hardcode them?
        # A hardcoded =15234.50 won't update if the data changes.
        sum_check, sum_fraction = self._check_summary_formulas(wb)
        checks.append(sum_check)

        # ── 6. Summary references Cleaned Data ────────────────────────────────
        # Did the Summary formulas pull from Cleaned Data (correct) or Raw Data
        # (wrong — raw data has nulls and duplicates that inflate counts)?
        ref_check, ref_ok = self._check_summary_references_cleaned(wb)
        checks.append(ref_check)

        # ── Row counts (informational, not scored) ────────────────────────────
        raw_count = (
            count_data_rows(wb["Raw Data"]) if "Raw Data" in wb.sheetnames else 0
        )
        cleaned_count = (
            count_data_rows(wb["Cleaned Data"]) if "Cleaned Data" in wb.sheetnames else 0
        )

        # ── Aggregate score ───────────────────────────────────────────────────
        sheets_check = next(c for c in checks if c.name == "required_sheets")
        score = (
            self.WEIGHTS["integrity"] * 1.0            +
            self.WEIGHTS["sheets"]    * sheets_check.score  +
            self.WEIGHTS["errors"]    * error_check.score   +
            self.WEIGHTS["revenue"]   * rev_fraction         +
            self.WEIGHTS["summary"]   * sum_fraction         +
            self.WEIGHTS["reference"] * float(ref_ok)
        )

        return ProgrammaticResult(
            score=min(max(score, 0.0), 1.0),
            file_integrity=True,
            sheets_present=[s for s in self.required_sheets if s in wb.sheetnames],
            sheets_missing=[s for s in self.required_sheets if s not in wb.sheetnames],
            formula_error_count=error_count,
            revenue_formula_fraction=rev_fraction,
            summary_formula_fraction=sum_fraction,
            summary_references_cleaned=ref_ok,
            cleaned_row_count=cleaned_count,
            raw_row_count=raw_count,
            checks=checks,
        )

    # ── Private checks ────────────────────────────────────────────────────────

    def _check_required_sheets(self, wb) -> CheckResult:
        """
        Verify all required sheets are present.

        Score = len(present) / len(required)
        So missing 1 of 3 sheets → 0.67, missing all → 0.0.
        """
        present = [s for s in self.required_sheets if s in wb.sheetnames]
        missing = [s for s in self.required_sheets if s not in wb.sheetnames]
        score = len(present) / len(self.required_sheets)
        return CheckResult(
            name="required_sheets",
            passed=len(missing) == 0,
            score=score,
            detail=f"Present: {present}. Missing: {missing}",
        )

    def _check_formula_errors(self, wb) -> tuple[CheckResult, int]:
        """
        Count cells containing Excel error values across all sheets.

        Score = max(0, 1.0 - count * 0.25)
        Each error costs 0.25 — 4 or more errors = 0.0.

        Returns (CheckResult, error_count) — count needed for ProgrammaticResult.
        """
        count = 0
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    # Formula errors are stored as their string representation
                    # e.g. cell.value == "#REF!" not as a special Excel error type
                    if isinstance(cell.value, str) and cell.value in EXCEL_ERRORS:
                        count += 1

        score = max(0.0, 1.0 - count * 0.25)
        return CheckResult(
            name="formula_errors",
            passed=count == 0,
            score=score,
            detail=f"{count} formula error(s) found across all sheets",
        ), count

    def _check_revenue_formulas(self, wb) -> tuple[CheckResult, float]:
        """
        Check whether revenue cells in Cleaned Data use formulas or hardcoded values.

        Expected: every cell in the revenue column (col 9 by default) is a formula
        like '=G2*H2'. A model that hardcodes 239.80 instead fails this check.

        Fraction = formula_cells / total_cells in revenue column (excluding header).
        Threshold for pass: >= 0.9 (allow 1 in 10 to be hardcoded before failing).

        This is the most interview-relevant check — it catches the subtle failure
        mode that looks correct visually but breaks if inputs change.
        """
        if "Cleaned Data" not in wb.sheetnames:
            return CheckResult("revenue_formulas", False, 0.0,
                               "Cleaned Data sheet missing"), 0.0

        ws = wb["Cleaned Data"]
        cells = get_column_cells(ws, col_idx=self.revenue_col, start_row=2)
        # Filter to non-empty cells only
        populated = [c for c in cells if c.value is not None]

        if not populated:
            return CheckResult("revenue_formulas", False, 0.0,
                               "No data found in revenue column"), 0.0

        formula_count = sum(
            1 for c in populated
            if isinstance(c.value, str) and c.value.startswith("=")
        )
        fraction = formula_count / len(populated)

        return CheckResult(
            name="revenue_formulas",
            passed=fraction >= 0.9,
            score=fraction,
            detail=f"{formula_count}/{len(populated)} revenue cells use formulas ({fraction:.0%})",
        ), fraction

    def _check_summary_formulas(self, wb) -> tuple[CheckResult, float]:
        """
        Check whether Summary sheet value cells use Excel formulas.

        Heuristic: in column B (the value column), cells that contain numbers
        (int or float) should be formulas. We scan column B rows 3+ and classify
        each cell as:
          - formula  (string starting with '=')  → correct
          - number   (int or float)              → hardcoded, penalized
          - string   (label or section header)   → skip
          - None     (empty)                     → skip

        The 2 intentional annotation hardcodes (duplicates_removed=5,
        nulls_filled=8) are also int values, so they count against the fraction.
        This is acceptable: a model that hardcodes EVERYTHING scores poorly;
        a model that uses formulas everywhere but leaves those 2 as annotations
        still scores ~0.85+.

        Threshold for pass: >= 0.7.
        """
        if "Summary" not in wb.sheetnames:
            return CheckResult("summary_formulas", False, 0.0,
                               "Summary sheet missing"), 0.0

        ws = wb["Summary"]
        total, formula_count = 0, 0

        # Column B (index 2), skip title rows (start at row 3)
        for row in ws.iter_rows(min_row=3, min_col=2, max_col=2):
            cell = row[0]
            is_formula = isinstance(cell.value, str) and cell.value.startswith("=")
            is_number  = isinstance(cell.value, (int, float))

            if is_formula or is_number:
                total += 1
                if is_formula:
                    formula_count += 1

        if total == 0:
            return CheckResult("summary_formulas", False, 0.0,
                               "No value cells found in Summary col B"), 0.0

        fraction = formula_count / total
        return CheckResult(
            name="summary_formulas",
            passed=fraction >= 0.7,
            score=fraction,
            detail=f"{formula_count}/{total} Summary value cells use formulas ({fraction:.0%})",
        ), fraction

    def _check_summary_references_cleaned(self, wb) -> tuple[CheckResult, bool]:
        """
        Verify that Summary sheet formulas reference 'Cleaned Data', not 'Raw Data'.

        A common model failure: SUMIF formulas that pull from Raw Data. These produce
        inflated totals because raw data has duplicates and null revenues counted as 0.

        Detection: scan all formula strings in Summary for the substring 'Cleaned Data'.
        One match is sufficient — if the model used Cleaned Data for revenue, it likely
        used it for everything.

        Cross-sheet reference syntax in Excel formulas: 'Sheet Name'!CellRef
        so we search for "Cleaned Data" (with or without surrounding quotes).
        """
        if "Summary" not in wb.sheetnames:
            return CheckResult("summary_references_cleaned", False, 0.0,
                               "Summary sheet missing"), False

        ws = wb["Summary"]
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    # Check for either quoted or unquoted sheet name in formula
                    if "Cleaned Data" in cell.value:
                        return CheckResult(
                            "summary_references_cleaned", True, 1.0,
                            f"Found reference to 'Cleaned Data' in {cell.coordinate}: {cell.value[:60]}"
                        ), True

        return CheckResult(
            "summary_references_cleaned", False, 0.0,
            "No formula in Summary references 'Cleaned Data' sheet"
        ), False
