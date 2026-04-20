"""
llm_judge.py — BlindedPairwiseJudge: Gemini Flash blinded pairwise comparison.

Design overview:
  1. Render both xlsx files to text using extract_to_text() (binary → LLM-readable).
  2. Run two judge calls with A/B order swapped (position-swap mitigation).
  3. Reconcile: if both agree on winner → use that score. If they disagree → 0.5 (tie).
  4. Return JudgeResult with raw scores, reconciled score, reasoning, position flag.

Score semantics:
  Raw scores (score_ab, score_ba) follow the "prefer B" convention:
    0 = prefer A,  0.5 = tie,  1 = prefer B
  Call 1: A=model output, B=gold  → model_wins_1 = 1.0 - score_ab
  Call 2: A=gold, B=model output  → model_wins_2 = score_ba
  Reconciled score = model_wins_1 if consistent, else 0.5.

Position-swap rationale:
  MT-Bench (Zheng et al. 2023) found LLM judges prefer whichever option appears first
  ~65% of the time without mitigation. Running both orderings and only declaring a winner
  when both agree reduces this to noise. A judge that's only consistent on 65% of swaps
  gives a ~0.65 × 0.65 = ~42% chance of correctly agreeing on a winner — still weak.
  We declare 0.5 (tie) when they disagree rather than averaging, because an average would
  import the position bias into the final score.

MockJudge:
  For testing without a Gemini API key. Returns configurable fixed scores, letting
  unit tests exercise the full oracle pipeline (scoring.py + judge integration) without
  making real API calls. Use MOCK_JUDGE=1 env var in the harness.

Usage:
    from llm_judge import BlindedPairwiseJudge, MockJudge
    from common.scoring import JudgeResult

    # Real judge
    judge = BlindedPairwiseJudge(api_key="AIza...")
    result = judge.compare(
        task_prompt="Clean the CSV and produce a 3-sheet xlsx...",
        output_path=Path("model_output.xlsx"),
        gold_path=Path("gold/task_001_gold.xlsx"),
    )
    print(result.score, result.position_consistent)

    # Mock judge (testing)
    mock = MockJudge(fixed_score_ab=0.0, fixed_score_ba=1.0)
    result = mock.compare(task_prompt, output_path, gold_path)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Add common/ to sys.path so this works when run from the data-cleaning-eval directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.scoring import JudgeResult  # noqa: E402
from common.xlsx_utils import load_workbook, extract_to_text  # noqa: E402


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert data analyst evaluating the quality of spreadsheet work.
You will be shown a task description, two spreadsheet deliverables labeled A and B,
and you must decide which is better — or call it a tie.

Evaluation criteria:
  1. Data cleaning completeness — were all three messiness issues addressed?
     (mixed date formats, duplicate order IDs, null revenue rows)
  2. Formula correctness — are computations done with Excel formulas, not hardcoded values?
  3. Summary accuracy — do the aggregate statistics correctly reflect the cleaned data?
  4. Structural clarity — are the three required sheets present and well-organized?

Scoring:
  0   = A is clearly better
  0.5 = tie or too close to call
  1   = B is clearly better

You MUST end your response with exactly one line in this format:
SCORE: [number]

where [number] is 0, 0.5, or 1. Nothing else on that line.
"""

JUDGE_USER_TEMPLATE = """\
## Task

{task_prompt}

---

## Spreadsheet A

{spreadsheet_a}

---

## Spreadsheet B

{spreadsheet_b}

---

Evaluate which spreadsheet better accomplishes the task.
End your response with:
SCORE: [0 or 0.5 or 1]
"""


# ── Score parsing ─────────────────────────────────────────────────────────────

_SCORE_PATTERN = re.compile(r"SCORE:\s*(0\.5|0|1)\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_score(response_text: str) -> float:
    """
    Extract the numeric score from a judge response.

    The judge is instructed to end with 'SCORE: [0, 0.5, or 1]'. This function
    finds the last matching line (in case the judge produces multiple draft lines)
    and returns the float value.

    Falls back to 0.5 (tie) if parsing fails — conservative: don't penalize a model
    for a judge formatting failure.

    Args:
        response_text: The full text returned by the judge model.

    Returns:
        0.0, 0.5, or 1.0.
    """
    matches = _SCORE_PATTERN.findall(response_text)
    if not matches:
        # Fallback: search for any standalone 0, 0.5, 1 near the word SCORE
        loose = re.search(r"SCORE[^0-9]*(0\.5|0|1)", response_text, re.IGNORECASE)
        if loose:
            return float(loose.group(1))
        return 0.5   # tie as fallback
    return float(matches[-1])   # use last match if there are multiple draft lines


def _reconcile(
    score_ab: float,
    score_ba: float,
) -> tuple[float, bool]:
    """
    Reconcile two position-swapped scores into a single model-wins score.

    Convention:
      score_ab: 0=prefer A, 0.5=tie, 1=prefer B  where A=model, B=gold
      score_ba: 0=prefer A, 0.5=tie, 1=prefer B  where A=gold,  B=model

    model_wins_1 = 1.0 - score_ab   (call 1: model wins if judge prefers A)
    model_wins_2 = score_ba          (call 2: model wins if judge prefers B)

    If both agree (same winner side): return model_wins_1, consistent=True.
    If they disagree:                 return 0.5,            consistent=False.

    "Agree" means both model_wins values are on the same side of 0.5:
      both > 0.5   → model won both orderings   → consistent, use model_wins_1
      both < 0.5   → gold won both orderings    → consistent, use model_wins_1
      one ≥ 0.5, one ≤ 0.5 with a clear difference → inconsistent, return 0.5
      either is exactly 0.5 → treat as tie → consistent, use 0.5

    Args:
        score_ab: Raw judge score, call 1 (A=model output, B=gold).
        score_ba: Raw judge score, call 2 (A=gold, B=model output).

    Returns:
        (reconciled_score [0,1], position_consistent bool)
    """
    model_wins_1 = 1.0 - score_ab
    model_wins_2 = score_ba

    # If either call is a tie, the overall result is a tie
    if model_wins_1 == 0.5 or model_wins_2 == 0.5:
        return 0.5, True   # consistent in the sense that ties don't contradict

    # Both agree: model won both, or gold won both
    if (model_wins_1 > 0.5) == (model_wins_2 > 0.5):
        return model_wins_1, True

    # They disagree — position bias may be the cause
    return 0.5, False


# ── Real judge (Gemini Flash) ─────────────────────────────────────────────────

class BlindedPairwiseJudge:
    """
    LLM-as-judge using Google Gemini Flash for blinded pairwise comparison.

    Why Gemini Flash (not GPT-4o)?
      Cross-family evaluation avoids self-preference bias. When evaluating GPT-4o outputs,
      a GPT-4o judge would be grading its own "style" — Wataoka et al. (2024) showed this
      produces a ~10% inflated win rate. Gemini Flash is a different architecture and
      training pipeline, so its preferences are uncorrelated with GPT-4o's output style.

    Why Flash (not Pro/Ultra)?
      Cost control. Each task requires 2 API calls. At eval scale (15 tasks × 2 models
      × 2 elicitation modes = 60 task runs), that's 120 API calls. Flash is ~20× cheaper
      than Pro-level models per token. The gate logic in HybridOracle means we only call
      the judge on structurally valid submissions (programmatic score ≥ 0.3), further
      reducing costs on bad outputs.

    Thread safety: not thread-safe. One instance per worker if parallelizing.
    """

    MODEL = "gemini-1.5-flash"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0   # seconds between retries

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
    ):
        """
        Args:
            api_key:  Gemini API key. Falls back to GEMINI_API_KEY env var.
            model:    Gemini model string. Default: gemini-1.5-flash.

        Raises:
            ImportError: if google-generativeai is not installed.
            ValueError:  if no API key is provided or found in environment.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for BlindedPairwiseJudge. "
                "Install with: pip install google-generativeai"
            ) from e

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key provided. Pass api_key= or set GEMINI_API_KEY env var."
            )

        genai.configure(api_key=resolved_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=JUDGE_SYSTEM_PROMPT,
        )
        self.model_name = model

    def compare(
        self,
        task_prompt: str,
        output_path: Path,
        gold_path: Path,
    ) -> JudgeResult:
        """
        Run blinded pairwise comparison between model output and gold deliverable.

        Runs two API calls with positions swapped, then reconciles. Both xlsx files
        are rendered to text via extract_to_text() before being sent to the judge.

        Args:
            task_prompt:  The original task prompt shown to the model (provides context).
            output_path:  Path to the model's .xlsx output.
            gold_path:    Path to the gold .xlsx deliverable.

        Returns:
            JudgeResult with reconciled score, raw scores, reasoning, and position flag.
        """
        output_text = self._render(output_path)
        gold_text   = self._render(gold_path)

        # Call 1: A=model output, B=gold
        prompt_ab = JUDGE_USER_TEMPLATE.format(
            task_prompt=task_prompt,
            spreadsheet_a=output_text,
            spreadsheet_b=gold_text,
        )
        reasoning_ab = self._call_api(prompt_ab)
        score_ab = _parse_score(reasoning_ab)

        # Call 2: A=gold, B=model output (positions swapped)
        prompt_ba = JUDGE_USER_TEMPLATE.format(
            task_prompt=task_prompt,
            spreadsheet_a=gold_text,
            spreadsheet_b=output_text,
        )
        reasoning_ba = self._call_api(prompt_ba)
        score_ba = _parse_score(reasoning_ba)

        score, consistent = _reconcile(score_ab, score_ba)

        return JudgeResult(
            score=score,
            score_ab=score_ab,
            score_ba=score_ba,
            position_consistent=consistent,
            reasoning_ab=reasoning_ab,
            reasoning_ba=reasoning_ba,
        )

    def _render(self, xlsx_path: Path) -> str:
        """
        Convert an .xlsx file to a text representation for the judge.

        Uses extract_to_text() from xlsx_utils. This reduces a binary xlsx to a
        human/LLM-readable string that includes sheet names, formula examples, and
        sample data rows. A hardcoded file shows '[none — all values may be hardcoded]'
        for the formula examples section, surfacing that structural signal to the judge.

        Args:
            xlsx_path: Path to the .xlsx file to render.

        Returns:
            Multi-line string representation of the workbook.
        """
        wb = load_workbook(xlsx_path, data_only=False)
        return extract_to_text(wb)

    def _call_api(self, prompt: str) -> str:
        """
        Call the Gemini Flash API with retry logic.

        Retries up to MAX_RETRIES times on API errors with exponential back-off
        (RETRY_DELAY * 2^attempt seconds). On final failure, returns a sentinel
        string containing 'SCORE: 0.5' so the caller still gets a parseable result.

        Args:
            prompt: The full user-turn prompt to send.

        Returns:
            The model's response text.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Return a fallback response so the pipeline doesn't crash
                    return f"[API error after {self.MAX_RETRIES} retries: {e}]\nSCORE: 0.5"

        return "SCORE: 0.5"   # unreachable but satisfies type checker


# ── Mock judge ────────────────────────────────────────────────────────────────

class MockJudge:
    """
    Deterministic mock judge for unit testing and CI without an API key.

    Allows setting fixed scores for each call order (AB and BA independently).
    This lets tests exercise every path in the scoring pipeline:

        MockJudge(fixed_score_ab=0.0, fixed_score_ba=1.0)
          → model_wins_1 = 1.0 - 0.0 = 1.0
          → model_wins_2 = 1.0
          → consistent, score = 1.0  (model wins both orderings)

        MockJudge(fixed_score_ab=1.0, fixed_score_ba=0.0)
          → model_wins_1 = 1.0 - 1.0 = 0.0
          → model_wins_2 = 0.0
          → consistent, score = 0.0  (gold wins both orderings)

        MockJudge(fixed_score_ab=0.0, fixed_score_ba=0.0)
          → model_wins_1 = 1.0,  model_wins_2 = 0.0
          → inconsistent, score = 0.5  (position bias case)

        MockJudge(fixed_score_ab=0.5, fixed_score_ba=0.5)
          → both ties → score = 0.5, consistent

    Rendering: skipped entirely (no file IO needed for unit tests).
    API calls: replaced with configurable fixed responses.
    """

    def __init__(
        self,
        fixed_score_ab: float = 0.0,
        fixed_score_ba: float = 1.0,
        reasoning_ab: str = "Mock reasoning: A is better.\nSCORE: 0",
        reasoning_ba: str = "Mock reasoning: B is better.\nSCORE: 1",
    ):
        """
        Args:
            fixed_score_ab:  Raw judge score for call 1 (A=model, B=gold). Default 0.0.
            fixed_score_ba:  Raw judge score for call 2 (A=gold, B=model). Default 1.0.
            reasoning_ab:    Mock reasoning text for call 1.
            reasoning_ba:    Mock reasoning text for call 2.

        Default (0.0, 1.0): model wins both orderings → score=1.0, consistent=True.
        This is the "perfect model" case for integration tests.
        """
        self.fixed_score_ab = fixed_score_ab
        self.fixed_score_ba = fixed_score_ba
        self.reasoning_ab   = reasoning_ab
        self.reasoning_ba   = reasoning_ba
        self.call_count     = 0   # tracks API-equivalent calls for assertions

    def compare(
        self,
        task_prompt: str,
        output_path: Path,
        gold_path: Path,
    ) -> JudgeResult:
        """
        Return a deterministic JudgeResult without touching the filesystem or an API.

        Args: Same as BlindedPairwiseJudge.compare (interface-compatible).

        Returns:
            JudgeResult built from fixed scores and canned reasoning.
        """
        self.call_count += 1   # each compare() = 2 logical API calls

        score, consistent = _reconcile(self.fixed_score_ab, self.fixed_score_ba)

        return JudgeResult(
            score=score,
            score_ab=self.fixed_score_ab,
            score_ba=self.fixed_score_ba,
            position_consistent=consistent,
            reasoning_ab=self.reasoning_ab,
            reasoning_ba=self.reasoning_ba,
        )
