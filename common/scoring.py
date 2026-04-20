"""
scoring.py — HybridOracle: combines programmatic checks + LLM judge scores.

Score semantics (higher = model did better):
  programmatic_score  [0, 1]  — 1.0 means all structural checks passed
  judge_score         [0, 1]  — 1.0 means judge strongly prefers model over gold
  hybrid_score        [0, 1]  — weighted sum of both layers

Agreement buckets (the primary research finding from Day 19):
  "agree-good"           both layers pass → model did well overall
  "agree-bad"            both layers fail → model clearly failed
  "judge-rescues"        prog fails, judge passes → good analysis but formula bugs
  "programmatic-catches" prog passes, judge fails → valid structure, weak analysis
  "gate-triggered"       programmatic score < gate → judge was never called

Usage:
    from common.scoring import HybridOracle

    oracle = HybridOracle()
    result = oracle.score(prog_score=0.85, judge=judge_result)
    print(result.hybrid_score, result.agreement_bucket)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Judge result ──────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    """
    Outcome of one blinded pairwise comparison with position-swap mitigation.

    Score semantics: 1.0 = judge prefers model output over gold (model wins).

    Raw scores (score_ab, score_ba) use the convention:
      0 = prefer A,  0.5 = tie,  1 = prefer B
    These are converted to model-wins perspective before storing in `score`.

    Call 1 (A=model, B=gold):  model_wins_1 = 1.0 - score_ab
    Call 2 (A=gold,  B=model): model_wins_2 = score_ba
    If both agree → score = model_wins_1. If they disagree → score = 0.5 (tie).
    """
    score: float                # [0, 1] model-wins perspective, after reconciliation
    score_ab: float             # raw judge score, A=model output, B=gold
    score_ba: float             # raw judge score, A=gold, B=model output
    position_consistent: bool   # True if both orderings agree on the winner
    reasoning_ab: str           # judge's written reasoning (A=model ordering)
    reasoning_ba: str           # judge's written reasoning (A=gold ordering)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "score_ab": self.score_ab,
            "score_ba": self.score_ba,
            "position_consistent": self.position_consistent,
            "reasoning_ab": self.reasoning_ab[:300],   # truncate for JSONL
            "reasoning_ba": self.reasoning_ba[:300],
        }


# ── Hybrid result ─────────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    """
    Combined output from all oracle layers for one task × model × elicitation run.

    This is the unit of analysis in Day 19-20. The `agreement_bucket` field is
    the primary research finding: it tells you *how* the model failed, not just
    whether it failed.
    """
    hybrid_score: float
    programmatic_score: float
    judge: Optional[JudgeResult]
    agreement_bucket: str   # see HybridOracle._classify() for possible values
    gate_triggered: bool    # True if programmatic gate prevented judge call

    def as_dict(self) -> dict:
        return {
            "hybrid_score": round(self.hybrid_score, 4),
            "programmatic_score": round(self.programmatic_score, 4),
            "judge": self.judge.as_dict() if self.judge else None,
            "agreement_bucket": self.agreement_bucket,
            "gate_triggered": self.gate_triggered,
        }


# ── Hybrid oracle ─────────────────────────────────────────────────────────────

class HybridOracle:
    """
    Combines programmatic checks and LLM judge into a single hybrid score.

    Design principles:
      1. Gate first: if programmatic score < gate threshold, skip the expensive
         judge call entirely. A structurally broken file isn't worth evaluating
         for content quality.
      2. Configurable weights: default 0.4/0.6 (prog/judge). Higher judge weight
         reflects that content quality matters more than formula mechanics for
         knowledge work, but programmatic checks provide the Goodhart-resistant anchor.
      3. Agreement classification: the four-bucket taxonomy is the research output.
         Aggregate win-rates hide whether models fail on structure vs. substance.

    Score semantics: higher = model did better (1.0 = perfect, 0.0 = total failure).
    """

    # Thresholds for agreement bucket classification
    PROG_PASS_THRESHOLD  = 0.7   # programmatic score >= this → "passes" structurally
    JUDGE_PASS_THRESHOLD = 0.5   # judge score >= this → "passes" on content quality

    def __init__(
        self,
        prog_weight: float = 0.4,
        judge_weight: float = 0.6,
        prog_gate: float = 0.3,
    ):
        """
        Args:
            prog_weight:  weight for programmatic score in hybrid (default 0.4)
            judge_weight: weight for judge score in hybrid (default 0.6)
            prog_gate:    if programmatic score < this, skip judge (default 0.3)

        Weights don't need to sum to 1.0 — they're independent scaling factors.
        Default 0.4 + 0.6 = 1.0, but e.g. 0.3 + 0.3 would produce scores in [0, 0.6].
        """
        assert prog_weight >= 0 and judge_weight >= 0
        self.prog_weight  = prog_weight
        self.judge_weight = judge_weight
        self.prog_gate    = prog_gate

    def score(
        self,
        prog_score: float,
        judge: Optional[JudgeResult] = None,
    ) -> HybridResult:
        """
        Compute hybrid score from programmatic result and optional judge result.

        If prog_score < prog_gate OR judge is None:
          → gate triggers, judge not called (or result discarded)
          → hybrid_score = prog_score * prog_weight only
          → agreement_bucket = "gate-triggered"

        Otherwise:
          → hybrid_score = prog_weight * prog_score + judge_weight * judge.score
          → agreement_bucket = one of four buckets based on pass/fail thresholds
        """
        gate_triggered = prog_score < self.prog_gate or judge is None

        if gate_triggered:
            return HybridResult(
                hybrid_score=prog_score * self.prog_weight,
                programmatic_score=prog_score,
                judge=None,
                agreement_bucket="gate-triggered",
                gate_triggered=True,
            )

        hybrid = self.prog_weight * prog_score + self.judge_weight * judge.score

        return HybridResult(
            hybrid_score=min(max(hybrid, 0.0), 1.0),
            programmatic_score=prog_score,
            judge=judge,
            agreement_bucket=self._classify(prog_score, judge.score),
            gate_triggered=False,
        )

    def _classify(self, prog_score: float, judge_score: float) -> str:
        """
        Classify the agreement between programmatic and judge layers.

        Four buckets — this is the research finding from Day 19:

          agree-good:           prog ≥ 0.7, judge ≥ 0.5
            → model did well on both structure AND content. Clean win.

          agree-bad:            prog < 0.7, judge < 0.5
            → model failed both layers. Comprehensive failure.

          judge-rescues:        prog < 0.7, judge ≥ 0.5
            → good analysis but formula bugs. More recoverable — the model
              understood the task but made structural mistakes. Targeted fix.

          programmatic-catches: prog ≥ 0.7, judge < 0.5
            → valid structure, weak analysis. The dangerous case — looks correct
              visually but the analysis is wrong or superficial.
        """
        prog_pass  = prog_score  >= self.PROG_PASS_THRESHOLD
        judge_pass = judge_score >= self.JUDGE_PASS_THRESHOLD

        if prog_pass and judge_pass:
            return "agree-good"
        elif not prog_pass and not judge_pass:
            return "agree-bad"
        elif not prog_pass and judge_pass:
            return "judge-rescues"
        else:                                 # prog_pass and not judge_pass
            return "programmatic-catches"
