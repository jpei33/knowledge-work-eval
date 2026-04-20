"""
harness.py — Eval harness for the data-cleaning knowledge work eval.

Orchestrates the full pipeline for one task × model × elicitation run:
  1. Load task config (task_001.json)
  2. Build the prompt (zero_shot or structured elicitation mode)
  3. Call the model (GPT-4o or Claude) and save its xlsx output
  4. Run ProgrammaticChecker → prog_score
  5. If gate passes, run BlindedPairwiseJudge → judge_result
  6. Feed both into HybridOracle → hybrid_score, agreement_bucket
  7. Append a result record to a JSONL log file

Design decisions:
  - ModelClient is an abstract base class. GPT4oClient and ClaudeClient are
    concrete implementations. The harness depends on the interface, not the
    vendor SDK. Swapping models is a one-line change.
  - MockModelClient enables full end-to-end pipeline testing without any API calls.
    Use MOCK_MODEL=1 env var. Returns a configurable .xlsx path instead of calling
    an API.
  - MOCK_JUDGE=1 env var swaps BlindedPairwiseJudge for MockJudge. Combined with
    MOCK_MODEL=1, the full pipeline runs in milliseconds in CI.
  - JSONL logging: one JSON object per line. Append-only. Each record is one run:
    one task × one model × one elicitation mode. Multiple runs accumulate in the
    same file for Day 19 analysis.
  - Outputs are saved to outputs/{model}/{elicitation}/{task_id}.xlsx so runs
    don't overwrite each other.

Usage:
    # Real run
    python harness.py --task tasks/task_001.json \\
                      --model gpt-4o \\
                      --elicitation structured \\
                      --output-dir outputs \\
                      --results-file results/results.jsonl

    # Dry run (both mocks, no API calls)
    MOCK_MODEL=1 MOCK_JUDGE=1 python harness.py \\
        --task tasks/task_001.json --model gpt-4o --elicitation structured

    # Run all tasks/models/elicitation combos (grid)
    python harness.py --grid --model gpt-4o claude-3-5-sonnet \\
                      --elicitation zero_shot structured
"""

from __future__ import annotations

import abc
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow running from repo root or from data-cleaning-eval/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.scoring import HybridOracle, HybridResult  # noqa: E402
from programmatic_checks import ProgrammaticChecker  # noqa: E402

# Conditional imports — judge and model clients
import importlib.util as _ilu  # noqa: E402
_judge_mod_path = Path(__file__).resolve().parent / "llm_judge.py"
_spec = _ilu.spec_from_file_location("llm_judge", _judge_mod_path)
_llm_judge = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_llm_judge)
BlindedPairwiseJudge = _llm_judge.BlindedPairwiseJudge
MockJudge            = _llm_judge.MockJudge


# ── Model client interface ────────────────────────────────────────────────────

class ModelClient(abc.ABC):
    """
    Abstract base class for model clients.

    The harness calls client.run(prompt, raw_csv_path) and gets back a Path
    pointing to the model's .xlsx output. Everything vendor-specific (API format,
    file upload, tool use) is hidden inside the concrete implementation.

    Why abstract base class (not duck typing)?
      Explicit interface makes the contract clear. Both GPT4oClient and
      ClaudeClient must implement run() with the same signature. If a new
      model is added, the developer gets an error at class definition time
      (not at runtime) if they forget to implement run().
    """

    @abc.abstractmethod
    def run(
        self,
        prompt: str,
        raw_csv_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Call the model with the task prompt and raw CSV, save its xlsx output.

        Args:
            prompt:       Full task prompt (zero_shot or structured).
            raw_csv_path: Path to the raw CSV to attach/upload.
            output_path:  Where to save the model's .xlsx response.

        Returns:
            Path to the saved .xlsx file (same as output_path if successful).

        Raises:
            RuntimeError: if the model fails to produce a valid .xlsx output.
        """
        ...

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Human-readable model identifier for logging (e.g. 'gpt-4o', 'claude-3-5-sonnet')."""
        ...


# ── Concrete model clients ────────────────────────────────────────────────────

class GPT4oClient(ModelClient):
    """
    Model client for GPT-4o via OpenAI API.

    Uses the Responses API with file upload: uploads the CSV as a file object,
    passes it as an attachment in the user message, requests an xlsx download.

    Note: GPT-4o can produce xlsx via code interpreter. The prompt instructs it
    to write and execute Python (openpyxl) to produce the file, then return it
    as a file attachment. The client downloads the attachment and saves it to
    output_path.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("No OpenAI API key. Pass api_key= or set OPENAI_API_KEY.")
        self._client = OpenAI(api_key=self._api_key)  # type: ignore[arg-type]
        self._model  = model

    @property
    def model_id(self) -> str:
        return self._model

    def run(self, prompt: str, raw_csv_path: Path, output_path: Path) -> Path:
        """
        Upload CSV, call GPT-4o with code interpreter, download the xlsx output.

        GPT-4o produces xlsx via its code interpreter tool: it writes Python to
        create the file, executes it in its sandbox, and returns the file as an
        attachment. This is the real tool-use asymmetry from GDPval: GPT-4o has
        web search + code interpreter; we're giving it code interpreter here to
        match the GDPval elicitation conditions.
        """
        # Upload the raw CSV as a file for GPT-4o to reference
        with open(raw_csv_path, "rb") as f:
            file_obj = self._client.files.create(file=f, purpose="assistants")

        try:
            response = self._client.responses.create(
                model=self._model,
                tools=[{"type": "code_interpreter"}],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_file",
                                "file_id": file_obj.id,
                                "filename": raw_csv_path.name,
                            },
                        ],
                    }
                ],
            )

            # Find the xlsx file attachment in the response outputs
            for output_item in response.output:
                if hasattr(output_item, "files"):
                    for file_ref in output_item.files:
                        if file_ref.filename.endswith(".xlsx"):
                            content = self._client.files.content(file_ref.id)
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_bytes(content.content)
                            return output_path

            raise RuntimeError(
                f"GPT-4o did not produce an xlsx file attachment. "
                f"Response output types: {[type(o).__name__ for o in response.output]}"
            )

        finally:
            # Always clean up the uploaded file to avoid storage accumulation
            self._client.files.delete(file_obj.id)


class ClaudeClient(ModelClient):
    """
    Model client for Claude via Anthropic API.

    Claude does not have a code interpreter tool in the standard API. Instead,
    we use the extended thinking / tool-use approach: provide Claude with a
    write_file tool that accepts base64-encoded file content. Claude calls the
    tool to return its xlsx output.

    Note: This reflects the GDPval tool-access asymmetry. Claude gets UI-based
    file creation while GPT-4o gets code interpreter. To make the comparison
    fair, we give Claude a write_file tool here — a deliberate elicitation
    standardization choice documented in the eval spec.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("No Anthropic API key. Pass api_key= or set ANTHROPIC_API_KEY.")

        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=self._api_key)
        self._model  = model

    @property
    def model_id(self) -> str:
        return self._model

    def run(self, prompt: str, raw_csv_path: Path, output_path: Path) -> Path:
        """
        Send the task prompt + CSV content to Claude, receive xlsx via tool call.

        The write_file tool accepts base64-encoded bytes so Claude can return
        binary files (xlsx) within a JSON tool_use block. Claude is instructed
        to produce the xlsx using Python (openpyxl) and call write_file with
        the result.
        """
        import base64

        csv_content = raw_csv_path.read_text(encoding="utf-8")

        full_prompt = (
            f"{prompt}\n\n"
            f"Here is the raw CSV data:\n\n```csv\n{csv_content}\n```\n\n"
            f"Use Python (openpyxl) to create the xlsx file as instructed. "
            f"When done, call the write_file tool with the base64-encoded xlsx bytes."
        )

        tools = [
            {
                "name": "write_file",
                "description": "Write the completed xlsx file. Pass the base64-encoded file bytes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content_base64": {"type": "string", "description": "Base64-encoded file bytes"},
                    },
                    "required": ["filename", "content_base64"],
                },
            }
        ]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            tools=tools,
            messages=[{"role": "user", "content": full_prompt}],
        )

        # Find the write_file tool call in the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "write_file":
                file_bytes = base64.b64decode(block.input["content_base64"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(file_bytes)
                return output_path

        raise RuntimeError(
            f"Claude did not call write_file tool. "
            f"Stop reason: {response.stop_reason}. "
            f"Content types: {[b.type for b in response.content]}"
        )


# ── Mock model client (testing) ───────────────────────────────────────────────

class MockModelClient(ModelClient):
    """
    Deterministic mock model client for pipeline testing without API calls.

    Instead of calling an API, returns a pre-existing fixture xlsx file.
    This lets the full harness pipeline — task loading, programmatic checking,
    gate logic, judge call, hybrid scoring, JSONL logging — run in tests
    without any network access.

    The fixture_path should be one of the existing test fixtures:
      - tests/fixtures/good_minimal.xlsx       (passes all checks)
      - tests/fixtures/bad_hardcoded_revenue.xlsx  (fails revenue check)
      - tests/fixtures/bad_missing_summary.xlsx    (fails sheet check)
      etc.

    This enables testing different pipeline paths by swapping fixtures.
    """

    def __init__(self, fixture_path: Path, model_name: str = "mock-model"):
        self._fixture_path = fixture_path
        self._model_name   = model_name
        self.call_count    = 0

    @property
    def model_id(self) -> str:
        return self._model_name

    def run(self, prompt: str, raw_csv_path: Path, output_path: Path) -> Path:
        """Copy the fixture file to output_path and return it."""
        import shutil
        self.call_count += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._fixture_path, output_path)
        return output_path


# ── JSONL logger ──────────────────────────────────────────────────────────────

class JSONLLogger:
    """
    Append-only JSONL logger. One JSON object per line, one line per eval run.

    JSONL (JSON Lines) format:
      - Each line is a valid, self-contained JSON object.
      - No comma between lines, no enclosing array.
      - Append-only: multiple runs accumulate in the same file.
      - Easy to stream-process with `for line in f` without loading all at once.

    Why JSONL over CSV?
      Each run record has nested structure (judge sub-dict with reasoning_ab,
      reasoning_ba, etc.). CSV would require flattening or escaping nested
      content. JSONL preserves the natural nesting from HybridResult.as_dict().

    Why append-only?
      Eval runs are expensive (real model calls). If the harness crashes mid-run,
      earlier results are preserved. Day 19 analysis reads the file from scratch
      each time, so accumulation is safe.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        """Append one record as a single JSON line."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        """Read all records. Returns empty list if file doesn't exist."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# ── Core run function ─────────────────────────────────────────────────────────

def run_one(
    task_path: Path,
    model_client: ModelClient,
    elicitation: str,
    output_dir: Path,
    logger: JSONLLogger,
    judge=None,      # BlindedPairwiseJudge | MockJudge | None (skips judge entirely)
    oracle: Optional[HybridOracle] = None,
) -> HybridResult:
    """
    Run the full eval pipeline for one task × model × elicitation combination.

    Steps:
      1. Load task config from task_path
      2. Build prompt from task config + elicitation mode
      3. Determine output path: output_dir/{model_id}/{elicitation}/{task_id}.xlsx
      4. Call model_client.run() → saves xlsx to output_path
      5. Run ProgrammaticChecker → ProgrammaticResult
      6. If gate passes AND judge provided: run judge.compare() → JudgeResult
      7. oracle.score(prog_score, judge_result) → HybridResult
      8. Build full record dict and write to JSONL log
      9. Return HybridResult

    Args:
        task_path:     Path to task .json config file.
        model_client:  ModelClient instance (real or mock).
        elicitation:   "zero_shot" or "structured" — selects prompt variant.
        output_dir:    Root directory for model output files.
        logger:        JSONLLogger instance for result logging.
        judge:         Judge instance (real or mock). If None, only prog scores logged.
        oracle:        HybridOracle instance. Defaults to HybridOracle() if None.

    Returns:
        HybridResult with hybrid_score, programmatic_score, judge, agreement_bucket.

    Raises:
        ValueError:   if elicitation mode not in task config prompts.
        RuntimeError: if model fails to produce a valid xlsx.
    """
    if oracle is None:
        oracle = HybridOracle()

    # 1. Load task
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = task["task_id"]

    # 2. Build prompt
    if elicitation not in task["prompts"]:
        raise ValueError(
            f"Elicitation mode '{elicitation}' not in task prompts. "
            f"Available: {list(task['prompts'].keys())}"
        )
    prompt = task["prompts"][elicitation]

    # 3. Output path: outputs/{model_id}/{elicitation}/{task_id}.xlsx
    output_path = output_dir / model_client.model_id / elicitation / f"{task_id}.xlsx"
    raw_csv_path = task_path.parent / task["reference_files"][0]
    gold_path    = task_path.parent.parent / "gold" / task["gold_file"]

    # 4. Call model
    started_at = datetime.now(timezone.utc).isoformat()
    model_error = None
    try:
        model_client.run(prompt, raw_csv_path, output_path)
    except Exception as e:
        model_error = str(e)

    # 5. Programmatic check
    checker = ProgrammaticChecker(task)
    if model_error or not output_path.exists():
        # Model failed to produce output — record a zero score
        hybrid = HybridResult(
            hybrid_score=0.0,
            programmatic_score=0.0,
            judge=None,
            agreement_bucket="gate-triggered",
            gate_triggered=True,
        )
        _log(logger, task, model_client, elicitation, started_at,
             prog_result=None, hybrid=hybrid, model_error=model_error)
        return hybrid

    prog_result = checker.check(output_path)

    # 6. Judge (conditional on gate)
    judge_result = None
    if judge is not None and prog_result.gate_passed:
        try:
            judge_result = judge.compare(prompt, output_path, gold_path)
        except Exception as e:
            # Judge failure should not abort the run — log and continue with prog only
            model_error = f"judge_error: {e}"

    # 7. Score
    hybrid = oracle.score(prog_result.score, judge_result)

    # 8. Log
    _log(logger, task, model_client, elicitation, started_at,
         prog_result=prog_result, hybrid=hybrid, model_error=model_error)

    return hybrid


def _log(
    logger: JSONLLogger,
    task: dict,
    model_client: ModelClient,
    elicitation: str,
    started_at: str,
    prog_result,
    hybrid: HybridResult,
    model_error: Optional[str],
) -> None:
    """
    Build a complete run record and write it to the JSONL log.

    Record schema:
      run_id:           "{task_id}__{model_id}__{elicitation}__{timestamp}"
      task_id:          from task config
      model:            model_client.model_id
      elicitation:      "zero_shot" | "structured"
      started_at:       ISO 8601 UTC timestamp
      model_error:      error string if model call failed, else null
      programmatic:     ProgrammaticResult.as_dict() or null
      hybrid:           HybridResult.as_dict()

    Why include started_at?
      Day 19 analysis may want to check for temporal patterns (e.g., did later
      runs differ from earlier ones — possible if the model was updated mid-eval).

    Why include raw programmatic result alongside hybrid?
      The per-check breakdown (integrity, sheets, errors, revenue, summary,
      reference) is essential for Day 20 error analysis. The hybrid score alone
      doesn't tell you which specific check failed.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{task['task_id']}__{model_client.model_id}__{elicitation}__{ts}"

    record = {
        "run_id":      run_id,
        "task_id":     task["task_id"],
        "model":       model_client.model_id,
        "elicitation": elicitation,
        "started_at":  started_at,
        "model_error": model_error,
        "programmatic": prog_result.as_dict() if prog_result is not None else None,
        "hybrid":       hybrid.as_dict(),
    }
    logger.write(record)


# ── Grid runner ───────────────────────────────────────────────────────────────

def run_grid(
    task_paths: list[Path],
    model_clients: list[ModelClient],
    elicitation_modes: list[str],
    output_dir: Path,
    logger: JSONLLogger,
    judge=None,
    oracle: Optional[HybridOracle] = None,
) -> list[HybridResult]:
    """
    Run all combinations of tasks × models × elicitation modes.

    The full factorial grid for Day 19:
      tasks:       [task_001, task_002, ..., task_015]   (15 tasks)
      models:      [gpt-4o, claude-3-5-sonnet]           (2 models)
      elicitation: [zero_shot, structured]               (2 modes)
      Total:       15 × 2 × 2 = 60 runs

    Results accumulate in the JSONL log. If the harness crashes mid-grid,
    completed runs are preserved and the grid can be resumed (check run_ids
    already in the log before running).

    Args:
        task_paths:        List of task .json paths.
        model_clients:     List of ModelClient instances.
        elicitation_modes: List of elicitation mode strings.
        output_dir:        Root directory for model output files.
        logger:            JSONLLogger instance.
        judge:             Shared judge instance (one judge per grid run).
        oracle:            Shared HybridOracle instance.

    Returns:
        List of HybridResult, one per combination, in grid order.
    """
    results = []
    total = len(task_paths) * len(model_clients) * len(elicitation_modes)
    done  = 0

    for task_path in task_paths:
        for client in model_clients:
            for elicitation in elicitation_modes:
                done += 1
                print(f"[{done}/{total}] {task_path.stem} × {client.model_id} × {elicitation}")
                try:
                    result = run_one(
                        task_path, client, elicitation,
                        output_dir, logger, judge, oracle,
                    )
                    results.append(result)
                    print(f"  → hybrid={result.hybrid_score:.3f}  bucket={result.agreement_bucket}")
                except Exception as e:
                    print(f"  → ERROR: {e}")
                    traceback.print_exc()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_clients(args) -> list[ModelClient]:
    """Build model clients from CLI args, respecting MOCK_MODEL env var."""
    if os.environ.get("MOCK_MODEL"):
        fixture = Path(__file__).parent / "tests" / "fixtures" / "good_minimal.xlsx"
        return [MockModelClient(fixture, name) for name in args.model]

    clients = []
    for model_name in args.model:
        if "gpt" in model_name:
            clients.append(GPT4oClient(model=model_name))
        elif "claude" in model_name:
            clients.append(ClaudeClient(model=model_name))
        else:
            raise ValueError(f"Unknown model: {model_name}. Expected 'gpt-*' or 'claude-*'.")
    return clients


def _build_judge(args):
    """Build judge from CLI args, respecting MOCK_JUDGE env var."""
    if args.no_judge:
        return None
    if os.environ.get("MOCK_JUDGE"):
        return MockJudge()
    return BlindedPairwiseJudge()


def main():
    parser = argparse.ArgumentParser(
        description="Run the data-cleaning eval harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python harness.py --task tasks/task_001.json --model gpt-4o --elicitation structured

  # Grid run (all combos)
  python harness.py --grid --model gpt-4o claude-3-5-sonnet --elicitation zero_shot structured

  # Dry run (no API calls)
  MOCK_MODEL=1 MOCK_JUDGE=1 python harness.py --task tasks/task_001.json --model gpt-4o --elicitation structured
        """
    )

    parser.add_argument("--task",       nargs="+", type=Path, help="Task JSON path(s)")
    parser.add_argument("--model",      nargs="+", default=["gpt-4o"], help="Model ID(s)")
    parser.add_argument("--elicitation",nargs="+", default=["structured"],
                        choices=["zero_shot", "structured"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), dest="output_dir")
    parser.add_argument("--results-file", type=Path, default=Path("results/results.jsonl"),
                        dest="results_file")
    parser.add_argument("--grid",       action="store_true",
                        help="Run all task × model × elicitation combinations")
    parser.add_argument("--no-judge",   action="store_true", dest="no_judge",
                        help="Skip judge calls (programmatic scoring only)")

    args = parser.parse_args()

    # Resolve task paths
    if args.grid:
        task_dir  = Path(__file__).parent / "tasks"
        task_paths = sorted(task_dir.glob("task_*.json"))
        if not task_paths:
            print(f"No task files found in {task_dir}")
            sys.exit(1)
    elif args.task:
        task_paths = args.task
    else:
        parser.error("Provide --task or --grid")

    clients = _build_clients(args)
    judge   = _build_judge(args)
    logger  = JSONLLogger(args.results_file)
    oracle  = HybridOracle()

    print(f"Tasks: {len(task_paths)}  |  Models: {[c.model_id for c in clients]}  "
          f"|  Elicitation: {args.elicitation}  |  Judge: {type(judge).__name__ if judge else 'None'}")
    print(f"Results → {args.results_file}\n")

    results = run_grid(task_paths, clients, args.elicitation, args.output_dir, logger, judge, oracle)

    # Summary
    if results:
        avg = sum(r.hybrid_score for r in results) / len(results)
        buckets = {}
        for r in results:
            buckets[r.agreement_bucket] = buckets.get(r.agreement_bucket, 0) + 1
        print(f"\n{'='*50}")
        print(f"Completed {len(results)} runs. Mean hybrid score: {avg:.3f}")
        print("Agreement buckets:", buckets)


if __name__ == "__main__":
    main()
