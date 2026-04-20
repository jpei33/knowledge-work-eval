# data-cleaning-eval

A knowledge work eval measuring whether frontier models can transform messy tabular data into a clean, well-structured `.xlsx` deliverable with accurate summary statistics.

Inspired by [GDPval](https://arxiv.org/abs/2510.04374) methodology. Demonstrates **hybrid scoring** — programmatic structural checks anchored to an LLM-as-judge blinded pairwise comparison — applied to a domain-agnostic spreadsheet task. Designed as a portable template for knowledge work evals where no single oracle suffices.

---

## Task

**Input:** A messy 75-row CSV with three data quality issues:
- Mixed date formats (ISO, MM/DD/YYYY, DD-Mon-YYYY)
- Duplicate order IDs (5 exact duplicates)
- Null revenue values (8 rows, computable from `quantity × unit_price`)

**Expected output:** A 3-sheet `.xlsx` deliverable:
- **Raw Data** — original CSV, unchanged
- **Cleaned Data** — 70 rows, standardized dates, deduplicates removed, revenue as `=G{n}*H{n}` formulas
- **Summary** — SUMIF/COUNTIF/AVERAGE formulas referencing Cleaned Data (not Raw Data)

**Elicitation modes:**
- `zero_shot` — minimal prompt, no format guidance
- `structured` — full format spec, explicit formula requirements, sheet layout instructions

The gap between modes is the **under-contextualization signal**: how much does explicit scaffolding improve performance? A large gap means the model can execute the task when well-described but cannot independently scope it from a minimal prompt.

---

## Oracle: Hybrid Scoring

No single oracle suffices for knowledge work. This eval uses a two-layer hybrid:

```
Layer 1 — Programmatic (programmatic_checks.py)
  6 checks, weighted sum → prog_score ∈ [0, 1]

  Check             Weight  What it catches
  ─────────────────────────────────────────────────────────────────
  file_integrity    0.10    Corrupt or unreadable .xlsx
  required_sheets   0.20    Missing Raw Data / Cleaned Data / Summary
  formula_errors    0.25    #REF!, #VALUE!, #DIV/0! anywhere in workbook
  revenue_formulas  0.20    Hardcoded values in revenue column (not =G*H)
  summary_formulas  0.15    Hardcoded numbers in Summary value cells
  summary_ref       0.10    Summary referencing Raw Data instead of Cleaned Data

  Gate: if prog_score < 0.3 → skip judge (file too broken to evaluate content)

Layer 2 — LLM Judge (llm_judge.py)
  Gemini Flash blinded pairwise comparison: model output vs. gold file
  Position-swap mitigation: two calls with A/B order reversed
  Reconciliation: agree → use winner, disagree → 0.5 (tie, not average)
  Cross-family judge (Gemini for GPT-4o/Claude evals) → avoids self-preference bias

Combiner (common/scoring.py)
  hybrid_score = 0.4 × prog_score + 0.6 × judge_score
```

### Agreement buckets

The oracle classifies every run into one of four buckets:

| Bucket | prog | judge | Interpretation |
|---|---|---|---|
| `agree-good` | ≥ 0.7 | ≥ 0.5 | Clean win — valid structure, good analysis |
| `agree-bad` | < 0.7 | < 0.5 | Comprehensive failure |
| `judge-rescues` | < 0.7 | ≥ 0.5 | Formula bugs, good analysis — **recoverable** |
| `programmatic-catches` | ≥ 0.7 | < 0.5 | Looks right, wrong analysis — **dangerous** |

`judge-rescues` is more recoverable than `programmatic-catches`. The former means the model understood the task but made mechanical mistakes fixable with better prompt formatting. The latter means the model produced correct-looking output with wrong analysis — a deeper capability problem.

---

## Key Findings (Synthetic Data, Days 19–20)

**Model comparison:**

| Model | Elicitation | Mean hybrid | 95% CI | Dominant bucket |
|---|---|---|---|---|
| claude-3-5-sonnet | structured | 0.753 | [0.681, 0.839] | agree-good |
| gpt-4o | structured | 0.696 | [0.591, 0.800] | agree-good |
| gpt-4o | zero_shot | 0.507 | [0.410, 0.577] | judge-rescues |
| claude-3-5-sonnet | zero_shot | 0.430 | [0.362, 0.497] | agree-bad |

**Elicitation gaps** (structured − zero_shot):
- Claude: +0.323 (large — highly prompt-sensitive)
- GPT-4o: +0.188 (large — prompt-sensitive)

Both models show large elicitation gaps, consistent with GDPval's under-contextualization finding. Neither model independently scopes all required cleaning steps from a minimal prompt.

**Failure mode taxonomy:**

| Category | Checks | Failure rate | Fix |
|---|---|---|---|
| Formula | revenue, reference | 32.5% | Add explicit formula instructions to prompt |
| Content | errors, summary | 30.0% | Capability gap — harder to fix |
| Structural | integrity, sheets | 5.0% | Already low |

Dominant failure: **Formula**. Models compute the right values but hardcode them instead of writing `=G2*H2`. The output looks correct to a human reviewer but fails the programmatic checker — exactly the failure mode that requires `data_only=False` to detect.

**Difficulty calibration:** All 5 synthetic tasks fall in the medium tier (std = 0.035 → `low_discrimination`). A production eval would add explicitly hard tasks (compound errors, no format hints) and easy tasks (single-column cleaning) to improve model separation.

**Model-task agreement (Spearman ρ = −0.70):** The two models disagree on which tasks are hard. In a real eval, tasks with high disagreement would be inspected for model-specific biases and potentially redesigned.

---

## Repo Structure

```
data-cleaning-eval/
├── tasks/
│   ├── task_001.json          # task config (required sheets, column names, prompts)
│   └── task_001_raw.csv       # messy input CSV
├── gold/
│   └── task_001_gold.xlsx     # known-correct deliverable (gitignored in production)
├── outputs/                   # model outputs, organised by model/elicitation/task
├── results/
│   └── synthetic_results.jsonl  # JSONL scoring log (one record per run)
├── scripts/
│   └── generate_synthetic_results.py  # synthetic data generator for dev/testing
├── tests/
│   ├── fixtures/              # one .xlsx per failure mode (6 fixtures)
│   ├── test_programmatic_checks.py   # 33 tests
│   ├── test_llm_judge.py             # 29 tests (2 skipped: require GEMINI_API_KEY)
│   ├── test_harness.py               # 24 tests
│   ├── test_analyze_results.py       # 40 tests
│   └── test_error_analysis.py        # 50 tests
├── harness.py                 # full eval pipeline: load → call model → score → log
├── programmatic_checks.py     # ProgrammaticChecker: 6 structural checks
├── llm_judge.py               # BlindedPairwiseJudge (Gemini Flash) + MockJudge
├── analyze_results.py         # bootstrap CIs, elicitation gap, bucket analysis
└── error_analysis.py          # failure mode taxonomy, difficulty calibration, Spearman ρ

common/                        # shared across evals
├── scoring.py                 # HybridOracle, HybridResult, JudgeResult
└── xlsx_utils.py              # load_workbook, iter_formula_cells, extract_to_text
```

---

## Running the Eval

**Install dependencies:**
```bash
pip install pytest openpyxl numpy
```

**Generate synthetic results (no API keys needed):**
```bash
cd Code/data-cleaning-eval
python scripts/generate_synthetic_results.py
```

**Run the full analysis report:**
```bash
python analyze_results.py results/synthetic_results.jsonl
```

**Run with real APIs:**
```bash
OPENAI_API_KEY=sk-... GEMINI_API_KEY=AIza... python harness.py \
    --task tasks/task_001.json \
    --model gpt-4o \
    --elicitation structured
```

**Dry run (both mocks, no API calls):**
```bash
MOCK_MODEL=1 MOCK_JUDGE=1 python harness.py \
    --task tasks/task_001.json --model gpt-4o --elicitation structured
```

**Run tests:**
```bash
pytest tests/ -v
# 176 passed, 2 skipped (skipped require GEMINI_API_KEY)
```

---

## Task Format

```json
{
  "task_id": "task_001",
  "domain": "retail_sales",
  "messiness_profile": ["mixed_date_formats", "duplicate_order_ids", "null_revenue_values"],
  "prompts": {
    "zero_shot": "Clean the attached CSV and produce a 3-sheet Excel file.",
    "structured": "Clean the attached CSV and produce a 3-sheet Excel file with the following requirements: ..."
  },
  "reference_files": ["task_001_raw.csv"],
  "gold_file": "task_001_gold.xlsx",
  "required_sheets": ["Raw Data", "Cleaned Data", "Summary"],
  "revenue_column": "I",
  "summary_sheet": "Summary",
  "cleaned_sheet": "Cleaned Data"
}
```

---

## Design Notes

**Why `data_only=False` for formula detection:**
openpyxl's `data_only=True` mode returns cached computed values — a hardcoded `239.97` and a formula `=G2*H2` look identical. `data_only=False` returns formula strings, making the check `isinstance(cell.value, str) and cell.value.startswith("=")` unambiguous. This is the core mechanism behind hardcoded value detection.

**Why Gemini Flash as judge (not GPT-4o):**
Self-preference bias (Wataoka et al. 2024) inflates same-family win rates by ~10%. Gemini Flash has uncorrelated stylistic preferences with GPT-4o and Claude outputs. Flash over Pro/Ultra for cost: with gate logic, each 60-run eval costs ~120 judge calls.

**Why position-swap (not single call):**
MT-Bench found LLM judges prefer the first option ~65% of the time without mitigation. Two calls with A/B swapped, reconciled by agreement: inconsistent calls return 0.5 (tie), not an average. Averaging would import the bias into the final score.

**Why JSONL over CSV:**
Crash-safe (each line is independently valid JSON), append-only (no header/enclosing structure to corrupt), supports nested fields (judge reasoning text, per-check arrays).

---

## Methodology Reference

- GDPval: [Evaluating AI on GDP-Level Tasks](https://arxiv.org/abs/2510.04374)
- MT-Bench / LLM-as-judge: [Zheng et al. 2023](https://arxiv.org/abs/2306.05685)
- Self-preference bias: [Wataoka et al. 2024](https://arxiv.org/abs/2404.13076)
- Bootstrap CIs for eval: [Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004)
