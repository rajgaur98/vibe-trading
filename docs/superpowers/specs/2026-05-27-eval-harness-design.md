# Design Spec — Eval Harness & Golden Set

## Problem

The Vibe Trading agents (`TechnicalVolumeAnalyst`, `HeadTrader`) are evaluated only via
unit tests on the surrounding code, and integration-tested via backtests. There is no
mechanism to measure whether a prompt change actually improves agent quality, whether
it regresses on edge cases that previously worked, or whether the structured outputs
still parse cleanly after a refactor.

**Rubric gap:** "Eval harness + golden sets — no evaluation harness running LLM-as-judge
or regression checking against a hand-labeled golden dataset."

**Hiring signal:** Demonstrate that prompt changes are gated by an automated quality
measurement that combines deterministic field scoring with LLM-as-judge prose
evaluation, persisted as a regression baseline in git.

## Solution — Hand-Labeled Golden Set + Hybrid Scoring CLI

A new `vibe_trading.eval.eval` module that:

1. Loads 30–50 hand-curated `(symbol, timestamp)` snapshots from `evals/snapshots/*.yaml`,
   each labeled with target `AnalystOutput` + target `HeadTraderOutput` field values
   plus must-mention / must-not-mention rubrics for the free-text fields.
2. For each case: rebuilds the deterministic market snapshot via `FeaturePipeline.run()`,
   runs the analyst (legacy single-shot snapshot path) and the trader (chained off the
   **labeled** analyst output, not the actual one — this isolates trader regressions
   from analyst regressions).
3. Scores each output field against its label:
   - **Deterministic** for categorical / numeric fields (exact match for enums; tolerance
     windows for floats with linear degradation).
   - **LLM-as-judge** (cheap Gemini Flash by default, swappable via env var) for the
     free-text `thesis` and `reasoning_summary` against the rubrics.
4. Aggregates per-case and suite-level scores, writes a timestamped JSON report under
   `data/reports/`, and compares against `evals/baseline.json` (the last approved run).
   Exits non-zero on regression so CI / pre-merge hooks can gate.
5. `--update-baseline` overwrites `evals/baseline.json` with current scores — the PR
   diff makes prompt impact reviewable.

### Data Flow (per case)

```
YAML case file                Eval Runner                     Scorer
   │                               │                              │
   ├─ symbol, timestamp ──────────►│                              │
   │                               ├─ FeaturePipeline.run() ──► snapshot dict
   │                               │   (DuckDB query, no LLM)     │
   │                               │                              │
   │                               ├─ TechnicalVolumeAnalyst      │
   │                               │   .analyze(symbol,           │
   │                               │            snapshot=…)       │
   │                               │   ── single LLM call ──► actual AnalystOutput
   │                               │                              │
   │                               ├─ HeadTrader.decide(          │
   │                               │     symbol,                  │
   │                               │     analyst_LABEL,  ◄── note: labeled, not actual
   │                               │     scorecard, positions)    │
   │                               │   ── single LLM call ──► actual HeadTraderOutput
   │                               │                              │
   │                               │                  actual + label ──►│
   │                               │                                    ├─ deterministic
   │                               │                                    │   field scoring
   │                               │                                    ├─ LLM judge on
   │                               │                                    │   rubric fields
   │                               │                                    │   (Gemini Flash)
   │                               │                                ◄── CaseScore
   │                               │                                    │
   │                               │◄─────────────────────────────────── │
   │                               │
   │                               ├─ aggregate → suite report
   │                               ├─ diff vs evals/baseline.json
   │                               └─ exit 0 (clean) or 1 (regression)
```

## File Layout

```
evals/                                    # NEW: golden set + baseline (committed)
├── snapshots/
│   ├── 001-btc-may22-breakout.yaml
│   ├── 002-eth-fakeout.yaml
│   └── …                                 # 30–50 cases
└── baseline.json                         # frozen scores from last approved run

src/vibe_trading/eval/
├── backtest.py                           # existing
├── eval.py                               # NEW: CLI entry point + orchestration
├── runner.py                             # NEW: loads cases, runs agents, captures outputs
├── scorer.py                             # NEW: deterministic + LLM-judge scoring
└── report.py                             # NEW: writes JSON report, baseline diff/update

tests/
├── fixtures/eval/                        # NEW: synthetic YAML fixtures for unit tests
│   ├── valid-case.yaml
│   └── malformed-case.yaml
└── test_eval.py                          # NEW: unit tests for loader/scorer/diff
```

Each module has one responsibility:

- **`eval.py`** — `argparse`, env validation, calls into runner → scorer → report. No business logic.
- **`runner.py`** — `EvalCase` pydantic loader + per-case execution. No scoring logic.
- **`scorer.py`** — pure functions: `score_categorical`, `score_numeric_tolerance`,
  `score_rubric_with_judge`, plus the per-case aggregator. No I/O.
- **`report.py`** — JSON serialization, baseline diff, terminal pretty-printer. No agent logic.

## Components

### 1. Golden-Set YAML Schema

```yaml
# evals/snapshots/001-btc-may22-breakout.yaml
id: 001-btc-may22-breakout
description: |
  BTC breaks above $30k resistance after 6 weeks of consolidation.
  Rising volume on the breakout candle confirms accumulation.
  Classic Murphy-style bullish setup.

# Source data — resolves via DuckDB candles at run time
symbol: BTC/USDT
timestamp: 2026-05-22T04:00:00Z

# Target analyst output
analyst_label:
  market_bias: bullish                   # exact-match enum
  volume_confirmation: confirmed         # exact-match enum
  nearest_support: 30100.0               # numeric, ±2% tolerance
  nearest_resistance: 32500.0            # numeric, ±2% tolerance
  confluence_score: 0.75                 # numeric, ±0.15 tolerance
  thesis_rubric:
    must_mention:
      - breakout above resistance
      - volume confirmation
      - bullish MA stack or rising RSI
    must_not_mention:
      - overbought warning
      - bearish reversal

# Target trader output (chained off the labeled analyst output above)
trader_label:
  action: long                           # exact-match enum
  stop_loss_strategy: 1.5_atr            # exact-match enum
  take_profit_strategy: next_resistance  # exact-match enum
  risk_reward_ratio: 2.0                 # numeric, ±0.5 tolerance
  hold_period_bias: medium               # exact-match enum
  reasoning_rubric:
    must_mention:
      - breakout
      - volume confluence
    must_not_mention:
      - risk aversion contradicting high confluence
```

Design choices:

- **Rubric in lists, not paragraph.** Each `must_mention` / `must_not_mention` entry
  is one criterion the LLM judge evaluates independently. Score per field is
  `passed_criteria / total_criteria`, which yields stable 0.0–1.0 numbers, surfaces
  partial matches, and is robust to paraphrase.
- **Tolerances are NOT in the YAML.** They live as constants in `scorer.py` so
  the golden set stays terse and tolerance tuning is one diff away from changing
  every case at once.
- **Filename = `<NNN>-<slug>.yaml`** for stable ordering, easy grep, easy reference
  from terminal output.

### 2. `src/vibe_trading/eval/runner.py` [NEW]

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import yaml
from pathlib import Path

from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput
from vibe_trading.agents.trader import HeadTrader
from vibe_trading.data.db import Database
from vibe_trading.features.pipeline import FeaturePipeline


class Rubric(BaseModel):
    must_mention: list[str] = []
    must_not_mention: list[str] = []


class AnalystLabel(BaseModel):
    market_bias: str
    volume_confirmation: str
    nearest_support: float
    nearest_resistance: float
    confluence_score: float
    thesis_rubric: Rubric


class TraderLabel(BaseModel):
    action: str
    stop_loss_strategy: str
    take_profit_strategy: str
    risk_reward_ratio: float
    hold_period_bias: str
    reasoning_rubric: Rubric


class EvalCase(BaseModel):
    id: str
    description: str
    symbol: str
    timestamp: datetime
    analyst_label: AnalystLabel
    trader_label: TraderLabel


class CaseResult(BaseModel):
    case_id: str
    snapshot_ok: bool                 # did FeaturePipeline produce a snapshot
    analyst_output: Optional[dict]    # raw dict, may be None on parse failure
    analyst_parsed: Optional[AnalystOutput]
    trader_output: Optional[dict]
    error: Optional[str] = None       # populated if a hard failure occurred
```

Public API:

- `load_cases(dir: Path) -> list[EvalCase]` — globs `*.yaml`, validates each via
  pydantic, raises descriptively on schema failures.
- `run_case(case: EvalCase, db: Database) -> CaseResult` — builds snapshot, runs
  analyst (legacy snapshot path), feeds **`case.analyst_label`** (not actual!) into
  the trader, returns captured outputs. All exceptions caught and stored on the result.

Notes:

- Runner constructs `TechnicalVolumeAnalyst()` with **no** `db`/`fetcher` so the
  tool-executor branch is NOT taken — eval explicitly exercises the legacy snapshot
  path for stable, single-LLM-call prompt-quality measurement.
- Trader is fed a synthetic `scorecard = {"accuracy": 0.55, "total_decisions": 100}`
  and empty `open_positions = []` — fixed across all cases so trader scoring measures
  prompt quality, not context-dependent behavior.

### 3. `src/vibe_trading/eval/scorer.py` [NEW]

Pure functions, no I/O, no LLM calls hardcoded — judge is injected.

```python
from pydantic import BaseModel
from typing import Callable, Optional


# Tolerance constants (tune once, applied to every case)
NUMERIC_TOLERANCES = {
    "nearest_support":      {"ok": 0.02, "zero": 0.05},  # 2% perfect, 5% fail
    "nearest_resistance":   {"ok": 0.02, "zero": 0.05},
    "confluence_score":     {"ok": 0.15, "zero": 0.30},  # absolute units
    "risk_reward_ratio":    {"ok": 0.5,  "zero": 1.5},   # absolute units
}


class FieldScore(BaseModel):
    field: str
    passed: bool
    score: float          # 0.0 .. 1.0
    note: str = ""


class CaseScore(BaseModel):
    case_id: str
    schema_ok: bool
    field_scores: list[FieldScore]
    analyst_score: float  # mean of analyst-field scores
    trader_score: float   # mean of trader-field scores
    total_score: float    # mean of all field scores
    error: Optional[str] = None


def score_categorical(field: str, actual, expected) -> FieldScore: ...
def score_numeric_tolerance(field: str, actual: float, expected: float) -> FieldScore: ...
def score_rubric(field: str, actual_text: str, rubric: Rubric, judge: Callable) -> FieldScore: ...
def score_case(result: CaseResult, case: EvalCase, judge: Callable) -> CaseScore: ...
```

Rules table:

| Field                   | Type        | Scorer                                                  |
|-------------------------|-------------|---------------------------------------------------------|
| `market_bias`           | enum        | exact match → 1.0, else 0.0                             |
| `volume_confirmation`   | enum        | exact match → 1.0, else 0.0                             |
| `nearest_support`       | float       | within 2% → 1.0; linear to 0.0 at 5%; 0.0 beyond        |
| `nearest_resistance`    | float       | same as `nearest_support`                               |
| `confluence_score`      | float       | within ±0.15 → 1.0; linear to 0.0 at ±0.30              |
| `thesis`                | rubric      | judge returns per-criterion pass list; score = passed/total |
| `action`                | enum        | exact match                                             |
| `stop_loss_strategy`    | enum        | exact match                                             |
| `take_profit_strategy`  | enum        | exact match                                             |
| `risk_reward_ratio`     | float       | within ±0.5 → 1.0; linear to 0.0 at ±1.5                |
| `hold_period_bias`      | enum        | exact match                                             |
| `reasoning_summary`     | rubric      | same as `thesis`                                        |
| **schema validity**     | meta        | Pydantic parsed OK → 1.0; failed parse short-circuits the entire case to total_score=0.0 |

`score_numeric_tolerance` handles edge cases:
- `expected == 0.0` falls back to absolute-distance scoring (treats `ok` / `zero`
  thresholds as absolute, not percentage).
- `NaN` / `None` in actual → score 0.0, note `"missing value"`.

**Linear degradation formula** (used for all numeric scorers):

```
For percentage-based tolerances (S/R prices):
    delta = abs(actual - expected) / abs(expected)
For absolute-distance tolerances (confluence_score, risk_reward_ratio):
    delta = abs(actual - expected)

if delta <= ok_threshold:        score = 1.0
elif delta >= zero_threshold:    score = 0.0
else:                            score = 1.0 - (delta - ok_threshold) / (zero_threshold - ok_threshold)

passed = (score == 1.0)
```

### 4. LLM Judge

```python
from pydantic import BaseModel
from vibe_trading.agents.client import LLMClient
import os


class CriterionEvaluation(BaseModel):
    criterion: str
    passed: bool
    justification: str


class JudgeOutput(BaseModel):
    must_mention_results: list[CriterionEvaluation]
    must_not_mention_results: list[CriterionEvaluation]


JUDGE_SYSTEM = """
You are a meticulous code-review-style evaluator. You will receive a piece of agent-generated
text and a rubric of must-mention and must-not-mention criteria. For each criterion, decide
whether the text satisfies it. Be strict on must-mention (the criterion must be clearly
present, not just hinted at) and strict on must-not-mention (any clear violation fails it).
Output strictly matches the JudgeOutput schema.
""".strip()


def build_judge(model_env: str = "EVAL_JUDGE_MODEL") -> Callable[[str, Rubric], JudgeOutput]:
    """Returns a closure that takes (actual_text, rubric) and returns a JudgeOutput.

    The underlying model is resolved at call time from the env var (default Gemini Flash),
    so swapping judges is one env var away.
    """
    client = LLMClient()
    default_model = "gemini-3.1-flash-lite"

    def judge(actual_text: str, rubric: Rubric) -> JudgeOutput:
        model_name = os.getenv(model_env, default_model)
        prompt = f"""TEXT:\n{actual_text}\n\nRUBRIC:\nMUST MENTION:\n""" + \
                 "\n".join(f"- {c}" for c in rubric.must_mention) + \
                 "\n\nMUST NOT MENTION:\n" + \
                 "\n".join(f"- {c}" for c in rubric.must_not_mention)
        raw = client.call_llm(
            model_name=model_name,
            system_instruction=JUDGE_SYSTEM,
            prompt=prompt,
            response_schema=JudgeOutput,
        )
        return JudgeOutput.model_validate_json(raw)
    return judge
```

The judge is a single function so unit tests can inject a stub. `temperature=0.1` is
inherited from existing `LLMClient.call_llm` defaults — judge variance run-to-run is
typically <2% which is accommodated by the regression thresholds below.

**Scoring formula for rubric fields:**

```
total_criteria  = len(rubric.must_mention) + len(rubric.must_not_mention)
passed_criteria = sum(1 for r in judge.must_mention_results if r.passed) +
                  sum(1 for r in judge.must_not_mention_results if r.passed)
score = passed_criteria / total_criteria  if total_criteria > 0 else 1.0
```

A rubric with zero criteria scores 1.0 (vacuously satisfied) — this lets cases opt
out of free-text scoring by leaving both lists empty.

### 5. `src/vibe_trading/eval/report.py` [NEW]

```python
from pathlib import Path
from datetime import datetime, timezone
import json


REGRESSION_THRESHOLDS = {
    "overall": 0.02,    # suite-level regression band
    "per_case": 0.05,   # per-case regression band
}


class SuiteReport(BaseModel):
    run_id: str
    run_at: datetime
    case_count: int
    overall_score: float
    analyst_score: float
    trader_score: float
    pass_rate: float       # fraction of cases where every FieldScore.passed == True
    schema_failures: int
    judge_errors: int
    per_case: dict[str, CaseScore]


def write_report(report: SuiteReport, reports_dir: Path) -> Path: ...
def load_baseline(baseline_path: Path) -> Optional[dict]: ...
def write_baseline(report: SuiteReport, baseline_path: Path) -> None: ...
def diff_against_baseline(report: SuiteReport, baseline: dict) -> DiffResult: ...
def print_summary(report: SuiteReport, diff: Optional[DiffResult]) -> None: ...


class DiffResult(BaseModel):
    overall_delta: float
    per_case_regressions: list[str]    # case ids that dropped > REGRESSION_THRESHOLDS["per_case"]
    per_case_improvements: list[str]
    new_schema_failures: list[str]     # cases that parsed in baseline but failed now
    is_regression: bool                # any of: overall < threshold, per_case_regressions, new_schema_failures
```

The exit code is computed from `DiffResult.is_regression` in `eval.py`.

### 6. `src/vibe_trading/eval/eval.py` [NEW] — CLI Entry Point

```python
import argparse
import sys
from pathlib import Path

from vibe_trading.data.db import Database
from vibe_trading.eval.runner import load_cases, run_case
from vibe_trading.eval.scorer import score_case, build_judge
from vibe_trading.eval.report import (
    SuiteReport, write_report, load_baseline, write_baseline,
    diff_against_baseline, print_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="vibe-eval")
    parser.add_argument("--snapshots", type=Path, default=Path("evals/snapshots"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline.json with current run results")
    args = parser.parse_args()

    cases = load_cases(args.snapshots)
    if not cases:
        print(f"No cases found in {args.snapshots}", file=sys.stderr)
        return 1

    db = Database()
    judge = build_judge()
    case_scores = []
    for case in cases:
        result = run_case(case, db)
        score = score_case(result, case, judge)
        case_scores.append(score)

    report = SuiteReport.from_scores(case_scores)
    report_path = write_report(report, args.reports_dir)

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"No baseline at {args.baseline}. Run with --update-baseline to seed it.")
        print_summary(report, diff=None)
        return 0

    diff = diff_against_baseline(report, baseline)
    print_summary(report, diff)

    if args.update_baseline:
        write_baseline(report, args.baseline)
        print(f"Baseline updated: {args.baseline}")

    return 1 if diff.is_regression else 0


if __name__ == "__main__":
    sys.exit(main())
```

CLI usage:

```bash
# Run the suite, see scores, compare against baseline. Non-zero exit on regression.
uv run python -m vibe_trading.eval.eval

# After a prompt change that you've reviewed and approved:
uv run python -m vibe_trading.eval.eval --update-baseline
```

## Regression Detection

`evals/baseline.json` shape:

```json
{
  "baseline_committed_at": "2026-05-27T18:00:00Z",
  "baseline_run_id": "eval-2026-05-27T17-58-12Z",
  "overall_score": 0.84,
  "analyst_score": 0.81,
  "trader_score": 0.87,
  "pass_rate": 0.72,
  "schema_failures": 0,
  "per_case": {
    "001-btc-may22-breakout": {"total": 0.91, "analyst": 0.88, "trader": 0.94, "schema_ok": true},
    "002-eth-fakeout":        {"total": 0.76, "analyst": 0.70, "trader": 0.82, "schema_ok": true}
  }
}
```

Comparison rules:

| Condition                                                                  | Effect            |
|----------------------------------------------------------------------------|-------------------|
| `report.overall_score < baseline.overall_score - 0.02`                     | regression: exit 1 |
| Any case where `report.per_case[id].total < baseline.per_case[id].total - 0.05` | regression: exit 1, list cases |
| New schema failures (case parsed in baseline, failed now)                  | regression: exit 1, list cases |
| Improvements (overall +0.02, per-case +0.05)                               | reported, never fail |
| Missing baseline                                                           | exit 0, suggest `--update-baseline` |

Thresholds live in `scorer.py` as `REGRESSION_THRESHOLDS`. The 0.02 / 0.05 numbers
accommodate ~1–2% LLM-judge non-determinism on the same input; revisit once we have
empirical drift data from real runs.

## Error Handling

| Scenario                                                          | Behavior |
|-------------------------------------------------------------------|----------|
| Malformed YAML in a snapshot                                      | Skip case, log warning, count as `schema_failure` |
| `(symbol, timestamp)` not in DuckDB candles                       | Skip case, log warning, bootstrap hint at end |
| `FeaturePipeline.run()` returns empty (insufficient candle history) | Same as above — skip + warn |
| Analyst LLM call fails (network / quota)                          | `CaseResult.error` populated; case total scores 0.0; suite continues |
| Trader LLM call fails                                             | Same — case total scores 0.0; suite continues |
| Pydantic parse fails on agent output                              | `schema_ok=False`; case total scores 0.0; tracked in `schema_failures` count |
| Judge LLM fails                                                   | Free-text field scores 0.5 with note "judge_error"; deterministic fields score normally; counted in `judge_errors` |
| Missing or malformed `evals/baseline.json`                        | Exit 0 with helpful message; never fails |
| `EVAL_JUDGE_MODEL` env var set to unknown model                   | LiteLLM raises on first judge call → caught as judge failure above |

**Principle:** the suite always completes a full pass. Individual failures degrade
their case score and surface in the report; they never abort the run. A prompt-change
PR therefore gets full signal even when a few cases are broken.

## Testing Strategy

Unit tests in `tests/test_eval.py`:

| Test                                              | Verifies |
|---------------------------------------------------|----------|
| `test_load_snapshot_yaml`                         | Round-trip fixture YAML → `EvalCase`; all fields hydrate |
| `test_load_snapshot_malformed_yaml`               | Malformed fixture → descriptive `ValidationError`, doesn't crash loader |
| `test_score_categorical_exact_match`              | Match → 1.0; mismatch → 0.0 |
| `test_score_numeric_within_tolerance`             | Exact, edge of tolerance, outside tolerance, NaN, expected==0 edge case |
| `test_score_rubric_with_mocked_judge`             | Mocks judge closure; verifies score is `passed/total`, prompt text shape |
| `test_score_rubric_judge_failure_returns_neutral` | Judge raises → score 0.5 with `judge_error` note; deterministic fields untouched |
| `test_schema_failure_short_circuits_case`         | Malformed agent JSON → `schema_ok=False`, total_score=0.0 |
| `test_runner_handles_missing_candles`             | Snapshot with no DuckDB candles → case skipped, suite continues |
| `test_runner_uses_legacy_snapshot_path`           | Verifies `TechnicalVolumeAnalyst` is instantiated without db/fetcher, so tool-loop isn't taken |
| `test_runner_feeds_LABELED_analyst_to_trader`     | Mocks both agents; verifies trader receives `case.analyst_label`, not actual analyst output |
| `test_baseline_diff_detects_regression`           | Synthetic scores → correct `DiffResult`, exit code logic |
| `test_baseline_diff_no_baseline_present`          | Missing baseline file → exit 0, warning emitted |
| `test_baseline_diff_new_schema_failure_flags_regression` | Baseline case passed parse, current run failed parse → regression |

Plus 2-3 synthetic YAML fixtures under `tests/fixtures/eval/`:

- `valid-case.yaml` — minimal well-formed case for loader tests.
- `malformed-case.yaml` — missing required field (e.g. `analyst_label`) for negative test.

Not tested:
- The actual golden-set quality — that's the user's craftsmanship, not code logic.
- Live LLM calls — eval suite hits real LLMs only in manual CLI mode; pytest stays hermetic.

## Operational Notes

- **First-run bootstrapping.** The first prompt change after merging this feature
  will need a `--update-baseline` commit. Document this in `README.md` under the
  development workflow section (separate task, not in scope for this spec).
- **DuckDB candle availability.** The eval requires historical candles to be already
  populated in DuckDB for every `(symbol, timestamp)` in the golden set. The
  scheduler's `bootstrap_if_needed` only bootstraps active trending symbols; the
  golden set may include retired symbols. Either (a) commit a small DuckDB seed
  file under `data/eval-fixtures.db` and have the runner open that DB instead, or
  (b) extend the bootstrap to honor a list of required symbols read from the golden
  set. Pick (b) — it's the path that stays in sync with prod fetcher logic.
- **Cost per full run** (50 cases × 2 LLM calls per case for agents + ~2 LLM
  judge calls per case for free-text fields) ≈ 200 calls × Gemini Flash @ ~$0.001 =
  ~$0.20 per full run. Affordable to run on every prompt PR.
- **Runtime** ≈ 50 cases × ~3 s per case = ~2-3 minutes for a full run on the
  legacy snapshot path. Tolerable in pre-merge CI.

## Backwards Compatibility

No changes to existing modules — eval is purely additive. Existing tests, backtest,
scheduler, and the analyst tool-loop path remain untouched.
