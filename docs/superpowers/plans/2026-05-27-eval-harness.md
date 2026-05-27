# Eval Harness & Golden Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI eval harness that scores both Vibe Trading agents (`TechnicalVolumeAnalyst`, `HeadTrader`) against a hand-labeled golden set of market snapshots, using hybrid scoring (deterministic field-level rules + LLM-as-judge for free-text), persisting a baseline in git so prompt-change PRs can be reviewed against measurable regression.

**Architecture:** A new `src/vibe_trading/eval/` module split into four single-responsibility files — `runner.py` (loads YAML cases and runs agents on the legacy snapshot path), `scorer.py` (pure scoring functions + injectable LLM judge), `report.py` (JSON output + baseline diff), `eval.py` (argparse CLI gluing them together). Golden-set YAMLs live under `evals/snapshots/` and the regression baseline lives at `evals/baseline.json` — both committed so prompt-change PRs surface their impact in the diff. The eval exercises the analyst's legacy single-shot snapshot path (not the multi-turn tool loop) for stable, single-LLM-call prompt-quality measurement; the trader is always fed the **labeled** analyst output (not the actual one) so trader regressions are isolated from analyst regressions.

**Tech Stack:** Python 3.12, pydantic v2, PyYAML, LiteLLM-backed `LLMClient` (existing), pytest. No new infrastructure dependencies.

---

## File Structure

```
evals/                                          # NEW: golden set + baseline (committed)
├── snapshots/
│   ├── 001-example-bullish-breakout.yaml       # Seed example (Task 12)
│   └── 002-example-bearish-fakeout.yaml        # Seed example (Task 12)
└── baseline.json                               # Empty/seed; user populates via --update-baseline

src/vibe_trading/eval/
├── backtest.py                                 # existing — untouched
├── runner.py                                   # NEW: pydantic models + load_cases + run_case
├── scorer.py                                   # NEW: pure scorers + build_judge + score_case
├── report.py                                   # NEW: SuiteReport, write_report, baseline diff
└── eval.py                                     # NEW: argparse CLI entry point (main)

tests/
├── fixtures/eval/
│   ├── valid-case.yaml                         # NEW (Task 2)
│   └── malformed-case.yaml                     # NEW (Task 3)
└── test_eval.py                                # NEW: all unit tests for the harness
```

Each module has one responsibility:
- `runner.py` — pydantic case schemas + YAML loading + agent invocation per case. No scoring.
- `scorer.py` — pure scoring functions + judge construction. No I/O, no agent invocation.
- `report.py` — suite aggregation + JSON serialization + baseline diff. No agent / no scoring logic.
- `eval.py` — argparse + module wiring only. No business logic.

---

### Task 1: Add PyYAML dependency and eval directory scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `evals/snapshots/.gitkeep` (empty marker file so the directory commits)
- Create: `tests/fixtures/eval/.gitkeep`

- [ ] **Step 1: Inspect current pyproject.toml**

Run: `cat /Users/raj/vibe-trading/pyproject.toml`
Expected: see existing `dependencies` block; `pyyaml` is likely missing as a direct dep (it comes transitively via langfuse, but the eval module needs an explicit declaration).

- [ ] **Step 2: Add the dependency**

Edit `pyproject.toml`. Locate the `dependencies = [...]` list. Add `"pyyaml>=6.0",` immediately after `"pydantic>=2.6.1",`. The result for that region should look like:

```toml
    "pydantic>=2.6.1",
    "pyyaml>=6.0",
    "TA-Lib>=0.4.28",
```

- [ ] **Step 3: Sync deps**

Run: `uv sync`
Expected: confirms `pyyaml` is satisfied (likely already installed transitively); no errors.

- [ ] **Step 4: Create empty directories so git tracks them**

```bash
mkdir -p /Users/raj/vibe-trading/evals/snapshots
mkdir -p /Users/raj/vibe-trading/tests/fixtures/eval
touch /Users/raj/vibe-trading/evals/snapshots/.gitkeep
touch /Users/raj/vibe-trading/tests/fixtures/eval/.gitkeep
```

- [ ] **Step 5: Verify import**

Run: `uv run python -c "import yaml; print(yaml.__version__)"`
Expected: a version string ≥ 6.0 prints; no error.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock evals/snapshots/.gitkeep tests/fixtures/eval/.gitkeep
git commit -m "chore: scaffold eval directory tree and declare pyyaml dependency"
```

(Stage only those files; the user has unrelated WIP edits elsewhere in the repo.)

---

### Task 2: Define pydantic models in runner.py + add valid fixture

**Files:**
- Create: `src/vibe_trading/eval/runner.py`
- Create: `tests/fixtures/eval/valid-case.yaml`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Create the valid fixture YAML**

Create `tests/fixtures/eval/valid-case.yaml`:

```yaml
id: fixture-001-bullish
description: |
  Synthetic test fixture for loader unit tests. Not a real market event.
symbol: BTC/USDT
timestamp: 2026-05-22T04:00:00Z

analyst_label:
  market_bias: bullish
  volume_confirmation: confirmed
  nearest_support: 30100.0
  nearest_resistance: 32500.0
  confluence_score: 0.75
  thesis_rubric:
    must_mention:
      - breakout above resistance
      - rising volume
    must_not_mention:
      - overbought warning

trader_label:
  action: long
  stop_loss_strategy: 1.5_atr
  take_profit_strategy: next_resistance
  risk_reward_ratio: 2.0
  hold_period_bias: medium
  reasoning_rubric:
    must_mention:
      - breakout
    must_not_mention:
      - risk aversion
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_eval.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from vibe_trading.eval.runner import (
    Rubric, AnalystLabel, TraderLabel, EvalCase,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eval"


def test_eval_case_pydantic_round_trip():
    """EvalCase.model_validate accepts the canonical fixture shape and hydrates all fields."""
    import yaml
    raw = yaml.safe_load((FIXTURES / "valid-case.yaml").read_text())
    case = EvalCase.model_validate(raw)

    assert case.id == "fixture-001-bullish"
    assert case.symbol == "BTC/USDT"
    assert case.timestamp == datetime(2026, 5, 22, 4, 0, 0, tzinfo=timezone.utc)

    assert isinstance(case.analyst_label, AnalystLabel)
    assert case.analyst_label.market_bias == "bullish"
    assert case.analyst_label.volume_confirmation == "confirmed"
    assert case.analyst_label.nearest_support == 30100.0
    assert case.analyst_label.confluence_score == 0.75
    assert isinstance(case.analyst_label.thesis_rubric, Rubric)
    assert "breakout above resistance" in case.analyst_label.thesis_rubric.must_mention
    assert "overbought warning" in case.analyst_label.thesis_rubric.must_not_mention

    assert isinstance(case.trader_label, TraderLabel)
    assert case.trader_label.action == "long"
    assert case.trader_label.risk_reward_ratio == 2.0
    assert "breakout" in case.trader_label.reasoning_rubric.must_mention
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py::test_eval_case_pydantic_round_trip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.eval.runner'`.

- [ ] **Step 4: Write minimal implementation**

Create `src/vibe_trading/eval/runner.py`:

```python
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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
    snapshot_ok: bool
    analyst_output: Optional[dict] = None    # parsed AnalystOutput as dict, or None on failure
    trader_output: Optional[dict] = None     # raw dict from trader.decide(), or None on failure
    analyst_schema_ok: bool = False
    trader_schema_ok: bool = False
    error: Optional[str] = None              # populated only on hard failures
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py::test_eval_case_pydantic_round_trip -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/eval/runner.py tests/test_eval.py tests/fixtures/eval/valid-case.yaml
git commit -m "feat(eval): add pydantic models for golden-set YAML cases"
```

---

### Task 3: Implement load_cases() with malformed-case handling

**Files:**
- Modify: `src/vibe_trading/eval/runner.py`
- Modify: `tests/test_eval.py`
- Create: `tests/fixtures/eval/malformed-case.yaml`

- [ ] **Step 1: Create the malformed fixture**

Create `tests/fixtures/eval/malformed-case.yaml` (missing required `trader_label`):

```yaml
id: fixture-002-malformed
description: Missing trader_label entirely.
symbol: ETH/USDT
timestamp: 2026-05-22T04:00:00Z

analyst_label:
  market_bias: bearish
  volume_confirmation: weak
  nearest_support: 1800.0
  nearest_resistance: 2000.0
  confluence_score: 0.4
  thesis_rubric:
    must_mention: []
    must_not_mention: []
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_eval.py`:

```python
import pytest
from vibe_trading.eval.runner import load_cases


def test_load_cases_loads_all_yaml_in_dir(tmp_path):
    """load_cases globs every *.yaml in the directory and returns parsed EvalCase objects."""
    import shutil
    shutil.copy(FIXTURES / "valid-case.yaml", tmp_path / "001.yaml")
    shutil.copy(FIXTURES / "valid-case.yaml", tmp_path / "002.yaml")

    cases = load_cases(tmp_path)
    assert len(cases) == 2
    assert all(c.id == "fixture-001-bullish" for c in cases)


def test_load_cases_returns_empty_for_empty_dir(tmp_path):
    """An empty directory returns an empty list (no error)."""
    assert load_cases(tmp_path) == []


def test_load_cases_raises_on_malformed_yaml(tmp_path):
    """A YAML that's missing required fields raises a descriptive ValidationError naming the case file."""
    import shutil
    shutil.copy(FIXTURES / "malformed-case.yaml", tmp_path / "bad.yaml")

    with pytest.raises(ValueError) as excinfo:
        load_cases(tmp_path)
    # The error message includes the path of the offending file so the user can find it.
    assert "bad.yaml" in str(excinfo.value)


def test_load_cases_ignores_dotfiles(tmp_path):
    """Files like .gitkeep are not loaded."""
    (tmp_path / ".gitkeep").touch()
    import shutil
    shutil.copy(FIXTURES / "valid-case.yaml", tmp_path / "good.yaml")
    cases = load_cases(tmp_path)
    assert len(cases) == 1
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_load_cases_loads_all_yaml_in_dir -v`
Expected: FAIL with `ImportError: cannot import name 'load_cases'`.

- [ ] **Step 4: Implement load_cases**

Append to `src/vibe_trading/eval/runner.py`:

```python
import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def load_cases(snapshots_dir: Path) -> list[EvalCase]:
    """Load every `*.yaml` file under `snapshots_dir` into a list of validated EvalCase objects.

    Files starting with `.` (e.g. .gitkeep) are skipped.
    Raises ValueError with the offending file path on any malformed YAML or schema violation.
    """
    snapshots_dir = Path(snapshots_dir)
    cases: list[EvalCase] = []

    for yaml_path in sorted(snapshots_dir.glob("*.yaml")):
        if yaml_path.name.startswith("."):
            continue
        try:
            raw = yaml.safe_load(yaml_path.read_text())
            case = EvalCase.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as e:
            raise ValueError(f"Failed to load eval case from {yaml_path}: {e}") from e
        cases.append(case)

    logger.info(f"Loaded {len(cases)} eval cases from {snapshots_dir}")
    return cases
```

- [ ] **Step 5: Run all four tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 5 PASS (round-trip + 4 load_cases tests).

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/eval/runner.py tests/test_eval.py tests/fixtures/eval/malformed-case.yaml
git commit -m "feat(eval): implement load_cases with descriptive error surfacing"
```

---

### Task 4: Implement run_case() with mocked agent test

**Files:**
- Modify: `src/vibe_trading/eval/runner.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from unittest.mock import patch, MagicMock
from vibe_trading.eval.runner import run_case
from vibe_trading.agents.analyst import AnalystOutput


def _load_valid_case():
    import yaml
    from vibe_trading.eval.runner import EvalCase
    raw = yaml.safe_load((FIXTURES / "valid-case.yaml").read_text())
    return EvalCase.model_validate(raw)


@patch("vibe_trading.eval.runner.HeadTrader")
@patch("vibe_trading.eval.runner.TechnicalVolumeAnalyst")
@patch("vibe_trading.eval.runner.FeaturePipeline")
def test_run_case_invokes_agents_with_snapshot_path(mock_pipeline_cls, mock_analyst_cls, mock_trader_cls):
    """run_case: builds snapshot via FeaturePipeline, runs analyst on snapshot path, then trader with LABELED analyst output."""
    case = _load_valid_case()
    mock_db = MagicMock()

    # FeaturePipeline.run returns a snapshot dict
    mock_pipeline_cls.return_value.run.return_value = {"symbol": case.symbol, "rsi_14": 55.0}

    # Analyst returns a parsed AnalystOutput
    actual_analyst = AnalystOutput(
        market_bias="bullish",
        volume_confirmation="confirmed",
        thesis="actual analyst thesis",
        nearest_support=30050.0,
        nearest_resistance=32600.0,
        confluence_score=0.78,
    )
    mock_analyst_cls.return_value.analyze.return_value = actual_analyst

    # Trader returns a hydrated proposal dict
    mock_trader_cls.return_value.decide.return_value = {
        "decision_id": "deadbeef",
        "timestamp": case.timestamp,
        "symbol": case.symbol,
        "action": "long",
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "next_resistance",
        "risk_reward_ratio": 2.0,
        "hold_period_bias": "medium",
        "reasoning_summary": "actual trader reasoning",
    }

    result = run_case(case, mock_db)

    assert result.case_id == case.id
    assert result.snapshot_ok is True
    assert result.analyst_schema_ok is True
    assert result.trader_schema_ok is True
    assert result.analyst_output["market_bias"] == "bullish"
    assert result.trader_output["action"] == "long"
    assert result.error is None

    # Analyst is invoked with the snapshot path (no tool-loop): keyword args symbol= and snapshot=
    analyst_call = mock_analyst_cls.return_value.analyze.call_args
    assert analyst_call.kwargs["symbol"] == case.symbol
    assert analyst_call.kwargs["snapshot"]["symbol"] == case.symbol

    # The trader receives the LABELED analyst output (not the actual one)
    trader_call = mock_trader_cls.return_value.decide.call_args
    trader_args = trader_call.args
    fed_analyst_output = trader_args[1]  # signature: (symbol, analyst_output, scorecard, open_positions)
    assert isinstance(fed_analyst_output, AnalystOutput)
    assert fed_analyst_output.market_bias == case.analyst_label.market_bias  # "bullish"
    assert fed_analyst_output.confluence_score == case.analyst_label.confluence_score  # 0.75 (label, not 0.78)


@patch("vibe_trading.eval.runner.HeadTrader")
@patch("vibe_trading.eval.runner.TechnicalVolumeAnalyst")
@patch("vibe_trading.eval.runner.FeaturePipeline")
def test_run_case_constructs_analyst_without_db_fetcher(mock_pipeline_cls, mock_analyst_cls, mock_trader_cls):
    """The analyst is constructed with no db/fetcher so the tool-loop path is NOT taken."""
    case = _load_valid_case()
    mock_pipeline_cls.return_value.run.return_value = {"symbol": case.symbol}
    mock_analyst_cls.return_value.analyze.return_value = AnalystOutput(
        market_bias="neutral", volume_confirmation="weak", thesis="",
        nearest_support=0.0, nearest_resistance=0.0, confluence_score=0.0,
    )
    mock_trader_cls.return_value.decide.return_value = {
        "decision_id": "x", "timestamp": case.timestamp, "symbol": case.symbol,
        "action": "flat", "stop_loss_strategy": "1.5_atr", "take_profit_strategy": "3.0_atr",
        "risk_reward_ratio": 1.5, "hold_period_bias": "medium", "reasoning_summary": "",
    }

    run_case(case, MagicMock())

    init_kwargs = mock_analyst_cls.call_args.kwargs
    assert init_kwargs.get("db") is None
    assert init_kwargs.get("fetcher") is None


@patch("vibe_trading.eval.runner.HeadTrader")
@patch("vibe_trading.eval.runner.TechnicalVolumeAnalyst")
@patch("vibe_trading.eval.runner.FeaturePipeline")
def test_run_case_handles_empty_snapshot(mock_pipeline_cls, mock_analyst_cls, mock_trader_cls):
    """FeaturePipeline returns {} (insufficient candles) -> snapshot_ok False, agents not invoked."""
    case = _load_valid_case()
    mock_pipeline_cls.return_value.run.return_value = {}

    result = run_case(case, MagicMock())

    assert result.snapshot_ok is False
    assert result.analyst_output is None
    assert result.trader_output is None
    mock_analyst_cls.return_value.analyze.assert_not_called()
    mock_trader_cls.return_value.decide.assert_not_called()


@patch("vibe_trading.eval.runner.HeadTrader")
@patch("vibe_trading.eval.runner.TechnicalVolumeAnalyst")
@patch("vibe_trading.eval.runner.FeaturePipeline")
def test_run_case_handles_analyst_exception(mock_pipeline_cls, mock_analyst_cls, mock_trader_cls):
    """Analyst raises -> error captured on result; trader not invoked."""
    case = _load_valid_case()
    mock_pipeline_cls.return_value.run.return_value = {"symbol": case.symbol}
    mock_analyst_cls.return_value.analyze.side_effect = ValueError("simulated analyst parse failure")

    result = run_case(case, MagicMock())

    assert result.snapshot_ok is True
    assert result.analyst_schema_ok is False
    assert result.analyst_output is None
    assert "simulated analyst parse failure" in result.error
    mock_trader_cls.return_value.decide.assert_not_called()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_run_case_invokes_agents_with_snapshot_path -v`
Expected: FAIL with `ImportError: cannot import name 'run_case'`.

- [ ] **Step 3: Implement run_case**

Append to `src/vibe_trading/eval/runner.py`:

```python
from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput
from vibe_trading.agents.trader import HeadTrader
from vibe_trading.data.db import Database
from vibe_trading.features.pipeline import FeaturePipeline


# Fixed inputs the trader sees in every eval case — keeps trader scoring independent of context.
FIXED_SCORECARD = {"accuracy": 0.55, "total_decisions": 100}
FIXED_OPEN_POSITIONS: list = []


def run_case(case: EvalCase, db: Database) -> CaseResult:
    """Run one eval case: build snapshot, run analyst (snapshot path), run trader with LABELED analyst output.

    The trader is intentionally fed `case.analyst_label` instead of the actual analyst output so
    trader scoring isolates trader-specific regressions from analyst-specific regressions.

    All exceptions are captured onto the result; the function never raises.
    """
    pipeline = FeaturePipeline(db)
    try:
        snapshot = pipeline.run(case.symbol, case.timestamp)
    except Exception as e:
        logger.warning(f"FeaturePipeline.run failed for {case.id}: {e}")
        return CaseResult(case_id=case.id, snapshot_ok=False, error=f"snapshot build failed: {e}")

    if not snapshot:
        logger.warning(f"FeaturePipeline.run returned empty snapshot for {case.id} ({case.symbol} @ {case.timestamp})")
        return CaseResult(case_id=case.id, snapshot_ok=False, error="empty snapshot — insufficient candle history?")

    # Analyst — explicit no db/fetcher so the tool-loop path is NOT taken
    analyst = TechnicalVolumeAnalyst(db=None, fetcher=None)
    try:
        actual_analyst_output = analyst.analyze(symbol=case.symbol, snapshot=snapshot)
    except Exception as e:
        logger.warning(f"Analyst failed for {case.id}: {e}")
        return CaseResult(
            case_id=case.id, snapshot_ok=True, analyst_schema_ok=False, error=f"analyst failed: {e}",
        )

    # Convert pydantic to dict for storage; keep parsed object only for the trader stage below
    analyst_output_dict = actual_analyst_output.model_dump()

    # Trader — fed the LABELED analyst output, not the actual one
    labeled_analyst_output = AnalystOutput(
        market_bias=case.analyst_label.market_bias,
        volume_confirmation=case.analyst_label.volume_confirmation,
        thesis="(label proxy — see rubric)",
        nearest_support=case.analyst_label.nearest_support,
        nearest_resistance=case.analyst_label.nearest_resistance,
        confluence_score=case.analyst_label.confluence_score,
    )

    trader = HeadTrader()
    try:
        trader_output = trader.decide(case.symbol, labeled_analyst_output, FIXED_SCORECARD, FIXED_OPEN_POSITIONS)
    except Exception as e:
        logger.warning(f"Trader failed for {case.id}: {e}")
        return CaseResult(
            case_id=case.id, snapshot_ok=True, analyst_schema_ok=True, analyst_output=analyst_output_dict,
            trader_schema_ok=False, error=f"trader failed: {e}",
        )

    return CaseResult(
        case_id=case.id, snapshot_ok=True,
        analyst_schema_ok=True, analyst_output=analyst_output_dict,
        trader_schema_ok=True, trader_output=trader_output,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 9 PASS (5 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/runner.py tests/test_eval.py
git commit -m "feat(eval): implement run_case driving analyst snapshot path + trader with labeled input"
```

---

### Task 5: Deterministic scorers (categorical + numeric tolerance)

**Files:**
- Create: `src/vibe_trading/eval/scorer.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.scorer import (
    FieldScore, score_categorical, score_numeric_tolerance, NUMERIC_TOLERANCES,
)


def test_score_categorical_match():
    fs = score_categorical("market_bias", "bullish", "bullish")
    assert fs.field == "market_bias"
    assert fs.passed is True
    assert fs.score == 1.0


def test_score_categorical_mismatch():
    fs = score_categorical("market_bias", "bearish", "bullish")
    assert fs.passed is False
    assert fs.score == 0.0
    assert "bearish" in fs.note and "bullish" in fs.note


def test_score_numeric_within_ok_threshold_pct():
    """nearest_support uses percentage tolerance; within 2% of expected scores 1.0."""
    fs = score_numeric_tolerance("nearest_support", actual=30050.0, expected=30100.0)
    assert fs.score == 1.0
    assert fs.passed is True


def test_score_numeric_linear_degradation_pct():
    """nearest_support 3.5% off (between 2% ok and 5% zero) -> linearly degraded to ~0.5."""
    # 3.5% above expected 30100.0 -> 31153.5; delta = 3.5%, ok=2%, zero=5%
    # score = 1.0 - (0.035 - 0.02) / (0.05 - 0.02) = 1.0 - 0.5 = 0.5
    fs = score_numeric_tolerance("nearest_support", actual=31153.5, expected=30100.0)
    assert 0.45 < fs.score < 0.55
    assert fs.passed is False  # passed is True only when score == 1.0


def test_score_numeric_outside_zero_threshold_pct():
    """nearest_support 8% off scores 0.0."""
    fs = score_numeric_tolerance("nearest_support", actual=32508.0, expected=30100.0)
    assert fs.score == 0.0
    assert fs.passed is False


def test_score_numeric_absolute_distance_for_confluence():
    """confluence_score uses absolute tolerance (±0.15 ok, ±0.30 zero)."""
    fs = score_numeric_tolerance("confluence_score", actual=0.7, expected=0.75)  # delta 0.05 -> within ok
    assert fs.score == 1.0
    fs2 = score_numeric_tolerance("confluence_score", actual=0.4, expected=0.75)  # delta 0.35 -> beyond zero
    assert fs2.score == 0.0


def test_score_numeric_expected_zero_uses_absolute_distance():
    """When expected == 0.0, fall back to absolute distance using ok/zero thresholds as absolute."""
    fs = score_numeric_tolerance("nearest_support", actual=0.01, expected=0.0)
    # NUMERIC_TOLERANCES['nearest_support'] = {'ok': 0.02, 'zero': 0.05} treated as absolute
    # delta = 0.01, within ok=0.02 -> score 1.0
    assert fs.score == 1.0


def test_score_numeric_handles_none_actual():
    fs = score_numeric_tolerance("nearest_support", actual=None, expected=30000.0)
    assert fs.score == 0.0
    assert "missing value" in fs.note


def test_score_numeric_handles_nan_actual():
    import math
    fs = score_numeric_tolerance("nearest_support", actual=math.nan, expected=30000.0)
    assert fs.score == 0.0
    assert "missing value" in fs.note


def test_numeric_tolerances_has_all_required_fields():
    """Smoke check: every numeric field used in the rules table has a tolerance entry."""
    assert set(NUMERIC_TOLERANCES.keys()) == {
        "nearest_support", "nearest_resistance", "confluence_score", "risk_reward_ratio",
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_score_categorical_match -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.eval.scorer'`.

- [ ] **Step 3: Implement scorer.py — deterministic scorers**

Create `src/vibe_trading/eval/scorer.py`:

```python
import math
from typing import Any, Optional

from pydantic import BaseModel


# Tolerance constants. `ok` -> score 1.0; `zero` -> score 0.0; linear in between.
# For nearest_support / nearest_resistance, thresholds are PERCENTAGES of expected.
# For confluence_score / risk_reward_ratio, thresholds are ABSOLUTE distances.
# When expected == 0.0, percentage thresholds collapse to absolute distance.
NUMERIC_TOLERANCES: dict[str, dict[str, float]] = {
    "nearest_support":    {"ok": 0.02, "zero": 0.05},
    "nearest_resistance": {"ok": 0.02, "zero": 0.05},
    "confluence_score":   {"ok": 0.15, "zero": 0.30},
    "risk_reward_ratio":  {"ok": 0.5,  "zero": 1.5},
}

PERCENTAGE_FIELDS = {"nearest_support", "nearest_resistance"}


class FieldScore(BaseModel):
    field: str
    passed: bool
    score: float
    note: str = ""


def score_categorical(field: str, actual: Any, expected: Any) -> FieldScore:
    """Exact-match scoring for enum-typed fields."""
    if actual == expected:
        return FieldScore(field=field, passed=True, score=1.0)
    return FieldScore(
        field=field,
        passed=False,
        score=0.0,
        note=f"expected '{expected}', got '{actual}'",
    )


def score_numeric_tolerance(field: str, actual: Optional[float], expected: float) -> FieldScore:
    """Tolerance-based numeric scoring with linear degradation.

    See NUMERIC_TOLERANCES + PERCENTAGE_FIELDS for the per-field thresholds.
    """
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return FieldScore(field=field, passed=False, score=0.0, note="missing value")

    thresholds = NUMERIC_TOLERANCES[field]
    ok = thresholds["ok"]
    zero = thresholds["zero"]

    if field in PERCENTAGE_FIELDS and expected != 0.0:
        delta = abs(actual - expected) / abs(expected)
    else:
        delta = abs(actual - expected)

    if delta <= ok:
        return FieldScore(field=field, passed=True, score=1.0)
    if delta >= zero:
        return FieldScore(field=field, passed=False, score=0.0,
                          note=f"delta {delta:.4f} beyond zero threshold {zero}")

    score = 1.0 - (delta - ok) / (zero - ok)
    return FieldScore(field=field, passed=False, score=score,
                      note=f"delta {delta:.4f} between thresholds [{ok}, {zero}]")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 19 PASS (9 prior + 10 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/scorer.py tests/test_eval.py
git commit -m "feat(eval): add deterministic categorical and numeric-tolerance scorers"
```

---

### Task 6: LLM judge and rubric scorer

**Files:**
- Modify: `src/vibe_trading/eval/scorer.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.scorer import (
    JudgeOutput, CriterionEvaluation, score_rubric, build_judge,
)
from vibe_trading.eval.runner import Rubric


def test_score_rubric_all_passed_returns_one():
    """All criteria pass -> score 1.0."""
    rubric = Rubric(must_mention=["breakout", "volume"], must_not_mention=["overbought"])

    def stub_judge(text: str, r: Rubric) -> JudgeOutput:
        return JudgeOutput(
            must_mention_results=[
                CriterionEvaluation(criterion="breakout", passed=True, justification="present"),
                CriterionEvaluation(criterion="volume", passed=True, justification="present"),
            ],
            must_not_mention_results=[
                CriterionEvaluation(criterion="overbought", passed=True, justification="absent"),
            ],
        )

    fs = score_rubric("thesis", "actual text", rubric, stub_judge)
    assert fs.field == "thesis"
    assert fs.score == 1.0
    assert fs.passed is True


def test_score_rubric_partial_pass():
    """2/3 criteria pass -> score 2/3."""
    rubric = Rubric(must_mention=["breakout", "volume"], must_not_mention=["overbought"])

    def stub_judge(text: str, r: Rubric) -> JudgeOutput:
        return JudgeOutput(
            must_mention_results=[
                CriterionEvaluation(criterion="breakout", passed=True, justification=""),
                CriterionEvaluation(criterion="volume", passed=False, justification="missing"),
            ],
            must_not_mention_results=[
                CriterionEvaluation(criterion="overbought", passed=True, justification=""),
            ],
        )

    fs = score_rubric("thesis", "actual text", rubric, stub_judge)
    assert abs(fs.score - (2 / 3)) < 1e-9
    assert fs.passed is False


def test_score_rubric_empty_rubric_vacuously_passes():
    """A rubric with no criteria scores 1.0 (vacuous truth)."""
    rubric = Rubric(must_mention=[], must_not_mention=[])

    def stub_judge(text: str, r: Rubric) -> JudgeOutput:
        return JudgeOutput(must_mention_results=[], must_not_mention_results=[])

    fs = score_rubric("thesis", "any text", rubric, stub_judge)
    assert fs.score == 1.0
    assert fs.passed is True


def test_score_rubric_judge_failure_returns_neutral():
    """If the judge raises, return score 0.5 with judge_error note (doesn't crash the run)."""
    rubric = Rubric(must_mention=["breakout"], must_not_mention=[])

    def bad_judge(text: str, r: Rubric) -> JudgeOutput:
        raise RuntimeError("simulated judge timeout")

    fs = score_rubric("thesis", "any text", rubric, bad_judge)
    assert fs.score == 0.5
    assert fs.passed is False
    assert "judge_error" in fs.note
    assert "simulated judge timeout" in fs.note


@patch("vibe_trading.eval.scorer.LLMClient")
@patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}, clear=False)
def test_build_judge_uses_default_model_when_env_unset(mock_client_cls):
    """build_judge() returns a callable that defaults to gemini-3.1-flash-lite when EVAL_JUDGE_MODEL is unset."""
    import os
    os.environ.pop("EVAL_JUDGE_MODEL", None)

    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.call_llm.return_value = (
        '{"must_mention_results": [{"criterion": "breakout", "passed": true, "justification": "ok"}],'
        ' "must_not_mention_results": []}'
    )
    mock_client_cls.return_value = mock_client

    judge = build_judge()
    rubric = Rubric(must_mention=["breakout"], must_not_mention=[])
    out = judge("some text", rubric)

    assert isinstance(out, JudgeOutput)
    assert out.must_mention_results[0].passed is True
    # Default model name reached LLMClient
    call_kwargs = mock_client.call_llm.call_args.kwargs
    assert call_kwargs["model_name"] == "gemini-3.1-flash-lite"


@patch("vibe_trading.eval.scorer.LLMClient")
@patch.dict("os.environ", {"GEMINI_API_KEY": "test_key", "EVAL_JUDGE_MODEL": "claude-3-5-sonnet-20241022"}, clear=False)
def test_build_judge_honors_env_override(mock_client_cls):
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.call_llm.return_value = '{"must_mention_results": [], "must_not_mention_results": []}'
    mock_client_cls.return_value = mock_client

    judge = build_judge()
    judge("text", Rubric())

    assert mock_client.call_llm.call_args.kwargs["model_name"] == "claude-3-5-sonnet-20241022"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_score_rubric_all_passed_returns_one -v`
Expected: FAIL with `ImportError: cannot import name 'JudgeOutput'`.

- [ ] **Step 3: Implement judge + rubric scorer**

Append to `src/vibe_trading/eval/scorer.py`:

```python
import os
from typing import Callable

from vibe_trading.agents.client import LLMClient
from vibe_trading.eval.runner import Rubric


class CriterionEvaluation(BaseModel):
    criterion: str
    passed: bool
    justification: str


class JudgeOutput(BaseModel):
    must_mention_results: list[CriterionEvaluation]
    must_not_mention_results: list[CriterionEvaluation]


_JUDGE_SYSTEM = """
You are a meticulous, code-review-style evaluator. You will receive a piece of agent-generated
text and a rubric of must-mention and must-not-mention criteria.

For each must-mention criterion: mark passed=true only if the criterion is clearly present in the
text (not just hinted at). Otherwise passed=false.

For each must-not-mention criterion: mark passed=true if the criterion is clearly absent from the
text. If the text clearly violates it, passed=false.

Output strictly matches the JudgeOutput JSON schema. Provide a one-sentence justification per
criterion.
""".strip()

_DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"


def build_judge() -> Callable[[str, Rubric], JudgeOutput]:
    """Returns a closure that, given (actual_text, rubric), calls an LLM judge and parses its output.

    Judge model is resolved per call from the EVAL_JUDGE_MODEL env var, defaulting to gemini-3.1-flash-lite.
    """
    client = LLMClient()

    def judge(actual_text: str, rubric: Rubric) -> JudgeOutput:
        model_name = os.getenv("EVAL_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
        prompt = "TEXT:\n" + actual_text + "\n\nRUBRIC:\nMUST MENTION:\n"
        prompt += "\n".join(f"- {c}" for c in rubric.must_mention) if rubric.must_mention else "(none)"
        prompt += "\n\nMUST NOT MENTION:\n"
        prompt += "\n".join(f"- {c}" for c in rubric.must_not_mention) if rubric.must_not_mention else "(none)"

        raw = client.call_llm(
            model_name=model_name,
            system_instruction=_JUDGE_SYSTEM,
            prompt=prompt,
            response_schema=JudgeOutput,
        )
        return JudgeOutput.model_validate_json(raw)

    return judge


def score_rubric(
    field: str,
    actual_text: str,
    rubric: Rubric,
    judge: Callable[[str, Rubric], JudgeOutput],
) -> FieldScore:
    """Score free-text against a rubric using the injected LLM judge.

    Score = passed_criteria / total_criteria across both must_mention and must_not_mention lists.
    Empty rubric -> score 1.0 (vacuous truth).
    Judge exception -> score 0.5 with judge_error note (suite continues).
    """
    total = len(rubric.must_mention) + len(rubric.must_not_mention)
    if total == 0:
        return FieldScore(field=field, passed=True, score=1.0, note="empty rubric")

    try:
        result = judge(actual_text, rubric)
    except Exception as e:
        return FieldScore(field=field, passed=False, score=0.5, note=f"judge_error: {e}")

    passed_count = (
        sum(1 for r in result.must_mention_results if r.passed)
        + sum(1 for r in result.must_not_mention_results if r.passed)
    )
    score = passed_count / total
    return FieldScore(field=field, passed=(score == 1.0), score=score,
                      note=f"{passed_count}/{total} criteria passed")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 25 PASS (19 prior + 6 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/scorer.py tests/test_eval.py
git commit -m "feat(eval): add LLM judge with env-configurable model and rubric scorer"
```

---

### Task 7: score_case() aggregator

**Files:**
- Modify: `src/vibe_trading/eval/scorer.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.scorer import CaseScore, score_case


def _passing_judge_stub(text: str, r: Rubric) -> JudgeOutput:
    return JudgeOutput(
        must_mention_results=[
            CriterionEvaluation(criterion=c, passed=True, justification="ok") for c in r.must_mention
        ],
        must_not_mention_results=[
            CriterionEvaluation(criterion=c, passed=True, justification="ok") for c in r.must_not_mention
        ],
    )


def _build_perfect_result(case):
    """Build a CaseResult whose outputs exactly match the labels (perfect score)."""
    from vibe_trading.eval.runner import CaseResult
    return CaseResult(
        case_id=case.id,
        snapshot_ok=True,
        analyst_schema_ok=True,
        analyst_output={
            "market_bias": case.analyst_label.market_bias,
            "volume_confirmation": case.analyst_label.volume_confirmation,
            "thesis": "perfect thesis (judge will pass via stub)",
            "nearest_support": case.analyst_label.nearest_support,
            "nearest_resistance": case.analyst_label.nearest_resistance,
            "confluence_score": case.analyst_label.confluence_score,
        },
        trader_schema_ok=True,
        trader_output={
            "action": case.trader_label.action,
            "stop_loss_strategy": case.trader_label.stop_loss_strategy,
            "take_profit_strategy": case.trader_label.take_profit_strategy,
            "risk_reward_ratio": case.trader_label.risk_reward_ratio,
            "hold_period_bias": case.trader_label.hold_period_bias,
            "reasoning_summary": "perfect reasoning",
        },
    )


def test_score_case_perfect_match_scores_one():
    case = _load_valid_case()
    result = _build_perfect_result(case)
    cs = score_case(result, case, _passing_judge_stub)

    assert cs.case_id == case.id
    assert cs.schema_ok is True
    assert cs.total_score == 1.0
    assert cs.analyst_score == 1.0
    assert cs.trader_score == 1.0
    assert all(fs.passed for fs in cs.field_scores)


def test_score_case_schema_failure_short_circuits():
    """analyst_schema_ok=False -> total_score=0.0, no field scoring attempted."""
    from vibe_trading.eval.runner import CaseResult
    case = _load_valid_case()
    result = CaseResult(
        case_id=case.id, snapshot_ok=True,
        analyst_schema_ok=False, analyst_output=None,
        trader_schema_ok=False, trader_output=None,
        error="analyst parse failed",
    )
    cs = score_case(result, case, _passing_judge_stub)
    assert cs.schema_ok is False
    assert cs.total_score == 0.0
    assert cs.analyst_score == 0.0
    assert cs.trader_score == 0.0
    assert cs.error == "analyst parse failed"


def test_score_case_one_mismatch_reduces_total():
    case = _load_valid_case()
    result = _build_perfect_result(case)
    # Flip market_bias from label "bullish" to "bearish" -> one field scores 0
    result.analyst_output["market_bias"] = "bearish"

    cs = score_case(result, case, _passing_judge_stub)
    # 11 scored fields total (6 analyst incl. thesis, 5 trader incl. reasoning). 1 fails -> 10/11.
    assert cs.total_score < 1.0
    assert cs.analyst_score < 1.0
    assert cs.trader_score == 1.0


def test_score_case_snapshot_failure_propagates():
    """snapshot_ok=False -> total_score=0.0, schema_ok=False, error preserved."""
    from vibe_trading.eval.runner import CaseResult
    case = _load_valid_case()
    result = CaseResult(case_id=case.id, snapshot_ok=False, error="empty snapshot")
    cs = score_case(result, case, _passing_judge_stub)
    assert cs.schema_ok is False
    assert cs.total_score == 0.0
    assert cs.error == "empty snapshot"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_score_case_perfect_match_scores_one -v`
Expected: FAIL with `ImportError: cannot import name 'CaseScore'`.

- [ ] **Step 3: Implement score_case**

Append to `src/vibe_trading/eval/scorer.py`:

```python
from vibe_trading.eval.runner import CaseResult, EvalCase


class CaseScore(BaseModel):
    case_id: str
    schema_ok: bool
    field_scores: list[FieldScore]
    analyst_score: float
    trader_score: float
    total_score: float
    error: Optional[str] = None


ANALYST_FIELDS = {"market_bias", "volume_confirmation", "nearest_support",
                  "nearest_resistance", "confluence_score", "thesis"}
TRADER_FIELDS = {"action", "stop_loss_strategy", "take_profit_strategy",
                 "risk_reward_ratio", "hold_period_bias", "reasoning_summary"}


def score_case(
    result: CaseResult,
    case: EvalCase,
    judge: Callable[[str, Rubric], JudgeOutput],
) -> CaseScore:
    """Aggregate per-field scores into a case-level CaseScore.

    Short-circuits to total_score=0.0 if snapshot or schema validation failed for either agent.
    """
    if not result.snapshot_ok or not result.analyst_schema_ok or not result.trader_schema_ok:
        return CaseScore(
            case_id=result.case_id, schema_ok=False, field_scores=[],
            analyst_score=0.0, trader_score=0.0, total_score=0.0,
            error=result.error,
        )

    field_scores: list[FieldScore] = [
        # Analyst — categorical
        score_categorical("market_bias", result.analyst_output["market_bias"], case.analyst_label.market_bias),
        score_categorical("volume_confirmation", result.analyst_output["volume_confirmation"], case.analyst_label.volume_confirmation),
        # Analyst — numeric
        score_numeric_tolerance("nearest_support", result.analyst_output["nearest_support"], case.analyst_label.nearest_support),
        score_numeric_tolerance("nearest_resistance", result.analyst_output["nearest_resistance"], case.analyst_label.nearest_resistance),
        score_numeric_tolerance("confluence_score", result.analyst_output["confluence_score"], case.analyst_label.confluence_score),
        # Analyst — free text
        score_rubric("thesis", result.analyst_output.get("thesis", ""), case.analyst_label.thesis_rubric, judge),
        # Trader — categorical
        score_categorical("action", result.trader_output["action"], case.trader_label.action),
        score_categorical("stop_loss_strategy", result.trader_output["stop_loss_strategy"], case.trader_label.stop_loss_strategy),
        score_categorical("take_profit_strategy", result.trader_output["take_profit_strategy"], case.trader_label.take_profit_strategy),
        score_categorical("hold_period_bias", result.trader_output["hold_period_bias"], case.trader_label.hold_period_bias),
        # Trader — numeric
        score_numeric_tolerance("risk_reward_ratio", float(result.trader_output["risk_reward_ratio"]), case.trader_label.risk_reward_ratio),
        # Trader — free text
        score_rubric("reasoning_summary", result.trader_output.get("reasoning_summary", ""), case.trader_label.reasoning_rubric, judge),
    ]

    analyst_field_scores = [fs for fs in field_scores if fs.field in ANALYST_FIELDS]
    trader_field_scores = [fs for fs in field_scores if fs.field in TRADER_FIELDS]

    analyst_score = sum(fs.score for fs in analyst_field_scores) / len(analyst_field_scores)
    trader_score = sum(fs.score for fs in trader_field_scores) / len(trader_field_scores)
    total = sum(fs.score for fs in field_scores) / len(field_scores)

    return CaseScore(
        case_id=result.case_id, schema_ok=True, field_scores=field_scores,
        analyst_score=analyst_score, trader_score=trader_score, total_score=total,
        error=None,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 29 PASS (25 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/scorer.py tests/test_eval.py
git commit -m "feat(eval): aggregate per-field scores into CaseScore with schema short-circuit"
```

---

### Task 8: SuiteReport + write_report()

**Files:**
- Create: `src/vibe_trading/eval/report.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
import json as json_mod
from vibe_trading.eval.report import SuiteReport, write_report
from vibe_trading.eval.scorer import CaseScore, FieldScore


def _make_score(case_id: str, total: float, analyst: float = None, trader: float = None,
                schema_ok: bool = True, error: str = None) -> CaseScore:
    return CaseScore(
        case_id=case_id, schema_ok=schema_ok, field_scores=[],
        analyst_score=analyst if analyst is not None else total,
        trader_score=trader if trader is not None else total,
        total_score=total, error=error,
    )


def test_suite_report_from_scores_aggregates_means():
    scores = [_make_score("a", 1.0), _make_score("b", 0.5), _make_score("c", 0.8)]
    report = SuiteReport.from_scores(scores)

    assert report.case_count == 3
    assert abs(report.overall_score - (1.0 + 0.5 + 0.8) / 3) < 1e-9
    assert "a" in report.per_case
    assert report.per_case["b"].total_score == 0.5


def test_suite_report_counts_schema_failures():
    scores = [_make_score("a", 1.0), _make_score("b", 0.0, schema_ok=False, error="parse")]
    report = SuiteReport.from_scores(scores)
    assert report.schema_failures == 1


def test_suite_report_pass_rate_only_counts_fully_passing_cases():
    """pass_rate = fraction of cases where every field passed."""
    cs_full_pass = CaseScore(
        case_id="a", schema_ok=True,
        field_scores=[FieldScore(field="market_bias", passed=True, score=1.0)],
        analyst_score=1.0, trader_score=1.0, total_score=1.0,
    )
    cs_partial = CaseScore(
        case_id="b", schema_ok=True,
        field_scores=[
            FieldScore(field="market_bias", passed=True, score=1.0),
            FieldScore(field="action", passed=False, score=0.0),
        ],
        analyst_score=1.0, trader_score=0.0, total_score=0.5,
    )
    report = SuiteReport.from_scores([cs_full_pass, cs_partial])
    assert report.pass_rate == 0.5


def test_write_report_writes_json_file(tmp_path):
    scores = [_make_score("only-case", 0.85)]
    report = SuiteReport.from_scores(scores)
    path = write_report(report, tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("eval-")
    assert path.suffix == ".json"
    data = json_mod.loads(path.read_text())
    assert data["case_count"] == 1
    assert data["per_case"]["only-case"]["total_score"] == 0.85
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_suite_report_from_scores_aggregates_means -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.eval.report'`.

- [ ] **Step 3: Implement report.py — SuiteReport + write_report**

Create `src/vibe_trading/eval/report.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from vibe_trading.eval.scorer import CaseScore


class SuiteReport(BaseModel):
    run_id: str
    run_at: datetime
    case_count: int
    overall_score: float
    analyst_score: float
    trader_score: float
    pass_rate: float
    schema_failures: int
    judge_errors: int
    per_case: dict[str, CaseScore]

    @classmethod
    def from_scores(cls, scores: list[CaseScore]) -> "SuiteReport":
        if not scores:
            run_at = datetime.now(timezone.utc)
            return cls(
                run_id=cls._run_id(run_at), run_at=run_at,
                case_count=0, overall_score=0.0, analyst_score=0.0, trader_score=0.0,
                pass_rate=0.0, schema_failures=0, judge_errors=0, per_case={},
            )

        run_at = datetime.now(timezone.utc)
        overall = sum(s.total_score for s in scores) / len(scores)
        analyst = sum(s.analyst_score for s in scores) / len(scores)
        trader = sum(s.trader_score for s in scores) / len(scores)
        schema_failures = sum(1 for s in scores if not s.schema_ok)
        pass_rate = sum(1 for s in scores if all(fs.passed for fs in s.field_scores)) / len(scores)
        judge_errors = sum(
            1 for s in scores
            for fs in s.field_scores
            if "judge_error" in fs.note
        )

        return cls(
            run_id=cls._run_id(run_at), run_at=run_at,
            case_count=len(scores), overall_score=overall,
            analyst_score=analyst, trader_score=trader, pass_rate=pass_rate,
            schema_failures=schema_failures, judge_errors=judge_errors,
            per_case={s.case_id: s for s in scores},
        )

    @staticmethod
    def _run_id(ts: datetime) -> str:
        return "eval-" + ts.strftime("%Y-%m-%dT%H-%M-%SZ")


def write_report(report: SuiteReport, reports_dir: Path) -> Path:
    """Write a timestamped JSON report under reports_dir. Returns the path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / (report.run_id + ".json")
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    return path
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 33 PASS (29 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/report.py tests/test_eval.py
git commit -m "feat(eval): aggregate CaseScores into SuiteReport and write timestamped JSON"
```

---

### Task 9: Baseline load + diff_against_baseline + write_baseline

**Files:**
- Modify: `src/vibe_trading/eval/report.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.report import (
    DiffResult, REGRESSION_THRESHOLDS, load_baseline, write_baseline, diff_against_baseline,
)


def test_load_baseline_returns_none_when_missing(tmp_path):
    assert load_baseline(tmp_path / "no-such-file.json") is None


def test_load_baseline_returns_none_when_malformed(tmp_path):
    p = tmp_path / "baseline.json"
    p.write_text("not valid json {")
    assert load_baseline(p) is None


def test_write_baseline_round_trips(tmp_path):
    scores = [_make_score("a", 0.85), _make_score("b", 0.7)]
    report = SuiteReport.from_scores(scores)
    p = tmp_path / "baseline.json"
    write_baseline(report, p)

    loaded = load_baseline(p)
    assert loaded is not None
    assert loaded["overall_score"] == report.overall_score
    assert loaded["per_case"]["a"]["total_score"] == 0.85


def test_diff_against_baseline_detects_overall_regression():
    baseline = {
        "overall_score": 0.85,
        "per_case": {"a": {"total_score": 0.9, "schema_ok": True}},
    }
    new = SuiteReport.from_scores([_make_score("a", 0.6)])  # overall 0.6 vs baseline 0.85 -> regression
    diff = diff_against_baseline(new, baseline)
    assert diff.is_regression is True
    assert diff.overall_delta < 0


def test_diff_against_baseline_detects_per_case_regression():
    baseline = {
        "overall_score": 0.85,
        "per_case": {
            "a": {"total_score": 0.9, "schema_ok": True},
            "b": {"total_score": 0.8, "schema_ok": True},
        },
    }
    # Overall barely moves but case 'a' drops by 0.10 (> 0.05 threshold) -> regression
    new = SuiteReport.from_scores([_make_score("a", 0.80), _make_score("b", 0.90)])
    diff = diff_against_baseline(new, baseline)
    assert "a" in diff.per_case_regressions
    assert diff.is_regression is True


def test_diff_against_baseline_flags_new_schema_failure():
    baseline = {
        "overall_score": 1.0,
        "per_case": {"a": {"total_score": 1.0, "schema_ok": True}},
    }
    new = SuiteReport.from_scores([_make_score("a", 0.0, schema_ok=False, error="parse")])
    diff = diff_against_baseline(new, baseline)
    assert "a" in diff.new_schema_failures
    assert diff.is_regression is True


def test_diff_against_baseline_reports_improvements_without_regression():
    baseline = {
        "overall_score": 0.7,
        "per_case": {"a": {"total_score": 0.6, "schema_ok": True}},
    }
    new = SuiteReport.from_scores([_make_score("a", 0.9)])
    diff = diff_against_baseline(new, baseline)
    assert diff.is_regression is False
    assert "a" in diff.per_case_improvements
    assert diff.overall_delta > 0


def test_regression_thresholds_exist():
    """Smoke check the constants live where the spec said they would."""
    assert REGRESSION_THRESHOLDS["overall"] == 0.02
    assert REGRESSION_THRESHOLDS["per_case"] == 0.05
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_load_baseline_returns_none_when_missing -v`
Expected: FAIL with `ImportError: cannot import name 'load_baseline'`.

- [ ] **Step 3: Implement baseline operations + diff**

Append to `src/vibe_trading/eval/report.py`:

```python
REGRESSION_THRESHOLDS = {
    "overall": 0.02,
    "per_case": 0.05,
}


class DiffResult(BaseModel):
    overall_delta: float
    per_case_regressions: list[str]
    per_case_improvements: list[str]
    new_schema_failures: list[str]
    is_regression: bool


def load_baseline(baseline_path: Path) -> Optional[dict]:
    """Load the baseline JSON. Returns None if file is missing or malformed."""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_baseline(report: SuiteReport, baseline_path: Path) -> None:
    """Overwrite baseline.json with the current report's scores."""
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "baseline_committed_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": report.run_id,
        "overall_score": report.overall_score,
        "analyst_score": report.analyst_score,
        "trader_score": report.trader_score,
        "pass_rate": report.pass_rate,
        "schema_failures": report.schema_failures,
        "per_case": {
            cid: {
                "total_score": cs.total_score,
                "analyst_score": cs.analyst_score,
                "trader_score": cs.trader_score,
                "schema_ok": cs.schema_ok,
            }
            for cid, cs in report.per_case.items()
        },
    }
    baseline_path.write_text(json.dumps(snapshot, indent=2, default=str))


def diff_against_baseline(report: SuiteReport, baseline: dict) -> DiffResult:
    """Compute regression status against a loaded baseline dict."""
    overall_delta = report.overall_score - baseline.get("overall_score", 0.0)

    per_case_regressions: list[str] = []
    per_case_improvements: list[str] = []
    new_schema_failures: list[str] = []

    baseline_cases = baseline.get("per_case", {})
    for case_id, current in report.per_case.items():
        prior = baseline_cases.get(case_id)
        if prior is None:
            continue  # new case not in baseline -> not graded against history

        delta = current.total_score - prior.get("total_score", 0.0)
        if delta < -REGRESSION_THRESHOLDS["per_case"]:
            per_case_regressions.append(case_id)
        elif delta > REGRESSION_THRESHOLDS["per_case"]:
            per_case_improvements.append(case_id)

        # Schema regression: case parsed in baseline but failed now
        if prior.get("schema_ok") is True and current.schema_ok is False:
            new_schema_failures.append(case_id)

    is_regression = (
        overall_delta < -REGRESSION_THRESHOLDS["overall"]
        or len(per_case_regressions) > 0
        or len(new_schema_failures) > 0
    )

    return DiffResult(
        overall_delta=overall_delta,
        per_case_regressions=per_case_regressions,
        per_case_improvements=per_case_improvements,
        new_schema_failures=new_schema_failures,
        is_regression=is_regression,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 41 PASS (33 prior + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/report.py tests/test_eval.py
git commit -m "feat(eval): baseline load/write and diff_against_baseline with regression thresholds"
```

---

### Task 10: print_summary terminal output

**Files:**
- Modify: `src/vibe_trading/eval/report.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.report import print_summary


def test_print_summary_includes_key_metrics(capsys):
    report = SuiteReport.from_scores([_make_score("only", 0.85, analyst=0.83, trader=0.87)])
    print_summary(report, diff=None)
    out = capsys.readouterr().out

    assert "Overall score" in out
    assert "0.85" in out
    assert "Analyst" in out and "0.83" in out
    assert "Trader" in out and "0.87" in out


def test_print_summary_lists_regressions_and_improvements(capsys):
    report = SuiteReport.from_scores([_make_score("good", 0.9), _make_score("bad", 0.5)])
    diff = DiffResult(
        overall_delta=-0.05,
        per_case_regressions=["bad"],
        per_case_improvements=["good"],
        new_schema_failures=[],
        is_regression=True,
    )
    print_summary(report, diff)
    out = capsys.readouterr().out

    assert "Per-case regressions" in out
    assert "bad" in out
    assert "Per-case improvements" in out
    assert "good" in out


def test_print_summary_handles_no_baseline_gracefully(capsys):
    report = SuiteReport.from_scores([_make_score("a", 0.7)])
    print_summary(report, diff=None)
    out = capsys.readouterr().out
    # Should not mention baseline comparisons
    assert "baseline" not in out.lower() or "no baseline" in out.lower()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_print_summary_includes_key_metrics -v`
Expected: FAIL with `ImportError: cannot import name 'print_summary'`.

- [ ] **Step 3: Implement print_summary**

Append to `src/vibe_trading/eval/report.py`:

```python
def print_summary(report: SuiteReport, diff: Optional[DiffResult]) -> None:
    """Print a human-readable summary to stdout."""
    print(f"Eval run: {report.run_id}   ({report.case_count} cases)")
    print("─" * 60)
    if diff is None:
        print(f"Overall score:    {report.overall_score:.2f}   (no baseline)")
        print(f"  Analyst:        {report.analyst_score:.2f}")
        print(f"  Trader:         {report.trader_score:.2f}")
    else:
        arrow = "▲" if diff.overall_delta > 0 else ("▼" if diff.overall_delta < 0 else "=")
        print(f"Overall score:    {report.overall_score:.2f}   "
              f"(delta {diff.overall_delta:+.2f} {arrow})")
        print(f"  Analyst:        {report.analyst_score:.2f}")
        print(f"  Trader:         {report.trader_score:.2f}")
    print(f"Pass rate:        {report.pass_rate * 100:.0f}%")
    print(f"Schema failures:  {report.schema_failures}")
    if report.judge_errors > 0:
        print(f"Judge errors:     {report.judge_errors}")
    print()

    if diff is not None:
        if diff.per_case_regressions:
            print("Per-case regressions (>5% drop):")
            for case_id in diff.per_case_regressions:
                print(f"  {case_id}  ▼")
            print()
        if diff.new_schema_failures:
            print("New schema failures:")
            for case_id in diff.new_schema_failures:
                print(f"  {case_id}")
            print()
        if diff.per_case_improvements:
            print("Per-case improvements (>5% gain):")
            for case_id in diff.per_case_improvements:
                print(f"  {case_id}  ▲")
            print()

    exit_code = 1 if (diff is not None and diff.is_regression) else 0
    status = "regression" if exit_code == 1 else "no regressions"
    print(f"Exit: {exit_code} ({status})")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 44 PASS (41 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/report.py tests/test_eval.py
git commit -m "feat(eval): human-readable terminal summary with diff arrows"
```

---

### Task 11: CLI entry point `eval.py`

**Files:**
- Create: `src/vibe_trading/eval/eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval.py`:

```python
from vibe_trading.eval.eval import main


@patch("vibe_trading.eval.eval.build_judge")
@patch("vibe_trading.eval.eval.run_case")
@patch("vibe_trading.eval.eval.Database")
@patch("vibe_trading.eval.eval.load_cases")
def test_main_returns_zero_with_no_baseline(
    mock_load_cases, mock_db_cls, mock_run_case, mock_build_judge, tmp_path
):
    """No baseline file -> exit 0, friendly message, report still written."""
    case = _load_valid_case()
    mock_load_cases.return_value = [case]
    mock_db_cls.return_value = MagicMock()
    mock_run_case.return_value = _build_perfect_result(case)
    mock_build_judge.return_value = _passing_judge_stub

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    reports = tmp_path / "reports"
    baseline = tmp_path / "baseline.json"  # does not exist

    rc = main([
        "--snapshots", str(snapshots),
        "--baseline", str(baseline),
        "--reports-dir", str(reports),
    ])
    assert rc == 0
    # A report file should have been written
    assert any(reports.glob("eval-*.json"))


@patch("vibe_trading.eval.eval.build_judge")
@patch("vibe_trading.eval.eval.run_case")
@patch("vibe_trading.eval.eval.Database")
@patch("vibe_trading.eval.eval.load_cases")
def test_main_returns_one_on_regression(
    mock_load_cases, mock_db_cls, mock_run_case, mock_build_judge, tmp_path
):
    """Baseline says 1.0, current run scores 0.0 on the same case -> exit 1."""
    case = _load_valid_case()
    mock_load_cases.return_value = [case]
    mock_db_cls.return_value = MagicMock()

    # Force a parse failure to drop the score to 0.0
    from vibe_trading.eval.runner import CaseResult
    mock_run_case.return_value = CaseResult(
        case_id=case.id, snapshot_ok=True, analyst_schema_ok=False, error="forced",
    )
    mock_build_judge.return_value = _passing_judge_stub

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    reports = tmp_path / "reports"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "overall_score": 1.0,
        "per_case": {case.id: {"total_score": 1.0, "schema_ok": True}},
    }))

    rc = main([
        "--snapshots", str(snapshots),
        "--baseline", str(baseline),
        "--reports-dir", str(reports),
    ])
    assert rc == 1


@patch("vibe_trading.eval.eval.build_judge")
@patch("vibe_trading.eval.eval.run_case")
@patch("vibe_trading.eval.eval.Database")
@patch("vibe_trading.eval.eval.load_cases")
def test_main_update_baseline_writes_baseline(
    mock_load_cases, mock_db_cls, mock_run_case, mock_build_judge, tmp_path
):
    case = _load_valid_case()
    mock_load_cases.return_value = [case]
    mock_db_cls.return_value = MagicMock()
    mock_run_case.return_value = _build_perfect_result(case)
    mock_build_judge.return_value = _passing_judge_stub

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    reports = tmp_path / "reports"
    baseline = tmp_path / "baseline.json"

    rc = main([
        "--snapshots", str(snapshots),
        "--baseline", str(baseline),
        "--reports-dir", str(reports),
        "--update-baseline",
    ])
    assert rc == 0
    assert baseline.exists()
    snapshot = json.loads(baseline.read_text())
    assert case.id in snapshot["per_case"]


@patch("vibe_trading.eval.eval.load_cases")
def test_main_returns_one_when_no_cases_found(mock_load_cases, tmp_path):
    mock_load_cases.return_value = []
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    rc = main([
        "--snapshots", str(snapshots),
        "--baseline", str(tmp_path / "baseline.json"),
        "--reports-dir", str(tmp_path / "reports"),
    ])
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_eval.py::test_main_returns_zero_with_no_baseline -v`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement eval.py**

Create `src/vibe_trading/eval/eval.py`:

```python
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from vibe_trading.data.db import Database
from vibe_trading.eval.runner import load_cases, run_case
from vibe_trading.eval.scorer import build_judge, score_case
from vibe_trading.eval.report import (
    SuiteReport, write_report, load_baseline, write_baseline,
    diff_against_baseline, print_summary,
)

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="vibe-eval")
    parser.add_argument("--snapshots", type=Path, default=Path("evals/snapshots"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline.json with current run scores")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.snapshots)
    except ValueError as e:
        print(f"ERROR loading cases: {e}", file=sys.stderr)
        return 1

    if not cases:
        print(f"ERROR: no cases found in {args.snapshots}", file=sys.stderr)
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
    logger.info(f"Wrote report to {report_path}")

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"No baseline at {args.baseline}. "
              f"Run with --update-baseline to seed it.")
        print_summary(report, diff=None)
        if args.update_baseline:
            write_baseline(report, args.baseline)
            print(f"Baseline written: {args.baseline}")
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

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 48 PASS (44 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/eval.py tests/test_eval.py
git commit -m "feat(eval): argparse CLI entry point with --update-baseline and regression exit code"
```

---

### Task 12: Seed example golden-set snapshots

**Files:**
- Create: `evals/snapshots/001-example-bullish-breakout.yaml`
- Create: `evals/snapshots/002-example-bearish-fakeout.yaml`
- Modify: `README.md` (add a short eval section pointing at the CLI + golden set)

- [ ] **Step 1: Write the first example case**

Create `evals/snapshots/001-example-bullish-breakout.yaml`:

```yaml
id: 001-example-bullish-breakout
description: |
  EXAMPLE CASE. Replace `(symbol, timestamp)` with real DuckDB-backed candles before
  treating scores from this case as meaningful.

  This case demonstrates a textbook bullish breakout: price clears a multi-week
  resistance on rising volume, with bullish MA stack and ADX in trending regime.
symbol: BTC/USDT
timestamp: 2026-04-01T04:00:00Z

analyst_label:
  market_bias: bullish
  volume_confirmation: confirmed
  nearest_support: 60000.0
  nearest_resistance: 67000.0
  confluence_score: 0.75
  thesis_rubric:
    must_mention:
      - breakout above resistance
      - rising volume confirming the move
    must_not_mention:
      - bearish reversal
      - overbought warning

trader_label:
  action: long
  stop_loss_strategy: 1.5_atr
  take_profit_strategy: next_resistance
  risk_reward_ratio: 2.0
  hold_period_bias: medium
  reasoning_rubric:
    must_mention:
      - breakout
      - volume confluence
    must_not_mention:
      - flat recommended despite high confluence
```

- [ ] **Step 2: Write the second example case**

Create `evals/snapshots/002-example-bearish-fakeout.yaml`:

```yaml
id: 002-example-bearish-fakeout
description: |
  EXAMPLE CASE. Replace `(symbol, timestamp)` with real DuckDB-backed candles before
  treating scores from this case as meaningful.

  This case demonstrates a bearish fakeout: price breaks above resistance on weak
  volume, immediately rejects, and closes back inside the range with bearish RSI
  divergence. The correct response is FLAT, not LONG.
symbol: ETH/USDT
timestamp: 2026-04-15T04:00:00Z

analyst_label:
  market_bias: bearish
  volume_confirmation: divergent
  nearest_support: 3000.0
  nearest_resistance: 3400.0
  confluence_score: 0.45
  thesis_rubric:
    must_mention:
      - failed breakout
      - bearish divergence
    must_not_mention:
      - strong bullish breakout
      - rising volume confirming the move

trader_label:
  action: flat
  stop_loss_strategy: 1.5_atr
  take_profit_strategy: next_resistance
  risk_reward_ratio: 1.5
  hold_period_bias: short
  reasoning_rubric:
    must_mention:
      - lack of edge
      - weak confluence
    must_not_mention:
      - enter long despite divergence
```

- [ ] **Step 3: Update README.md with an eval section**

Add this section to the bottom of `/Users/raj/vibe-trading/README.md` (or create the file with this content if it does not yet contain a heading). Locate an appropriate spot near the existing usage docs:

```markdown
## Eval Harness

Score the analyst + trader prompts against the hand-labeled golden set under `evals/snapshots/`:

```bash
# Run the eval against the current baseline, prints summary, non-zero exit on regression
uv run python -m vibe_trading.eval.eval

# After a prompt change you've reviewed and approved:
uv run python -m vibe_trading.eval.eval --update-baseline

# Use a different judge model (any LiteLLM-compatible identifier)
EVAL_JUDGE_MODEL=claude-3-5-haiku-20241022 uv run python -m vibe_trading.eval.eval
```

The golden-set YAMLs live in `evals/snapshots/`. The first two cases are illustrative
examples — replace their `(symbol, timestamp)` with real DuckDB-backed candles, and add
30-50 hand-curated cases covering breakouts, fakeouts, ranges, and FOMC reactions to
make the suite meaningful.

Reports land in `data/reports/eval-<timestamp>.json`. The regression yardstick is
`evals/baseline.json`, committed to git so prompt-impact diffs are reviewable in PRs.
```

- [ ] **Step 4: Verify the example cases parse**

Run: `uv run python -c "from pathlib import Path; from vibe_trading.eval.runner import load_cases; cases = load_cases(Path('evals/snapshots')); print(f'Loaded {len(cases)} cases:', [c.id for c in cases])"`
Expected: prints `Loaded 2 cases: ['001-example-bullish-breakout', '002-example-bearish-fakeout']`.

- [ ] **Step 5: Commit**

```bash
git add evals/snapshots/001-example-bullish-breakout.yaml evals/snapshots/002-example-bearish-fakeout.yaml README.md
git commit -m "docs(eval): seed two example golden-set cases and document the CLI"
```

---

### Task 13: Final full-suite verification

**Files:**
- (None modified — verification only.)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS — at least 48 new in `test_eval.py` plus existing suites green.

- [ ] **Step 2: Smoke-import all new modules**

Run: `uv run python -c "from vibe_trading.eval import eval, runner, scorer, report; print('OK')"`
Expected: prints `OK` with no errors.

- [ ] **Step 3: Dry-run the CLI without a baseline**

Run: `uv run python -m vibe_trading.eval.eval --snapshots evals/snapshots --baseline evals/baseline.json --reports-dir data/reports/eval`
Expected (paraphrased):
- Either: agents run against real LLM (consumes a small budget) — produces a report and prints "No baseline ... Run with --update-baseline to seed it." Exit 0.
- Or: if `GEMINI_API_KEY` (or whatever provider is configured) isn't set in the shell, error mentions missing API key.

This step is informational — if the user wants to dry-run, they can. The harness is verified by the unit tests in Step 1.

- [ ] **Step 4: Commit any tweaks if necessary**

If anything surfaced during verification (e.g. lint nits), commit them:

```bash
git status
git add -p
git commit -m "chore: post-implementation eval-harness cleanup"
```

Otherwise this task ends without a commit.

---

## Spec coverage check (self-review)

- **Pydantic case schema (Rubric, AnalystLabel, TraderLabel, EvalCase, CaseResult):** Task 2.
- **`load_cases()` with descriptive error on malformed YAML:** Task 3.
- **`run_case()` driving analyst snapshot path + trader-with-labeled-input:** Task 4.
- **Deterministic categorical scorer:** Task 5.
- **Deterministic numeric-tolerance scorer with linear-degradation formula + NaN/None/expected-zero edges:** Task 5.
- **`build_judge()` with EVAL_JUDGE_MODEL env override:** Task 6.
- **`score_rubric()` with empty-rubric vacuous-pass + judge-failure-neutral semantics:** Task 6.
- **`score_case()` aggregator with schema short-circuit:** Task 7.
- **`SuiteReport.from_scores()` with overall / analyst / trader / pass_rate / schema_failures / judge_errors:** Task 8.
- **`write_report()` timestamped JSON output:** Task 8.
- **`load_baseline()`, `write_baseline()`, `diff_against_baseline()` with REGRESSION_THRESHOLDS = {overall: 0.02, per_case: 0.05}:** Task 9.
- **`print_summary()` terminal output with diff arrows:** Task 10.
- **`eval.py` argparse CLI with --update-baseline and non-zero exit on regression:** Task 11.
- **Seed example golden-set cases + README documentation:** Task 12.
- **Final verification:** Task 13.
- **`evals/` + `tests/fixtures/eval/` directory scaffolding + pyyaml dep:** Task 1.

All spec requirements have at least one task that implements them, no placeholders remain in the plan, and type signatures are consistent across tasks (e.g. `Rubric`, `FieldScore`, `CaseScore`, `SuiteReport`, `DiffResult` names match every site they appear in).
