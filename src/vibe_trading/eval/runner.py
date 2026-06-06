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


from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput
from vibe_trading.agents.trader import HeadTrader
from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher
from vibe_trading.features.pipeline import FeaturePipeline


# Fixed inputs the trader sees in every eval case — keeps trader scoring independent of context.
FIXED_SCORECARD = {"accuracy": 0.55, "total_decisions": 100}
FIXED_OPEN_POSITIONS: list = []


def run_case(case: EvalCase, db: Database, analyst_path: str = "snapshot") -> CaseResult:
    """Run one eval case: build snapshot, run analyst, run trader with LABELED analyst output.

    `analyst_path` selects which analyst code path to exercise:
      - "snapshot" (default): the fast, single-call legacy path — analyze(symbol, snapshot=...).
        Deterministic, ~1 LLM call/case; the regression-gate default.
      - "tool-loop": the SAME multi-turn tool-use path production runs —
        analyze(symbol, timestamp) with a live Database+DataFetcher-backed ToolExecutor.
        Slower / more LLM calls, but verifies exactly what ships.
    The snapshot is always built (cheap, deterministic) for the trader's current_price and
    the snapshot_ok gate, regardless of which analyst path runs.

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

    # Analyst — snapshot path (no db/fetcher) by default; tool-loop path (db+fetcher,
    # timestamp-driven) when analyst_path="tool-loop" to mirror exactly what prod runs.
    try:
        if analyst_path == "tool-loop":
            analyst = TechnicalVolumeAnalyst(db=db, fetcher=DataFetcher())
            actual_analyst_output = analyst.analyze(symbol=case.symbol, timestamp=case.timestamp)
        else:
            analyst = TechnicalVolumeAnalyst(db=None, fetcher=None)
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

    try:
        trader = HeadTrader()
        # Current price from the snapshot drives the trader's proximity-based stop selection.
        trader_output = trader.decide(
            case.symbol, labeled_analyst_output, FIXED_SCORECARD, FIXED_OPEN_POSITIONS,
            current_price=float(snapshot.get("close", 0.0)),
        )
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
