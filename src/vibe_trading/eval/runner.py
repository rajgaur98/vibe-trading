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
