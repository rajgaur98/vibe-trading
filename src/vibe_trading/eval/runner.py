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
