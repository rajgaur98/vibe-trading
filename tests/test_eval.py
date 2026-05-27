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
