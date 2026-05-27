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
