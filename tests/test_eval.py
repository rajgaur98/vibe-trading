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
