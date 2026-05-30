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
@patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}, clear=True)
def test_build_judge_defaults_to_client_model_when_env_unset(mock_client_cls):
    """With EVAL_JUDGE_MODEL unset, the judge inherits LLMClient.model so it always
    targets the currently-configured provider (no risk of pointing at a Gemini
    model when LLM_PROVIDER=groq).
    """
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
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
    call_kwargs = mock_client.call_llm.call_args.kwargs
    assert call_kwargs["model_name"] == "gemini-3.1-flash-lite"


@patch("vibe_trading.eval.scorer.LLMClient")
@patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=True)
def test_build_judge_defaults_to_client_model_under_groq(mock_client_cls):
    """Regression guard for the swap from a hardcoded gemini default to client.model
    inheritance — under LLM_PROVIDER=groq the judge must target the Groq model,
    not a Gemini one.
    """
    mock_client = MagicMock()
    mock_client.provider = "groq"
    mock_client.model = "llama-3.3-70b-versatile"
    mock_client.call_llm.return_value = (
        '{"must_mention_results": [], "must_not_mention_results": []}'
    )
    mock_client_cls.return_value = mock_client

    judge = build_judge()
    judge("text", Rubric())

    assert mock_client.call_llm.call_args.kwargs["model_name"] == "llama-3.3-70b-versatile"


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
    # Valid fixture is directional (long), so all 12 fields scored; 1 analyst field fails.
    assert cs.total_score < 1.0
    assert cs.analyst_score < 1.0
    assert cs.trader_score == 1.0


def test_score_case_flat_label_skips_entry_strategy_fields():
    """On a 'flat' label the trader is scored only on action + reasoning_summary.

    The entry-strategy fields (stop_loss_strategy, take_profit_strategy,
    hold_period_bias, risk_reward_ratio) have no ground truth on a no-trade
    decision, so they are excluded — scoring them against placeholder labels was
    measuring noise.
    """
    case = _load_valid_case()
    case.trader_label.action = "flat"
    result = _build_perfect_result(case)
    result.trader_output["action"] = "flat"  # correctly declines
    # The LLM still emits (irrelevant) entry-strategy values — they must be IGNORED.
    result.trader_output["stop_loss_strategy"] = "2.0_atr"
    result.trader_output["take_profit_strategy"] = "4.0_atr"
    result.trader_output["hold_period_bias"] = "long"
    result.trader_output["risk_reward_ratio"] = 9.9

    cs = score_case(result, case, _passing_judge_stub)
    scored = {fs.field for fs in cs.field_scores}

    # Entry-strategy fields excluded from scoring on a flat label
    assert "stop_loss_strategy" not in scored
    assert "take_profit_strategy" not in scored
    assert "hold_period_bias" not in scored
    assert "risk_reward_ratio" not in scored
    # action + reasoning_summary still scored
    assert "action" in scored
    assert "reasoning_summary" in scored
    # Trader score = mean(action=1.0, reasoning=1.0) = 1.0 despite the wrong entry-strategy values
    assert cs.trader_score == 1.0


def test_score_case_directional_label_scores_all_trader_fields():
    """On a directional (long/short) label, all six trader fields are scored."""
    case = _load_valid_case()
    case.trader_label.action = "long"
    result = _build_perfect_result(case)

    cs = score_case(result, case, _passing_judge_stub)
    scored = {fs.field for fs in cs.field_scores}
    for f in ["action", "stop_loss_strategy", "take_profit_strategy",
              "hold_period_bias", "risk_reward_ratio", "reasoning_summary"]:
        assert f in scored, f"{f} should be scored on a directional label"


def test_score_case_snapshot_failure_propagates():
    """snapshot_ok=False -> total_score=0.0, schema_ok=False, error preserved."""
    from vibe_trading.eval.runner import CaseResult
    case = _load_valid_case()
    result = CaseResult(case_id=case.id, snapshot_ok=False, error="empty snapshot")
    cs = score_case(result, case, _passing_judge_stub)
    assert cs.schema_ok is False
    assert cs.total_score == 0.0
    assert cs.error == "empty snapshot"


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


import json
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
        "--throttle-seconds", "0",
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
        "--throttle-seconds", "0",
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
        "--throttle-seconds", "0",
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
        "--throttle-seconds", "0",
    ])
    assert rc == 1


@patch("vibe_trading.eval.eval.build_judge")
@patch("vibe_trading.eval.eval.run_case")
@patch("vibe_trading.eval.eval.Database")
@patch("vibe_trading.eval.eval.load_cases")
def test_main_scores_all_cases_in_parallel(
    mock_load_cases, mock_db_cls, mock_run_case, mock_build_judge, tmp_path
):
    """Parallel orchestration: every case is scored, results preserve case identity,
    and each worker gets its own Database (thread isolation)."""
    import copy
    base = _load_valid_case()
    cases = []
    for i in range(3):
        c = copy.deepcopy(base)
        c.id = f"case-{i}"
        cases.append(c)
    mock_load_cases.return_value = cases
    mock_db_cls.return_value = MagicMock()
    mock_run_case.side_effect = lambda case, db: _build_perfect_result(case)
    mock_build_judge.return_value = _passing_judge_stub

    reports = tmp_path / "reports"
    rc = main([
        "--snapshots", str(tmp_path / "snap"),
        "--baseline", str(tmp_path / "baseline.json"),
        "--reports-dir", str(reports),
        "--throttle-seconds", "0",
        "--max-workers", "3",
    ])
    assert rc == 0

    report_files = list(reports.glob("eval-*.json"))
    assert report_files, "a report should have been written"
    data = json.loads(report_files[0].read_text())
    assert data["case_count"] == 3
    assert set(data["per_case"].keys()) == {"case-0", "case-1", "case-2"}
    # One Database per case → confirms per-thread isolation (not a shared connection)
    assert mock_db_cls.call_count == 3


@patch("vibe_trading.eval.runner.HeadTrader")
@patch("vibe_trading.eval.runner.TechnicalVolumeAnalyst")
@patch("vibe_trading.eval.runner.FeaturePipeline")
def test_run_case_handles_analyst_construction_failure(mock_pipeline_cls, mock_analyst_cls, mock_trader_cls):
    """If TechnicalVolumeAnalyst() raises at construction time (e.g. missing API key), the suite continues."""
    case = _load_valid_case()
    mock_pipeline_cls.return_value.run.return_value = {"symbol": case.symbol}
    mock_analyst_cls.side_effect = ValueError("GEMINI_API_KEY environment variable is not set")

    # Must not raise — the failure should be captured on the CaseResult.
    result = run_case(case, MagicMock())

    assert result.snapshot_ok is True
    assert result.analyst_schema_ok is False
    assert "GEMINI_API_KEY" in result.error
    mock_trader_cls.return_value.decide.assert_not_called()


def test_suite_report_pass_rate_excludes_schema_failures():
    """A case with schema_ok=False and field_scores=[] must NOT count toward pass_rate."""
    cs_passing = CaseScore(
        case_id="good", schema_ok=True,
        field_scores=[FieldScore(field="market_bias", passed=True, score=1.0)],
        analyst_score=1.0, trader_score=1.0, total_score=1.0,
    )
    cs_schema_fail = CaseScore(
        case_id="broken", schema_ok=False, field_scores=[],
        analyst_score=0.0, trader_score=0.0, total_score=0.0, error="parse",
    )
    report = SuiteReport.from_scores([cs_passing, cs_schema_fail])
    assert report.pass_rate == 0.5  # 1 of 2 cases passes; the schema-failed one is NOT a pass
    assert report.schema_failures == 1
