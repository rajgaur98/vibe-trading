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
