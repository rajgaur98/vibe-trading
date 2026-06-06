"""Schema-robustness tests (rubric: every LLM call returns a Pydantic-validated object;
track schema-compliance rate).

Covers:
- a successful structured parse returns a validated model/dict;
- a first-call malformed response triggers ONE corrective retry whose valid response is accepted;
- two consecutive failures raise the typed SchemaValidationError (never a bare KeyError);
- schema_ok is recorded on the cost event for both the pass and the fail paths.
"""
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from vibe_trading.agents.client import LLMClient, SchemaValidationError
from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput
from vibe_trading.agents.trader import HeadTrader, HeadTraderOutput


# --- shared helpers ----------------------------------------------------------

VALID_ANALYST_JSON = (
    '{"market_bias": "bullish", "volume_confirmation": "confirmed", '
    '"thesis": "ok", "nearest_support": 95.0, "nearest_resistance": 105.0, '
    '"confluence_score": 0.7}'
)
VALID_TRADER_JSON = (
    '{"action": "long", "stop_loss_strategy": "1.5_atr", '
    '"take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, '
    '"hold_period_bias": "medium", "reasoning_summary": "Strong trend."}'
)
# Malformed: valid JSON syntactically but missing required fields (KeyError/ValidationError risk).
MALFORMED_TRADER_JSON = '{"action": "long"}'
# Not even JSON.
NOT_JSON = "I think you should go long, sorry no JSON here."


class _CollectSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


def _make_client_mock(provider="gemini", model="gemini-3.1-flash-lite"):
    c = MagicMock()
    c.provider = provider
    c.model = model
    return c


# --- (1) success path: validated model is returned ---------------------------

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_trader_decide_returns_validated_model_fields():
    """A clean structured response yields a HeadTrader proposal built off a validated
    HeadTraderOutput (not a hand-built dict-by-key)."""
    client = _make_client_mock()
    client.call_llm.return_value = VALID_TRADER_JSON
    trader = HeadTrader(client=client)
    analyst = AnalystOutput(
        market_bias="bullish", volume_confirmation="confirmed", thesis="t",
        nearest_support=95.0, nearest_resistance=105.0, confluence_score=0.8,
    )
    proposal = trader.decide("BTC/USDT", analyst, {}, [], current_price=100.0)
    assert proposal["action"] == "long"
    assert proposal["stop_loss_strategy"] == "1.5_atr"
    assert str(proposal["risk_reward_ratio"]) == "2.0"
    client.call_llm.assert_called_once()


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_analyst_snapshot_returns_validated_model():
    client = _make_client_mock()
    client.call_llm.return_value = VALID_ANALYST_JSON
    analyst = TechnicalVolumeAnalyst(client=client)
    res = analyst.analyze(symbol="BTC/USDT", snapshot={"symbol": "BTC/USDT"})
    assert isinstance(res, AnalystOutput)
    assert res.market_bias == "bullish"
    client.call_llm.assert_called_once()


# --- (2) corrective retry: bad-then-good is accepted -------------------------

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_trader_first_malformed_then_valid_triggers_one_retry():
    """First structured response is malformed -> ONE corrective re-call -> valid response accepted."""
    client = _make_client_mock()
    client.call_llm.side_effect = [MALFORMED_TRADER_JSON, VALID_TRADER_JSON]
    trader = HeadTrader(client=client)
    analyst = AnalystOutput(
        market_bias="bullish", volume_confirmation="confirmed", thesis="t",
        nearest_support=95.0, nearest_resistance=105.0, confluence_score=0.8,
    )
    proposal = trader.decide("BTC/USDT", analyst, {}, [], current_price=100.0)
    assert proposal["action"] == "long"
    assert client.call_llm.call_count == 2
    # The retry must include a corrective instruction in the prompt.
    retry_prompt = client.call_llm.call_args_list[1].kwargs["prompt"]
    assert "did not match" in retry_prompt.lower() or "valid json" in retry_prompt.lower()


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_analyst_snapshot_first_not_json_then_valid_triggers_retry():
    client = _make_client_mock()
    client.call_llm.side_effect = [NOT_JSON, VALID_ANALYST_JSON]
    analyst = TechnicalVolumeAnalyst(client=client)
    res = analyst.analyze(symbol="BTC/USDT", snapshot={"symbol": "BTC/USDT"})
    assert isinstance(res, AnalystOutput)
    assert res.market_bias == "bullish"
    assert client.call_llm.call_count == 2


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_analyst_tool_loop_first_malformed_then_valid_triggers_retry():
    """The tool-loop final-answer path is also wrapped in the corrective-retry helper."""
    client = _make_client_mock()
    client.call_llm_with_tools.side_effect = ["```json\n{not valid}\n```", VALID_ANALYST_JSON]
    db, fetcher = MagicMock(), MagicMock()
    analyst = TechnicalVolumeAnalyst(client=client, db=db, fetcher=fetcher)
    res = analyst.analyze(symbol="BTC/USDT", timestamp=datetime(2026, 5, 26))
    assert isinstance(res, AnalystOutput)
    assert res.market_bias == "bullish"
    assert client.call_llm_with_tools.call_count == 2


# --- (3) two consecutive failures -> typed SchemaValidationError (NOT KeyError) -

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_trader_two_failures_raise_schema_validation_error_not_keyerror():
    client = _make_client_mock()
    client.call_llm.side_effect = [MALFORMED_TRADER_JSON, MALFORMED_TRADER_JSON]
    trader = HeadTrader(client=client)
    analyst = AnalystOutput(
        market_bias="bullish", volume_confirmation="confirmed", thesis="t",
        nearest_support=95.0, nearest_resistance=105.0, confluence_score=0.8,
    )
    with pytest.raises(SchemaValidationError):
        trader.decide("BTC/USDT", analyst, {}, [], current_price=100.0)
    assert client.call_llm.call_count == 2


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_analyst_two_failures_raise_schema_validation_error():
    client = _make_client_mock()
    client.call_llm.side_effect = [NOT_JSON, NOT_JSON]
    analyst = TechnicalVolumeAnalyst(client=client)
    with pytest.raises(SchemaValidationError):
        analyst.analyze(symbol="BTC/USDT", snapshot={"symbol": "BTC/USDT"})
    assert client.call_llm.call_count == 2


def test_schema_validation_error_is_not_keyerror():
    """Defensive: the typed error is its own exception, not a KeyError subclass."""
    assert not issubclass(SchemaValidationError, KeyError)


# --- (4) schema_ok recorded on the cost event for pass AND fail --------------

@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_schema_ok_true_recorded_on_cost_event_for_valid_parse(mock_completion):
    """A structured call whose output validates records schema_ok=True on its cost event."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = VALID_TRADER_JSON
    resp.choices[0].message.tool_calls = None
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
    mock_completion.return_value = resp

    sink = _CollectSink()
    LLMClient.set_cost_sink(sink)
    try:
        trader = HeadTrader()  # real LLMClient under the hood
        analyst = AnalystOutput(
            market_bias="bullish", volume_confirmation="confirmed", thesis="t",
            nearest_support=95.0, nearest_resistance=105.0, confluence_score=0.8,
        )
        trader.decide("BTC/USDT", analyst, {}, [], current_price=100.0)
    finally:
        LLMClient.set_cost_sink(None)

    assert len(sink.events) == 1
    assert sink.events[0].schema_ok is True


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_schema_ok_false_recorded_on_both_failed_attempts(mock_completion):
    """Two malformed responses -> two cost events, both with schema_ok=False, then raise."""
    def malformed(*a, **k):
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = MALFORMED_TRADER_JSON
        r.choices[0].message.tool_calls = None
        r.usage = MagicMock(prompt_tokens=100, completion_tokens=10)
        return r
    mock_completion.side_effect = malformed

    sink = _CollectSink()
    LLMClient.set_cost_sink(sink)
    try:
        trader = HeadTrader()
        analyst = AnalystOutput(
            market_bias="bullish", volume_confirmation="confirmed", thesis="t",
            nearest_support=95.0, nearest_resistance=105.0, confluence_score=0.8,
        )
        with pytest.raises(SchemaValidationError):
            trader.decide("BTC/USDT", analyst, {}, [], current_price=100.0)
    finally:
        LLMClient.set_cost_sink(None)

    assert len(sink.events) == 2
    assert all(ev.schema_ok is False for ev in sink.events)
