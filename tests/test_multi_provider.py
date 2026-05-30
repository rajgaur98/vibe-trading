import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from vibe_trading.agents.client import LLMClient, get_litellm_model_string

def test_get_litellm_model_string():
    assert get_litellm_model_string("gemini", "gemini-3.1-flash-lite") == "gemini/gemini-3.1-flash-lite"
    assert get_litellm_model_string("openai", "gpt-4o") == "openai/gpt-4o"
    assert get_litellm_model_string("anthropic", "claude-3") == "anthropic/claude-3"
    assert get_litellm_model_string("ollama", "qwen2.5:14b") == "ollama/qwen2.5:14b"
    assert get_litellm_model_string("groq", "llama-3.3-70b-versatile") == "groq/llama-3.3-70b-versatile"
    assert get_litellm_model_string("other", "model") == "model"


@patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test_groq_key"})
def test_llm_client_initialization_groq():
    client = LLMClient()
    assert client.provider == "groq"


@patch.dict("os.environ", {"LLM_PROVIDER": "groq"}, clear=True)
def test_llm_client_initialization_groq_missing_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is not set"):
        LLMClient()

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"}, clear=True)
def test_llm_client_initialization_gemini():
    """With LLM_MODEL unset, LLMClient falls back to the hardcoded default."""
    client = LLMClient()
    assert client.provider == "gemini"
    assert client.model == "gemini-3.1-flash-lite"

@patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test_openai_key"})
def test_llm_client_initialization_openai():
    client = LLMClient()
    assert client.provider == "openai"

@patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=True)
def test_llm_client_initialization_openai_missing_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
        LLMClient()

@patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test_anthropic_key"})
def test_llm_client_initialization_anthropic():
    client = LLMClient()
    assert client.provider == "anthropic"

@patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}, clear=True)
def test_llm_client_initialization_anthropic_missing_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is not set"):
        LLMClient()


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_min_call_interval_defaults_to_zero():
    """No throttle by default — live trading behavior unchanged."""
    client = LLMClient()
    assert client.min_call_interval == 0.0


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k", "LLM_MIN_CALL_INTERVAL_SECONDS": "4.5"}, clear=True)
def test_min_call_interval_parsed_from_env():
    client = LLMClient()
    assert client.min_call_interval == 4.5


@patch("vibe_trading.agents.client.time")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k", "LLM_MIN_CALL_INTERVAL_SECONDS": "4.0"}, clear=True)
def test_throttle_sleeps_when_calls_too_close(mock_time):
    """A second call within the interval sleeps for the remaining time; the gate is
    shared across instances (the class-level last-call timestamp) so analyst/trader/
    judge clients collectively respect one provider rate limit."""
    LLMClient._last_call_at = 0.0  # reset shared class state
    # monotonic() sequence: first _throttle sees t=100 (huge elapsed -> no sleep, sets last=100),
    # second _throttle sees t=101 (1s elapsed -> wait 3s), then records t=104.
    mock_time.monotonic.side_effect = [100.0, 100.0, 101.0, 104.0]
    client = LLMClient()

    client._throttle()
    mock_time.sleep.assert_not_called()   # first call: no prior call to space from

    client._throttle()
    mock_time.sleep.assert_called_once()
    waited = mock_time.sleep.call_args[0][0]
    assert abs(waited - 3.0) < 1e-6        # 4.0 interval - 1.0 elapsed


@patch("vibe_trading.agents.client.time")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_throttle_noop_when_interval_zero(mock_time):
    """interval=0 -> never sleeps (default / live path)."""
    LLMClient._last_call_at = 0.0
    mock_time.monotonic.return_value = 100.0
    client = LLMClient()
    client._throttle()
    client._throttle()
    mock_time.sleep.assert_not_called()

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key", "LLM_MODEL": "custom-gemini-model"})
def test_llm_client_initialization_model_override():
    client = LLMClient()
    assert client.model == "custom-gemini-model"

@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_call_llm(mock_completion):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"result": "success"}'
    mock_completion.return_value = mock_response

    client = LLMClient()
    res = client.call_llm("test-model", "system prompt", "user prompt")
    assert res == '{"result": "success"}'
    mock_completion.assert_called_once_with(
        model="gemini/test-model",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"}
        ],
        temperature=0.1
    )

from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput, _extract_json


def test_extract_json_passthrough_for_clean_json():
    """A bare JSON object is returned unchanged (legacy response_format path)."""
    assert _extract_json('{"market_bias": "bullish"}') == '{"market_bias": "bullish"}'


def test_extract_json_strips_json_fences():
    """Markdown ```json fences (Gemma's tool-loop final answer) are stripped."""
    fenced = '```json\n{"market_bias": "bearish", "confluence_score": 0.5}\n```'
    assert _extract_json(fenced) == '{"market_bias": "bearish", "confluence_score": 0.5}'


def test_extract_json_strips_bare_fences():
    """Bare ``` fences (no language tag) are stripped."""
    fenced = '```\n{"a": 1}\n```'
    assert _extract_json(fenced) == '{"a": 1}'


def test_extract_json_strips_surrounding_whitespace():
    """Leading/trailing whitespace and newlines around fenced JSON are removed."""
    fenced = '\n\n```json\n{"a": 1}\n```\n\n'
    assert _extract_json(fenced) == '{"a": 1}'


def test_extract_json_handles_none():
    """None input returns empty string (so json.loads fails with a clear error, not a TypeError)."""
    assert _extract_json(None) == ""


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_tool_loop_parses_fenced_json():
    """End-to-end: when the tool-loop returns markdown-fenced JSON, analyze() still parses it."""
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemma-4-31b-it"
    mock_client.call_llm_with_tools.return_value = (
        '```json\n{"market_bias": "bearish", "volume_confirmation": "divergent", '
        '"thesis": "fenced thesis", "nearest_support": 60000.0, '
        '"nearest_resistance": 67000.0, "confluence_score": 0.5}\n```'
    )
    db, fetcher = MagicMock(), MagicMock()
    analyst = TechnicalVolumeAnalyst(client=mock_client, db=db, fetcher=fetcher)
    res = analyst.analyze(symbol="BTC/USDT", timestamp=datetime(2026, 5, 26))
    assert isinstance(res, AnalystOutput)
    assert res.market_bias == "bearish"
    assert res.confluence_score == 0.5
    mock_client.call_llm_with_tools.assert_called_once()


@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_integration():
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
    # Mock the raw json response matching AnalystOutput schema
    mock_client.call_llm.return_value = '{"market_bias": "bullish", "volume_confirmation": "confirmed", "thesis": "Strong breakout on high volume.", "nearest_support": 95.0, "nearest_resistance": 105.0, "confluence_score": 0.8}'
    
    analyst = TechnicalVolumeAnalyst(client=mock_client)
    snapshot = {"symbol": "BTC/USDT"}
    res = analyst.analyze(symbol="BTC/USDT", snapshot=snapshot)
    
    assert isinstance(res, AnalystOutput)
    assert res.market_bias == "bullish"
    mock_client.call_llm.assert_called_once()


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_end_to_end_with_client_mock(mock_completion):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"market_bias": "bullish", "volume_confirmation": "confirmed", "thesis": "Strong breakout.", "nearest_support": 95.0, "nearest_resistance": 105.0, "confluence_score": 0.8}'
    mock_completion.return_value = mock_response

    analyst = TechnicalVolumeAnalyst()
    res = analyst.analyze(symbol="BTC/USDT", snapshot={"symbol": "BTC/USDT"})
    
    assert res.market_bias == "bullish"
    mock_completion.assert_called_once()
    # verify model prefix formatting
    call_kwargs = mock_completion.call_args[1]
    assert "gemini" in call_kwargs["model"]


from vibe_trading.agents.trader import HeadTrader, HeadTraderOutput

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_trader_integration():
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
    mock_client.call_llm.return_value = '{"action": "long", "stop_loss_strategy": "1.5_atr", "take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, "hold_period_bias": "medium", "reasoning_summary": "Strong trend confirmation."}'
    
    trader = HeadTrader(client=mock_client)
    analyst_res = AnalystOutput(
        market_bias="bullish",
        volume_confirmation="confirmed",
        thesis="Strong breakout",
        nearest_support=95.0,
        nearest_resistance=105.0,
        confluence_score=0.8
    )
    
    proposal = trader.decide("BTC/USDT", analyst_res, {}, [])
    assert proposal["action"] == "long"
    mock_client.call_llm.assert_called_once()


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"}, clear=True)
def test_trader_end_to_end_with_client_mock(mock_completion):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"action": "long", "stop_loss_strategy": "1.5_atr", "take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, "hold_period_bias": "medium", "reasoning_summary": "Strong trend."}'
    mock_completion.return_value = mock_response

    trader = HeadTrader()
    analyst_res = AnalystOutput(
        market_bias="bullish",
        volume_confirmation="confirmed",
        thesis="Strong breakout",
        nearest_support=95.0,
        nearest_resistance=105.0,
        confluence_score=0.8
    )
    proposal = trader.decide("BTC/USDT", analyst_res, {}, [])
    assert proposal["action"] == "long"
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "gemini/gemini-3.1-flash-lite"


@patch.dict("os.environ", {}, clear=True)
def test_default_env_load():
    # If no env vars are defined, initializing LLMClient should raise ValueError about GEMINI_API_KEY
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
        LLMClient()


from vibe_trading.agents.tools import ANALYST_TOOLS

def test_analyst_tools_schema_shape():
    """Verify ANALYST_TOOLS exposes the six expected tools in OpenAI function-calling format."""
    assert isinstance(ANALYST_TOOLS, list)
    assert len(ANALYST_TOOLS) == 6

    names = {t["function"]["name"] for t in ANALYST_TOOLS}
    assert names == {
        "get_candles",
        "get_indicators",
        "get_support_resistance",
        "get_candlestick_patterns",
        "get_derivatives",
        "get_market_sentiment",
    }

    for tool in ANALYST_TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]

from vibe_trading.agents.tools import ToolExecutor

def _make_executor():
    """Build a ToolExecutor with mocked Database and DataFetcher."""
    db = MagicMock()
    fetcher = MagicMock()
    return ToolExecutor(db=db, fetcher=fetcher), db, fetcher

def test_tool_executor_dispatch():
    """Every tool name routes to a callable handler in the dispatch table."""
    executor, _, _ = _make_executor()
    expected = {
        "get_candles",
        "get_indicators",
        "get_support_resistance",
        "get_candlestick_patterns",
        "get_derivatives",
        "get_market_sentiment",
    }
    assert set(executor._dispatch.keys()) == expected
    for name, handler in executor._dispatch.items():
        assert callable(handler), f"{name} handler must be callable"

def test_tool_executor_unknown_tool():
    """Unknown tool names return a structured error JSON without raising."""
    executor, _, _ = _make_executor()
    result = executor.execute("not_a_real_tool", {})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "Unknown tool" in parsed["error"]

def test_tool_executor_exception_handling():
    """Handler exceptions are caught and returned as error JSON."""
    executor, _, _ = _make_executor()
    def boom(**kwargs):
        raise RuntimeError("simulated DB failure")
    executor._dispatch["get_candles"] = boom

    result = executor.execute("get_candles", {"symbol": "BTC/USDT", "timeframe": "4h"})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "Tool execution failed" in parsed["error"]
    assert "simulated DB failure" in parsed["error"]


def test_get_market_sentiment():
    """Mock the Fear & Greed Index HTTP endpoint; verify parsing."""
    executor, _, _ = _make_executor()

    fake_payload = json.dumps({
        "data": [
            {"value": "72", "value_classification": "Greed", "timestamp": "1700000000"}
        ]
    }).encode("utf-8")

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse(fake_payload)) as mock_urlopen:
        result_str = executor.execute("get_market_sentiment", {})

    parsed = json.loads(result_str)
    assert parsed["value"] == 72
    assert parsed["classification"] == "Greed"
    assert parsed["timestamp"] == "1700000000"
    # Verify the correct URL was requested
    call_args = mock_urlopen.call_args
    request_obj = call_args[0][0]
    assert "api.alternative.me/fng" in request_obj.full_url


import pandas as pd
from datetime import datetime

def test_get_candles_clamps_limit_and_returns_records():
    """Handler clamps limit to 50, uses pinned timestamp, returns row dicts as JSON."""
    executor, _, _ = _make_executor()
    pinned_ts = datetime(2026, 5, 27, 12, 0, 0)
    executor.set_timestamp(pinned_ts)

    fake_df = pd.DataFrame([
        {"timestamp": datetime(2026, 5, 27, 8), "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1234.0},
        {"timestamp": datetime(2026, 5, 27, 12), "open": 104.0, "high": 108.0, "low": 103.0, "close": 107.0, "volume": 2345.0},
    ])
    executor.pipeline._get_candles = MagicMock(return_value=fake_df)

    # Request limit=999 -> should clamp to 50 in the call to pipeline._get_candles
    result_str = executor.execute("get_candles", {"symbol": "BTC/USDT", "timeframe": "4h", "limit": 999})
    parsed = json.loads(result_str)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[1]["close"] == 107.0

    # Verify clamping + timestamp pin
    call_args = executor.pipeline._get_candles.call_args
    assert call_args[0][0] == "BTC/USDT"
    assert call_args[0][1] == "4h"
    assert call_args[0][2] == pinned_ts
    assert call_args[1]["limit"] == 50


def test_get_indicators_returns_latest_with_regimes():
    """Handler queries 300 candles, runs the indicator pipeline, returns latest row + regimes."""
    executor, _, _ = _make_executor()
    executor.set_timestamp(datetime(2026, 5, 27, 12))

    # Build a candle df that already has indicator columns populated (mocking _calculate_indicators)
    n = 60
    fake_raw = pd.DataFrame({
        "timestamp": [datetime(2026, 5, 27) + pd.Timedelta(hours=4 * i) for i in range(n)],
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })
    fake_feats = fake_raw.copy()
    fake_feats["rsi_14"] = [50.0] * (n - 4) + [55.0, 65.0, 70.0, 72.0]
    fake_feats["macd"] = [0.1] * n
    fake_feats["macd_signal"] = [0.05] * n
    fake_feats["macd_hist"] = [0.05] * (n - 4) + [0.06, 0.07, 0.075, 0.08]
    fake_feats["adx_14"] = [20.0] * (n - 4) + [22.0, 24.0, 26.0, 28.0]
    fake_feats["obv"] = [1000.0 + i * 5 for i in range(n - 4)] + [1000.0, 1100.0, 1200.0, 1300.0]
    fake_feats["ma20"] = [100.0] * n
    fake_feats["ma50"] = [99.0] * n
    fake_feats["ma200"] = [98.0] * n

    executor.pipeline._get_candles = MagicMock(return_value=fake_raw)
    executor.pipeline._calculate_indicators = MagicMock(return_value=fake_feats)

    result_str = executor.execute("get_indicators", {"symbol": "BTC/USDT", "timeframe": "4h"})
    parsed = json.loads(result_str)

    assert parsed["rsi_14"] == 72.0
    assert parsed["rsi_regime"] == "overbought"  # >= 70
    assert parsed["macd_hist"] == 0.08
    assert parsed["macd_regime"] == "bullish_momentum_expanding"
    assert parsed["adx_regime"] == "strong_trend"  # >= 25
    assert "obv_trend" in parsed
    assert parsed["ma20"] == 100.0

    # _get_candles called with limit=300
    call_kwargs = executor.pipeline._get_candles.call_args[1]
    assert call_kwargs["limit"] == 300

def test_get_indicators_returns_error_when_insufficient_candles():
    """If fewer than 50 candles available, return an error dict (not crash)."""
    executor, _, _ = _make_executor()
    executor.pipeline._get_candles = MagicMock(return_value=pd.DataFrame())

    result_str = executor.execute("get_indicators", {"symbol": "BTC/USDT", "timeframe": "4h"})
    parsed = json.loads(result_str)
    assert "error" in parsed


def test_get_support_resistance_returns_levels_and_proximity():
    """Handler runs scipy-based S/R detection plus proximity calculations."""
    executor, _, _ = _make_executor()
    executor.set_timestamp(datetime(2026, 5, 27, 12))

    fake_df = pd.DataFrame({
        "timestamp": [datetime(2026, 5, 27, h) for h in range(0, 16, 4)],
        "open": [100.0] * 4,
        "high": [105.0] * 4,
        "low": [95.0] * 4,
        "close": [100.0, 101.0, 102.0, 103.0],
        "volume": [1000.0] * 4,
    })
    executor.pipeline._get_candles = MagicMock(return_value=fake_df)
    executor.pipeline._detect_support_resistance = MagicMock(
        return_value={"supports": [95.0], "resistances": [110.0]}
    )

    result_str = executor.execute("get_support_resistance", {"symbol": "BTC/USDT"})
    parsed = json.loads(result_str)

    assert parsed["current_price"] == 103.0
    assert parsed["support_price"] == 95.0
    assert parsed["resistance_price"] == 110.0
    assert "support_proximity" in parsed
    assert "resistance_proximity" in parsed

    # Calls _get_candles for 4h with limit=300
    call_args = executor.pipeline._get_candles.call_args
    assert call_args[0][1] == "4h"
    assert call_args[1]["limit"] == 300


def test_get_candlestick_patterns_returns_pattern_string():
    """Handler invokes TA-Lib pattern recognition on the last 30 4h candles."""
    executor, _, _ = _make_executor()
    executor.set_timestamp(datetime(2026, 5, 27, 12))

    fake_df = pd.DataFrame({
        "timestamp": [datetime(2026, 5, 27, h) for h in range(0, 20, 4)],
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0] * 5,
        "low": [99.0] * 5,
        "close": [104.0] * 5,
        "volume": [1000.0] * 5,
    })
    executor.pipeline._get_candles = MagicMock(return_value=fake_df)
    executor.pipeline._recognize_candlesticks = MagicMock(return_value="engulfing_bullish, hammer_bullish")

    result_str = executor.execute("get_candlestick_patterns", {"symbol": "BTC/USDT"})
    parsed = json.loads(result_str)

    assert parsed["pattern"] == "engulfing_bullish, hammer_bullish"

    # Calls _get_candles for 4h with limit=30
    call_args = executor.pipeline._get_candles.call_args
    assert call_args[0][1] == "4h"
    assert call_args[1]["limit"] == 30

def test_get_candlestick_patterns_returns_none_when_no_data():
    """Empty candle DF -> pattern 'none'."""
    executor, _, _ = _make_executor()
    executor.pipeline._get_candles = MagicMock(return_value=pd.DataFrame())

    result_str = executor.execute("get_candlestick_patterns", {"symbol": "BTC/USDT"})
    parsed = json.loads(result_str)
    assert parsed["pattern"] == "none"


def test_get_derivatives_delegates_to_fetcher():
    """Handler is a thin pass-through to DataFetcher.fetch_funding_rate_and_oi."""
    executor, _, fetcher = _make_executor()
    fetcher.fetch_funding_rate_and_oi.return_value = {
        "funding_rate": "0.0123% (neutral)",
        "open_interest_trend": "5,000,000 USD value (active)",
    }

    result_str = executor.execute("get_derivatives", {"symbol": "BTC/USDT"})
    parsed = json.loads(result_str)

    assert parsed["funding_rate"] == "0.0123% (neutral)"
    assert parsed["open_interest_trend"] == "5,000,000 USD value (active)"
    fetcher.fetch_funding_rate_and_oi.assert_called_once_with("BTC/USDT")


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_call_llm_with_tools_single_turn(mock_completion):
    """LLM returns tool_calls once, then a final content message — loop runs twice and returns content."""
    # Turn 1: model asks for one tool call
    msg1 = MagicMock()
    msg1.tool_calls = [MagicMock()]
    msg1.tool_calls[0].id = "call_1"
    msg1.tool_calls[0].function.name = "get_market_sentiment"
    msg1.tool_calls[0].function.arguments = "{}"

    # Turn 2: model returns final JSON content, no tool calls
    msg2 = MagicMock()
    msg2.tool_calls = None
    msg2.content = '{"market_bias": "bullish", "volume_confirmation": "confirmed", "thesis": "ok", "nearest_support": 1.0, "nearest_resistance": 2.0, "confluence_score": 0.5}'

    resp1 = MagicMock(); resp1.choices = [MagicMock(message=msg1)]
    resp2 = MagicMock(); resp2.choices = [MagicMock(message=msg2)]
    mock_completion.side_effect = [resp1, resp2]

    tool_executor = MagicMock()
    tool_executor.execute.return_value = '{"value": 70, "classification": "Greed"}'

    client = LLMClient()
    result = client.call_llm_with_tools(
        model_name="test-model",
        system_instruction="sys",
        prompt="usr",
        tools=[{"type": "function", "function": {"name": "get_market_sentiment", "parameters": {}}}],
        tool_executor=tool_executor,
    )

    assert "bullish" in result
    assert mock_completion.call_count == 2
    tool_executor.execute.assert_called_once_with("get_market_sentiment", {})


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_call_llm_with_tools_multi_turn(mock_completion):
    """Two consecutive turns with tool_calls, then a final content message."""
    # Turn 1: call get_candles
    msg1 = MagicMock()
    msg1.tool_calls = [MagicMock()]
    msg1.tool_calls[0].id = "call_1"
    msg1.tool_calls[0].function.name = "get_candles"
    msg1.tool_calls[0].function.arguments = '{"symbol": "BTC/USDT", "timeframe": "4h"}'

    # Turn 2: call get_indicators
    msg2 = MagicMock()
    msg2.tool_calls = [MagicMock()]
    msg2.tool_calls[0].id = "call_2"
    msg2.tool_calls[0].function.name = "get_indicators"
    msg2.tool_calls[0].function.arguments = '{"symbol": "BTC/USDT", "timeframe": "4h"}'

    # Turn 3: final answer
    msg3 = MagicMock()
    msg3.tool_calls = None
    msg3.content = '{"market_bias": "neutral"}'

    mock_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)]),
        MagicMock(choices=[MagicMock(message=msg3)]),
    ]

    tool_executor = MagicMock()
    tool_executor.execute.side_effect = ['{"candles": []}', '{"rsi_14": 50}']

    client = LLMClient()
    result = client.call_llm_with_tools(
        model_name="m",
        system_instruction="sys",
        prompt="usr",
        tools=[],
        tool_executor=tool_executor,
    )

    assert "neutral" in result
    assert mock_completion.call_count == 3
    assert tool_executor.execute.call_count == 2
    # Verify order of tool calls
    assert tool_executor.execute.call_args_list[0][0][0] == "get_candles"
    assert tool_executor.execute.call_args_list[1][0][0] == "get_indicators"


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_call_llm_with_tools_max_iterations(mock_completion):
    """If the LLM never stops requesting tool calls, RuntimeError is raised."""
    # Always return tool_calls (never a final content)
    def looping_response(*args, **kwargs):
        msg = MagicMock()
        tc = MagicMock()
        tc.id = "looped"
        tc.function.name = "get_market_sentiment"
        tc.function.arguments = "{}"
        msg.tool_calls = [tc]
        msg.content = None
        return MagicMock(choices=[MagicMock(message=msg)])
    mock_completion.side_effect = looping_response

    tool_executor = MagicMock()
    tool_executor.execute.return_value = '{"value": 50}'

    client = LLMClient()
    with pytest.raises(RuntimeError, match="exceeded max tool-call iterations"):
        client.call_llm_with_tools(
            model_name="m",
            system_instruction="sys",
            prompt="usr",
            tools=[],
            tool_executor=tool_executor,
            max_iterations=3,
        )

    assert mock_completion.call_count == 3
    assert tool_executor.execute.call_count == 3


def _valid_analyst_json():
    return ('{"market_bias": "bullish", "volume_confirmation": "confirmed", '
            '"thesis": "ok", "nearest_support": 95.0, "nearest_resistance": 105.0, '
            '"confluence_score": 0.7}')

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_tool_use_integration():
    """When db+fetcher are provided, analyze(symbol, timestamp) uses the tool-loop path."""
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
    mock_client.call_llm_with_tools.return_value = _valid_analyst_json()

    db = MagicMock()
    fetcher = MagicMock()
    analyst = TechnicalVolumeAnalyst(client=mock_client, db=db, fetcher=fetcher)

    ts = datetime(2026, 5, 27, 12)
    result = analyst.analyze(symbol="BTC/USDT", timestamp=ts)

    assert isinstance(result, AnalystOutput)
    assert result.market_bias == "bullish"

    # Verify the tool-loop method was invoked (not the legacy call_llm)
    mock_client.call_llm_with_tools.assert_called_once()
    mock_client.call_llm.assert_not_called()

    # Verify the tool_executor was pinned to the analyst's timestamp before invocation
    assert analyst.tool_executor.current_timestamp == ts

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_legacy_snapshot_fallback():
    """When no db/fetcher are provided, analyze(snapshot=...) uses the legacy call_llm path."""
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
    mock_client.call_llm.return_value = _valid_analyst_json()

    analyst = TechnicalVolumeAnalyst(client=mock_client)
    snapshot = {"symbol": "BTC/USDT", "rsi_14": 55.0}
    result = analyst.analyze(symbol="BTC/USDT", snapshot=snapshot)

    assert isinstance(result, AnalystOutput)
    assert result.market_bias == "bullish"
    mock_client.call_llm.assert_called_once()
    mock_client.call_llm_with_tools.assert_not_called()


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_call_llm_with_tools_malformed_args(mock_completion):
    """Malformed JSON in tool_call.function.arguments yields an error tool result, loop continues."""
    msg1 = MagicMock()
    msg1.tool_calls = [MagicMock()]
    msg1.tool_calls[0].id = "call_bad"
    msg1.tool_calls[0].function.name = "get_candles"
    msg1.tool_calls[0].function.arguments = "{not valid json"

    msg2 = MagicMock()
    msg2.tool_calls = None
    msg2.content = '{"market_bias": "neutral"}'

    mock_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2)]),
    ]

    tool_executor = MagicMock()
    client = LLMClient()
    result = client.call_llm_with_tools(
        model_name="m",
        system_instruction="sys",
        prompt="usr",
        tools=[],
        tool_executor=tool_executor,
    )

    # Loop completed; executor was never invoked (args failed to parse)
    assert "neutral" in result
    tool_executor.execute.assert_not_called()
    # Second LLM call must include a tool message with an error payload
    second_call_messages = mock_completion.call_args_list[1][1]["messages"]
    tool_msgs = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    parsed = json.loads(tool_msgs[0]["content"])
    assert "error" in parsed
    assert "Malformed tool arguments" in parsed["error"]
