import json
import pytest
from unittest.mock import patch, MagicMock
from vibe_trading.agents.client import LLMClient, get_litellm_model_string

def test_get_litellm_model_string():
    assert get_litellm_model_string("gemini", "gemini-3.1-flash-lite") == "gemini/gemini-3.1-flash-lite"
    assert get_litellm_model_string("openai", "gpt-4o") == "openai/gpt-4o"
    assert get_litellm_model_string("anthropic", "claude-3") == "anthropic/claude-3"
    assert get_litellm_model_string("other", "model") == "model"

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_llm_client_initialization_gemini():
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

from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput

@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
def test_analyst_integration():
    mock_client = MagicMock()
    mock_client.provider = "gemini"
    mock_client.model = "gemini-3.1-flash-lite"
    # Mock the raw json response matching AnalystOutput schema
    mock_client.call_llm.return_value = '{"market_bias": "bullish", "volume_confirmation": "confirmed", "thesis": "Strong breakout on high volume.", "nearest_support": 95.0, "nearest_resistance": 105.0, "confluence_score": 0.8}'
    
    analyst = TechnicalVolumeAnalyst(client=mock_client)
    snapshot = {"symbol": "BTC/USDT"}
    res = analyst.analyze(snapshot)
    
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
    res = analyst.analyze({"symbol": "BTC/USDT"})
    
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
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
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


from io import BytesIO

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
