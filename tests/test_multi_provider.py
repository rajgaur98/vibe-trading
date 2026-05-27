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



