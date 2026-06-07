from unittest.mock import MagicMock
from types import SimpleNamespace

from vibe_trading.agents.trader import HeadTrader
from vibe_trading.journal import Precedent


def _analyst():
    return SimpleNamespace(model_dump=lambda: {"market_bias": "bullish"})


def _client_returning(content):
    c = MagicMock()
    c.provider = "gemini"
    c.model = "gemma-4-31b-it"
    c.call_llm.return_value = content
    return c


_VALID = ('{"action":"flat","stop_loss_strategy":"1.5_atr","take_profit_strategy":"3.0_atr",'
          '"risk_reward_ratio":2.0,"hold_period_bias":"medium","reasoning_summary":"no edge"}')


def test_decide_injects_precedents_into_prompt():
    client = _client_returning(_VALID)
    trader = HeadTrader(client=client)
    precedents = [Precedent("SOL/USDT", "long", "2026-05-01", 0.91, "closed", 2.3,
                            "traded long -> win +2.3%")]
    trader.decide("BTC/USDT", _analyst(), {"accuracy": 0.5}, [], current_price=100.0,
                  precedents=precedents)
    prompt = client.call_llm.call_args.kwargs["prompt"]
    assert "PRECEDENTS" in prompt
    assert "traded long -> win +2.3%" in prompt


def test_decide_without_precedents_omits_block():
    client = _client_returning(_VALID)
    trader = HeadTrader(client=client)
    trader.decide("BTC/USDT", _analyst(), {"accuracy": 0.5}, [], current_price=100.0)
    prompt = client.call_llm.call_args.kwargs["prompt"]
    assert "PRECEDENTS" not in prompt
