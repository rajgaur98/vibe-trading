"""Tests for the scheduler's exec-price resolution (the only new pure logic).
The full sync_and_evaluate loop is network-heavy and covered by manual verification."""
from unittest.mock import MagicMock

from vibe_trading.runtime.scheduler import TradingScheduler


def _scheduler_without_init():
    """Build a TradingScheduler instance without running __init__ (which needs DBs/LLM)."""
    return TradingScheduler.__new__(TradingScheduler)


def test_resolve_exec_price_uses_broker_mark_when_available():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = 250.0
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 250.0


def test_resolve_exec_price_falls_back_when_mark_none():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = None
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0


def test_resolve_exec_price_falls_back_on_broker_error():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.side_effect = Exception("boom")
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0
