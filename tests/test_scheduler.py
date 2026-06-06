"""Tests for the scheduler's exec-price resolution (the only new pure logic).
The full sync_and_evaluate loop is network-heavy and covered by manual verification."""
from datetime import datetime
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


def test_record_closed_trades_inserts_and_alerts(monkeypatch):
    sched = _scheduler_without_init()
    fake_conn = MagicMock()
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    factory = MagicMock(return_value=fake_pg)
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    trades = [{
        "trade_id": "t1", "symbol": "BTC/USDT", "action": "long",
        "entry_time": datetime(2026, 6, 1), "entry_price": 100.0,
        "close_time": datetime(2026, 6, 2), "close_price": 110.0,
        "size_usd": 1000.0, "realized_pnl": 99.6, "result": "win",
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1            # one INSERT
    assert fake_pg.connect.called and fake_pg.close.called  # own connection lifecycle
    assert len(alerts) == 1 and "BTC/USDT" in alerts[0]


def test_record_closed_trades_empty_is_noop(monkeypatch):
    sched = _scheduler_without_init()
    factory = MagicMock()
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    sched._record_closed_trades([])
    assert factory.call_count == 0  # no connection opened
    assert alerts == []
