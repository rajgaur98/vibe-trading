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
        "decision_id": "dec-42",
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1            # one INSERT
    assert fake_pg.connect.called and fake_pg.close.called  # own connection lifecycle
    assert len(alerts) == 1 and "BTC/USDT" in alerts[0]

    # the INSERT must link the trade back to the decision that opened it
    insert_call = fake_conn.execute.call_args
    insert_sql, insert_params = insert_call.args[0], insert_call.args[1]
    assert "decision_id" in insert_sql              # column present in the INSERT list
    assert "dec-42" in insert_params                # value threaded through


def test_record_closed_trades_decision_id_defaults_none(monkeypatch):
    """A closed trade lacking a decision_id (e.g. an orphan reconcile) must still
    INSERT cleanly — decision_id falls back to None rather than KeyError-ing."""
    sched = _scheduler_without_init()
    fake_conn = MagicMock()
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase",
                        MagicMock(return_value=fake_pg))
    sched._send_discord_alert = lambda msg: None

    trades = [{
        "trade_id": "t2", "symbol": "ETH/USDT", "action": "short",
        "entry_time": datetime(2026, 6, 1), "entry_price": 100.0,
        "close_time": datetime(2026, 6, 2), "close_price": 90.0,
        "size_usd": 500.0, "realized_pnl": 49.0, "result": "win",
        # NOTE: no "decision_id" key
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1
    insert_params = fake_conn.execute.call_args.args[1]
    assert None in insert_params  # decision_id defaulted to None, no KeyError


def test_record_closed_trades_empty_is_noop(monkeypatch):
    sched = _scheduler_without_init()
    factory = MagicMock()
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    sched._record_closed_trades([])
    assert factory.call_count == 0  # no connection opened
    assert alerts == []


def test_maybe_start_ws_listener_none_when_not_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    sched = _scheduler_without_init()
    assert sched._maybe_start_ws_listener() is None


def test_maybe_start_ws_listener_starts_in_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", lambda *a, **k: MagicMock())
    monkeypatch.setattr("vibe_trading.runtime.scheduler.BinanceFuturesBroker", lambda *a, **k: MagicMock())

    started = {}

    class FakeListener:
        def __init__(self, broker, record_fn):
            started["init"] = True

        def start(self):
            started["start"] = True

    monkeypatch.setattr("vibe_trading.runtime.ws_listener.UserDataStreamListener", FakeListener)

    listener = sched._maybe_start_ws_listener()
    assert isinstance(listener, FakeListener)
    assert started.get("init") and started.get("start")


def test_maybe_start_ws_listener_failopen_returns_none(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None

    def _boom(*a, **k):
        raise RuntimeError("no creds")

    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", _boom)
    # A listener-construction failure must NOT propagate (scheduler keeps running).
    assert sched._maybe_start_ws_listener() is None
