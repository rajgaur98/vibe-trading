"""Unit tests for the User Data Stream listener. No real websocket / asyncio / network:
the ccxt.pro client is injected via build_client, and _handle_orders/_is_exit_fill are
sync and tested directly."""
import time
from unittest.mock import MagicMock

from vibe_trading.runtime.ws_listener import _is_exit_fill, UserDataStreamListener


def test_is_exit_fill_filled_reduce_only_true():
    assert _is_exit_fill({"status": "closed", "reduceOnly": True, "type": "market"}) is True


def test_is_exit_fill_filled_bracket_type_true():
    assert _is_exit_fill({"status": "filled", "type": "take_profit_market"}) is True
    assert _is_exit_fill({"status": "closed", "type": "stop_market"}) is True


def test_is_exit_fill_filled_entry_false():
    # a filled non-reduce-only market entry is NOT an exit
    assert _is_exit_fill({"status": "closed", "reduceOnly": False, "type": "market"}) is False


def test_is_exit_fill_open_bracket_false():
    # an unfilled (resting) bracket order is not a fill
    assert _is_exit_fill({"status": "open", "type": "stop_market"}) is False


def _listener(broker, record_fn):
    return UserDataStreamListener(broker, record_fn, build_client=lambda: MagicMock())


def test_handle_orders_exit_fill_triggers_reconcile_and_records():
    broker = MagicMock()
    broker.update_positions.return_value = [{"symbol": "BTC/USDT", "realized_pnl": 5.0}]
    recorded = []
    listener = _listener(broker, recorded.append)

    listener._handle_orders([{"status": "closed", "reduceOnly": True, "type": "take_profit_market"}])

    broker.update_positions.assert_called_once_with({})
    assert recorded == [[{"symbol": "BTC/USDT", "realized_pnl": 5.0}]]


def test_handle_orders_non_exit_does_nothing():
    broker = MagicMock()
    listener = _listener(broker, lambda c: None)
    listener._handle_orders([{"status": "open", "type": "limit"}])
    broker.update_positions.assert_not_called()


def test_handle_orders_exit_fill_no_closed_trades_skips_record():
    broker = MagicMock()
    broker.update_positions.return_value = []  # reconcile found nothing newly closed
    recorded = []
    listener = _listener(broker, recorded.append)
    listener._handle_orders([{"status": "closed", "reduceOnly": True}])
    broker.update_positions.assert_called_once_with({})
    assert recorded == []  # nothing to record


def test_reconcile_and_record_swallows_broker_error():
    broker = MagicMock()
    broker.update_positions.side_effect = Exception("boom")
    recorded = []
    listener = _listener(broker, recorded.append)
    listener._reconcile_and_record()  # must not raise
    assert recorded == []


def test_start_is_idempotent_and_stop_clears_running(monkeypatch):
    broker = MagicMock()
    listener = _listener(broker, lambda c: None)

    async def _noop():
        return  # don't open a real websocket

    monkeypatch.setattr(listener, "_run", _noop)

    listener.start()
    assert listener._running is True
    first_thread = listener._thread
    assert first_thread is not None

    listener.start()  # idempotent: must NOT spawn a second thread
    assert listener._thread is first_thread

    listener.stop()
    assert listener._running is False
    time.sleep(0.05)  # let the daemon thread wind down


def test_default_client_routes_to_demo():
    # Construction is offline (no load_markets); just assert the demo routing is applied.
    ex = UserDataStreamListener._default_client()
    assert "demo-fapi.binance.com" in ex.urls["api"]["fapiPrivate"]
    assert ex.urls["api"]["ws"]["future"] == "wss://demo-fstream.binance.com/ws"
