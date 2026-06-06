"""Unit tests for the User Data Stream listener. No real websocket / asyncio / network:
the ccxt.pro client is injected via build_client, and _handle_orders/_is_exit_fill are
sync and tested directly."""
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
