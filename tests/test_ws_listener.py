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
