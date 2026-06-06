"""Unit tests for BinanceFuturesBroker. A MagicMock ccxt exchange is injected via the
constructor's `exchange=` param, so no live network calls happen in pytest.
"""
import os
from unittest.mock import MagicMock

import pytest

from vibe_trading.brokers.binance_futures import BinanceFuturesBroker, _to_ccxt_symbol


def _mock_exchange():
    """A ccxt-like mock with sensible defaults for the happy path."""
    ex = MagicMock()
    # precision helpers echo their input (as ccxt does, but returning a string)
    ex.amount_to_precision.side_effect = lambda sym, x: f"{float(x):.6f}"
    ex.price_to_precision.side_effect = lambda sym, x: f"{float(x):.2f}"
    # generous limits so the happy path is not rejected
    ex.market.return_value = {"limits": {"cost": {"min": 5.0}, "amount": {"min": 0.0001}}}
    ex.fetch_ticker.return_value = {"last": 100.0}
    # market entry fills at avg 100.0
    ex.create_order.return_value = {"id": "x1", "average": 100.0, "price": 100.0}
    return ex


def test_to_ccxt_symbol():
    assert _to_ccxt_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert _to_ccxt_symbol("ETH/USDT") == "ETH/USDT:USDT"


def test_init_injected_exchange_does_not_touch_network():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.exchange is ex
    ex.set_sandbox_mode.assert_not_called()  # injection path skips real setup
    ex.load_markets.assert_not_called()


def test_init_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "false")
    with pytest.raises(ValueError, match="BINANCE_TESTNET_API_KEY"):
        BinanceFuturesBroker(db=None)  # no injection → real path → creds required
