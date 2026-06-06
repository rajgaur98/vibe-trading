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


def test_submit_order_long_places_entry_and_brackets():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "success"
    assert res["entry_price"] == 100.0
    ex.set_leverage.assert_called_once_with(1, "BTC/USDT:USDT")

    # Three orders: market entry, TAKE_PROFIT_MARKET, STOP_MARKET
    calls = ex.create_order.call_args_list
    assert len(calls) == 3

    # 1) market BUY of size_usd/mark = 1000/100 = 10.0 (precision-rounded)
    a0 = calls[0]
    assert a0.args[0] == "BTC/USDT:USDT"
    assert a0.args[1] == "market"
    assert a0.args[2] == "buy"
    assert float(a0.args[3]) == 10.0

    # 2) TAKE_PROFIT_MARKET SELL closePosition @ tp
    a1 = calls[1]
    assert a1.args[1] == "TAKE_PROFIT_MARKET"
    assert a1.args[2] == "sell"
    assert a1.kwargs["params"]["closePosition"] is True
    assert float(a1.kwargs["params"]["stopPrice"]) == 110.0

    # 3) STOP_MARKET SELL closePosition @ sl
    a2 = calls[2]
    assert a2.args[1] == "STOP_MARKET"
    assert a2.args[2] == "sell"
    assert a2.kwargs["params"]["closePosition"] is True
    assert float(a2.kwargs["params"]["stopPrice"]) == 95.0


def test_submit_order_short_flips_sides():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker.submit_order(
        symbol="ETH/USDT", action="short", size_usd=1000.0,
        stop_price=110.0, take_profit_price=90.0, entry_price=100.0,
    )
    calls = ex.create_order.call_args_list
    assert calls[0].args[2] == "sell"   # entry SELL for a short
    assert calls[1].args[1] == "TAKE_PROFIT_MARKET"
    assert calls[1].args[2] == "buy"    # exit side BUY
    assert calls[2].args[1] == "STOP_MARKET"
    assert calls[2].args[2] == "buy"


def test_submit_order_rejects_below_min_notional():
    ex = _mock_exchange()
    ex.market.return_value = {"limits": {"cost": {"min": 5000.0}, "amount": {"min": 0.0001}}}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=100.0,  # notional 100 < min 5000
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "rejected"
    assert "minimum" in res["reason"]
    ex.create_order.assert_not_called()  # no entry order placed


def test_submit_order_dry_run_places_nothing(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "true")
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "dry_run"
    ex.create_order.assert_not_called()


def test_submit_order_rounds_via_precision_helpers():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.123456, take_profit_price=110.987654, entry_price=100.0,
    )
    # stopPrice on the bracket orders must come from price_to_precision (2dp here)
    calls = ex.create_order.call_args_list
    assert calls[1].kwargs["params"]["stopPrice"] == "110.99"
    assert calls[2].kwargs["params"]["stopPrice"] == "95.12"
    ex.amount_to_precision.assert_called()  # qty rounded too


def test_get_mark_price_returns_last():
    ex = _mock_exchange()
    ex.fetch_ticker.return_value = {"last": 123.45}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_mark_price("BTC/USDT") == 123.45
    ex.fetch_ticker.assert_called_with("BTC/USDT:USDT")


def test_get_mark_price_none_on_error():
    ex = _mock_exchange()
    ex.fetch_ticker.side_effect = Exception("network down")
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_mark_price("BTC/USDT") is None


def test_get_balance_dry_run_is_10000(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "true")
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_balance() == 10000.0
    assert broker.peak_balance == 10000.0  # peak tracked in-memory


def test_get_balance_reads_usdt_total_and_tracks_peak():
    ex = _mock_exchange()
    ex.fetch_balance.return_value = {"USDT": {"total": 8500.0}}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_balance() == 8500.0
    assert broker.peak_balance == 8500.0
    # balance drops; peak holds
    ex.fetch_balance.return_value = {"USDT": {"total": 8000.0}}
    assert broker.get_balance() == 8000.0
    assert broker.peak_balance == 8500.0


def test_get_open_positions_maps_exchange_and_brackets():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long",
         "entryPrice": 100.0, "notional": 50.0, "markPrice": 105.0},
        {"symbol": "DOGE/USDT:USDT", "contracts": 0.0},  # flat → skipped
    ]
    ex.fetch_open_orders.return_value = [
        {"type": "stop_market", "stopPrice": 95.0},
        {"type": "take_profit_market", "stopPrice": 110.0},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    positions = broker.get_open_positions()

    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "BTC/USDT"          # un-converted (plain)
    assert p["side"] == "long"
    assert p["entry_price"] == 100.0
    assert p["size_usd"] == 50.0
    assert p["stop_price"] == 95.0
    assert p["take_profit_price"] == 110.0
    assert p["current_price"] == 105.0


def test_get_open_positions_empty_on_error():
    ex = _mock_exchange()
    ex.fetch_positions.side_effect = Exception("boom")
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_open_positions() == []
