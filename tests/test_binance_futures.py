"""Unit tests for BinanceFuturesBroker. A MagicMock ccxt exchange is injected via the
constructor's `exchange=` param, so no live network calls happen in pytest.
"""
import os
from unittest.mock import MagicMock

import pytest

from datetime import datetime as _dt

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
    # default: a position is already open so submit_order's _await_position poll returns
    # immediately (tests that exercise reconcile/close override this explicitly).
    ex.fetch_positions.return_value = [{"contracts": 1.0}]
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


def test_submit_order_rolls_back_entry_if_brackets_fail():
    ex = _mock_exchange()
    # entry succeeds; the TP bracket raises → we must roll back the naked entry.
    def _ce(*args, **kwargs):
        if args[1] == "TAKE_PROFIT_MARKET":
            raise Exception("bracket boom")
        return {"id": "o", "average": 100.0, "price": 100.0}
    ex.create_order.side_effect = _ce
    ex.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long"}]

    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "rejected"
    assert "bracket" in res["reason"]
    # rollback issued a reduce-only market close + cancelled leftover orders
    ex.cancel_all_orders.assert_called_once_with("BTC/USDT:USDT")
    last = ex.create_order.call_args_list[-1]
    assert last.args[1] == "market" and last.args[2] == "sell"
    assert last.kwargs["params"]["reduceOnly"] is True


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
    # closePosition brackets come back as CONDITIONAL orders (type normalized to "market",
    # trigger via triggerPrice, reduceOnly). TP/SL distinguished by trigger vs entry (100).
    ex.fetch_open_orders.return_value = [
        {"type": "market", "triggerPrice": 110.0, "reduceOnly": True},  # above entry → TP
        {"type": "market", "triggerPrice": 95.0, "reduceOnly": True},   # below entry → SL
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
    # conditional orders must be fetched with the stop/trigger flag, not the plain call
    assert ex.fetch_open_orders.call_args.kwargs.get("params") == {"stop": True}


def test_get_open_positions_empty_on_error():
    ex = _mock_exchange()
    ex.fetch_positions.side_effect = Exception("boom")
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_open_positions() == []


def test_close_position_reduce_only_and_cancels_brackets():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long"},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.close_position("BTC/USDT")

    assert res["status"] == "success"
    # reduce-only market SELL of the abs contracts
    close_call = ex.create_order.call_args_list[-1]
    assert close_call.args[0] == "BTC/USDT:USDT"
    assert close_call.args[1] == "market"
    assert close_call.args[2] == "sell"
    assert float(close_call.args[3]) == 0.5
    assert close_call.kwargs["params"]["reduceOnly"] is True
    ex.cancel_all_orders.assert_called_once_with("BTC/USDT:USDT")


def test_close_position_no_position_returns_rejected():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT", "contracts": 0.0}]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.close_position("BTC/USDT")
    assert res["status"] == "rejected"


def test_update_positions_reconciles_closed_trade():
    ex = _mock_exchange()
    # Ledger says BTC is open; exchange shows it flat → it was closed by a bracket.
    ex.fetch_positions.return_value = []  # nothing open on the exchange
    ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 110.0, "amount": 10.0, "fee": {"cost": 0.4}},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    # Inject a ledger row (db is None, so _load_ledger would be empty otherwise)
    broker._load_ledger = lambda: [{
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1, 0, 0, 0),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }]

    closed = broker.update_positions({})
    assert len(closed) == 1
    t = closed[0]
    assert t["symbol"] == "BTC/USDT"
    assert t["action"] == "long"
    assert t["entry_price"] == 100.0
    assert t["close_price"] == 110.0
    # qty = 1000/100 = 10 ; pnl = (110-100)*10 - 0.4 = 99.6
    assert round(t["realized_pnl"], 2) == 99.6
    assert t["result"] == "win"
    assert {"trade_id", "close_time", "size_usd"} <= set(t.keys())


def test_update_positions_keeps_still_open_position():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5},  # still open
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker._load_ledger = lambda: [{
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }]
    assert broker.update_positions({}) == []


def test_update_positions_empty_ledger_no_calls():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)  # db None → empty ledger
    assert broker.update_positions({}) == []
    ex.fetch_positions.assert_not_called()


def test_delete_position_claim_semantics():
    # db=None → True (nothing to contend over)
    broker = BinanceFuturesBroker(db=None, exchange=_mock_exchange())
    assert broker._delete_position("BTC/USDT") is True

    # db present → returns rowcount > 0
    fake_cur = MagicMock()
    fake_cur.rowcount = 1
    fake_conn = MagicMock()
    fake_conn.execute.return_value = fake_cur
    fake_db = MagicMock()
    fake_db.conn = fake_conn
    broker2 = BinanceFuturesBroker(db=fake_db, exchange=_mock_exchange())
    assert broker2._delete_position("BTC/USDT") is True
    fake_cur.rowcount = 0
    assert broker2._delete_position("BTC/USDT") is False


def test_update_positions_idempotent_under_concurrent_claim():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = []  # symbol flat on exchange
    ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 110.0, "amount": 10.0, "fee": {"cost": 0.0}},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    row = {
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }
    broker._load_ledger = lambda: [row]
    # First reconcile claims the row (True); a racing second one loses it (False).
    claims = iter([True, False])
    broker._delete_position = lambda symbol: next(claims)

    first = broker.update_positions({})
    second = broker.update_positions({})
    assert len(first) == 1 and first[0]["symbol"] == "BTC/USDT"
    assert second == []  # not recorded twice
