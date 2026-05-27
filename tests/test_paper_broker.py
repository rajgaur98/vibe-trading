"""Tests for PaperBroker entry-price handling.

Covers the fix where submit_order accepts an optional entry_price so the live
paper-trading path can mark fills at submission time (using the current market
price computed by RiskManager), while the backtester keeps its delayed-fill
semantic by not passing entry_price (default 0.0 → lazy-fill on next tick).
"""

from vibe_trading.brokers.paper import PaperBroker


def test_submit_order_with_entry_price_sets_it_immediately():
    """Live path: passing entry_price means the position is filled at submission."""
    broker = PaperBroker(initial_balance=10000.0, db=None)
    res = broker.submit_order(
        symbol="BTC/USDT",
        action="long",
        size_usd=100.0,
        stop_price=50000.0,
        take_profit_price=55000.0,
        entry_price=52000.0,
    )
    assert res["status"] == "success"
    pos = broker.get_open_positions()[0]
    assert pos["symbol"] == "BTC/USDT"
    assert pos["entry_price"] == 52000.0  # filled immediately, not deferred


def test_submit_order_without_entry_price_defaults_to_lazy_fill():
    """Backtest path: omitting entry_price keeps the existing 0.0 placeholder + lazy-fill."""
    broker = PaperBroker(initial_balance=10000.0, db=None)
    broker.submit_order(
        symbol="ETH/USDT",
        action="short",
        size_usd=100.0,
        stop_price=2200.0,
        take_profit_price=1800.0,
    )
    pos = broker.get_open_positions()[0]
    assert pos["entry_price"] == 0.0  # placeholder, lazy-fill happens on first update_positions

    # First update_positions tick: no SL/TP hit, but entry_price is filled in
    closed = broker.update_positions({"ETH/USDT": 2000.0})
    assert closed == []
    assert broker.positions["ETH/USDT"]["entry_price"] == 2000.0


def test_submit_order_filled_position_pnl_math_uses_immediate_entry():
    """When entry_price is set at submit time, the next update tick should be able to
    resolve SL/TP without doing the lazy-fill detour."""
    broker = PaperBroker(initial_balance=10000.0, db=None)
    broker.submit_order(
        symbol="SOL/USDT",
        action="long",
        size_usd=100.0,
        stop_price=90.0,
        take_profit_price=110.0,
        entry_price=100.0,
    )
    # Price ticks to take-profit on the very next update — entry was already set
    closed = broker.update_positions({"SOL/USDT": 110.0})
    assert len(closed) == 1
    trade = closed[0]
    assert trade["symbol"] == "SOL/USDT"
    assert trade["entry_price"] == 100.0
    assert trade["close_price"] == 110.0
    assert trade["result"] == "win"
