import pytest
import pandas as pd
from decimal import Decimal
from vibe_trading.brokers.risk import RiskManager

def test_risk_manager_drawdown_circuit_breaker():
    manager = RiskManager(max_drawdown_pct=0.15)
    
    # 20% drawdown (peak 10,000, current 8,000)
    proposal = {
        "symbol": "BTC/USDT",
        "action": "long",
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "risk_reward_multiplier",
        "risk_reward_ratio": 2.0
    }
    
    # Mock candle DataFrame
    df = pd.DataFrame({
        "high": [100.0] * 20,
        "low": [98.0] * 20,
        "close": [99.0] * 20
    })
    
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}
    
    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=99.0,
        df_4h=df,
        account_balance=8000.0,
        peak_balance=10000.0,
        open_positions=[],
        snapshot=snapshot
    )
    
    assert res["approved"] is False
    assert "drawdown" in res["reason"]

def test_risk_manager_max_positions_limit():
    manager = RiskManager(max_concurrent_trades=3)
    
    # Active positions is already 3
    open_positions = [
        {"symbol": "ETH/USDT"},
        {"symbol": "SOL/USDT"},
        {"symbol": "ADA/USDT"}
    ]
    
    proposal = {
        "symbol": "BTC/USDT",
        "action": "long",
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "risk_reward_multiplier",
        "risk_reward_ratio": 2.0
    }
    
    df = pd.DataFrame({
        "high": [100.0] * 20,
        "low": [98.0] * 20,
        "close": [99.0] * 20
    })
    
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}
    
    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=99.0,
        df_4h=df,
        account_balance=10000.0,
        peak_balance=10000.0,
        open_positions=open_positions,
        snapshot=snapshot
    )
    
    assert res["approved"] is False
    assert "concurrent" in res["reason"]

def test_position_sizing_calculation():
    # Risk 1% of $10,000 = $100
    # Price = $100, Stop Loss = $95 (5% distance)
    # Position size = $100 / 0.05 = $2000
    manager = RiskManager(max_risk_per_trade_pct=0.01, maker_fee_pct=0.0, slippage_buffer_pct=0.0)
    
    proposal = {
        "symbol": "BTC/USDT",
        "action": "long",
        "stop_loss_strategy": "tight_atr",  # 1.0 ATR
        "take_profit_strategy": "risk_reward_multiplier",
        "risk_reward_ratio": 2.0
    }
    
    # Setup dataframe so ATR(14) calculation returns 5.0
    # For simplicity, we mock highs/lows/closes with clear range
    df = pd.DataFrame({
        "high": [105.0] * 20,
        "low": [100.0] * 20,
        "close": [102.5] * 20
    })
    
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}
    
    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=100.0,
        df_4h=df,
        account_balance=10000.0,
        peak_balance=10000.0,
        open_positions=[],
        snapshot=snapshot
    )
    
    assert res["approved"] is True
    # Stop price should be Entry - 1.0 * ATR. With ATR around 5, Stop price should be ~95
    assert res["stop_price"] < 100.0
    assert res["take_profit_price"] > 100.0
    assert res["size_usd"] > 0.0


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

import numpy as np
from hypothesis import given, settings, strategies as st

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df_4h(base_price: float) -> pd.DataFrame:
    """
    Build a minimal 20-row 4-h OHLC dataframe centred around *base_price*.
    The high/low spread is fixed at 2 % of base_price so talib.ATR(14) always
    converges to a clean, non-NaN value proportional to the price.
    """
    spread = base_price * 0.02
    return pd.DataFrame({
        "high":  [base_price + spread] * 20,
        "low":   [base_price - spread] * 20,
        "close": [base_price] * 20,
    })


def _base_proposal(action: str = "long") -> dict:
    """Return a minimal, valid proposal dict using ATR-based strategies."""
    return {
        "symbol": "BTC/USDT",
        "action": action,
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "risk_reward_multiplier",
        "risk_reward_ratio": 2.0,
    }


def _healthy_manager() -> RiskManager:
    """RiskManager with default caps, no fee friction (keeps sizes large enough)."""
    return RiskManager(
        max_risk_per_trade_pct=0.01,
        max_drawdown_pct=0.15,
        max_exposure_pct=0.50,
        max_concurrent_trades=5,
        maker_fee_pct=0.0,
        slippage_buffer_pct=0.0,
    )


# Reusable strategy for prices that are realistic and won't overflow Decimal
_price_st = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)
_balance_st = st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Invariant 1: Long stop/target ordering
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    current_price=_price_st,
    account_balance=_balance_st,
)
def test_prop_long_stop_target_ordering(current_price, account_balance):
    """
    For an approved LONG, stop_price < entry_price < take_profit_price.
    """
    manager = _healthy_manager()
    proposal = _base_proposal(action="long")
    df = _make_df_4h(current_price)
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}

    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=current_price,
        df_4h=df,
        account_balance=account_balance,
        peak_balance=account_balance,  # no drawdown
        open_positions=[],
        snapshot=snapshot,
    )

    if res["approved"]:
        assert res["stop_price"] < res["entry_price"], (
            f"Long: stop_price ({res['stop_price']}) must be < entry_price ({res['entry_price']})"
        )
        assert res["take_profit_price"] > res["entry_price"], (
            f"Long: take_profit ({res['take_profit_price']}) must be > entry_price ({res['entry_price']})"
        )


# ---------------------------------------------------------------------------
# Invariant 2: Short stop/target ordering
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    current_price=_price_st,
    account_balance=_balance_st,
)
def test_prop_short_stop_target_ordering(current_price, account_balance):
    """
    For an approved SHORT, stop_price > entry_price > take_profit_price.
    """
    manager = _healthy_manager()
    proposal = _base_proposal(action="short")
    df = _make_df_4h(current_price)
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}

    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=current_price,
        df_4h=df,
        account_balance=account_balance,
        peak_balance=account_balance,  # no drawdown
        open_positions=[],
        snapshot=snapshot,
    )

    if res["approved"]:
        assert res["stop_price"] > res["entry_price"], (
            f"Short: stop_price ({res['stop_price']}) must be > entry_price ({res['entry_price']})"
        )
        assert res["take_profit_price"] < res["entry_price"], (
            f"Short: take_profit ({res['take_profit_price']}) must be < entry_price ({res['entry_price']})"
        )


# ---------------------------------------------------------------------------
# Invariant 3: Exposure cap
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    current_price=_price_st,
    account_balance=_balance_st,
)
def test_prop_exposure_cap(current_price, account_balance):
    """
    size_usd must never exceed max_exposure_pct * account_balance for any approved proposal.
    """
    max_exposure_pct = 0.50
    manager = RiskManager(
        max_exposure_pct=max_exposure_pct,
        maker_fee_pct=0.0,
        slippage_buffer_pct=0.0,
    )
    proposal = _base_proposal(action="long")
    df = _make_df_4h(current_price)
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}

    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=current_price,
        df_4h=df,
        account_balance=account_balance,
        peak_balance=account_balance,
        open_positions=[],
        snapshot=snapshot,
    )

    if res["approved"]:
        max_allowed = max_exposure_pct * account_balance
        epsilon = 0.02  # allow 2-cent rounding from Decimal.quantize
        assert res["size_usd"] <= max_allowed + epsilon, (
            f"size_usd ({res['size_usd']}) exceeds cap ({max_allowed})"
        )


# ---------------------------------------------------------------------------
# Invariant 4: Drawdown circuit breaker
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    current_price=_price_st,
    # peak_balance strictly > account_balance so there is always a drawdown
    peak_balance=st.floats(min_value=1000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    drawdown_extra=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
)
def test_prop_drawdown_circuit_breaker(current_price, peak_balance, drawdown_extra):
    """
    When account_balance has dropped by at least max_drawdown_pct from peak_balance,
    evaluate_proposal must return approved=False regardless of anything else.
    """
    max_drawdown_pct = 0.15
    manager = RiskManager(max_drawdown_pct=max_drawdown_pct)

    # Force drawdown to be exactly (max_drawdown_pct + drawdown_extra)
    drawdown_fraction = max_drawdown_pct + drawdown_extra
    # Clamp so account_balance stays positive
    drawdown_fraction = min(drawdown_fraction, 0.999)
    account_balance = peak_balance * (1.0 - drawdown_fraction)

    proposal = _base_proposal(action="long")
    df = _make_df_4h(current_price)
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}

    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=current_price,
        df_4h=df,
        account_balance=account_balance,
        peak_balance=peak_balance,
        open_positions=[],
        snapshot=snapshot,
    )

    assert res["approved"] is False, (
        f"Expected circuit breaker rejection; drawdown={drawdown_fraction:.2%} "
        f"(threshold={max_drawdown_pct:.2%})"
    )


# ---------------------------------------------------------------------------
# Invariant 5: Max concurrent trades
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    current_price=_price_st,
    account_balance=_balance_st,
    extra_positions=st.integers(min_value=0, max_value=10),
)
def test_prop_max_concurrent_trades(current_price, account_balance, extra_positions):
    """
    When len(open_positions) >= max_concurrent_trades (and the symbol is new),
    evaluate_proposal must return approved=False.
    """
    max_concurrent = 5
    manager = RiskManager(max_concurrent_trades=max_concurrent)

    # Build open_positions with exactly (max_concurrent + extra_positions) entries,
    # each with a DIFFERENT symbol from the proposal's symbol.
    total_open = max_concurrent + extra_positions
    open_positions = [{"symbol": f"COIN{i}/USDT"} for i in range(total_open)]

    proposal = _base_proposal(action="long")  # symbol = "BTC/USDT", not in open_positions
    df = _make_df_4h(current_price)
    snapshot = {"support_price": 0.0, "resistance_price": 0.0}

    res = manager.evaluate_proposal(
        proposal=proposal,
        current_price=current_price,
        df_4h=df,
        account_balance=account_balance,
        peak_balance=account_balance,
        open_positions=open_positions,
        snapshot=snapshot,
    )

    assert res["approved"] is False, (
        f"Expected rejection: {total_open} open positions >= max {max_concurrent}"
    )


def test_next_resistance_tp_falls_back_to_rr_when_resistance_below_entry():
    """Breakout into price discovery: a LONG with take_profit_strategy='next_resistance'
    when the only resistance is BELOW entry must NOT floor the take-profit to ~entry.
    It should fall back to the risk/reward multiplier target (rr x stop distance above
    entry). Regression for the ALLO/USDT TP-at-entry bug."""
    manager = RiskManager()
    proposal = {
        "symbol": "ALLO/USDT",
        "action": "long",
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "next_resistance",
        "risk_reward_ratio": 2.0,
    }
    # 20 candles, constant 0.02 true range -> ATR ~0.02
    df = pd.DataFrame({"high": [0.41] * 20, "low": [0.39] * 20, "close": [0.40] * 20})
    # Resistance (0.36) is BELOW the entry (0.40) — breakout to new highs; support distant.
    snapshot = {"support_price": 0.30, "resistance_price": 0.36}

    res = manager.evaluate_proposal(
        proposal=proposal, current_price=0.40, df_4h=df,
        account_balance=10000.0, peak_balance=10000.0, open_positions=[], snapshot=snapshot,
    )

    assert res["approved"] is True
    entry = 0.40
    tp = float(res["take_profit_price"])
    stop = float(res["stop_price"])
    # TP must be a real target above entry, not the trivial floor (~entry).
    assert tp > entry + 0.5 * (entry - stop), f"TP {tp} collapsed toward entry {entry}"
    # Specifically: TP distance == rr_ratio * stop distance (the RR fallback).
    assert abs((tp - entry) - 2.0 * (entry - stop)) < 1e-6


def test_next_support_tp_falls_back_to_rr_when_support_above_entry():
    """Symmetric short case: breakdown below all support. A SHORT with 'next_resistance'
    (which targets support for shorts) when support is ABOVE entry must fall back to the
    RR multiplier, not floor the TP to ~entry."""
    manager = RiskManager()
    proposal = {
        "symbol": "ALLO/USDT",
        "action": "short",
        "stop_loss_strategy": "1.5_atr",
        "take_profit_strategy": "next_resistance",
        "risk_reward_ratio": 2.0,
    }
    df = pd.DataFrame({"high": [0.41] * 20, "low": [0.39] * 20, "close": [0.40] * 20})
    # Support (0.44) is ABOVE entry (0.40) — breakdown to new lows; resistance distant above.
    snapshot = {"support_price": 0.44, "resistance_price": 0.50}

    res = manager.evaluate_proposal(
        proposal=proposal, current_price=0.40, df_4h=df,
        account_balance=10000.0, peak_balance=10000.0, open_positions=[], snapshot=snapshot,
    )

    assert res["approved"] is True
    entry = 0.40
    tp = float(res["take_profit_price"])
    stop = float(res["stop_price"])
    assert tp < entry - 0.5 * (stop - entry), f"TP {tp} collapsed toward entry {entry}"
    assert abs((entry - tp) - 2.0 * (stop - entry)) < 1e-6
