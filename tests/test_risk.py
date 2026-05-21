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
