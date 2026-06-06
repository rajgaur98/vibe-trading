import logging
from decimal import Decimal
import numpy as np
import talib
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(
        self,
        max_risk_per_trade_pct: float = 0.01,  # Risk 1% of equity per trade
        max_drawdown_pct: float = 0.15,        # 15% Max drawdown circuit breaker
        max_exposure_pct: float = 0.50,        # Max 50% account exposure per asset
        max_concurrent_trades: int = 5,         # Max 5 open positions at a time
        maker_fee_pct: float = 0.004,          # Exchange fee buffer (e.g., 0.4% Coinbase Advanced)
        slippage_buffer_pct: float = 0.001     # Slippage buffer (0.1%)
    ):
        self.max_risk_per_trade_pct = Decimal(str(max_risk_per_trade_pct))
        self.max_drawdown_pct = Decimal(str(max_drawdown_pct))
        self.max_exposure_pct = Decimal(str(max_exposure_pct))
        self.max_concurrent_trades = max_concurrent_trades
        self.fee_and_slip_buffer = Decimal(str(maker_fee_pct + slippage_buffer_pct))

    def evaluate_proposal(
        self,
        proposal: dict,
        current_price: float,
        df_4h: pd.DataFrame,
        account_balance: float,
        peak_balance: float,
        open_positions: List[Dict[str, Any]],
        snapshot: dict
    ) -> Dict[str, Any]:
        """
        Evaluates a Head Trader proposal and calculates exact stop, take-profit, and size.
        Returns a dict indicating if approved and the filled trade parameter details.
        """
        symbol = proposal["symbol"]
        action = proposal["action"]
        
        balance_dec = Decimal(str(account_balance))
        peak_dec = Decimal(str(peak_balance))

        # 1. Circuit Breaker Check (Max Drawdown)
        current_drawdown = (peak_dec - balance_dec) / peak_dec
        if current_drawdown >= self.max_drawdown_pct:
            logger.warning(f"RISK VETO: Circuit breaker triggered. Drawdown: {current_drawdown:.2%}")
            return {"approved": False, "reason": "Max drawdown circuit breaker breached", "size_usd": Decimal("0")}

        # 2. Portfolio Size Constraint (Concurrent Trades)
        active_symbols = [pos['symbol'] for pos in open_positions]
        if len(open_positions) >= self.max_concurrent_trades:
            if symbol not in active_symbols:
                logger.warning("RISK VETO: Max concurrent trades reached.")
                return {"approved": False, "reason": "Max concurrent trades reached", "size_usd": Decimal("0")}

        # 3. Calculate volatility context (ATR)
        highs = df_4h['high'].values
        lows = df_4h['low'].values
        closes = df_4h['close'].values
        
        atr_values = talib.ATR(highs, lows, closes, timeperiod=14)
        atr = atr_values[-1] if not np.isnan(atr_values[-1]) else (0.01 * current_price)
        
        # 4. Resolve Stop-Loss Price
        sl_strategy = proposal["stop_loss_strategy"]
        entry_price = Decimal(str(current_price))
        
        if sl_strategy == "1.5_atr":
            sl_dist = Decimal(str(1.5 * atr))
        elif sl_strategy == "2.0_atr":
            sl_dist = Decimal(str(2.0 * atr))
        elif sl_strategy == "tight_atr":
            sl_dist = Decimal(str(1.0 * atr))
        elif sl_strategy == "swing_low":
            # For long, use support. For short, use resistance.
            level = snapshot["support_price"] if action == "long" else snapshot["resistance_price"]
            if level > 0:
                # Add a 0.2% buffer beyond the S/R level
                buffer = Decimal(str(level)) * Decimal("0.002")
                if action == "long":
                    sl_dist = entry_price - (Decimal(str(level)) - buffer)
                else:
                    sl_dist = (Decimal(str(level)) + buffer) - entry_price
            else:
                sl_dist = Decimal(str(1.5 * atr))
        else:
            sl_dist = Decimal(str(1.5 * atr))
            
        # Ensure stop distance is positive and sane
        sl_dist = max(sl_dist, Decimal(str(0.001 * current_price)))
        
        if action == "long":
            stop_price = entry_price - sl_dist
        else:
            stop_price = entry_price + sl_dist

        # 5. Resolve Take-Profit Price
        tp_strategy = proposal["take_profit_strategy"]
        rr_ratio = Decimal(str(proposal["risk_reward_ratio"]))
        
        if tp_strategy == "risk_reward_multiplier":
            tp_dist = sl_dist * rr_ratio
        elif tp_strategy == "3.0_atr":
            tp_dist = Decimal(str(3.0 * atr))
        elif tp_strategy == "4.0_atr":
            tp_dist = Decimal(str(4.0 * atr))
        elif tp_strategy == "next_resistance":
            level = snapshot["resistance_price"] if action == "long" else snapshot["support_price"]
            if level > 0:
                buffer = Decimal(str(level)) * Decimal("0.002")
                if action == "long":
                    tp_dist = (Decimal(str(level)) + buffer) - entry_price
                else:
                    tp_dist = entry_price - (Decimal(str(level)) - buffer)
            else:
                tp_dist = sl_dist * rr_ratio
            # The structural level may not be BEYOND entry in the trade direction — e.g. a
            # breakout into price discovery above all resistance (long) or a breakdown below
            # all support (short). Then tp_dist <= 0 and the old floor would clamp the
            # take-profit to ~entry, capping a winner at break-even. Fall back to the
            # risk/reward multiplier so the target is a real rr x stop distance away.
            if tp_dist <= 0:
                tp_dist = sl_dist * rr_ratio
        else:
            tp_dist = sl_dist * rr_ratio

        tp_dist = max(tp_dist, Decimal(str(0.001 * current_price)))
        
        if action == "long":
            take_profit_price = entry_price + tp_dist
        else:
            take_profit_price = entry_price - tp_dist

        # 6. Calculate Position Size
        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        risk_amount_usd = balance_dec * self.max_risk_per_trade_pct
        
        # Position Size = Risk Amount / Stop Distance %
        raw_position_size_usd = risk_amount_usd / stop_distance_pct
        
        # Adjust for fees and slippage friction
        adjusted_position_size_usd = raw_position_size_usd * (Decimal("1") - self.fee_and_slip_buffer)
        
        # Cap exposure
        max_exposure_usd = balance_dec * self.max_exposure_pct
        final_position_size_usd = min(adjusted_position_size_usd, max_exposure_usd)
        
        # Reject dust trades
        if final_position_size_usd < Decimal("10.0"):
            return {
                "approved": False,
                "reason": f"Calculated position size (${final_position_size_usd:.2f}) is below the $10 minimum",
                "size_usd": Decimal("0")
            }

        return {
            "approved": True,
            "reason": "Risk limits approved",
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
            "take_profit_price": float(take_profit_price),
            "size_usd": float(final_position_size_usd.quantize(Decimal("0.01"))),
            "risk_amount_usd": float(risk_amount_usd)
        }
