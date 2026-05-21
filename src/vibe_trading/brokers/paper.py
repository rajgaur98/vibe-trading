import logging
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
from vibe_trading.brokers.base import BaseBroker

logger = logging.getLogger(__name__)

class PaperBroker(BaseBroker):
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

    def get_balance(self) -> float:
        return self.balance

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return list(self.positions.values())

    def submit_order(
        self,
        symbol: str,
        action: str,
        size_usd: float,
        stop_price: float,
        take_profit_price: float
    ) -> Dict[str, Any]:
        if symbol in self.positions:
            logger.warning(f"PaperBroker: Position already exists for {symbol}. Skipping order.")
            return {"status": "rejected", "reason": "Position exists"}
            
        entry_price = stop_price  # placeholder or will be populated by current price
        
        position = {
            "symbol": symbol,
            "side": action,
            "entry_price": 0.0,  # Will be set when executing
            "size_usd": size_usd,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "entry_time": datetime.utcnow()
        }
        
        self.positions[symbol] = position
        logger.info(f"PaperBroker: Submitted {action} order for {symbol} (Size: ${size_usd:.2f}, SL: {stop_price}, TP: {take_profit_price})")
        return {"status": "success", "position": position}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.positions:
            return {"status": "rejected", "reason": "No open position"}
            
        pos = self.positions.pop(symbol)
        logger.info(f"PaperBroker: Closed position for {symbol}")
        return {"status": "success", "closed_position": pos}

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Updates active positions against current prices.
        Checks if stop-loss or take-profit has been hit and resolves the PnL.
        """
        closed_trades = []
        for symbol, pos in list(self.positions.items()):
            if symbol not in current_prices:
                continue
                
            price = current_prices[symbol]
            side = pos["side"]
            sl = pos["stop_price"]
            tp = pos["take_profit_price"]
            
            # If entry price wasn't set, set it on first update tick
            if pos["entry_price"] == 0.0:
                pos["entry_price"] = price
                logger.info(f"PaperBroker: Filled {side} entry for {symbol} at ${price:.2f}")
                continue
                
            entry = pos["entry_price"]
            size_usd = pos["size_usd"]
            
            hit_sl = False
            hit_tp = False
            
            if side == "long":
                if price <= sl:
                    hit_sl = True
                elif price >= tp:
                    hit_tp = True
            elif side == "short":
                if price >= sl:
                    hit_sl = True
                elif price <= tp:
                    hit_tp = True
                    
            if hit_sl or hit_tp:
                # Calculate return
                exit_price = sl if hit_sl else tp
                price_return = (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
                
                # Fees buffer (approx 0.4% maker/taker fee)
                fees = size_usd * 0.004
                pnl = (size_usd * price_return) - fees
                
                # Update account balance
                self.balance += pnl
                self.peak_balance = max(self.peak_balance, self.balance)
                
                closed_info = {
                    "trade_id": str(uuid4()),
                    "symbol": symbol,
                    "action": side,
                    "entry_time": pos["entry_time"],
                    "entry_price": entry,
                    "close_time": datetime.utcnow(),
                    "close_price": exit_price,
                    "size_usd": size_usd,
                    "realized_pnl": pnl,
                    "result": "win" if pnl > 0 else "loss"
                }
                
                self.trade_history.append(closed_info)
                self.positions.pop(symbol)
                closed_trades.append(closed_info)
                
                trigger_name = "Stop Loss" if hit_sl else "Take Profit"
                logger.info(f"PaperBroker: {trigger_name} HIT for {symbol}. Exit Price: ${exit_price:.2f}, PnL: ${pnl:.2f}")
                
        return closed_trades
