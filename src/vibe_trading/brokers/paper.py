import logging
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
from vibe_trading.brokers.base import BaseBroker

logger = logging.getLogger(__name__)

class PaperBroker(BaseBroker):
    def __init__(self, initial_balance: float = 10000.0, db = None):
        self.db = db
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

        # If database is provided, load persistent state
        if self.db:
            self._load_state()

    def _load_state(self):
        """Loads balance, peak balance, and open positions from DuckDB."""
        if not self.db:
            return
        try:
            self.db.connect()
            # 1. Load portfolio state
            res = self.db.conn.execute(
                "SELECT balance, peak_balance FROM portfolio_state ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if res:
                self.balance = res[0]
                self.peak_balance = res[1]
                logger.info(f"PaperBroker: Loaded balance ${self.balance:.2f} and peak balance ${self.peak_balance:.2f} from DB")
            else:
                self.db.conn.execute("""
                    INSERT INTO portfolio_state (timestamp, balance, peak_balance)
                    VALUES (CURRENT_TIMESTAMP, ?, ?)
                """, (self.balance, self.peak_balance))

            # 2. Load open positions
            rows = self.db.conn.execute(
                "SELECT symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price FROM open_positions"
            ).fetchall()
            for r in rows:
                symbol = r[0]
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": r[1],
                    "entry_time": r[2],
                    "entry_price": r[3],
                    "size_usd": r[4],
                    "stop_price": r[5],
                    "take_profit_price": r[6]
                }
            if rows:
                logger.info(f"PaperBroker: Loaded {len(rows)} open positions from DB")
        except Exception as e:
            logger.error(f"PaperBroker: Failed to load state: {e}")
        finally:
            self.db.close()

    def _save_portfolio_state(self):
        """Saves current balance and peak_balance to DuckDB."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute("""
                INSERT INTO portfolio_state (timestamp, balance, peak_balance)
                VALUES (CURRENT_TIMESTAMP, ?, ?)
            """, (self.balance, self.peak_balance))
        except Exception as e:
            logger.error(f"PaperBroker: Failed to save portfolio state: {e}")
        finally:
            self.db.close()

    def _save_position(self, pos: Dict[str, Any]):
        """Persists or updates an open position in DuckDB."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute("""
                INSERT OR REPLACE INTO open_positions (symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pos["symbol"], pos["side"], pos["entry_time"], pos["entry_price"], pos["size_usd"], pos["stop_price"], pos["take_profit_price"]))
        except Exception as e:
            logger.error(f"PaperBroker: Failed to save position for {pos['symbol']}: {e}")
        finally:
            self.db.close()

    def _delete_position(self, symbol: str):
        """Removes an open position from DuckDB."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute(
                "DELETE FROM open_positions WHERE symbol = ?", (symbol,)
            )
        except Exception as e:
            logger.error(f"PaperBroker: Failed to delete position for {symbol}: {e}")
        finally:
            self.db.close()

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
        
        self._save_position(position)
        return {"status": "success", "position": position}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.positions:
            return {"status": "rejected", "reason": "No open position"}
            
        pos = self.positions.pop(symbol)
        self._delete_position(symbol)
        
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
                self._save_position(pos)
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
                
                self._delete_position(symbol)
                self._save_portfolio_state()
                
                closed_trades.append(closed_info)
                
                trigger_name = "Stop Loss" if hit_sl else "Take Profit"
                logger.info(f"PaperBroker: {trigger_name} HIT for {symbol}. Exit Price: ${exit_price:.2f}, PnL: ${pnl:.2f}")
                
        return closed_trades
