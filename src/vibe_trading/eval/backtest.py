import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import quantstats as qs
from pathlib import Path

from vibe_trading.data.db import Database
from vibe_trading.features.pipeline import FeaturePipeline
from vibe_trading.brokers.risk import RiskManager
from vibe_trading.brokers.paper import PaperBroker
from vibe_trading.agents.analyst import AnalystOutput

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, db: Database, symbols: List[str], initial_balance: float = 10000.0):
        self.db = db
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.pipeline = FeaturePipeline(db)
        self.risk_manager = RiskManager()
        self.broker = PaperBroker(initial_balance)
        self.equity_curve: List[Dict[str, Any]] = []

    def run(self, start_date: datetime, end_date: datetime, use_live_agents: bool = False) -> Dict[str, Any]:
        """
        Runs the backtest simulation loop.
        Iterates through 4h candles chronologically.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}...")
        self.db.connect()

        # Get all 4h candle timestamps in range
        # Note: We sort ascending to process in chronological order
        timestamps_res = self.db.conn.execute("""
            SELECT DISTINCT timestamp 
            FROM candles 
            WHERE timeframe = '4h' AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (start_date, end_date)).fetchall()

        timestamps = [r[0] for r in timestamps_res]
        if not timestamps:
            logger.error("No historical 4h candles found in database for the given date range.")
            self.db.close()
            return {}

        logger.info(f"Running backtest over {len(timestamps)} time periods...")
        
        # Warm up the broker balance
        self.equity_curve.append({
            "timestamp": start_date,
            "balance": self.broker.get_balance()
        })

        for i, ts in enumerate(timestamps):
            # 1. Update prices for active positions from the current candle's open/close
            current_prices = {}
            for sym in self.symbols:
                price_res = self.db.conn.execute("""
                    SELECT close 
                    FROM candles 
                    WHERE symbol = ? AND timeframe = '4h' AND timestamp = ?
                """, (sym, ts)).fetchone()
                if price_res:
                    current_prices[sym] = price_res[0]

            # Update broker positions and capture closed trades
            closed_trades = self._update_and_resolve_brackets(ts, current_prices)
            
            # Log equity daily value
            if ts.hour == 0 or i == len(timestamps) - 1:
                self.equity_curve.append({
                    "timestamp": ts,
                    "balance": self.broker.get_balance()
                })

            # 2. Check for new signals
            for sym in self.symbols:
                # If we already have 3 open positions, skip new entries
                if len(self.broker.get_open_positions()) >= 3:
                    continue
                    
                # Skip if position already exists for this symbol
                if any(pos['symbol'] == sym for pos in self.broker.get_open_positions()):
                    continue

                # Run feature pipeline
                snapshot = self.pipeline.run(sym, ts)
                if not snapshot:
                    continue

                # Get mock or live decision (timestamp drives the tool-loop's no-look-ahead cutoff)
                proposal = self._get_decision(sym, snapshot, ts, use_live_agents)
                if not proposal or proposal["action"] == "flat":
                    continue

                # Evaluate proposal through Risk Manager
                # Fetch 4h historical candles for volatility/ATR calculation
                df_4h = self.pipeline._get_candles(sym, "4h", ts, limit=30)
                risk_res = self.risk_manager.evaluate_proposal(
                    proposal=proposal,
                    current_price=snapshot["close"],
                    df_4h=df_4h,
                    account_balance=self.broker.get_balance(),
                    peak_balance=self.broker.peak_balance,
                    open_positions=self.broker.get_open_positions(),
                    snapshot=snapshot
                )

                if risk_res["approved"]:
                    # Submit order to paper broker
                    self.broker.submit_order(
                        symbol=sym,
                        action=proposal["action"],
                        size_usd=risk_res["size_usd"],
                        stop_price=risk_res["stop_price"],
                        take_profit_price=risk_res["take_profit_price"]
                    )

        self.db.close()
        logger.info("Backtest complete.")
        
        # Calculate performance stats
        return self._generate_reports()

    def _get_decision(self, symbol: str, snapshot: dict, timestamp: datetime, use_live_agents: bool) -> dict:
        """Helper to return agent proposals. Simulates trading signals deterministically in backtest to save API costs."""
        if use_live_agents:
            from vibe_trading.agents.trader import HeadTrader
            from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
            from vibe_trading.data.fetcher import DataFetcher

            analyst = TechnicalVolumeAnalyst(db=self.db, fetcher=DataFetcher())
            trader = HeadTrader()

            scorecard = {"accuracy": 0.55, "total_decisions": 100}
            open_positions = self.broker.get_open_positions()

            analyst_res = analyst.analyze(symbol=symbol, timestamp=timestamp)
            proposal = trader.decide(symbol, analyst_res, scorecard, open_positions)
            return proposal
        else:
            # Deterministic Mock Trading Agent (Simulates technical vibes)
            rsi = snapshot["rsi_14"]
            obv_trend = snapshot["obv_trend"]
            macd_regime = snapshot["macd_regime"]

            action = "flat"
            if rsi < 40 or (obv_trend == "accumulation" and "bullish" in macd_regime):
                action = "long"
            elif rsi > 70 or (obv_trend == "distribution" and "bearish" in macd_regime):
                action = "short"

            return {
                "symbol": symbol,
                "action": action,
                "stop_loss_strategy": "1.5_atr",
                "take_profit_strategy": "risk_reward_multiplier",
                "risk_reward_ratio": 2.0,
                "hold_period_bias": "medium",
                "reasoning_summary": "Mock technical backtest signal",
            }

    def _update_and_resolve_brackets(self, timestamp: datetime, current_prices: Dict[str, float]) -> List[dict]:
        """
        Updates positions checking SL/TP.
        Resolves OCO overlaps using sub-candle logic.
        """
        closed_trades = []
        for sym, pos in list(self.broker.positions.items()):
            if sym not in current_prices:
                continue
                
            # Fetch the candle's High and Low for the 4h timeframe
            candle_res = self.db.conn.execute("""
                SELECT high, low, close 
                FROM candles 
                WHERE symbol = ? AND timeframe = '4h' AND timestamp = ?
            """, (sym, timestamp)).fetchone()
            
            if not candle_res:
                continue
                
            high, low, close = candle_res
            sl = pos["stop_price"]
            tp = pos["take_profit_price"]
            side = pos["side"]
            
            # Scenario A: Both Stop Loss and Take Profit are breached inside this 4h candle
            if (side == "long" and low <= sl and high >= tp) or (side == "short" and high >= sl and low <= tp):
                # Resolve with sub-candle data
                exit_price = self._resolve_subcandle_path(sym, timestamp, sl, tp, side)
                
                # Close the position using the resolved exit price
                pnl_mult = 1 if side == "long" else -1
                price_return = (exit_price - pos["entry_price"]) / pos["entry_price"] * pnl_mult
                fees = pos["size_usd"] * 0.004
                pnl = (pos["size_usd"] * price_return) - fees
                
                self.broker.balance += pnl
                self.broker.peak_balance = max(self.broker.peak_balance, self.broker.balance)
                
                closed_info = {
                    "trade_id": str(uuid4()),
                    "symbol": sym,
                    "action": side,
                    "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"],
                    "close_time": timestamp,
                    "close_price": exit_price,
                    "size_usd": pos["size_usd"],
                    "realized_pnl": pnl,
                    "result": "win" if pnl > 0 else "loss"
                }
                self.broker.trade_history.append(closed_info)
                self.broker.positions.pop(sym)
                closed_trades.append(closed_info)
                
                logger.info(f"Backtest: Resolved overlapping OCO brackets for {sym} using sub-candle logic. Exit Price: ${exit_price:.2f}, Result: {closed_info['result']}")
            
            # Scenario B: Normal single-bracket breach
            else:
                # Let PaperBroker execute normal checks
                trades = self.broker.update_positions({sym: close})
                if trades:
                    closed_trades.extend(trades)
                    
        return closed_trades

    def _resolve_subcandle_path(
        self,
        symbol: str,
        timestamp: datetime,
        stop_price: float,
        take_profit_price: float,
        side: str
    ) -> float:
        """
        Fetches 1-minute or 5-minute candles to determine if Stop Loss or Take Profit was hit first.
        Falls back to conservative Stop Loss hit if data is missing.
        """
        # Calculate the 4h range end time
        range_end = timestamp + timedelta(hours=4)
        
        # Look for 1m or 5m candles in that 4h range
        sub_candles = self.db.conn.execute("""
            SELECT high, low, close 
            FROM candles 
            WHERE symbol = ? AND timeframe IN ('1m', '5m') AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC
        """, (symbol, timestamp, range_end)).fetchall()
        
        if not sub_candles:
            # CONSERVATIVE FALLBACK: Assume stop-loss was hit first
            logger.warning(f"No sub-candle data for {symbol} at {timestamp}. Applying conservative stop-loss fallback.")
            return stop_price
            
        for high, low, close in sub_candles:
            if side == "long":
                if low <= stop_price:
                    return stop_price
                if high >= take_profit_price:
                    return take_profit_price
            else:
                if high >= stop_price:
                    return stop_price
                if low <= take_profit_price:
                    return take_profit_price
                    
        # If neither hit on the sub-candles (unlikely), return close
        return sub_candles[-1][2]

    def _generate_reports(self) -> Dict[str, Any]:
        """Compiles stats and saves a QuantStats report."""
        if not self.broker.trade_history:
            logger.warning("No trades were made during this backtest.")
            return {}
            
        df_equity = pd.DataFrame(self.equity_curve)
        df_equity.set_index('timestamp', inplace=True)
        
        # Resample daily returns
        daily_equity = df_equity['balance'].resample('1D').last().ffill()
        returns = daily_equity.pct_change().dropna()
        
        # Save QuantStats Report
        report_path = Path("data/reports/backtest_report.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving QuantStats tear sheet to {report_path.absolute()}...")
        
        # We wrap in try-except because QuantStats can error on extremely small datasets
        try:
            qs.reports.html(returns, output=str(report_path), title="Vibe Trading Backtest Report")
        except Exception as e:
            logger.error(f"Failed to generate QuantStats HTML report: {e}")

        # Basic summary metrics
        trades_df = pd.DataFrame(self.broker.trade_history)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'win'])
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['realized_pnl'].sum()
        
        summary = {
            "total_trades": total_trades,
            "win_rate": f"{win_rate:.2%}",
            "total_pnl_usd": f"${total_pnl:.2f}",
            "final_balance": f"${self.broker.get_balance():.2f}",
            "report_file": str(report_path.absolute())
        }
        
        logger.info(f"Backtest Summary: {summary}")
        return summary
from uuid import uuid4
