import os
import time
import logging
from datetime import datetime
import urllib.request
import json
from apscheduler.schedulers.blocking import BlockingScheduler
from langfuse import observe, propagate_attributes

from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher
from vibe_trading.features.pipeline import FeaturePipeline
from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
from vibe_trading.agents.trader import HeadTrader
from vibe_trading.brokers.risk import RiskManager
from vibe_trading.brokers.paper import PaperBroker
from vibe_trading.brokers.coinbase import CoinbaseBroker

logger = logging.getLogger(__name__)

class TradingScheduler:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.db = Database()
        
        # Load correct broker dependency based on TRADING_MODE
        mode = os.getenv("TRADING_MODE", "PAPER").upper()
        if mode == "LIVE_SANDBOX":
            self.broker = CoinbaseBroker()
        else:
            self.broker = PaperBroker(db=self.db)
        
        self.fetcher = DataFetcher()
        self.pipeline = FeaturePipeline(self.db)
        self.risk_manager = RiskManager()
            
        # Initialize agents
        self.analyst = TechnicalVolumeAnalyst()
        self.trader = HeadTrader()
        
        # Keep track of analyst scorecard (mock in v1)
        self.scorecard = {"accuracy": 0.55, "total_decisions": 50}

    def start(self):
        """Starts the main scheduling loop."""
        # 1. Run immediate bootstrap/sync on startup
        logger.info("Initializing startup data synchronization...")
        self.sync_and_evaluate()
        
        # 2. Setup recurring 4-hour scheduler
        scheduler = BlockingScheduler()
        # Schedule to run at the start of every 4h block (00:00, 04:00, 08:00, etc.)
        scheduler.add_job(self.sync_and_evaluate, 'cron', hour='*/4', minute=1)
        
        logger.info("Scheduler started. Running on 4h intervals.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")

    @observe()
    def sync_and_evaluate(self):
        """Syncs latest candles, updates broker status, and triggers agent evaluations."""
        with propagate_attributes(
            trace_name="ExecutionWindow-sync-and-evaluate",
            tags=["live-tick"],
            metadata={"symbols": ",".join(self.symbols)}
        ):
            now = datetime.utcnow()
            logger.info(f"--- Execution Window: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
            
            try:
                # 1. Fetch latest candles and write to DB (internally connects and closes DB)
                self.fetcher.incremental_update(self.db, self.symbols, ["1d", "4h"], limit=15)
                
                # 2. Update existing broker positions (for OCO fills in paper/sandbox)
                self.db.connect()
                current_prices = {}
                try:
                    for sym in self.symbols:
                        price_res = self.db.conn.execute(
                            "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' ORDER BY timestamp DESC LIMIT 1",
                            (sym,)
                        ).fetchone()
                        if price_res:
                            current_prices[sym] = price_res[0]
                finally:
                    self.db.close()
                        
                # Update positions (internally connects and closes DuckDB inside PaperBroker)
                closed_trades = self.broker.update_positions(current_prices)
                if closed_trades:
                    self.db.connect()
                    try:
                        for trade in closed_trades:
                            # Log closed trade to DB
                            self.db.conn.execute("""
                                INSERT INTO trades (trade_id, symbol, action, entry_time, entry_price, close_time, close_price, size_usd, realized_pnl, result)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (trade["trade_id"], trade["symbol"], trade["action"], trade["entry_time"], trade["entry_price"],
                                  trade["close_time"], trade["close_price"], trade["size_usd"], trade["realized_pnl"], trade["result"]))
                            
                            self._send_discord_alert(
                                f"🔄 **TRADE CLOSED:** {trade['symbol']} ({trade['action'].upper()})\n"
                                f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['close_price']:.2f}\n"
                                f"PnL: **${trade['realized_pnl']:.2f}** ({trade['result'].upper()})"
                            )
                    finally:
                        self.db.close()
                
                # 3. Check for new entry signals
                for sym in self.symbols:
                    # Limit open positions to max portfolio sizing
                    if len(self.broker.get_open_positions()) >= 3:
                        logger.info("Max concurrent portfolio exposure reached. Skipping new evaluations.")
                        break
                        
                    # Skip if position already exists for this symbol
                    if any(pos['symbol'] == sym for pos in self.broker.get_open_positions()):
                        continue
                    
                    # Fetch latest 4h candle timestamp
                    self.db.connect()
                    try:
                        last_candle_ts_res = self.db.conn.execute(
                            "SELECT timestamp, close FROM candles WHERE symbol = ? AND timeframe = '4h' ORDER BY timestamp DESC LIMIT 1",
                            (sym,)
                        ).fetchone()
                    finally:
                        self.db.close()
                    
                    if not last_candle_ts_res:
                        continue
                    
                    last_ts, current_price = last_candle_ts_res
                    
                    # Generate market snapshot (internally connects and closes DuckDB)
                    snapshot = self.pipeline.run(sym, last_ts)
                    if not snapshot:
                        continue
                    
                    # Step 1: Analyst agent evaluation (database is fully closed during this slow API call!)
                    analyst_report = self.analyst.analyze(snapshot)
                    
                    # Step 2: Head Trader decision (database is fully closed during this slow API call!)
                    open_positions = self.broker.get_open_positions()
                    proposal = self.trader.decide(sym, analyst_report, self.scorecard, open_positions)
                    
                    # Step 3: Log decision to database
                    self.db.connect()
                    try:
                        self.db.conn.execute("""
                            INSERT OR IGNORE INTO decision_log (decision_id, timestamp, symbol, action, stop_loss_strategy, take_profit_strategy, risk_reward_ratio, reasoning_summary, agent_transcripts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (proposal["decision_id"], proposal["timestamp"], proposal["symbol"], proposal["action"],
                              proposal["stop_loss_strategy"], proposal["take_profit_strategy"], float(proposal["risk_reward_ratio"]),
                              proposal["reasoning_summary"], json.dumps(snapshot, default=str)))
                    finally:
                        self.db.close()
                    
                    if proposal["action"] == "flat":
                        logger.info(f"Head Trader decided FLAT for {sym}. Reasoning: {proposal['reasoning_summary']}")
                        continue
                    
                    # Step 4: Risk Manager evaluation (pipeline._get_candles internally connects and closes DB)
                    df_4h = self.pipeline._get_candles(sym, "4h", last_ts, limit=30)
                    risk_res = self.risk_manager.evaluate_proposal(
                        proposal=proposal,
                        current_price=current_price,
                        df_4h=df_4h,
                        account_balance=self.broker.get_balance(),
                        peak_balance=self.broker.peak_balance,
                        open_positions=open_positions,
                        snapshot=snapshot
                    )
                    
                    if risk_res["approved"]:
                        # Submit order to broker (internally connects and closes DuckDB inside PaperBroker)
                        self.broker.submit_order(
                            symbol=sym,
                            action=proposal["action"],
                            size_usd=risk_res["size_usd"],
                            stop_price=risk_res["stop_price"],
                            take_profit_price=risk_res["take_profit_price"]
                        )
                        
                        self._send_discord_alert(
                            f"🚀 **NEW TRADE ENTERED:** {sym} ({proposal['action'].upper()})\n"
                            f"Entry Price: ${risk_res['entry_price']:.2f}\n"
                            f"Stop Loss: ${risk_res['stop_price']:.2f} ({proposal['stop_loss_strategy']})\n"
                            f"Take Profit: ${risk_res['take_profit_price']:.2f} ({proposal['take_profit_strategy']})\n"
                            f"Position Size: **${risk_res['size_usd']:.2f}** (Risking 1.00% equity)\n"
                            f"Reasoning: *{proposal['reasoning_summary']}*"
                        )
                    else:
                        logger.warning(f"Risk Manager rejected proposal for {sym}: {risk_res['reason']}")
                        self._send_discord_alert(
                            f"⚠️ **RISK VETO:** Rejected {proposal['action'].upper()} on {sym}.\n"
                            f"Reason: {risk_res['reason']}"
                        )
            except Exception as e:
                logger.error(f"Error in scheduler tick: {e}", exc_info=True)
                self._send_discord_alert(f"🔴 **SCHEDULER ERROR:** {str(e)}")

    def _send_discord_alert(self, message: str):
        """Sends an alert to Discord webhook if configured."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url or "your_discord_webhook_url" in webhook_url:
            return
            
        data = json.dumps({"content": message}).encode('utf-8')
        req = urllib.request.Request(
            webhook_url, 
            data=data, 
            headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                pass
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
