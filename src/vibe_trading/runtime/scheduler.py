import os
import time
import logging
from datetime import datetime, date
import urllib.request
import json
from apscheduler.schedulers.blocking import BlockingScheduler
from langfuse import observe, propagate_attributes

from vibe_trading import audit
from vibe_trading.data.db import Database, PostgresDatabase
from vibe_trading.data.fetcher import DataFetcher
from vibe_trading.features.pipeline import FeaturePipeline
from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
from vibe_trading.agents.trader import HeadTrader
from vibe_trading.agents.client import LLMClient
from vibe_trading.agents.cost import PostgresCostLogger, daily_summary, should_alarm, should_block_trading
from vibe_trading.brokers.risk import RiskManager
from vibe_trading.brokers.paper import PaperBroker
from vibe_trading.brokers.coinbase import CoinbaseBroker
from vibe_trading.brokers.binance_futures import BinanceFuturesBroker
from vibe_trading.runtime.decision_pipeline import DecisionPipeline

logger = logging.getLogger(__name__)

class TradingScheduler:
    def __init__(self, symbols: list = None):
        self.symbols = symbols or []
        self.db = Database()
        self.pg_db = PostgresDatabase()

        # Route every LLM call's cost into Postgres (own pooled connection, not self.pg_db).
        LLMClient.set_cost_sink(PostgresCostLogger())
        self._cost_alarmed_on: date | None = None
        self._cost_blocked_on: date | None = None

        # Load correct broker dependency based on TRADING_MODE
        mode = os.getenv("TRADING_MODE", "PAPER").upper()
        if mode == "LIVE_SANDBOX":
            self.broker = CoinbaseBroker()
        elif mode == "LIVE_TESTNET":
            self.broker = BinanceFuturesBroker(db=self.pg_db)
        else:
            self.broker = PaperBroker(db=self.pg_db)
        
        self.fetcher = DataFetcher()
        self.pipeline = FeaturePipeline(self.db)
        self.risk_manager = RiskManager()
            
        # Initialize agents
        self.analyst = TechnicalVolumeAnalyst(db=self.db, fetcher=self.fetcher)
        self.trader = HeadTrader()
        
        # Keep track of analyst scorecard (mock in v1)
        self.scorecard = {"accuracy": 0.55, "total_decisions": 50}

        # The per-symbol agent decision graph (analyst -> trader -> risk), extracted
        # from sync_and_evaluate into a small, testable runner. All side effects
        # (persistence, audit, order submission, alerts) stay in the scheduler.
        self.decision_pipeline = DecisionPipeline(
            self.analyst, self.trader, self.risk_manager, self.pipeline, self.broker,
            scorecard=self.scorecard, trace_id_fn=self._current_trace_id,
        )

    def start(self):
        """Starts the main scheduling loop."""
        # 1. Start real-time fill bookkeeping FIRST (LIVE_TESTNET only) so the User Data
        #    Stream is live immediately — not gated behind the slow initial sync below.
        self.ws_listener = self._maybe_start_ws_listener()

        # 2. Run immediate bootstrap/sync on startup
        logger.info("Initializing startup data synchronization...")
        self.sync_and_evaluate()

        # 3. Setup recurring 4-hour scheduler
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
        # 1. Resolve trading symbols dynamically
        is_dynamic = len(self.symbols) == 0
        if is_dynamic:
            try:
                trending_symbols = self.fetcher.fetch_trending_symbols(limit=10)
            except Exception as e:
                logger.error(f"Error fetching trending symbols: {e}")
                trending_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "NEAR/USDT", "BNB/USDT"]
        else:
            trending_symbols = self.symbols

        open_positions = self.broker.get_open_positions()
        all_active_symbols = list(set(trending_symbols + [pos['symbol'] for pos in open_positions]))

        # 2. Run bootstrapping if needed for any active symbols (ensures warm-up history)
        try:
            self.fetcher.bootstrap_if_needed(self.db, all_active_symbols, ["1d", "4h"])
        except Exception as e:
            logger.error(f"Error running bootstrap check: {e}")

        with propagate_attributes(
            trace_name="ExecutionWindow-sync-and-evaluate",
            tags=["live-tick"],
            metadata={"symbols": ",".join(all_active_symbols)}
        ):
            now = datetime.utcnow()
            logger.info(f"--- Execution Window: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
            
            try:
                # 0. Alarm if today's LLM spend has spiked past the configured cap.
                self._check_cost_alarm()

                # 1. Fetch latest candles and write to DB (internally connects and closes DB)
                self.fetcher.incremental_update(self.db, all_active_symbols, ["1d", "4h"], limit=15)
                
                # 2. Update existing broker positions (for OCO fills in paper/sandbox)
                self.db.connect()
                current_prices = {}
                try:
                    for sym in all_active_symbols:
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
                self._record_closed_trades(closed_trades)
                
                # 3. Check for new entry signals — unless the hard LLM-spend cap is hit.
                # Existing positions were already updated above and keep being managed;
                # only NEW-entry evaluation (the expensive analyst/trader LLM calls) is blocked.
                trading_blocked = self._trading_blocked_by_cost()
                for sym in trending_symbols:
                    if trading_blocked:
                        break

                    # Limit open positions to max concurrent portfolio exposure (5 positions)
                    if len(self.broker.get_open_positions()) >= 5:
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

                    # Execution-critical price: futures mark in LIVE_TESTNET, else spot close.
                    # TA still uses spot candles; only the entry/proximity price is aligned.
                    exec_price = self._resolve_exec_price(sym, current_price)

                    # Run the agent decision graph (analyst -> snapshot -> trader -> risk).
                    # Pure orchestration; every side effect below is owned by the scheduler.
                    result = self.decision_pipeline.run_symbol(sym, last_ts, exec_price)
                    if result.status in ("analyst_failed", "no_snapshot"):
                        continue

                    proposal = result.proposal
                    snapshot = result.snapshot
                    trace_id = result.trace_id
                    analyst_report = result.analyst_report

                    # Log decision to database (every decision, including FLAT).
                    self.pg_db.connect()
                    try:
                        self.pg_db.conn.execute("""
                            INSERT OR IGNORE INTO decision_log (decision_id, timestamp, symbol, action, stop_loss_strategy, take_profit_strategy, risk_reward_ratio, reasoning_summary, agent_transcripts, trace_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (proposal["decision_id"], proposal["timestamp"], proposal["symbol"], proposal["action"],
                              proposal["stop_loss_strategy"], proposal["take_profit_strategy"], float(proposal["risk_reward_ratio"]),
                              proposal["reasoning_summary"], json.dumps(snapshot, default=str), trace_id))
                    finally:
                        self.pg_db.close()

                    # Append-only Parquet audit record for EVERY decision (including FLAT).
                    # Unlike decision_log.agent_transcripts (which stores the feature
                    # snapshot), this captures the REAL analyst + trader reasoning so the
                    # full decision corpus is queryable later.
                    audit.append_decision({
                        "decision_id": proposal["decision_id"],
                        "timestamp": proposal["timestamp"],
                        "symbol": proposal["symbol"],
                        "action": proposal["action"],
                        "stop_loss_strategy": proposal["stop_loss_strategy"],
                        "take_profit_strategy": proposal["take_profit_strategy"],
                        "risk_reward_ratio": float(proposal["risk_reward_ratio"]),
                        "reasoning_summary": proposal["reasoning_summary"],
                        "trace_id": trace_id,
                        "agent_transcripts": {
                            "analyst": analyst_report.model_dump(),
                            "trader": proposal,
                        },
                        "snapshot": snapshot,
                    })

                    if result.status == "flat":
                        logger.info(f"Head Trader decided FLAT for {sym}. Reasoning: {proposal['reasoning_summary']}")
                        continue

                    # Risk Manager outcome (computed inside the decision pipeline).
                    risk_res = result.risk_result
                    if risk_res["approved"]:
                        # Submit order to broker (internally connects and closes DuckDB inside PaperBroker)
                        # Pass entry_price so the live paper-trading path fills the position
                        # immediately at the RiskManager-computed mark, not on the next tick.
                        self.broker.submit_order(
                            symbol=sym,
                            action=proposal["action"],
                            size_usd=risk_res["size_usd"],
                            stop_price=risk_res["stop_price"],
                            take_profit_price=risk_res["take_profit_price"],
                            entry_price=risk_res["entry_price"],
                            decision_id=proposal["decision_id"],
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

    def _maybe_start_ws_listener(self):
        """Start the User Data Stream websocket listener for real-time fill bookkeeping
        (LIVE_TESTNET only). Fail-open: any construction error is logged and returns None
        so the scheduler still runs (the 4h reconcile remains the safety net)."""
        if os.getenv("TRADING_MODE", "PAPER").upper() != "LIVE_TESTNET":
            return None
        try:
            from vibe_trading.runtime.ws_listener import UserDataStreamListener
            ws_broker = BinanceFuturesBroker(db=PostgresDatabase())  # own conn (thread-safe)
            listener = UserDataStreamListener(ws_broker, self._record_closed_trades)
            listener.start()
            return listener
        except Exception as e:
            logger.error(f"Failed to start User Data Stream listener "
                         f"(continuing without it): {e}")
            return None

    def _record_closed_trades(self, closed_trades: list):
        """Persist closed trades to `trades` and send Discord alerts. Thread-safe: opens
        its OWN pooled PostgresDatabase connection per call (the web layer uses this same
        pattern), so the 4h-tick thread and the ws-listener thread can both call it."""
        if not closed_trades:
            return
        pg = PostgresDatabase()
        pg.connect()
        try:
            for trade in closed_trades:
                pg.conn.execute("""
                    INSERT INTO trades (trade_id, symbol, action, entry_time, entry_price, close_time, close_price, size_usd, realized_pnl, result, decision_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade["trade_id"], trade["symbol"], trade["action"], trade["entry_time"], trade["entry_price"],
                      trade["close_time"], trade["close_price"], trade["size_usd"], trade["realized_pnl"], trade["result"],
                      trade.get("decision_id")))
        finally:
            pg.close()
        for trade in closed_trades:
            self._send_discord_alert(
                f"🔄 **TRADE CLOSED:** {trade['symbol']} ({trade['action'].upper()})\n"
                f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['close_price']:.2f}\n"
                f"PnL: **${trade['realized_pnl']:.2f}** ({trade['result'].upper()})"
            )

    def _current_trace_id(self):
        """Return the active Langfuse trace id (Langfuse 4.x) so a decision can be
        joined to its trace. Returns None on any failure — observability must never
        break the trading loop."""
        try:
            from langfuse import get_client
            return get_client().get_current_trace_id()
        except Exception as e:
            logger.warning(f"Could not capture Langfuse trace id (non-fatal): {e}")
            return None

    def _resolve_exec_price(self, sym: str, fallback: float) -> float:
        """Execution-critical price: the broker's futures mark when available
        (LIVE_TESTNET), else the DuckDB spot 4h close fallback (PAPER/eval).
        TA still runs on spot candles — only the entry/proximity price is aligned."""
        try:
            mark = self.broker.get_mark_price(sym)
        except Exception as e:
            logger.warning(f"get_mark_price failed for {sym}: {e}; using spot fallback.")
            mark = None
        return mark if mark is not None else fallback

    def _check_cost_alarm(self):
        """Discord-alarm once per UTC day when LLM spend exceeds LLM_DAILY_COST_ALARM_USD."""
        threshold = float(os.getenv("LLM_DAILY_COST_ALARM_USD", "5.0"))
        try:
            self.pg_db.connect()
            summary = daily_summary(self.pg_db.conn)
        except Exception as e:
            logger.warning(f"cost alarm check skipped (non-fatal): {e}")
            return
        finally:
            try:
                self.pg_db.close()
            except Exception:
                pass

        today = datetime.utcnow().date()
        already = self._cost_alarmed_on == today
        if should_alarm(summary["today_usd"], threshold, already):
            self._cost_alarmed_on = today
            self._send_discord_alert(
                f"💸 **LLM COST ALARM:** today's spend ${summary['today_usd']:.2f} "
                f"exceeded ${threshold:.2f} ({summary['calls']} calls, "
                f"~${summary['projected_monthly_usd']:.2f}/mo projected)."
            )

    def _trading_blocked_by_cost(self) -> bool:
        """Hard daily LLM-spend kill switch. True when today's spend has hit
        LLM_DAILY_COST_CAP_USD (default $10; <= 0 disables), meaning new-entry
        evaluation should be skipped for the rest of the UTC day. Notifies Discord
        once per day when the cap first engages.

        Fail-open: on any error reading spend, returns False — a logging/DB hiccup
        must never halt trading (that would be a worse failure than slight overspend).
        """
        cap = float(os.getenv("LLM_DAILY_COST_CAP_USD", "10.0"))
        if cap <= 0:
            return False
        try:
            self.pg_db.connect()
            summary = daily_summary(self.pg_db.conn)
        except Exception as e:
            logger.warning(f"cost cap check skipped (non-fatal, fail-open): {e}")
            return False
        finally:
            try:
                self.pg_db.close()
            except Exception:
                pass

        if not should_block_trading(summary["today_usd"], cap):
            return False

        today = datetime.utcnow().date()
        if self._cost_blocked_on != today:
            self._cost_blocked_on = today
            self._send_discord_alert(
                f"🛑 **LLM COST CAP REACHED:** today's spend ${summary['today_usd']:.2f} "
                f"hit the ${cap:.2f} cap. Blocking NEW trade evaluation until tomorrow "
                f"(open positions are still managed)."
            )
        logger.warning(
            f"LLM daily cost cap (${cap:.2f}) reached — skipping new-entry evaluation this tick."
        )
        return True

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
