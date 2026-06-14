import argparse
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher
from vibe_trading.eval.backtest import BacktestEngine
from vibe_trading.runtime.scheduler import TradingScheduler
from vibe_trading.runtime import state_sync, monitoring

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("vibe_trading.cli")


def _flush_langfuse():
    try:
        from langfuse import get_client
        logger.info("Flushing Langfuse traces...")
        get_client().flush()
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")


def execute_trade_once(symbols):
    """One scheduled execution window: warm state from the cache, run a single
    sync+evaluate, ping the dead-man's-switch on success, and always flush traces
    and push state back (even if evaluation raised)."""
    state_sync.pull()
    try:
        scheduler = TradingScheduler(symbols)
        scheduler.sync_and_evaluate()
        monitoring.ping_healthcheck(success=True)
    finally:
        _flush_langfuse()
        state_sync.push()


def main():
    # Load dotenv keys
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Vibe Trading CLI — Systematic Crypto Agentic Trading Bot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # 1. Bootstrap command
    bootstrap_parser = subparsers.add_parser("bootstrap", help="Download historical candles into DuckDB")
    bootstrap_parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
        help="List of symbols to bootstrap (e.g. BTC/USDT ETH/USDT)"
    )

    # 2. Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run historical simulation backtest")
    backtest_parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
        help="Symbols to backtest"
    )
    backtest_parser.add_argument(
        "--start", type=str, default=(datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format"
    )
    backtest_parser.add_argument(
        "--end", type=str, default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format"
    )
    backtest_parser.add_argument(
        "--live-agents", action="store_true", default=False,
        help="If set, calls Gemini APIs instead of local technical mocks"
    )

    # 3. Live command
    live_parser = subparsers.add_parser("live", help="Start the live recurring trading scheduler")
    live_parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to monitor and trade (default: dynamic trending coins)"
    )

    # 4. Trade-once command (on demand bypass)
    trade_once_parser = subparsers.add_parser("trade-once", help="Trigger a single sync and evaluation window immediately on demand")
    trade_once_parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to monitor and trade (default: dynamic trending coins)"
    )

    args = parser.parse_args()

    if args.command == "bootstrap":
        logger.info(f"Starting bootstrap for: {args.symbols}")
        db = Database()
        fetcher = DataFetcher()
        fetcher.bootstrap(db, args.symbols, ["1d", "4h"])
        logger.info("Bootstrap complete.")

    elif args.command == "backtest":
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        
        logger.info(f"Starting backtest for {args.symbols} ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})")
        db = Database()
        engine = BacktestEngine(db, args.symbols)
        results = engine.run(start_dt, end_dt, use_live_agents=args.live_agents)
        print("\n=== Backtest Summary ===")
        for k, v in results.items():
            print(f"{k}: {v}")

    elif args.command == "live":
        logger.info(f"Starting recurring 4-hour live scheduler for: {args.symbols}")
        scheduler = TradingScheduler(args.symbols)
        scheduler.start()

    elif args.command == "trade-once":
        logger.info(f"Triggering on-demand trading execution window for: {args.symbols}")
        execute_trade_once(args.symbols)
        logger.info("On-demand execution window completed.")

if __name__ == "__main__":
    main()
