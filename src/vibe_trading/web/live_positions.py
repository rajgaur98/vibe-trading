"""Live-exchange position reads for the dashboard (LIVE_TESTNET mode).

Kept separate from web.main so it can be imported and unit-tested without triggering
main.py's module-level PostgresDatabase() pool initialization.
"""
import logging

logger = logging.getLogger(__name__)

_live_broker = None


def _get_live_broker():
    """Lazily construct and cache a read-only BinanceFuturesBroker for the dashboard."""
    global _live_broker
    if _live_broker is None:
        from vibe_trading.brokers.binance_futures import BinanceFuturesBroker
        _live_broker = BinanceFuturesBroker(db=None)
    return _live_broker


def live_testnet_positions():
    """Live exchange positions for the dashboard, or None to signal fallback to Postgres."""
    try:
        return _get_live_broker().get_open_positions()
    except Exception as e:
        logger.warning(f"live_testnet_positions failed, falling back to ledger: {e}")
        return None
