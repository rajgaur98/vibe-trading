import os
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime

import ccxt

from vibe_trading.brokers.base import BaseBroker

logger = logging.getLogger(__name__)


def _to_ccxt_symbol(symbol: str) -> str:
    """'BTC/USDT' -> ccxt USDⓂ-futures unified symbol 'BTC/USDT:USDT'."""
    base, quote = symbol.split("/")
    return f"{base}/{quote}:{quote}"


def _to_plain_symbol(ccxt_symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC/USDT' (drop the settle suffix)."""
    return ccxt_symbol.split(":")[0]


class BinanceFuturesBroker(BaseBroker):
    def __init__(self, db=None, exchange=None):
        self.db = db  # PostgresDatabase — the reconciliation ledger (open_positions)
        self.peak_balance = 0.0  # tracked in-memory; updated in get_balance()
        self.dry_run = os.getenv("BINANCE_TESTNET_DRY_RUN", "false").lower() == "true"
        self.leverage = int(os.getenv("BINANCE_TESTNET_LEVERAGE", "1"))

        if exchange is not None:
            # Injected (tests / pre-configured): use as-is, no network, no creds check.
            self.exchange = exchange
            return

        key = os.getenv("BINANCE_TESTNET_API_KEY")
        secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not self.dry_run and (not key or not secret):
            raise ValueError(
                "BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET must be set for "
                "TRADING_MODE=LIVE_TESTNET (or set BINANCE_TESTNET_DRY_RUN=true)."
            )
        self.exchange = ccxt.binance({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.exchange.set_sandbox_mode(True)  # routes to testnet.binancefuture.com
        self.exchange.load_markets()
        logger.info(f"BinanceFuturesBroker initialized (dry_run={self.dry_run}, leverage={self.leverage}x)")

    # --- BaseBroker abstract methods (filled in by subsequent tasks) ---
    def submit_order(
        self,
        symbol: str,
        action: str,
        size_usd: float,
        stop_price: float,
        take_profit_price: float,
        entry_price: float = 0.0,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_open_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_balance(self) -> float:
        raise NotImplementedError

    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        raise NotImplementedError
