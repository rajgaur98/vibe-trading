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
        sym = _to_ccxt_symbol(symbol)
        try:
            self.exchange.set_leverage(self.leverage, sym)

            mark = entry_price if entry_price > 0 else float(self.exchange.fetch_ticker(sym)["last"])
            qty = float(self.exchange.amount_to_precision(sym, size_usd / mark))

            market = self.exchange.market(sym)
            limits = market.get("limits", {}) or {}
            min_cost = (limits.get("cost", {}) or {}).get("min")
            min_amount = (limits.get("amount", {}) or {}).get("min")
            if (min_cost is not None and qty * mark < min_cost) or \
               (min_amount is not None and qty < min_amount):
                logger.warning(f"BinanceFuturesBroker: {symbol} below exchange minimum "
                               f"(qty={qty}, notional={qty * mark:.2f}). Skipping.")
                return {"status": "rejected", "reason": "below exchange minimum"}

            entry_side = "buy" if action == "long" else "sell"
            exit_side = "sell" if action == "long" else "buy"
            tp = self.exchange.price_to_precision(sym, take_profit_price)
            sl = self.exchange.price_to_precision(sym, stop_price)

            if self.dry_run:
                logger.info(f"[DRY_RUN] {symbol} {action} qty={qty} entry~{mark} TP={tp} SL={sl}")
                self._persist_position(symbol, action, mark, size_usd, stop_price, take_profit_price)
                return {"status": "dry_run", "entry_price": mark, "order_ids": {}}

            entry_order = self.exchange.create_order(sym, "market", entry_side, qty)
            tp_order = self.exchange.create_order(
                sym, "TAKE_PROFIT_MARKET", exit_side, None,
                params={"stopPrice": tp, "closePosition": True},
            )
            sl_order = self.exchange.create_order(
                sym, "STOP_MARKET", exit_side, None,
                params={"stopPrice": sl, "closePosition": True},
            )

            avg = float(entry_order.get("average") or entry_order.get("price") or mark)
            self._persist_position(symbol, action, avg, size_usd, stop_price, take_profit_price)
            logger.info(f"BinanceFuturesBroker: opened {action} {symbol} @ {avg} "
                        f"(TP={tp}, SL={sl}, size=${size_usd:.2f})")
            return {
                "status": "success",
                "entry_price": avg,
                "order_ids": {
                    "entry": entry_order.get("id"),
                    "take_profit": tp_order.get("id"),
                    "stop": sl_order.get("id"),
                },
            }
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: submit_order failed for {symbol}: {e}")
            return {"status": "rejected", "reason": str(e)}

    def _persist_position(self, symbol, side, entry_price, size_usd, stop_price, take_profit_price):
        """Write the open position to the Postgres ledger (no-op when db is None).
        Reuses the exact SQL PaperBroker uses, so translate_query handles the dialect."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, side, datetime.utcnow(), entry_price, size_usd, stop_price, take_profit_price),
            )
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: failed to persist position {symbol}: {e}")
        finally:
            self.db.close()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_balance(self) -> float:
        if self.dry_run:
            self.peak_balance = max(self.peak_balance, 10000.0)
            return 10000.0
        try:
            bal = float(self.exchange.fetch_balance()["USDT"]["total"])
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: get_balance failed: {e}")
            return 0.0
        self.peak_balance = max(self.peak_balance, bal)
        return bal

    def get_mark_price(self, symbol: str) -> Optional[float]:
        try:
            return float(self.exchange.fetch_ticker(_to_ccxt_symbol(symbol))["last"])
        except Exception as e:
            logger.warning(f"BinanceFuturesBroker: get_mark_price failed for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        raise NotImplementedError
