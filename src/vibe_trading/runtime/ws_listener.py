import asyncio
import logging
import os
import threading

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


def _is_exit_fill(order: dict) -> bool:
    """True when an order update represents a FILLED position-closing (bracket) order —
    the signal that a position likely just closed and a reconcile should run now.
    Errs toward triggering: a redundant reconcile is harmless (update_positions is
    idempotent), a missed one delays bookkeeping to the next 4h tick."""
    status = (order.get("status") or "").lower()
    if status not in ("closed", "filled"):
        return False
    info = order.get("info", {}) or {}
    reduce_only = order.get("reduceOnly") or info.get("R") or info.get("closePosition")
    otype = (order.get("type") or info.get("o") or "").upper()
    is_bracket = ("STOP" in otype) or ("TAKE_PROFIT" in otype)
    return bool(reduce_only) or is_bracket


class UserDataStreamListener:
    def __init__(self, reconcile_broker, record_fn, build_client=None):
        """`reconcile_broker`: a BinanceFuturesBroker used ONLY for update_positions()
        (its own sync ccxt client + own Postgres connection — never shared with the
        scheduler). `record_fn(closed_trades: list)`: scheduler._record_closed_trades.
        `build_client`: factory returning a ccxt.pro client (injected in tests)."""
        self.broker = reconcile_broker
        self.record_fn = record_fn
        self._build_client = build_client or self._default_client
        self._running = False
        self._thread = None
        self._exchange = None

    @staticmethod
    def _default_client():
        ex = ccxtpro.binance({
            "apiKey": os.getenv("BINANCE_TESTNET_API_KEY"),
            "secret": os.getenv("BINANCE_TESTNET_API_SECRET"),
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        ex.set_sandbox_mode(True)
        return ex

    def _handle_orders(self, orders: list):
        """Sync, unit-testable. Triggers a reconcile when any update is an exit fill."""
        if any(_is_exit_fill(o) for o in orders):
            self._reconcile_and_record()

    def _reconcile_and_record(self):
        try:
            closed = self.broker.update_positions({})
        except Exception as e:
            logger.error(f"ws_listener reconcile failed: {e}")
            return
        if closed:
            self.record_fn(closed)
