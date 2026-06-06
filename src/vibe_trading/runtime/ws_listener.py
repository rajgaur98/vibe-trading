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


class UserDataStreamListener:  # placeholder — full implementation added in the next task
    pass
