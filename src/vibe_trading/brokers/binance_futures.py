import os
import time
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime

import ccxt

from vibe_trading.brokers.base import BaseBroker

logger = logging.getLogger(__name__)


# Binance Futures DEMO TRADING REST base (demo.binance.com → "Futures Demo API Base
# Endpoint"). Demo keys are valid ONLY here — not on the deprecated futures testnet
# (testnet.binancefuture.com) and not on production (fapi.binance.com).
DEMO_FAPI_URL = os.getenv("BINANCE_DEMO_FAPI_URL", "https://demo-fapi.binance.com")


def _route_to_demo(exchange) -> None:
    """Point ccxt's USDⓂ-futures REST URLs at the demo-trading endpoint, replacing the
    deprecated set_sandbox_mode (which ccxt 4.5+ removed for futures)."""
    versions = {
        "fapiPublic": "v1", "fapiPublicV2": "v2", "fapiPublicV3": "v3",
        "fapiPrivate": "v1", "fapiPrivateV2": "v2", "fapiPrivateV3": "v3",
    }
    for url_key, ver in versions.items():
        if url_key in exchange.urls["api"]:
            exchange.urls["api"][url_key] = f"{DEMO_FAPI_URL}/fapi/{ver}"
    if "fapiData" in exchange.urls["api"]:
        exchange.urls["api"]["fapiData"] = f"{DEMO_FAPI_URL}/futures/data"


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
            # A futures-only demo key cannot authenticate the SPOT/margin SAPI endpoints
            # that load_markets() hits by default (-2008). Skip currencies and restrict
            # market loading to USDⓂ futures ("linear") — both load via public fapi endpoints.
            "options": {
                "defaultType": "future",
                "fetchCurrencies": False,
                "fetchMarkets": {"types": ["linear"]},
            },
        })
        # Route to Binance Futures DEMO TRADING (demo-fapi.binance.com). We do NOT call
        # set_sandbox_mode (ccxt 4.5+ dropped it for futures); demo keys are scoped to a
        # demo account and only authenticate against this endpoint — no real funds.
        _route_to_demo(self.exchange)
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
        decision_id: str = None,
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
                self._persist_position(symbol, action, mark, size_usd, stop_price, take_profit_price, decision_id)
                return {"status": "dry_run", "entry_price": mark, "order_ids": {}}

            entry_order = self.exchange.create_order(sym, "market", entry_side, qty)
            # The closePosition brackets are rejected (-4509) if the market entry's fill
            # hasn't registered as a position yet — a real propagation race on demo/prod.
            # Wait (bounded) for the position before attaching the brackets, and HONOR the
            # result: if the fill never registers, attaching brackets would only -4509 and
            # leave a naked, unprotected entry. Roll back instead of risking that.
            if not self._await_position(sym):
                logger.error(f"BinanceFuturesBroker: entry fill for {symbol} never registered "
                             f"as a position; rolling back rather than risk a naked entry.")
                self._rollback_entry(sym)
                return {"status": "rejected", "reason": "entry fill never registered as a position"}
            try:
                # Each bracket is placed via a helper that retries ONLY on -4509 (the
                # registration race) — a clean rejection means nothing was placed, so the
                # retry is duplicate-safe. Any other error is a real failure and propagates.
                tp_order = self._place_close_order(sym, "TAKE_PROFIT_MARKET", exit_side, tp)
                sl_order = self._place_close_order(sym, "STOP_MARKET", exit_side, sl)
            except Exception as e:
                # Entry filled but a bracket failed → we'd be holding a NAKED, unprotected
                # position. Roll the entry back immediately rather than leave it exposed.
                logger.error(f"BinanceFuturesBroker: bracket placement failed for {symbol}, "
                             f"rolling back the entry: {e}")
                self._rollback_entry(sym)
                return {"status": "rejected", "reason": f"bracket placement failed: {e}"}

            avg = float(entry_order.get("average") or entry_order.get("price") or mark)
            self._persist_position(symbol, action, avg, size_usd, stop_price, take_profit_price, decision_id)
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

    def _place_close_order(self, sym: str, order_type: str, exit_side: str, trigger,
                           attempts: int = 6, delay: float = 0.5):
        """Place a single closePosition trigger order (TAKE_PROFIT_MARKET or STOP_MARKET),
        retrying ONLY on -4509 ("TIF GTE can only be used with open positions"). That error
        means the just-filled entry hasn't propagated to a position yet — a clean rejection
        with nothing placed, so we re-await the position and retry (duplicate-safe). Any
        other error is a genuine failure and re-raises immediately for the caller to handle."""
        last_err = None
        for _ in range(attempts):
            try:
                return self.exchange.create_order(
                    sym, order_type, exit_side, None,
                    params={"stopPrice": trigger, "closePosition": True},
                )
            except Exception as e:
                if "-4509" not in str(e):
                    raise  # real failure — don't mask it behind retries
                last_err = e
                self._await_position(sym)  # give the fill more time to register
                time.sleep(delay)
        raise last_err

    def _rollback_entry(self, sym: str) -> None:
        """Undo a just-filled entry whose protective brackets failed to attach, so we never
        hold a naked, unprotected position. Crucially it WAITS for the fill to register
        first — the same propagation lag that breaks bracket placement also hides the
        position from a naive close — then issues a reduce-only market close, cancels
        leftover orders, and VERIFIES the position is flat. If it cannot be flattened it
        escalates to a CRITICAL operator alarm rather than silently leaving a naked entry."""
        self._await_position(sym)  # the fill may not be visible yet; wait before closing
        try:
            positions = self.exchange.fetch_positions([sym])
            pos = next((p for p in positions if float(p.get("contracts") or 0) != 0), None)
            if pos:
                contracts = abs(float(pos["contracts"]))
                opp = "sell" if pos.get("side") == "long" else "buy"
                self.exchange.create_order(sym, "market", opp, contracts, params={"reduceOnly": True})
            self.exchange.cancel_all_orders(sym)
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: rollback close/cancel failed for {sym}: {e}")
        if not self._confirm_flat(sym):
            logger.critical(
                f"BinanceFuturesBroker: NAKED POSITION — rollback could not flatten {sym}; "
                f"manual intervention required (an open position may have NO protective brackets)."
            )

    def _await_position(self, sym: str, attempts: int = 20, delay: float = 0.5) -> bool:
        """Poll (bounded, ~10s) until the just-submitted market entry registers as an open
        position, so the closePosition brackets aren't rejected with -4509. Returns True
        once a non-zero position is seen, False if it never appears within the budget."""
        for _ in range(attempts):
            try:
                positions = self.exchange.fetch_positions([sym])
            except Exception:
                return False
            if any(abs(float(p.get("contracts") or 0)) > 0 for p in positions):
                return True
            time.sleep(delay)
        return False

    def _confirm_flat(self, sym: str, attempts: int = 6, delay: float = 0.5) -> bool:
        """Poll (bounded) until no open position remains for `sym`. Returns True once flat;
        False if a position is still showing after the budget — the caller should treat that
        as a naked-position risk and alarm."""
        for _ in range(attempts):
            try:
                positions = self.exchange.fetch_positions([sym])
            except Exception:
                return False
            if not any(abs(float(p.get("contracts") or 0)) > 0 for p in positions):
                return True
            time.sleep(delay)
        return False

    def _persist_position(self, symbol, side, entry_price, size_usd, stop_price, take_profit_price, decision_id=None):
        """Write the open position to the Postgres ledger (no-op when db is None).
        Reuses the exact SQL PaperBroker uses, so translate_query handles the dialect.
        `decision_id` links the position back to the decision that opened it."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price, decision_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, side, datetime.utcnow(), entry_price, size_usd, stop_price, take_profit_price, decision_id),
            )
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: failed to persist position {symbol}: {e}")
        finally:
            self.db.close()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            raw = self.exchange.fetch_positions()
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: fetch_positions failed: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for p in raw:
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                continue
            ccxt_sym = p.get("symbol", "")
            plain = _to_plain_symbol(ccxt_sym)
            side = p.get("side")
            entry = float(p.get("entryPrice") or 0)
            notional = abs(float(p.get("notional") or (contracts * entry)))
            mark = float(p.get("markPrice") or 0) or None

            stop_price = None
            take_profit_price = None
            try:
                # closePosition TP/SL are CONDITIONAL (trigger) orders: the plain
                # fetch_open_orders returns nothing — params={'stop': True} fetches them.
                # ccxt normalizes their `type` to "market", so we classify TP vs SL by the
                # trigger price relative to the entry + side, not the type string.
                ref = entry or mark or 0.0
                for o in self.exchange.fetch_open_orders(ccxt_sym, params={"stop": True}):
                    trig = o.get("triggerPrice") or o.get("stopPrice")
                    if trig is None:
                        continue
                    trig = float(trig)
                    if side == "short":
                        is_tp = trig <= ref
                    else:  # long (default)
                        is_tp = trig >= ref
                    if is_tp:
                        take_profit_price = trig
                    else:
                        stop_price = trig
            except Exception as e:
                logger.warning(f"BinanceFuturesBroker: fetch conditional orders failed for {plain}: {e}")

            out.append({
                "symbol": plain,
                "side": p.get("side"),
                "entry_price": entry,
                "size_usd": notional,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "current_price": mark,
            })
        return out

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

    def _load_ledger(self) -> List[Dict[str, Any]]:
        """Read all open positions from the Postgres ledger ([] when db is None)."""
        if not self.db:
            return []
        try:
            self.db.connect()
            rows = self.db.conn.execute(
                "SELECT symbol, side, entry_time, entry_price, size_usd, stop_price, "
                "take_profit_price, decision_id FROM open_positions"
            ).fetchall()
            return [{
                "symbol": r[0], "side": r[1], "entry_time": r[2], "entry_price": r[3],
                "size_usd": r[4], "stop_price": r[5], "take_profit_price": r[6],
                "decision_id": r[7],
            } for r in rows]
        finally:
            self.db.close()

    def _delete_position(self, symbol: str) -> bool:
        """Remove a position from the Postgres ledger. Returns True if THIS call claimed
        (actually deleted) the row — the atomic gate that makes concurrent reconciles
        record a close exactly once. When db is None (backtest/test), returns True
        (nothing to contend over)."""
        if not self.db:
            return True
        try:
            self.db.connect()
            cur = self.db.conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
            return bool(getattr(cur, "rowcount", 0) and cur.rowcount > 0)
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: failed to delete ledger row {symbol}: {e}")
            return False
        finally:
            self.db.close()

    def close_position(self, symbol: str) -> Dict[str, Any]:
        sym = _to_ccxt_symbol(symbol)
        try:
            positions = self.exchange.fetch_positions([sym])
            pos = next((p for p in positions if float(p.get("contracts") or 0) != 0), None)
            if not pos:
                self._delete_position(symbol)
                return {"status": "rejected", "reason": "no open position"}
            contracts = abs(float(pos["contracts"]))
            opposite = "sell" if pos.get("side") == "long" else "buy"
            self.exchange.create_order(sym, "market", opposite, contracts, params={"reduceOnly": True})
            self.exchange.cancel_all_orders(sym)
            self._delete_position(symbol)
            logger.info(f"BinanceFuturesBroker: closed {symbol} ({contracts} contracts)")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: close_position failed for {symbol}: {e}")
            return {"status": "rejected", "reason": str(e)}

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Reconcile the Postgres ledger against live exchange positions. Any ledger
        symbol that is no longer open on the exchange was closed by its bracket →
        emit a closed_trade and drop it from the ledger. `current_prices` is ignored
        (the exchange is the source of truth). Never raises — returns [] on any error."""
        ledger = self._load_ledger()
        if not ledger:
            return []
        try:
            live = self.exchange.fetch_positions()
            open_syms = {
                _to_plain_symbol(p.get("symbol", ""))
                for p in live if float(p.get("contracts") or 0) != 0
            }
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: reconcile fetch_positions failed: {e}")
            return []

        closed: List[Dict[str, Any]] = []
        for row in ledger:
            if row["symbol"] in open_syms:
                continue
            try:
                trade = self._build_closed_trade(row)
                if self._delete_position(row["symbol"]):  # atomic claim → record once
                    closed.append(trade)
            except Exception as e:
                logger.error(f"BinanceFuturesBroker: failed to build closed trade for "
                             f"{row['symbol']}: {e}")
        return closed

    def _build_closed_trade(self, row: Dict[str, Any]) -> Dict[str, Any]:
        side = row["side"]
        entry_price = float(row["entry_price"])
        size_usd = float(row["size_usd"])
        entry_time = row["entry_time"]
        qty = size_usd / entry_price if entry_price else 0.0

        close_price = entry_price
        realized_pnl = 0.0
        try:
            since = int(entry_time.timestamp() * 1000) if hasattr(entry_time, "timestamp") else None
            fills = self.exchange.fetch_my_trades(_to_ccxt_symbol(row["symbol"]), since=since)
            exit_side = "sell" if side == "long" else "buy"
            closing = [f for f in fills if f.get("side") == exit_side]
            if closing:
                total_amt = sum(float(f["amount"]) for f in closing)
                total_cost = sum(float(f["price"]) * float(f["amount"]) for f in closing)
                close_price = total_cost / total_amt if total_amt else entry_price
                fees = sum(float((f.get("fee") or {}).get("cost") or 0) for f in closing)
                if side == "long":
                    realized_pnl = (close_price - entry_price) * qty - fees
                else:
                    realized_pnl = (entry_price - close_price) * qty - fees
        except Exception as e:
            logger.warning(f"BinanceFuturesBroker: could not fetch closing fills for "
                           f"{row['symbol']}: {e}")

        return {
            "trade_id": str(uuid4()),
            "symbol": row["symbol"],
            "action": side,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "close_time": datetime.utcnow(),
            "close_price": close_price,
            "size_usd": size_usd,
            "realized_pnl": realized_pnl,
            "result": "win" if realized_pnl > 0 else "loss",
            "decision_id": row.get("decision_id"),
        }
