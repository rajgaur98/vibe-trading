# Design Spec — User Data Stream Websocket (real-time fill bookkeeping)

> **Initiative:** real exchange execution. This is **sub-project 2 of 2**, building on
> sub-project 1 (`BinanceFuturesBroker`, native brackets — see
> `2026-06-06-binance-futures-broker-design.md`). It produces working software on its own:
> bracket-closed trades are recorded to the `trades` table + Discord **in real time**
> instead of waiting up to one 4h reconcile tick.

## Problem

Sub-project 1 made exits real-time on the *exchange* (native `TAKE_PROFIT_MARKET` /
`STOP_MARKET` brackets) and made the dashboard read exchange truth. But the **trade-history
log + Discord close alert** still only fire when `update_positions()` runs — on startup and
each 4h tick. So a position that hits TP at 00:05 isn't *recorded* (PnL booked, alert sent)
until the 04:01 tick — up to ~4h of lag in the books, even though the money already moved.

## Solution

A `UserDataStreamListener` runs in a **daemon thread** started from
`TradingScheduler.start()` (LIVE_TESTNET only). It holds its own asyncio loop calling
`await exchange.watch_orders()` on its **own ccxt.pro client** (`set_sandbox_mode(True)`);
ccxt.pro internally obtains the listenKey, sends the ~30-min keepalive, and reconnects on
drop. When a reduce-only / `closePosition` bracket order **fills**, the listener triggers the
broker's existing `update_positions()` reconcile **immediately** and forwards any closed
trades through the **same record path** the 4h tick uses. The websocket computes nothing
itself — it is a low-latency *trigger* for the exchange-truth reconcile. The 4h tick remains
the safety net; the exchange stays the single source of truth.

**Out of scope (always):** mainnet / real funds (testnet only); independent PnL math in the
ws handler (reconcile owns it); a separate ws process/container (in-process thread by design).

## Components

### 1. `src/vibe_trading/runtime/ws_listener.py` [NEW] — `UserDataStreamListener`

```python
import asyncio, logging, threading
import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


def _is_exit_fill(order: dict) -> bool:
    """True when an order update represents a filled position-closing (bracket) order —
    the signal that a position likely just closed and a reconcile should run now."""
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
        (its own sync ccxt client — never shared with the scheduler's broker).
        `record_fn(closed_trades: list)`: the scheduler's _record_closed_trades.
        `build_client`: optional factory returning a ccxt.pro client (injected in tests)."""
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

    def start(self):
        """Spawn the daemon thread running the asyncio loop. Idempotent."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()),
                                        name="ws-user-data-stream", daemon=True)
        self._thread.start()
        logger.info("UserDataStreamListener started.")

    async def _run(self):
        self._exchange = self._build_client()
        try:
            while self._running:
                try:
                    # On (re)connect, reconcile once to catch fills missed during any gap.
                    self._reconcile_and_record()
                    while self._running:
                        orders = await self._exchange.watch_orders()
                        self._handle_orders(orders)
                except Exception as e:
                    logger.warning(f"ws_listener stream error (will retry): {e}")
                    await asyncio.sleep(5)
        finally:
            try:
                await self._exchange.close()
            except Exception:
                pass

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

    def stop(self):
        """Best-effort graceful shutdown (the thread is a daemon, so this is optional)."""
        self._running = False
```

`import os` at module top (used by `_default_client`).

**Trigger semantics:** over-triggering is harmless — `update_positions()` is idempotent
(claim-delete, §3) and a redundant reconcile just re-reads exchange truth. So `_is_exit_fill`
errs toward triggering on any reduce-only / bracket fill rather than trying to perfectly
classify the event.

### 2. `src/vibe_trading/runtime/scheduler.py` [MODIFY]

**(a) Extract closed-trade recording.** The inline block in `sync_and_evaluate` that INSERTs
each closed trade into `trades` and sends the Discord close alert moves into a method:

```python
def _record_closed_trades(self, closed_trades: list):
    """Persist closed trades to `trades` and send Discord alerts. Thread-safe: opens its
    OWN pooled PostgresDatabase connection per call (the web layer uses this same pattern),
    so the 4h-tick thread and the ws thread can both call it concurrently."""
    if not closed_trades:
        return
    pg = PostgresDatabase()
    pg.connect()
    try:
        for trade in closed_trades:
            pg.conn.execute("""
                INSERT INTO trades (trade_id, symbol, action, entry_time, entry_price,
                                    close_time, close_price, size_usd, realized_pnl, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade["trade_id"], trade["symbol"], trade["action"], trade["entry_time"],
                  trade["entry_price"], trade["close_time"], trade["close_price"],
                  trade["size_usd"], trade["realized_pnl"], trade["result"]))
    finally:
        pg.close()
    for trade in closed_trades:
        self._send_discord_alert(
            f"🔄 **TRADE CLOSED:** {trade['symbol']} ({trade['action'].upper()})\n"
            f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['close_price']:.2f}\n"
            f"PnL: **${trade['realized_pnl']:.2f}** ({trade['result'].upper()})"
        )
```

The tick's existing block becomes `self._record_closed_trades(closed_trades)` (behavior
identical; it now uses a fresh pooled connection instead of `self.pg_db`, which is safe).

**(b) Start the listener (LIVE_TESTNET only).** In `start()`, before `scheduler.start()`:

```python
if os.getenv("TRADING_MODE", "PAPER").upper() == "LIVE_TESTNET":
    from vibe_trading.runtime.ws_listener import UserDataStreamListener
    # Its OWN PostgresDatabase() instance (not self.pg_db): PostgresDatabase holds a single
    # mutable .conn per instance, so the ws thread must not share the scheduler's. Both draw
    # from the same shared pool. The ccxt client is likewise its own (no cross-thread sharing).
    ws_broker = BinanceFuturesBroker(db=PostgresDatabase())
    self.ws_listener = UserDataStreamListener(ws_broker, self._record_closed_trades)
    self.ws_listener.start()
```

No other structural change. The startup/immediate `sync_and_evaluate()` (already in `start()`)
plus the listener's own on-connect reconcile both cover the "fills while the bot was down" case.

### 3. `src/vibe_trading/brokers/binance_futures.py` [MODIFY] — concurrency hardening

Make `update_positions` **idempotent under concurrent reconciles** (the ws thread and the 4h
tick can reconcile the same close at the same time). The ledger row delete becomes the atomic
**claim** that decides who records the trade:

- `_delete_position(symbol) -> bool`: after executing the `DELETE`, return whether a row was
  actually removed via the psycopg2 cursor `rowcount` (`cur.rowcount > 0`). The cursor is the
  object returned by `self.db.conn.execute(...)`. **When `self.db` is None** (no ledger to
  contend over — backtest / unit-test paths), return `True`: there is nothing to claim against,
  so the caller proceeds exactly as in sub-project 1. `close_position` already ignores the
  return value, so this change is backward-compatible there.
- `update_positions`: for each closed symbol, build the `closed_trade`, then call
  `_delete_position`; **append it to the results only if the delete claimed the row**
  (returned `True`). A concurrent reconciler that lost the claim gets `rowcount == 0` and
  skips it — so the trade is recorded exactly once.

This is the single idempotency gate; no app-level lock is needed because the two reconcile
paths use **separate** ccxt clients and **separate** pooled Postgres connections — the only
shared state is the `open_positions` row, and the DB delete arbitrates it atomically.

## Concurrency model

| Resource | Isolation |
|---|---|
| ccxt **sync** client | scheduler-broker and ws-reconcile-broker are **separate** instances; never shared across threads |
| ccxt.pro **async** client | owned solely by the ws thread's asyncio loop |
| Postgres | every write opens a fresh **pooled** connection (`_record_closed_trades`, `_delete_position` via the broker's own `PostgresDatabase`); no shared mutable `conn` |
| `open_positions` ledger row | the atomic claim-delete (`rowcount`) is the one gate that makes double-recording impossible |

**Invariant:** a websocket failure can never corrupt or block the scheduler — different
thread, different clients, different connections. Worst case it degrades to the existing
≤4h reconcile latency.

**Note:** `_reconcile_and_record` calls the synchronous `update_positions()` (REST
`fetch_positions`/`fetch_my_trades`) from inside the asyncio loop, briefly blocking
`watch_orders` for the duration of those calls. This is acceptable: the thread does nothing
else, fills are infrequent, and the call is a short REST round-trip. Keeping reconcile
synchronous (reusing the broker as-is) is simpler than offloading to a thread executor, and
the brief pause only delays the *next* event read, never drops it (ccxt.pro buffers).

## Data Flow (a TP fills intraday, LIVE_TESTNET)

```
exchange fills TP bracket (closePosition, reduce-only) → position flat
  → User Data Stream pushes ORDER_TRADE_UPDATE
  → ccxt.pro watch_orders() yields the filled reduce-only order
  → listener._handle_orders: _is_exit_fill? → ws_broker.update_positions({})
       reconcile: ledger has SYM, exchange shows flat → build closed_trade,
       claim-delete the ledger row (rowcount==1 → claimed)
  → scheduler._record_closed_trades(closed): INSERT trades + Discord  (SAME path as 4h tick)
[safety net]  4h tick also reconciles; if it races, claim-delete yields rowcount==0 → no dup
[reconnect]   on ws (re)connect, listener reconciles once to catch fills missed during the gap
[bot down]    start()'s immediate sync_and_evaluate + the listener's on-connect reconcile catch it
```

## Error Handling

| Scenario | Behavior |
|---|---|
| ws stream error / disconnect | caught in `_run`; log + `sleep(5)` + retry; ccxt.pro re-establishes the listenKey/connection. On reconnect, reconcile once. |
| listenKey expiry | handled inside ccxt.pro (keepalive + re-create); a surfaced error just re-loops. |
| reconcile (`update_positions`) error | caught in `_reconcile_and_record`; logged; the 4h tick will catch the close later. |
| `record_fn` (DB/Discord) error | DB insert wrapped by its own connection scope; a failure logs but never crashes the ws thread (and the 4h tick is the backstop). |
| not LIVE_TESTNET / missing creds | listener is never constructed/started — pure no-op. |
| duplicate reconcile (ws + tick) | atomic claim-delete records exactly once. |

## Testing (no live network / no real asyncio in pytest)

`tests/test_ws_listener.py`:
1. `_is_exit_fill`: filled reduce-only order → True; filled `STOP_MARKET`/`TAKE_PROFIT_MARKET`
   → True; a filled **entry** (non-reduce-only market) → False; an **open** (unfilled) bracket
   → False.
2. `_handle_orders` with an exit-fill event + a mock broker (`update_positions` returns one
   closed trade) + a mock `record_fn` → asserts `update_positions` called once and `record_fn`
   received the closed trades.
3. `_handle_orders` with only non-exit events → `update_positions` **not** called.
4. `_reconcile_and_record` swallows a broker exception (no raise; `record_fn` not called).
5. `start()` spawns a thread and is idempotent (second `start()` doesn't spawn a second);
   `stop()` clears `_running`. (`_run` is patched / `build_client` injected so no real ws.)

`tests/test_binance_futures.py` (extend):
6. claim-delete idempotency: a fake db whose delete reports `rowcount` 1 then 0 → the first
   `update_positions` returns the closed trade, the second returns `[]` for the same symbol.

`tests/test_scheduler.py` (extend):
7. `_record_closed_trades`: patch `PostgresDatabase` + `_send_discord_alert`; assert one INSERT
   and one Discord alert per closed trade, and that it opens/closes its own connection.

**Live verification (manual, needs your testnet keys):** with `TRADING_MODE=LIVE_TESTNET`,
open a position via `trade-once` (or the smoke script), then trigger its TP/SL on
`testnet.binancefuture.com`. Confirm the close is recorded in `trades` + a Discord alert fires
within seconds — not at the next 4h tick.

## Backwards Compatibility

- Additive: PAPER / LIVE_SANDBOX unchanged; the listener only starts in LIVE_TESTNET.
- `_record_closed_trades` is a pure extraction of the existing tick block — same SQL, same
  Discord message — so the 4h-tick behavior is unchanged (it just uses a fresh pooled
  connection, which the connection pool already supports).
- `update_positions` becomes idempotent; its single-reconcile behavior (the only case in
  sub-project 1) is unchanged — the claim simply always succeeds when there's no contender.
- No new dependency: `ccxt.pro` ships inside the existing `ccxt>=4.2.0` (bundled, free).
