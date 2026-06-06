# User Data Stream Websocket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record bracket-closed trades to the `trades` table + Discord **in real time** (instead of waiting up to one 4h reconcile tick) by triggering the existing exchange-truth reconcile the instant a `closePosition` order fills on the Binance futures testnet.

**Architecture:** A `UserDataStreamListener` runs in a daemon thread started from `TradingScheduler.start()` (LIVE_TESTNET only). Its asyncio loop calls `await exchange.watch_orders()` on its own ccxt.pro client; on a reduce-only/bracket fill it triggers `BinanceFuturesBroker.update_positions()` (exchange truth) and forwards any closed trades through the same `_record_closed_trades` path the 4h tick uses. Idempotency under concurrent reconciles comes from an atomic claim-delete on the `open_positions` ledger row.

**Tech Stack:** Python 3.13, `ccxt.pro` (bundled in the existing `ccxt>=4.2.0`, v4.5.54 — `watch_orders`, `set_sandbox_mode`), `asyncio` + `threading`, `psycopg2` pooled Postgres, `pytest` + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-06-06-user-data-stream-websocket-design.md`

---

## File Structure

- **Modify `src/vibe_trading/brokers/binance_futures.py`** — make `update_positions` idempotent: `_delete_position` returns whether it actually claimed (deleted) the ledger row (`rowcount`; `db is None → True`), and `update_positions` records a closed trade only if it claimed the row.
- **Modify `src/vibe_trading/runtime/scheduler.py`** — extract `_record_closed_trades` (own pooled connection, thread-safe), used by both the 4h tick and the ws callback; add `_maybe_start_ws_listener` and call it from `start()` (LIVE_TESTNET only).
- **Create `src/vibe_trading/runtime/ws_listener.py`** — `_is_exit_fill` (pure) + `UserDataStreamListener` (own ccxt.pro client, daemon thread, reconcile trigger).
- **Create `tests/test_ws_listener.py`** — `_is_exit_fill`, `_handle_orders`, `_reconcile_and_record`, `start`/`stop`.
- **Modify `tests/test_binance_futures.py`** — claim-delete semantics + idempotency.
- **Modify `tests/test_scheduler.py`** — `_record_closed_trades` + `_maybe_start_ws_listener` gating.
- **Modify `README.md`** — flip the "websocket is a follow-on" note to "implemented".

**Interface decisions (locked here, used across tasks):**
- `_delete_position(symbol) -> bool` (was `-> None`): `True` when a row was claimed/deleted (or when `db is None`), else `False`. `close_position` ignores the return (unchanged behavior).
- `UserDataStreamListener(reconcile_broker, record_fn, build_client=None)` — `reconcile_broker` is a `BinanceFuturesBroker` used only for `update_positions()`; `record_fn(closed_trades: list)` is `scheduler._record_closed_trades`; `build_client` is an injectable factory returning a ccxt.pro client (mocked in tests).
- The ws reconcile broker is constructed with its **own** `PostgresDatabase()` instance (NOT the scheduler's `self.pg_db`), because `PostgresDatabase` stores a single mutable `.conn` per instance — sharing it across the scheduler thread and the ws thread would race. Both instances draw from the same shared pool. *(This corrects a detail in the spec, which wrote `db=self.pg_db`.)*

---

### Task 1: Broker idempotent claim-delete

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_binance_futures.py`:

```python
def test_delete_position_claim_semantics():
    # db=None → True (nothing to contend over)
    broker = BinanceFuturesBroker(db=None, exchange=_mock_exchange())
    assert broker._delete_position("BTC/USDT") is True

    # db present → returns rowcount > 0
    fake_cur = MagicMock()
    fake_cur.rowcount = 1
    fake_conn = MagicMock()
    fake_conn.execute.return_value = fake_cur
    fake_db = MagicMock()
    fake_db.conn = fake_conn
    broker2 = BinanceFuturesBroker(db=fake_db, exchange=_mock_exchange())
    assert broker2._delete_position("BTC/USDT") is True
    fake_cur.rowcount = 0
    assert broker2._delete_position("BTC/USDT") is False


def test_update_positions_idempotent_under_concurrent_claim():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = []  # symbol flat on exchange
    ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 110.0, "amount": 10.0, "fee": {"cost": 0.0}},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    row = {
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }
    broker._load_ledger = lambda: [row]
    # First reconcile claims the row (True); a racing second one loses it (False).
    claims = iter([True, False])
    broker._delete_position = lambda symbol: next(claims)

    first = broker.update_positions({})
    second = broker.update_positions({})
    assert len(first) == 1 and first[0]["symbol"] == "BTC/USDT"
    assert second == []  # not recorded twice
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_binance_futures.py -k "claim or idempotent" -v`
Expected: FAIL — `_delete_position` currently returns `None` (so `is True` fails) and `update_positions` appends before/without checking the claim.

- [ ] **Step 3: Make `_delete_position` return a claim bool**

In `src/vibe_trading/brokers/binance_futures.py`, replace the whole `_delete_position` method:

```python
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
```

- [ ] **Step 4: Gate recording on the claim in `update_positions`**

In the same file, replace the `for row in ledger:` loop body inside `update_positions` with:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass (and no regression)**

Run: `PYTHONPATH=src uv run pytest tests/test_binance_futures.py -v`
Expected: PASS — the two new tests plus all prior broker tests (`test_update_positions_reconciles_closed_trade` still passes because `db is None → _delete_position returns True`).

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): idempotent update_positions via atomic claim-delete"
```

---

### Task 2: Scheduler `_record_closed_trades` extraction

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scheduler.py` (add this import at the top of the file if missing):

```python
from datetime import datetime
```

```python
def test_record_closed_trades_inserts_and_alerts(monkeypatch):
    sched = _scheduler_without_init()
    fake_conn = MagicMock()
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    factory = MagicMock(return_value=fake_pg)
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    trades = [{
        "trade_id": "t1", "symbol": "BTC/USDT", "action": "long",
        "entry_time": datetime(2026, 6, 1), "entry_price": 100.0,
        "close_time": datetime(2026, 6, 2), "close_price": 110.0,
        "size_usd": 1000.0, "realized_pnl": 99.6, "result": "win",
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1            # one INSERT
    assert fake_pg.connect.called and fake_pg.close.called  # own connection lifecycle
    assert len(alerts) == 1 and "BTC/USDT" in alerts[0]


def test_record_closed_trades_empty_is_noop(monkeypatch):
    sched = _scheduler_without_init()
    factory = MagicMock()
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    sched._record_closed_trades([])
    assert factory.call_count == 0  # no connection opened
    assert alerts == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -k record_closed -v`
Expected: FAIL with `AttributeError: 'TradingScheduler' object has no attribute '_record_closed_trades'`

- [ ] **Step 3: Add the `_record_closed_trades` method**

In `src/vibe_trading/runtime/scheduler.py`, add this method just above `_resolve_exec_price`:

```python
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
                    INSERT INTO trades (trade_id, symbol, action, entry_time, entry_price, close_time, close_price, size_usd, realized_pnl, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade["trade_id"], trade["symbol"], trade["action"], trade["entry_time"], trade["entry_price"],
                      trade["close_time"], trade["close_price"], trade["size_usd"], trade["realized_pnl"], trade["result"]))
        finally:
            pg.close()
        for trade in closed_trades:
            self._send_discord_alert(
                f"🔄 **TRADE CLOSED:** {trade['symbol']} ({trade['action'].upper()})\n"
                f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['close_price']:.2f}\n"
                f"PnL: **${trade['realized_pnl']:.2f}** ({trade['result'].upper()})"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -k record_closed -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Replace the inline tick block with the new method**

In `sync_and_evaluate`, replace this block (currently lines ~126-143):

```python
                if closed_trades:
                    self.pg_db.connect()
                    try:
                        for trade in closed_trades:
                            # Log closed trade to DB
                            self.pg_db.conn.execute("""
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
                        self.pg_db.close()
```

with:

```python
                self._record_closed_trades(closed_trades)
```

- [ ] **Step 6: Run the full suite to verify no regression**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py tests/test_scheduler.py
git commit -m "refactor(scheduler): extract thread-safe _record_closed_trades"
```

---

### Task 3: `ws_listener.py` — `_is_exit_fill` pure function

**Files:**
- Create: `src/vibe_trading/runtime/ws_listener.py`
- Test: `tests/test_ws_listener.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ws_listener.py`:

```python
"""Unit tests for the User Data Stream listener. No real websocket / asyncio / network:
the ccxt.pro client is injected via build_client, and _handle_orders/_is_exit_fill are
sync and tested directly."""
from unittest.mock import MagicMock

from vibe_trading.runtime.ws_listener import _is_exit_fill, UserDataStreamListener


def test_is_exit_fill_filled_reduce_only_true():
    assert _is_exit_fill({"status": "closed", "reduceOnly": True, "type": "market"}) is True


def test_is_exit_fill_filled_bracket_type_true():
    assert _is_exit_fill({"status": "filled", "type": "take_profit_market"}) is True
    assert _is_exit_fill({"status": "closed", "type": "stop_market"}) is True


def test_is_exit_fill_filled_entry_false():
    # a filled non-reduce-only market entry is NOT an exit
    assert _is_exit_fill({"status": "closed", "reduceOnly": False, "type": "market"}) is False


def test_is_exit_fill_open_bracket_false():
    # an unfilled (resting) bracket order is not a fill
    assert _is_exit_fill({"status": "open", "type": "stop_market"}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -k is_exit_fill -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.runtime.ws_listener'`

- [ ] **Step 3: Create the module with imports + `_is_exit_fill`**

Create `src/vibe_trading/runtime/ws_listener.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -k is_exit_fill -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/ws_listener.py tests/test_ws_listener.py
git commit -m "feat(ws): _is_exit_fill bracket-fill classifier"
```

---

### Task 4: `UserDataStreamListener` core (handle + reconcile)

**Files:**
- Modify: `src/vibe_trading/runtime/ws_listener.py`
- Test: `tests/test_ws_listener.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ws_listener.py`:

```python
def _listener(broker, record_fn):
    return UserDataStreamListener(broker, record_fn, build_client=lambda: MagicMock())


def test_handle_orders_exit_fill_triggers_reconcile_and_records():
    broker = MagicMock()
    broker.update_positions.return_value = [{"symbol": "BTC/USDT", "realized_pnl": 5.0}]
    recorded = []
    listener = _listener(broker, recorded.append)

    listener._handle_orders([{"status": "closed", "reduceOnly": True, "type": "take_profit_market"}])

    broker.update_positions.assert_called_once_with({})
    assert recorded == [[{"symbol": "BTC/USDT", "realized_pnl": 5.0}]]


def test_handle_orders_non_exit_does_nothing():
    broker = MagicMock()
    listener = _listener(broker, lambda c: None)
    listener._handle_orders([{"status": "open", "type": "limit"}])
    broker.update_positions.assert_not_called()


def test_handle_orders_exit_fill_no_closed_trades_skips_record():
    broker = MagicMock()
    broker.update_positions.return_value = []  # reconcile found nothing newly closed
    recorded = []
    listener = _listener(broker, recorded.append)
    listener._handle_orders([{"status": "closed", "reduceOnly": True}])
    broker.update_positions.assert_called_once_with({})
    assert recorded == []  # nothing to record


def test_reconcile_and_record_swallows_broker_error():
    broker = MagicMock()
    broker.update_positions.side_effect = Exception("boom")
    recorded = []
    listener = _listener(broker, recorded.append)
    listener._reconcile_and_record()  # must not raise
    assert recorded == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -k "handle_orders or reconcile" -v`
Expected: FAIL with `TypeError: UserDataStreamListener() takes no arguments` (class not defined yet)

- [ ] **Step 3: Add the class core**

Append to `src/vibe_trading/runtime/ws_listener.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -v`
Expected: PASS (all `_is_exit_fill` + handle/reconcile tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/ws_listener.py tests/test_ws_listener.py
git commit -m "feat(ws): UserDataStreamListener reconcile-trigger core"
```

---

### Task 5: `UserDataStreamListener` lifecycle (`start`/`stop`/`_run`)

**Files:**
- Modify: `src/vibe_trading/runtime/ws_listener.py`
- Test: `tests/test_ws_listener.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ws_listener.py`:

```python
import time


def test_start_is_idempotent_and_stop_clears_running(monkeypatch):
    broker = MagicMock()
    listener = _listener(broker, lambda c: None)

    async def _noop():
        return  # don't open a real websocket

    monkeypatch.setattr(listener, "_run", _noop)

    listener.start()
    assert listener._running is True
    first_thread = listener._thread
    assert first_thread is not None

    listener.start()  # idempotent: must NOT spawn a second thread
    assert listener._thread is first_thread

    listener.stop()
    assert listener._running is False
    time.sleep(0.05)  # let the daemon thread wind down
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -k start_is_idempotent -v`
Expected: FAIL with `AttributeError: 'UserDataStreamListener' object has no attribute 'start'`

- [ ] **Step 3: Add `start`, `_run`, `stop`**

Append to the `UserDataStreamListener` class in `src/vibe_trading/runtime/ws_listener.py`:

```python
    def start(self):
        """Spawn the daemon thread running the asyncio loop. Idempotent."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run()),
            name="ws-user-data-stream", daemon=True,
        )
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

    def stop(self):
        """Best-effort graceful shutdown (the thread is a daemon, so this is optional)."""
        self._running = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_ws_listener.py -v`
Expected: PASS (all listener tests). `_run` itself is exercised by manual live verification (it needs a real websocket); the unit tests cover the logic it calls.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/ws_listener.py tests/test_ws_listener.py
git commit -m "feat(ws): listener start/stop + asyncio watch_orders loop"
```

---

### Task 6: Wire the listener into the scheduler + docs + full verification

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py`
- Modify: `README.md`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scheduler.py`:

```python
def test_maybe_start_ws_listener_none_when_not_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    sched = _scheduler_without_init()
    assert sched._maybe_start_ws_listener() is None


def test_maybe_start_ws_listener_starts_in_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", lambda *a, **k: MagicMock())
    monkeypatch.setattr("vibe_trading.runtime.scheduler.BinanceFuturesBroker", lambda *a, **k: MagicMock())

    started = {}

    class FakeListener:
        def __init__(self, broker, record_fn):
            started["init"] = True

        def start(self):
            started["start"] = True

    monkeypatch.setattr("vibe_trading.runtime.ws_listener.UserDataStreamListener", FakeListener)

    listener = sched._maybe_start_ws_listener()
    assert isinstance(listener, FakeListener)
    assert started.get("init") and started.get("start")


def test_maybe_start_ws_listener_failopen_returns_none(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None

    def _boom(*a, **k):
        raise RuntimeError("no creds")

    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", _boom)
    # A listener-construction failure must NOT propagate (scheduler keeps running).
    assert sched._maybe_start_ws_listener() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -k maybe_start_ws -v`
Expected: FAIL with `AttributeError: 'TradingScheduler' object has no attribute '_maybe_start_ws_listener'`

- [ ] **Step 3: Add `_maybe_start_ws_listener`**

In `src/vibe_trading/runtime/scheduler.py`, add this method just above `_record_closed_trades`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -k maybe_start_ws -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Call it from `start()`**

In `start()`, insert the listener start right after the immediate `self.sync_and_evaluate()` and before building the `BlockingScheduler`:

```python
    def start(self):
        """Starts the main scheduling loop."""
        # 1. Run immediate bootstrap/sync on startup
        logger.info("Initializing startup data synchronization...")
        self.sync_and_evaluate()

        # 1b. Real-time fill bookkeeping via the User Data Stream websocket (LIVE_TESTNET only)
        self.ws_listener = self._maybe_start_ws_listener()

        # 2. Setup recurring 4-hour scheduler
        scheduler = BlockingScheduler()
```

(Leave the rest of `start()` unchanged.)

- [ ] **Step 6: Update the README follow-on note**

In `README.md`, in the "Live Testnet Execution" section, replace the trailing blockquote:

```markdown
> Real-time fill push via the User Data Stream websocket is a separate follow-on; until
> then the trade-history log + close alert may lag up to one 4h tick (the exit itself and
> the dashboard are already real-time).
```

with:

```markdown
> **Real-time bookkeeping:** a User Data Stream websocket listener (ccxt.pro `watch_orders`,
> a daemon thread started with the scheduler) records bracket-closed trades to `trades` +
> Discord within seconds of the fill, instead of at the next 4h tick. The 4h reconcile
> remains the safety net, and an atomic ledger claim-delete prevents any double-recording.
```

- [ ] **Step 7: Run the FULL suite**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: PASS — the prior 176 tests plus the new ws-listener (~9), broker idempotency (2), and scheduler (5) tests, no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py README.md tests/test_scheduler.py
git commit -m "feat(scheduler): start User Data Stream listener in LIVE_TESTNET"
```

---

## Manual Live Verification (you, after merge — needs your testnet keys)

1. With testnet keys in `.env` and `TRADING_MODE=LIVE_TESTNET`, start the bot (`docker compose up -d vibe-bot` or `uv run python -m vibe_trading.cli live`). Confirm the log line `UserDataStreamListener started.`
2. Open a position (`trade-once` or `scripts/binance_testnet_smoke.py`).
3. On `testnet.binancefuture.com`, manually move the mark through the TP or SL (or wait for a real trigger).
4. Confirm the close is recorded in the `trades` table **and** a Discord "TRADE CLOSED" alert fires within seconds — not at the next 4h tick.
5. Bounce the bot mid-position and confirm the on-connect reconcile (and the immediate `sync_and_evaluate`) catch any fill that happened while it was down, recording it exactly once (no duplicate row).
```
