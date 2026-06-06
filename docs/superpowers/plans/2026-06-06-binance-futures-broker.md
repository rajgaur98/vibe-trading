# Binance Futures Testnet Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `BinanceFuturesBroker` that executes real orders on the Binance USDⓂ Futures **testnet** with native exchange brackets (TP + SL), so exits fill in real time even when the bot is offline.

**Architecture:** A new `BinanceFuturesBroker(BaseBroker)` talks to `testnet.binancefuture.com` via authenticated `ccxt` (`set_sandbox_mode(True)`, leverage pinned 1×). Entry = market order + two reduce-only `closePosition` bracket orders (`TAKE_PROFIT_MARKET`, `STOP_MARKET`); the exchange fills whichever triggers first and auto-cancels the sibling. The Postgres `open_positions` table is the reconciliation ledger; closed trades are detected by comparing the ledger to live exchange positions each tick. Selected via `TRADING_MODE=LIVE_TESTNET`. TA stays on **spot** candles; only the execution-critical price aligns to the futures mark via a new optional `BaseBroker.get_mark_price()`.

**Tech Stack:** Python 3.13, `ccxt>=4.2.0` (already a dependency), `psycopg2` Postgres (existing `PostgresDatabase`), `pytest` + `unittest.mock` (tests inject a mocked ccxt exchange — no live network calls in pytest).

**Spec:** `docs/superpowers/specs/2026-06-06-binance-futures-broker-design.md`

---

## File Structure

- **Create `src/vibe_trading/brokers/binance_futures.py`** — the entire broker: `_to_ccxt_symbol` helper + `BinanceFuturesBroker(BaseBroker)` (constructor with ccxt-exchange dependency injection for testability, the five `BaseBroker` methods, `get_mark_price`, and private ledger helpers `_persist_position`/`_load_ledger`/`_delete_position`/`_build_closed_trade`).
- **Modify `src/vibe_trading/brokers/base.py`** — add one optional concrete method `get_mark_price` (default `None`) so existing brokers need no change.
- **Modify `src/vibe_trading/runtime/scheduler.py`** — broker-selection branch for `LIVE_TESTNET`; a `_resolve_exec_price` helper; feed the resolved exec price into `trader.decide` + `risk_manager.evaluate_proposal`.
- **Modify `src/vibe_trading/web/main.py`** — `/api/positions` reads the exchange directly in `LIVE_TESTNET` mode (via a unit-testable `live_testnet_positions()` helper), falling back to the Postgres path on any error.
- **Create `tests/test_binance_futures.py`** — all broker unit tests (mocked ccxt).
- **Modify `tests/test_paper_broker.py`** — assert `PaperBroker.get_mark_price` returns `None`.
- **Create `scripts/binance_testnet_smoke.py`** — manual live verification (needs your testnet keys; not run in pytest).
- **Modify `.env.example`** and **`README.md`** — config + docs.

**Key interface decisions (locked here, used by every task):**
- Plain symbols (`"BTC/USDT"`) cross the `BaseBroker` boundary; `BinanceFuturesBroker` converts to/from ccxt's `"BTC/USDT:USDT"` internally. `get_open_positions` returns **plain** symbols.
- Constructor signature: `BinanceFuturesBroker(db=None, exchange=None)`. When `exchange` is provided (tests), it is used as-is and **no** ccxt construction / network / creds check happens. When `exchange is None` (production), it builds the real ccxt futures client, calls `set_sandbox_mode(True)` and `load_markets()`, and requires creds unless dry-run.
- `get_open_positions()` dict shape (matches the dashboard + PaperBroker): `{"symbol", "side", "entry_price", "size_usd", "stop_price", "take_profit_price", "current_price"}`.
- `update_positions()` closed-trade dict shape (matches the `trades` table + the scheduler's existing INSERT): `{"trade_id", "symbol", "action", "entry_time", "entry_price", "close_time", "close_price", "size_usd", "realized_pnl", "result"}`.
- The broker exposes a `peak_balance` attribute (the scheduler reads `self.broker.peak_balance` for the drawdown circuit breaker), updated inside `get_balance()`.

---

### Task 1: `BaseBroker.get_mark_price` optional default

**Files:**
- Modify: `src/vibe_trading/brokers/base.py`
- Test: `tests/test_paper_broker.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_paper_broker.py`:

```python
def test_paper_broker_get_mark_price_is_none():
    """PaperBroker inherits the BaseBroker default get_mark_price() -> None, so the
    scheduler falls back to the DuckDB spot close (PAPER/eval behavior unchanged)."""
    broker = PaperBroker(initial_balance=10000.0, db=None)
    assert broker.get_mark_price("BTC/USDT") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_paper_broker.py::test_paper_broker_get_mark_price_is_none -v`
Expected: FAIL with `AttributeError: 'PaperBroker' object has no attribute 'get_mark_price'`

- [ ] **Step 3: Add the optional method to BaseBroker**

In `src/vibe_trading/brokers/base.py`, change the import line and add a concrete (non-abstract) method at the end of the class. The full new import line:

```python
from typing import Dict, Any, List, Optional
```

Add this method after `update_positions` (it is intentionally **not** decorated with `@abstractmethod`, so existing brokers inherit it unchanged):

```python
    def get_mark_price(self, symbol: str) -> Optional[float]:
        """Execution-critical price for `symbol`, or None if this broker has no
        live price source. The scheduler uses this in LIVE_TESTNET to align entry/
        proximity decisions to the traded instrument's mark; brokers that return
        None cause the scheduler to fall back to the DuckDB spot close.
        """
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_paper_broker.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/base.py tests/test_paper_broker.py
git commit -m "feat(broker): add optional BaseBroker.get_mark_price (default None)"
```

---

### Task 2: `_to_ccxt_symbol` + broker constructor

**Files:**
- Create: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_binance_futures.py`:

```python
"""Unit tests for BinanceFuturesBroker. A MagicMock ccxt exchange is injected via the
constructor's `exchange=` param, so no live network calls happen in pytest.
"""
import os
from unittest.mock import MagicMock

import pytest

from vibe_trading.brokers.binance_futures import BinanceFuturesBroker, _to_ccxt_symbol


def _mock_exchange():
    """A ccxt-like mock with sensible defaults for the happy path."""
    ex = MagicMock()
    # precision helpers echo their input (as ccxt does, but returning a string)
    ex.amount_to_precision.side_effect = lambda sym, x: f"{float(x):.6f}"
    ex.price_to_precision.side_effect = lambda sym, x: f"{float(x):.2f}"
    # generous limits so the happy path is not rejected
    ex.market.return_value = {"limits": {"cost": {"min": 5.0}, "amount": {"min": 0.0001}}}
    ex.fetch_ticker.return_value = {"last": 100.0}
    # market entry fills at avg 100.0
    ex.create_order.return_value = {"id": "x1", "average": 100.0, "price": 100.0}
    return ex


def test_to_ccxt_symbol():
    assert _to_ccxt_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert _to_ccxt_symbol("ETH/USDT") == "ETH/USDT:USDT"


def test_init_injected_exchange_does_not_touch_network():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.exchange is ex
    ex.set_sandbox_mode.assert_not_called()  # injection path skips real setup
    ex.load_markets.assert_not_called()


def test_init_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "false")
    with pytest.raises(ValueError, match="BINANCE_TESTNET_API_KEY"):
        BinanceFuturesBroker(db=None)  # no injection → real path → creds required
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.brokers.binance_futures'`

- [ ] **Step 3: Create the module with the helper + constructor**

Create `src/vibe_trading/brokers/binance_futures.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): BinanceFuturesBroker constructor + symbol helpers"
```

---

### Task 3: `submit_order` — entry + native brackets (long & short)

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_binance_futures.py`:

```python
def test_submit_order_long_places_entry_and_brackets():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "success"
    assert res["entry_price"] == 100.0
    ex.set_leverage.assert_called_once_with(1, "BTC/USDT:USDT")

    # Three orders: market entry, TAKE_PROFIT_MARKET, STOP_MARKET
    calls = ex.create_order.call_args_list
    assert len(calls) == 3

    # 1) market BUY of size_usd/mark = 1000/100 = 10.0 (precision-rounded)
    a0 = calls[0]
    assert a0.args[0] == "BTC/USDT:USDT"
    assert a0.args[1] == "market"
    assert a0.args[2] == "buy"
    assert float(a0.args[3]) == 10.0

    # 2) TAKE_PROFIT_MARKET SELL closePosition @ tp
    a1 = calls[1]
    assert a1.args[1] == "TAKE_PROFIT_MARKET"
    assert a1.args[2] == "sell"
    assert a1.kwargs["params"]["closePosition"] is True
    assert float(a1.kwargs["params"]["stopPrice"]) == 110.0

    # 3) STOP_MARKET SELL closePosition @ sl
    a2 = calls[2]
    assert a2.args[1] == "STOP_MARKET"
    assert a2.args[2] == "sell"
    assert a2.kwargs["params"]["closePosition"] is True
    assert float(a2.kwargs["params"]["stopPrice"]) == 95.0


def test_submit_order_short_flips_sides():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker.submit_order(
        symbol="ETH/USDT", action="short", size_usd=1000.0,
        stop_price=110.0, take_profit_price=90.0, entry_price=100.0,
    )
    calls = ex.create_order.call_args_list
    assert calls[0].args[2] == "sell"   # entry SELL for a short
    assert calls[1].args[1] == "TAKE_PROFIT_MARKET"
    assert calls[1].args[2] == "buy"    # exit side BUY
    assert calls[2].args[1] == "STOP_MARKET"
    assert calls[2].args[2] == "buy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k submit_order -v`
Expected: FAIL with `AttributeError: 'BinanceFuturesBroker' object has no attribute 'submit_order'`

- [ ] **Step 3: Implement `submit_order` + `_persist_position`**

Append to the `BinanceFuturesBroker` class in `src/vibe_trading/brokers/binance_futures.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k submit_order -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): submit_order with native TP/SL brackets"
```

---

### Task 4: `submit_order` — min-notional rejection, dry_run, precision

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py` (no code change expected — logic already implemented in Task 3; this task verifies it)
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_binance_futures.py`:

```python
def test_submit_order_rejects_below_min_notional():
    ex = _mock_exchange()
    ex.market.return_value = {"limits": {"cost": {"min": 5000.0}, "amount": {"min": 0.0001}}}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=100.0,  # notional 100 < min 5000
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "rejected"
    assert "minimum" in res["reason"]
    ex.create_order.assert_not_called()  # no entry order placed


def test_submit_order_dry_run_places_nothing(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "true")
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.0, take_profit_price=110.0, entry_price=100.0,
    )
    assert res["status"] == "dry_run"
    ex.create_order.assert_not_called()


def test_submit_order_rounds_via_precision_helpers():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker.submit_order(
        symbol="BTC/USDT", action="long", size_usd=1000.0,
        stop_price=95.123456, take_profit_price=110.987654, entry_price=100.0,
    )
    # stopPrice on the bracket orders must come from price_to_precision (2dp here)
    calls = ex.create_order.call_args_list
    assert calls[1].kwargs["params"]["stopPrice"] == "110.99"
    assert calls[2].kwargs["params"]["stopPrice"] == "95.12"
    ex.amount_to_precision.assert_called()  # qty rounded too
```

- [ ] **Step 2: Run tests to verify they pass (logic already present)**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k "min_notional or dry_run or precision" -v`
Expected: PASS (3 tests). If `test_submit_order_dry_run_places_nothing` fails because `BINANCE_TESTNET_DRY_RUN` was read at construction before `monkeypatch.setenv`, confirm the test sets the env var **before** constructing the broker (it does). PASS.

- [ ] **Step 3: (No implementation needed)**

Task 3 already implemented min-notional rejection, dry_run, and precision rounding. If any test fails, fix `submit_order` to satisfy it rather than weakening the test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_binance_futures.py
git commit -m "test(broker): submit_order rejection/dry_run/precision coverage"
```

---

### Task 5: `get_balance` (+ peak_balance) and `get_mark_price`

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_binance_futures.py`:

```python
def test_get_mark_price_returns_last():
    ex = _mock_exchange()
    ex.fetch_ticker.return_value = {"last": 123.45}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_mark_price("BTC/USDT") == 123.45
    ex.fetch_ticker.assert_called_with("BTC/USDT:USDT")


def test_get_mark_price_none_on_error():
    ex = _mock_exchange()
    ex.fetch_ticker.side_effect = Exception("network down")
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_mark_price("BTC/USDT") is None


def test_get_balance_dry_run_is_10000(monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET_DRY_RUN", "true")
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_balance() == 10000.0
    assert broker.peak_balance == 10000.0  # peak tracked in-memory


def test_get_balance_reads_usdt_total_and_tracks_peak():
    ex = _mock_exchange()
    ex.fetch_balance.return_value = {"USDT": {"total": 8500.0}}
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_balance() == 8500.0
    assert broker.peak_balance == 8500.0
    # balance drops; peak holds
    ex.fetch_balance.return_value = {"USDT": {"total": 8000.0}}
    assert broker.get_balance() == 8000.0
    assert broker.peak_balance == 8500.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k "mark_price or get_balance" -v`
Expected: FAIL with `AttributeError` on `get_balance` / `get_mark_price` not being the overridden versions.

- [ ] **Step 3: Implement `get_balance` and `get_mark_price`**

Append to the `BinanceFuturesBroker` class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k "mark_price or get_balance" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): get_balance (peak tracking) + get_mark_price"
```

---

### Task 6: `get_open_positions` — map exchange state to dashboard shape

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_binance_futures.py`:

```python
def test_get_open_positions_maps_exchange_and_brackets():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long",
         "entryPrice": 100.0, "notional": 50.0, "markPrice": 105.0},
        {"symbol": "DOGE/USDT:USDT", "contracts": 0.0},  # flat → skipped
    ]
    ex.fetch_open_orders.return_value = [
        {"type": "stop_market", "stopPrice": 95.0},
        {"type": "take_profit_market", "stopPrice": 110.0},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    positions = broker.get_open_positions()

    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "BTC/USDT"          # un-converted (plain)
    assert p["side"] == "long"
    assert p["entry_price"] == 100.0
    assert p["size_usd"] == 50.0
    assert p["stop_price"] == 95.0
    assert p["take_profit_price"] == 110.0
    assert p["current_price"] == 105.0


def test_get_open_positions_empty_on_error():
    ex = _mock_exchange()
    ex.fetch_positions.side_effect = Exception("boom")
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    assert broker.get_open_positions() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k open_positions -v`
Expected: FAIL — `get_open_positions` not yet overridden / `AttributeError`.

- [ ] **Step 3: Implement `get_open_positions`**

Append to the `BinanceFuturesBroker` class:

```python
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
            entry = float(p.get("entryPrice") or 0)
            notional = abs(float(p.get("notional") or (contracts * entry)))
            mark = float(p.get("markPrice") or 0) or None

            stop_price = None
            take_profit_price = None
            try:
                for o in self.exchange.fetch_open_orders(ccxt_sym):
                    otype = (o.get("type") or "").upper()
                    sp = o.get("stopPrice") or (o.get("info", {}) or {}).get("stopPrice")
                    if sp is None:
                        continue
                    if "TAKE_PROFIT" in otype:
                        take_profit_price = float(sp)
                    elif "STOP" in otype:
                        stop_price = float(sp)
            except Exception as e:
                logger.warning(f"BinanceFuturesBroker: fetch_open_orders failed for {plain}: {e}")

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k open_positions -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): get_open_positions maps exchange + brackets to dashboard shape"
```

---

### Task 7: Ledger helpers + `close_position`

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_binance_futures.py`:

```python
def test_close_position_reduce_only_and_cancels_brackets():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5, "side": "long"},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.close_position("BTC/USDT")

    assert res["status"] == "success"
    # reduce-only market SELL of the abs contracts
    close_call = ex.create_order.call_args_list[-1]
    assert close_call.args[0] == "BTC/USDT:USDT"
    assert close_call.args[1] == "market"
    assert close_call.args[2] == "sell"
    assert float(close_call.args[3]) == 0.5
    assert close_call.kwargs["params"]["reduceOnly"] is True
    ex.cancel_all_orders.assert_called_once_with("BTC/USDT:USDT")


def test_close_position_no_position_returns_rejected():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT", "contracts": 0.0}]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    res = broker.close_position("BTC/USDT")
    assert res["status"] == "rejected"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k close_position -v`
Expected: FAIL — `close_position` is abstract/missing override.

- [ ] **Step 3: Implement ledger helpers + `close_position`**

Append to the `BinanceFuturesBroker` class:

```python
    def _load_ledger(self) -> List[Dict[str, Any]]:
        """Read all open positions from the Postgres ledger ([] when db is None)."""
        if not self.db:
            return []
        try:
            self.db.connect()
            rows = self.db.conn.execute(
                "SELECT symbol, side, entry_time, entry_price, size_usd, stop_price, "
                "take_profit_price FROM open_positions"
            ).fetchall()
            return [{
                "symbol": r[0], "side": r[1], "entry_time": r[2], "entry_price": r[3],
                "size_usd": r[4], "stop_price": r[5], "take_profit_price": r[6],
            } for r in rows]
        finally:
            self.db.close()

    def _delete_position(self, symbol: str):
        """Remove a position from the Postgres ledger (no-op when db is None)."""
        if not self.db:
            return
        try:
            self.db.connect()
            self.db.conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
        except Exception as e:
            logger.error(f"BinanceFuturesBroker: failed to delete ledger row {symbol}: {e}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k close_position -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): ledger helpers + close_position (reduce-only + cancel brackets)"
```

---

### Task 8: `update_positions` reconcile + `_build_closed_trade`

**Files:**
- Modify: `src/vibe_trading/brokers/binance_futures.py`
- Test: `tests/test_binance_futures.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_binance_futures.py`:

```python
from datetime import datetime as _dt


def test_update_positions_reconciles_closed_trade():
    ex = _mock_exchange()
    # Ledger says BTC is open; exchange shows it flat → it was closed by a bracket.
    ex.fetch_positions.return_value = []  # nothing open on the exchange
    ex.fetch_my_trades.return_value = [
        {"side": "sell", "price": 110.0, "amount": 10.0, "fee": {"cost": 0.4}},
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    # Inject a ledger row (db is None, so _load_ledger would be empty otherwise)
    broker._load_ledger = lambda: [{
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1, 0, 0, 0),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }]

    closed = broker.update_positions({})
    assert len(closed) == 1
    t = closed[0]
    assert t["symbol"] == "BTC/USDT"
    assert t["action"] == "long"
    assert t["entry_price"] == 100.0
    assert t["close_price"] == 110.0
    # qty = 1000/100 = 10 ; pnl = (110-100)*10 - 0.4 = 99.6
    assert round(t["realized_pnl"], 2) == 99.6
    assert t["result"] == "win"
    assert {"trade_id", "close_time", "size_usd"} <= set(t.keys())


def test_update_positions_keeps_still_open_position():
    ex = _mock_exchange()
    ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.5},  # still open
    ]
    broker = BinanceFuturesBroker(db=None, exchange=ex)
    broker._load_ledger = lambda: [{
        "symbol": "BTC/USDT", "side": "long", "entry_time": _dt(2026, 6, 1),
        "entry_price": 100.0, "size_usd": 1000.0, "stop_price": 95.0, "take_profit_price": 110.0,
    }]
    assert broker.update_positions({}) == []


def test_update_positions_empty_ledger_no_calls():
    ex = _mock_exchange()
    broker = BinanceFuturesBroker(db=None, exchange=ex)  # db None → empty ledger
    assert broker.update_positions({}) == []
    ex.fetch_positions.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -k update_positions -v`
Expected: FAIL — `update_positions` is abstract/missing override.

- [ ] **Step 3: Implement `update_positions` + `_build_closed_trade`**

Append to the `BinanceFuturesBroker` class:

```python
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
                closed.append(self._build_closed_trade(row))
                self._delete_position(row["symbol"])
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
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_binance_futures.py -v`
Expected: PASS (all broker tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/brokers/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(broker): update_positions reconcile + closed-trade builder"
```

---

### Task 9: Scheduler wiring — LIVE_TESTNET branch + exec-price helper

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py`
- Test: `tests/test_scheduler.py` (create)

> **Note:** the spec's "startup reconcile" is already satisfied — `start()` calls `sync_and_evaluate()` immediately, whose `update_positions()` call reconciles the futures broker and logs any closes through the existing `trades`+Discord path (scheduler lines ~122-140). Do **not** add a second reconcile call; it would double-log.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scheduler.py`:

```python
"""Tests for the scheduler's exec-price resolution (the only new pure logic).
The full sync_and_evaluate loop is network-heavy and covered by manual verification."""
from unittest.mock import MagicMock

from vibe_trading.runtime.scheduler import TradingScheduler


def _scheduler_without_init():
    """Build a TradingScheduler instance without running __init__ (which needs DBs/LLM)."""
    return TradingScheduler.__new__(TradingScheduler)


def test_resolve_exec_price_uses_broker_mark_when_available():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = 250.0
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 250.0


def test_resolve_exec_price_falls_back_when_mark_none():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = None
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0


def test_resolve_exec_price_falls_back_on_broker_error():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.side_effect = Exception("boom")
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_scheduler.py -v`
Expected: FAIL with `AttributeError: 'TradingScheduler' object has no attribute '_resolve_exec_price'`

- [ ] **Step 3: Add the broker import, selection branch, and helper**

In `src/vibe_trading/runtime/scheduler.py`:

(a) Add the import near the other broker imports (after the `CoinbaseBroker` import on line ~19):

```python
from vibe_trading.brokers.binance_futures import BinanceFuturesBroker
```

(b) Extend the broker-selection block (currently lines ~35-39) to:

```python
        mode = os.getenv("TRADING_MODE", "PAPER").upper()
        if mode == "LIVE_SANDBOX":
            self.broker = CoinbaseBroker()
        elif mode == "LIVE_TESTNET":
            self.broker = BinanceFuturesBroker(db=self.pg_db)
        else:
            self.broker = PaperBroker(db=self.pg_db)
```

(c) Add the helper method (place it just above `_check_cost_alarm`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_scheduler.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Use the exec price in the entry loop**

In `sync_and_evaluate`, after the line `last_ts, current_price = last_candle_ts_res` (line ~172), insert:

```python
                    # Execution-critical price: futures mark in LIVE_TESTNET, else spot close.
                    exec_price = self._resolve_exec_price(sym, current_price)
```

Then change the two call sites to use `exec_price` instead of `current_price`:

- The trader call (line ~189):
```python
                    proposal = self.trader.decide(sym, analyst_report, self.scorecard, open_positions, current_price=exec_price)
```
- The risk-manager call (line ~211, the `current_price=` kwarg):
```python
                        current_price=exec_price,
```

(Leave `submit_order(entry_price=risk_res["entry_price"], ...)` as-is — `risk_res["entry_price"]` is already derived from the price passed into `evaluate_proposal`, so it now reflects the futures mark automatically.)

- [ ] **Step 6: Run the full suite to verify nothing regressed**

Run: `PYTHONPATH=src pytest tests/test_scheduler.py tests/test_binance_futures.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): LIVE_TESTNET broker branch + futures-mark exec price"
```

---

### Task 10: Web `/api/positions` reads the exchange in LIVE_TESTNET

**Files:**
- Create: `src/vibe_trading/web/live_positions.py`
- Modify: `src/vibe_trading/web/main.py`
- Test: `tests/test_web_positions.py` (create)

> **Why a new module:** importing `vibe_trading.web.main` runs `PostgresDatabase()` at module load (it opens a live Supabase pool), so a unit test cannot import it without network. Putting the helper in its own side-effect-free module (`live_positions.py` — no module-level DB/ccxt) keeps the test hermetic and the responsibility focused.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_positions.py`:

```python
"""Tests for the LIVE_TESTNET positions helper. We test the helper module directly
(no HTTP, no PostgresDatabase import), patching the cached broker so no ccxt/network
is touched."""
from unittest.mock import MagicMock

import vibe_trading.web.live_positions as lp


def test_live_testnet_positions_returns_exchange_positions(monkeypatch):
    fake_broker = MagicMock()
    fake_broker.get_open_positions.return_value = [{"symbol": "BTC/USDT", "side": "long"}]
    monkeypatch.setattr(lp, "_get_live_broker", lambda: fake_broker)
    assert lp.live_testnet_positions() == [{"symbol": "BTC/USDT", "side": "long"}]


def test_live_testnet_positions_returns_none_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("exchange unreachable")
    monkeypatch.setattr(lp, "_get_live_broker", _boom)
    assert lp.live_testnet_positions() is None  # None signals fallback to Postgres
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_web_positions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.web.live_positions'`

- [ ] **Step 3: Create the helper module**

Create `src/vibe_trading/web/live_positions.py` (no module-level DB or ccxt — both are imported lazily inside the function, so importing this module is side-effect-free):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_web_positions.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire the endpoint in `main.py`**

In `src/vibe_trading/web/main.py`:

(a) Add the import near the other `vibe_trading` imports (after line ~9):

```python
from vibe_trading.web.live_positions import live_testnet_positions
```

(b) At the very top of the `get_positions()` function body (line ~165, before `positions = []`), add the LIVE_TESTNET short-circuit:

```python
    if os.getenv("TRADING_MODE", "PAPER").upper() == "LIVE_TESTNET":
        live = live_testnet_positions()
        if live is not None:
            return live
        # else: fall through to the Postgres ledger path below
```

- [ ] **Step 6: Verify import wiring (no network needed)**

Run: `PYTHONPATH=src python -c "import ast; ast.parse(open('src/vibe_trading/web/main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK` (avoids importing main.py, which would init the Postgres pool).

- [ ] **Step 7: Commit**

```bash
git add src/vibe_trading/web/live_positions.py src/vibe_trading/web/main.py tests/test_web_positions.py
git commit -m "feat(web): /api/positions reads exchange directly in LIVE_TESTNET"
```

---

### Task 11: Config, docs, smoke script, full verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Create: `scripts/binance_testnet_smoke.py`

- [ ] **Step 1: Add config to `.env.example`**

Append to `.env.example`:

```env

# --- Binance USDⓂ Futures testnet execution (TRADING_MODE=LIVE_TESTNET) ---
# Create keys at https://testnet.binancefuture.com/ (Account → API Key).
# Testnet only — never point this at mainnet / real funds.
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
BINANCE_TESTNET_DRY_RUN=false      # true = log intended orders, place none (safe wiring check)
BINANCE_TESTNET_LEVERAGE=1
# TRADING_MODE=LIVE_TESTNET         # uncomment to activate the futures broker
```

- [ ] **Step 2: Add a README section**

Add a subsection under the trading-mode docs in `README.md` (after the Docker section):

```markdown
## Live Testnet Execution (Binance USDⓂ Futures)

Set `TRADING_MODE=LIVE_TESTNET` to execute real orders on the **Binance USDⓂ Futures
testnet** (`testnet.binancefuture.com`) with native exchange brackets. On entry the broker
places a market order plus two reduce-only `closePosition` orders — `TAKE_PROFIT_MARKET`
and `STOP_MARKET` — so the exchange fills whichever triggers first and cancels the sibling,
**even if the bot is offline**. Leverage is pinned to 1× so risk/sizing semantics match the
paper model. Futures (not spot) is used because the trader emits **short** as well as long.

- **Setup:** create testnet API keys, put them in `BINANCE_TESTNET_API_KEY` /
  `BINANCE_TESTNET_API_SECRET`, and set `TRADING_MODE=LIVE_TESTNET`.
- **Dry run:** `BINANCE_TESTNET_DRY_RUN=true` logs intended orders without placing any —
  a safe way to verify wiring before sending real testnet orders.
- **Dashboard:** in this mode `/api/positions` reads open positions **directly from the
  exchange** (always accurate), falling back to the Postgres ledger on any exchange error.
- **Bookkeeping:** the Postgres `open_positions` table is the reconciliation ledger; each
  tick compares it to live exchange positions and records any bracket-closed trade.
- **TA still uses spot candles** — only the execution price aligns to the futures mark.
- **Smoke test (manual):** `python scripts/binance_testnet_smoke.py` opens a tiny position
  with a bracket, prints it, and closes it (requires your testnet keys).

> Real-time fill push via the User Data Stream websocket is a separate follow-on; until
> then the trade-history log + close alert may lag up to one 4h tick (the exit itself and
> the dashboard are already real-time).
```

- [ ] **Step 3: Create the manual smoke script**

Create `scripts/binance_testnet_smoke.py`:

```python
"""Manual live smoke test for BinanceFuturesBroker against the Binance futures testnet.

NOT run by pytest. Requires BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in your
environment (or .env). Places a tiny BTC long with a bracket, prints the resulting position
and open orders, then closes it.

Usage:
    python scripts/binance_testnet_smoke.py
"""
import logging
import time

from dotenv import load_dotenv

from vibe_trading.brokers.binance_futures import BinanceFuturesBroker

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main():
    broker = BinanceFuturesBroker(db=None)  # real testnet via ccxt set_sandbox_mode

    symbol = "BTC/USDT"
    mark = broker.get_mark_price(symbol)
    print(f"Mark price for {symbol}: {mark}")
    if not mark:
        raise SystemExit("Could not read mark price — check creds / connectivity.")

    # ~$200 notional; TP +2%, SL -2%
    res = broker.submit_order(
        symbol=symbol, action="long", size_usd=200.0,
        stop_price=mark * 0.98, take_profit_price=mark * 1.02, entry_price=mark,
    )
    print(f"submit_order → {res}")
    if res["status"] not in ("success", "dry_run"):
        raise SystemExit(f"Order not placed: {res}")

    time.sleep(2)
    print("Open positions on exchange:")
    for p in broker.get_open_positions():
        print(f"  {p}")

    print("Closing position...")
    print(f"close_position → {broker.close_position(symbol)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the FULL test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: PASS — the prior 151 tests plus the new broker (≈13), scheduler (3), web (2), and base/paper (1) tests, with no regressions.

- [ ] **Step 5: Verify the smoke script imports cleanly (no live call)**

Run: `PYTHONPATH=src python -c "import scripts.binance_testnet_smoke" 2>/dev/null || PYTHONPATH=src python -c "import ast; ast.parse(open('scripts/binance_testnet_smoke.py').read()); print('smoke script parses OK')"`
Expected: `smoke script parses OK` (or clean import) — confirms no syntax errors without hitting the network.

- [ ] **Step 6: Commit**

```bash
git add .env.example README.md scripts/binance_testnet_smoke.py
git commit -m "docs(broker): LIVE_TESTNET config, README, manual smoke script"
```

---

## Manual Live Verification (you, after merge — needs your testnet keys)

1. Put testnet keys in `.env`, set `TRADING_MODE=LIVE_TESTNET`, optionally `BINANCE_TESTNET_DRY_RUN=true` first.
2. `python scripts/binance_testnet_smoke.py` → confirm a position opens with both brackets, then closes.
3. Set `BINANCE_TESTNET_DRY_RUN=false`, run a real `trade-once` (or let a tick fire) and confirm on `testnet.binancefuture.com` that the entry + two bracket orders appear, and the dashboard `/api/positions` shows the live position.
4. Manually trigger a TP/SL on the testnet UI (or wait) and confirm the next tick records the closed trade in `trades` + sends the Discord close alert.
```
