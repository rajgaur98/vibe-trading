# Design Spec — Binance Futures Testnet Broker (native brackets)

> **Initiative:** real exchange execution with native OCO/bracket orders. This is
> **sub-project 1 of 2**. Sub-project 2 (User Data Stream websocket listener for
> real-time fill bookkeeping) builds on this and gets its own spec/plan. This spec
> produces working software on its own: real-time bracket *exits* on the exchange,
> a live exchange-truth dashboard, and reconcile-based trade bookkeeping.

## Problem

Today the bot only "executes" via `PaperBroker` (a simulation) or a stub `CoinbaseBroker`.
SL/TP are checked by the bot polling the 4h candle close every 4 hours — so exits are
not real-time, intraday wicks through a level are missed, and if the bot is down there is
no stop-loss protection at all. The correct model for live trading is to submit the entry
plus an attached **bracket** (take-profit + stop) so the **exchange's matching engine**
fills the exit in real time, even when the bot is offline.

## Solution

A new `BinanceFuturesBroker` implementing the existing `BaseBroker` interface, talking to
the **Binance USDⓂ Futures testnet** via authenticated `ccxt` (`set_sandbox_mode(True)`).
On entry it places a market order plus two reduce-only `closePosition` bracket orders
(`TAKE_PROFIT_MARKET`, `STOP_MARKET`); Binance fills whichever triggers first and
auto-cancels the sibling. Futures is required because the trader emits **short** as well
as long (spot cannot short); leverage is pinned to **1×** so risk/sizing semantics match
the current spot-like model. Selected by `TRADING_MODE=LIVE_TESTNET`. The dashboard reads
open positions **directly from the exchange** (always accurate). Trade history is recorded
by **reconciliation** (on startup + each 4h tick): any position we recorded as open that is
no longer open on the exchange was closed by its bracket → record the closed trade.

**Out of scope (sub-project 2):** the User Data Stream websocket for real-time fill
push. Until then, the *trade-history log + Discord close alert* may lag up to one tick;
the *exit itself* and the *dashboard* are already real-time.

**Out of scope (always):** mainnet / real funds. Testnet only.

## Components

### 1. `src/vibe_trading/brokers/binance_futures.py` [NEW] — `BinanceFuturesBroker(BaseBroker)`

```python
import os, logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4
import ccxt

logger = logging.getLogger(__name__)

def _to_ccxt_symbol(symbol: str) -> str:
    """'BTC/USDT' -> ccxt USDⓂ-futures unified symbol 'BTC/USDT:USDT'."""
    base, quote = symbol.split("/")
    return f"{base}/{quote}:{quote}"

class BinanceFuturesBroker(BaseBroker):
    def __init__(self, db=None):
        self.db = db  # PostgresDatabase — the reconciliation ledger (open_positions table)
        self.dry_run = os.getenv("BINANCE_TESTNET_DRY_RUN", "false").lower() == "true"
        key = os.getenv("BINANCE_TESTNET_API_KEY")
        secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not self.dry_run and (not key or not secret):
            raise ValueError(
                "BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET must be set for "
                "TRADING_MODE=LIVE_TESTNET (or set BINANCE_TESTNET_DRY_RUN=true)."
            )
        self.exchange = ccxt.binance({
            "apiKey": key, "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.exchange.set_sandbox_mode(True)   # routes to testnet.binancefuture.com
        self._markets = self.exchange.load_markets()
        self.leverage = int(os.getenv("BINANCE_TESTNET_LEVERAGE", "1"))
```

Interface methods (all symbols converted via `_to_ccxt_symbol`; all prices/amounts rounded
via `exchange.price_to_precision` / `exchange.amount_to_precision`):

- **`submit_order(symbol, action, size_usd, stop_price, take_profit_price, entry_price=0.0) -> dict`**
  1. `set_leverage(self.leverage, sym)`.
  2. `mark = entry_price or exchange.fetch_ticker(sym)["last"]`; `qty = amount_to_precision(sym, size_usd / mark)`.
  3. Reject if `qty * mark < market["limits"]["cost"]["min"]` or `qty < market["limits"]["amount"]["min"]` → return `{"status": "rejected", "reason": "below exchange minimum"}`.
  4. Entry: `entry_side = "buy" if action == "long" else "sell"`; `create_order(sym, "market", entry_side, qty)`.
  5. Brackets (reduce-only, whole-position), `exit_side = "sell" if action == "long" else "buy"`:
     - `create_order(sym, "TAKE_PROFIT_MARKET", exit_side, None, params={"stopPrice": price_to_precision(sym, take_profit_price), "closePosition": True})`
     - `create_order(sym, "STOP_MARKET", exit_side, None, params={"stopPrice": price_to_precision(sym, stop_price), "closePosition": True})`
  6. Persist to the Postgres `open_positions` ledger (symbol, side, entry_time, actual avg fill price, size_usd, stop_price, take_profit_price) so reconcile + the scheduler's max-concurrent / skip-existing checks work.
  7. `dry_run`: log all three intended orders, persist a simulated position, place nothing, return `{"status": "dry_run", ...}`.
  8. Return `{"status": "success", "entry_price": <actual avg fill>, "order_ids": {...}}`. Any `ccxt` error → log + return `{"status": "rejected", "reason": str(e)}` (caller already guards on status).

- **`get_open_positions() -> list[dict]`**: `fetch_positions()` filtered to non-zero `contracts`; for each, `fetch_open_orders(sym)` to read the `STOP_MARKET`/`TAKE_PROFIT_MARKET` `stopPrice`s. Map to the dashboard's shape: `{symbol (un-converted to 'BTC/USDT'), side, entry_price, size_usd (abs notional), stop_price, take_profit_price, current_price (markPrice)}`.

- **`get_balance() -> float`**: `fetch_balance()["USDT"]["total"]` (dry_run → 10000.0).

- **`close_position(symbol) -> dict`**: read the live position; `create_order(sym, "market", opposite_side, abs(contracts), params={"reduceOnly": True})`; `cancel_all_orders(sym)` to clear leftover brackets; remove from the ledger.

- **`update_positions(current_prices) -> list[dict]`** (reconciliation; `current_prices` ignored — the exchange is truth): for each symbol in the Postgres `open_positions` ledger, if it is **not** currently open on the exchange (`fetch_positions`), its bracket filled → fetch the closing fill / realized PnL (`fetch_my_trades(sym, since=entry_time_ms)`, take reduce-only closing fills; or position-close income), build a `closed_trade` dict matching the existing `trades` schema (`trade_id, symbol, action, entry_time, entry_price, close_time, close_price, size_usd, realized_pnl, result`), delete it from the ledger, and append to the returned list. The scheduler logs these to `trades` + Discord (existing path). Exchange/network error → log, return `[]` (never crash a tick).

### 2. `src/vibe_trading/runtime/scheduler.py` [MODIFY]

Broker selection gains a branch:
```python
mode = os.getenv("TRADING_MODE", "PAPER").upper()
if mode == "LIVE_SANDBOX":
    self.broker = CoinbaseBroker()
elif mode == "LIVE_TESTNET":
    self.broker = BinanceFuturesBroker(db=self.pg_db)
else:
    self.broker = PaperBroker(db=self.pg_db)
```
Add a **startup reconcile**: in `start()`, before the recurring loop, call `self.broker.update_positions({})` once and log/record any closes that happened while the bot was down. (For PaperBroker this is a harmless no-op-ish call; guard so it only matters for the futures broker — or simply call it for all brokers since `update_positions({})` with empty prices is safe.) The existing per-tick `update_positions(current_prices)` call already covers periodic reconcile. No other structural change — `submit_order` already passes `entry_price=risk_res["entry_price"]`.

### 3. `src/vibe_trading/web/main.py` [MODIFY]

`/api/positions`: when `TRADING_MODE == "LIVE_TESTNET"`, return a read-only
`BinanceFuturesBroker(db=None).get_open_positions()` (live exchange truth, always accurate);
otherwise keep the current Postgres path. Wrap in try/except → fall back to the Postgres
path (or `[]`) on any exchange error so the dashboard never 500s. (Construct the broker
per-request like `get_pg_conn`, or cache a module-level read-only instance.)

### 4. Config

`.env` / `.env.example`:
```
# LIVE_TESTNET execution against Binance USDⓂ Futures testnet (testnet.binancefuture.com)
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
BINANCE_TESTNET_DRY_RUN=false      # true = log intended orders, place none (safe wiring check)
BINANCE_TESTNET_LEVERAGE=1
# TRADING_MODE=LIVE_TESTNET        # to activate this broker
```

## Data Flow (one approved long, LIVE_TESTNET)

```
scheduler tick → analyst → trader → RiskManager (approved; entry/stop/tp/size_usd)
  → BinanceFuturesBroker.submit_order(...)
       set_leverage(1) ; qty = size_usd / mark (rounded)
       create_order market BUY qty                         → position opens on exchange
       create_order TAKE_PROFIT_MARKET SELL closePosition @ tp   ┐ native bracket held by
       create_order STOP_MARKET        SELL closePosition @ sl   ┘ the exchange (real-time)
       persist position to Postgres open_positions ledger
  ── price hits TP intraday ──► EXCHANGE fills TP instantly, auto-cancels SL  (bot uninvolved)
  ── next tick / startup ──► update_positions(): ledger has SYM, exchange shows it flat
       → fetch realized PnL → record closed trade in `trades` + Discord → drop from ledger
dashboard /api/positions ──► BinanceFuturesBroker.get_open_positions() (live, always current)
```

## Error Handling

| Scenario | Behavior |
|---|---|
| Missing testnet creds (and not dry_run) | Construction raises `ValueError` — fail fast, don't silently no-op |
| Order rejected (precision / min-notional / margin) | Caught; `submit_order` returns `{"status": "rejected", "reason": ...}`; scheduler logs + Discord; nothing persisted. RiskManager caps + the LLM kill switch still gate *before* submission |
| Exchange/network error in `get_open_positions` (dashboard) | Caught; fall back to Postgres ledger / `[]`; API never 500s |
| Exchange/network error in `update_positions` (reconcile) | Caught; return `[]`; tick continues; the close is caught on a later reconcile |
| `dry_run=true` | No order calls; logs intended orders; simulated success + ledger row |
| Bracket sibling not auto-cancelled | `close_position` / reconcile defensively `cancel_all_orders(sym)` |

**Invariant:** every `ccxt` call is wrapped; a broker error degrades to a logged rejection/empty result — it never crashes a scheduler tick or the web API.

## Testing (all unit tests use a mocked `ccxt` exchange — no live calls in pytest)

`tests/test_binance_futures.py`:
1. `submit_order` long: `set_leverage(1)`, market `buy` with precision-rounded qty, then `TAKE_PROFIT_MARKET`/`STOP_MARKET` `sell` `closePosition` with rounded `stopPrice`s = tp/sl.
2. `submit_order` short: market `sell`; brackets `buy` side.
3. min-notional / min-amount rejection → `status="rejected"`, no entry order placed.
4. `dry_run=true` → zero `create_order` calls; returns `status="dry_run"`; ledger row written.
5. Missing creds (not dry_run) → `__init__` raises `ValueError`.
6. `get_open_positions` maps `fetch_positions` + `fetch_open_orders` → dashboard shape incl. stop/tp from the bracket orders; symbol un-converted to `BTC/USDT`.
7. `update_positions` reconcile: ledger has SYM, `fetch_positions` shows it closed → emits one `closed_trade` with realized PnL + removes it from the ledger.
8. `close_position`: reduce-only market opposite-side + `cancel_all_orders`.
9. Precision: a price/qty needing rounding is passed through `price_to_precision`/`amount_to_precision`.

**Live verification (manual, needs your testnet keys):** a `scripts/binance_testnet_smoke.py` that, with `BINANCE_TESTNET_*` set, places a tiny long with a bracket, prints the resulting position + open orders, and closes it. I cannot run this — it requires credentials I don't have.

## Backwards Compatibility

- Additive: `PAPER` (default) and `LIVE_SANDBOX` paths are unchanged. The new broker only activates on `TRADING_MODE=LIVE_TESTNET`.
- `BaseBroker` interface is unchanged — `BinanceFuturesBroker` implements the same five methods, so the scheduler needs only the selection branch (+ the startup reconcile call, which is safe for all brokers).
- The `open_positions` Postgres table is reused as the reconciliation ledger; no schema change.

## Follow-on (sub-project 2, separate spec)

User Data Stream websocket: obtain `listenKey`, hold a persistent ws to the futures testnet
stream, handle `ORDER_TRADE_UPDATE` (bracket fills) → record the closed trade + Discord in
**real time**; `PUT`-keepalive the `listenKey` (~30 min); reconnect-and-reconcile on drop.
This broker's `update_positions` reconcile becomes the documented safety net for ws gaps and
bot downtime. The exchange remains the single source of truth.
