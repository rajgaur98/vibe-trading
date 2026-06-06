# Design Spec — LLM Cost & Token Tracking

## Problem

The project makes LLM calls (analyst, trader, eval judge) but captures nothing about
their cost. The AI-engineering rubric flags this as a Tier-1 must-have ("Cost and
latency are first-class — track them per call, per request, per user. Alarm on spike.")
and it maps directly to an interview question the project currently cannot answer:
*"What does it cost per decision?"*

`LLMClient` is the single chokepoint for every model call (`call_llm` and
`call_llm_with_tools` both go through `litellm.completion`), so capture can be added in
exactly one place.

## Solution — Capture in the client, persist to Postgres, surface on the dashboard

After each `litellm.completion`, `LLMClient` computes token usage + dollar cost + latency
and hands a `CostEvent` to a pluggable, class-level **cost sink**. In production the
scheduler sets the sink to a `PostgresCostLogger` that appends to an `llm_cost_log` table;
in eval/tests the sink is unset (no DB coupling, no behavior change). A `/api/costs`
endpoint aggregates the table and a dashboard tile shows $/day and projected $/month. A
per-tick spike alarm sends a Discord alert when daily spend exceeds a configurable
threshold.

**Out of scope (deferred to guardrail item #8):** the hard daily-$ kill switch that
*blocks trading*. This spec only *measures and alarms*.

## Why a class-level sink (not constructor injection)

`LLMClient` is currently a pure, DB-free transport used identically by tests, the eval
harness, and production. Threading a logger through every construction site (analyst,
trader, and the eval judge each build their own `LLMClient`) would be invasive and would
couple the eval/test paths to Postgres. Instead, mirror the existing per-call throttle
(`LLMClient._last_call_at`, a class-level shared field): a class-level `_cost_sink`
defaulting to `None`. The app wires it once at startup; everything else is untouched.

## Components

### 1. `src/vibe_trading/agents/cost.py` [NEW]

```python
from datetime import datetime, timezone
from typing import Optional, Protocol
from uuid import uuid4
import logging
import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Shadow prices (USD per token) for models LiteLLM does not price — notably the
# free-tier Gemma models the bot runs. Lets projected $/month stay meaningful
# ("what this would cost on a paid tier") instead of reading $0. Tune as needed.
PRICE_OVERRIDES: dict[str, tuple[float, float]] = {
    # model substring : (input_cost_per_token, output_cost_per_token)
    "gemma-4-31b-it": (0.20e-6, 0.40e-6),
    "gemma-4-26b-a4b-it": (0.15e-6, 0.30e-6),
}


def usage_cost(model_str: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a call. LiteLLM pricing first; PRICE_OVERRIDES fallback; else 0.0.

    `model_str` is the LiteLLM-format id (e.g. 'gemini/gemma-4-31b-it').
    Never raises — returns 0.0 on any failure.
    """
    try:
        info = litellm.get_model_info(model_str)
        in_c = info.get("input_cost_per_token")
        out_c = info.get("output_cost_per_token")
        if in_c is not None and out_c is not None:
            return prompt_tokens * in_c + completion_tokens * out_c
    except Exception:
        pass
    for needle, (in_c, out_c) in PRICE_OVERRIDES.items():
        if needle in model_str:
            return prompt_tokens * in_c + completion_tokens * out_c
    return 0.0


class CostEvent(BaseModel):
    call_id: str
    timestamp: datetime
    provider: str
    model: str            # litellm-format model string
    call_type: str        # "single" | "tool_loop"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float

    @classmethod
    def build(cls, *, provider, model, call_type, prompt_tokens, completion_tokens, latency_ms):
        total = prompt_tokens + completion_tokens
        return cls(
            call_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            provider=provider, model=model, call_type=call_type,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=usage_cost(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )


class CostSink(Protocol):
    def record(self, event: CostEvent) -> None: ...


class PostgresCostLogger:
    """Appends CostEvents to llm_cost_log. Best-effort: never raises to the caller.

    Holds its OWN PostgresDatabase instance (sharing the class-level pool) rather than
    the scheduler's, so its connect/close cycle can't alias `self.conn` on an instance
    the scheduler is concurrently using during a tick.
    """
    def __init__(self, db=None):
        from vibe_trading.data.db import PostgresDatabase
        self.pg_db = db or PostgresDatabase()

    def record(self, event: CostEvent) -> None:
        try:
            self.pg_db.connect()
            self.pg_db.conn.execute(
                """INSERT OR IGNORE INTO llm_cost_log
                   (call_id, timestamp, provider, model, call_type,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.call_id, event.timestamp, event.provider, event.model, event.call_type,
                 event.prompt_tokens, event.completion_tokens, event.total_tokens,
                 event.cost_usd, event.latency_ms),
            )
        except Exception as e:
            logger.warning(f"cost logging failed (non-fatal): {e}")
        finally:
            self.pg_db.close()


def daily_summary(pg_db) -> dict:
    """Aggregate today's (UTC) spend for the dashboard. Returns zeros if empty/unavailable."""
    # SUM(cost_usd), COUNT(*), SUM(total_tokens) WHERE timestamp >= start-of-day-UTC
    # plus per-model breakdown. Returns:
    # {"today_usd","calls","tokens","avg_cost_per_call","projected_monthly_usd","by_model":[...]}
```

Note: `INSERT OR IGNORE` is the project's DuckDB-dialect idiom that the
`PostgresConnectionWrapper.translate_query` already converts to
`INSERT ... ON CONFLICT (call_id) DO NOTHING`. **The translation layer must be extended
to handle `llm_cost_log`** (see Component 3).

### 2. `src/vibe_trading/agents/client.py` [MODIFY]

Add class-level sink state alongside the existing throttle fields, and emit after each call.

```python
class LLMClient:
    _last_call_at: float = 0.0
    _throttle_lock = threading.Lock()
    _cost_sink = None  # set once at app startup; None => no-op (tests/eval)

    @classmethod
    def set_cost_sink(cls, sink) -> None:
        cls._cost_sink = sink

    def _emit_cost(self, response, model_str, call_type, latency_ms) -> None:
        """Best-effort: build a CostEvent from the litellm response and hand to the sink.
        Never raises — cost logging must not break an LLM call or a trade."""
        sink = LLMClient._cost_sink
        if sink is None:
            return
        try:
            usage = getattr(response, "usage", None) or {}
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            event = CostEvent.build(
                provider=self.provider, model=model_str, call_type=call_type,
                prompt_tokens=pt, completion_tokens=ct, latency_ms=latency_ms,
            )
            sink.record(event)
        except Exception as e:
            logger.warning(f"cost emit failed (non-fatal): {e}")
```

- In `call_llm`: wrap the `litellm.completion(**kwargs)` with a `time.monotonic()` start/end,
  then `self._emit_cost(response, model_str, "single", latency_ms)` before returning content.
- In `call_llm_with_tools`: same around each per-iteration `litellm.completion`, with
  `call_type="tool_loop"`.
- Latency uses `time.monotonic()` (already imported for the throttle).

### 3. `src/vibe_trading/data/db.py` [MODIFY]

- Add `llm_cost_log` to `PostgresDatabase._create_tables`:
  ```sql
  CREATE TABLE IF NOT EXISTS llm_cost_log (
      call_id VARCHAR PRIMARY KEY,
      timestamp TIMESTAMP,
      provider VARCHAR,
      model VARCHAR,
      call_type VARCHAR,
      prompt_tokens INTEGER,
      completion_tokens INTEGER,
      total_tokens INTEGER,
      cost_usd DOUBLE PRECISION,
      latency_ms DOUBLE PRECISION
  )
  ```
- Extend `translate_query()` so `INSERT OR IGNORE INTO llm_cost_log` →
  `INSERT INTO llm_cost_log ... ON CONFLICT (call_id) DO NOTHING` (same pattern as the
  existing `decision_log` branch).

### 4. `src/vibe_trading/web/main.py` [MODIFY]

Add `GET /api/costs` using the existing `get_pg_conn()` context manager:

```json
{
  "today_usd": 0.0123,
  "calls": 47,
  "tokens": 91234,
  "avg_cost_per_call": 0.00026,
  "projected_monthly_usd": 0.369,
  "by_model": [{"model": "gemini/gemma-4-31b-it", "calls": 47, "cost_usd": 0.0123}]
}
```

Defaults to zeros if the table is empty or unavailable (mirrors the defensive try/except
already used by `/api/metrics`).

### 5. `web/src/components/` [MODIFY]

Add a cost tile to the dashboard mirroring the existing `MetricsGrid` card pattern: show
**Today's Spend**, **Projected / mo**, and **Calls / Tokens**. Fetch `/api/costs` on the
same 30s poll the dashboard already uses. (Per `web/AGENTS.md`, follow the existing Next.js
component conventions; do not introduce new patterns.)

### 6. Spike alarm — `src/vibe_trading/runtime/scheduler.py` [MODIFY]

- At the top of `sync_and_evaluate()` (or once per tick), call `_check_cost_alarm()`:
  query today's spend via `cost.daily_summary(self.pg_db)`; if `today_usd >`
  `float(os.getenv("LLM_DAILY_COST_ALARM_USD", "5.0"))`, send a Discord alert via the
  existing `_send_discord_alert`. De-dup: alarm at most once per UTC day (track the last
  alarmed date on the scheduler instance).
- Wire the sink once in `TradingScheduler.__init__`:
  `LLMClient.set_cost_sink(PostgresCostLogger())` (the logger owns its own pooled
  `PostgresDatabase`, independent of `self.pg_db`).
- Also wire it in the CLI entry path that starts the scheduler so any process that trades
  logs cost. (The scheduler `__init__` is sufficient since both `live` and `trade-once`
  construct a `TradingScheduler`.)

## Data Flow

```
analyst/trader/judge → LLMClient.call_llm[_with_tools]
        │  litellm.completion() ──► response (.usage)
        │  measure latency (monotonic)
        ├─ _emit_cost(): CostEvent.build() → usage_cost() [litellm price | override | 0]
        │       └─ LLMClient._cost_sink.record(event)   (None in eval/tests → skipped)
        ▼
PostgresCostLogger → INSERT OR IGNORE INTO llm_cost_log   (best-effort)
        ▼
/api/costs ── daily_summary() ──► dashboard cost tile (30s poll)
scheduler tick ── daily_summary() ──► Discord spike alarm if over LLM_DAILY_COST_ALARM_USD
```

## Error Handling

| Failure | Behavior |
|---|---|
| Model not in LiteLLM pricing and no override | `usage_cost` returns 0.0; tokens still logged |
| `response.usage` missing/None | tokens default to 0; event still recorded (cost 0.0) |
| Cost sink / Postgres write fails | `record()` and `_emit_cost` catch, log a warning, return; **the LLM call returns normally** |
| `daily_summary` query fails | returns zeros; `/api/costs` serves zeros; no alarm |
| `LLM_DAILY_COST_ALARM_USD` unset | defaults to 5.0 |

**Invariant:** cost tracking is strictly observational — no path through it can raise into
an LLM call, an agent decision, or a trade.

## Same-Code-Path Note

Capture lives in `LLMClient`, which eval and prod share, so token/cost *capture* is
identical across paths. Only the *sink* is prod-only (set by the scheduler), so the eval
doesn't write trading-account cost rows — correct, since eval cost ≠ live trading cost.
This does not widen the existing eval-vs-prod analyst-path divergence (that's a separate item).

## Testing Strategy

Unit tests (`tests/test_cost.py` new; extend `tests/test_multi_provider.py`):

1. `usage_cost` — known model (LiteLLM price) computes input×in + output×out.
2. `usage_cost` — unknown model matching a PRICE_OVERRIDES needle uses the override.
3. `usage_cost` — unknown model, no override → 0.0; never raises.
4. `CostEvent.build` — totals and cost populate; `total_tokens == prompt + completion`.
5. `LLMClient` emits a CostEvent to a stub sink on `call_llm` when a sink is set
   (mock `litellm.completion` with a `.usage`), with `call_type="single"`.
6. `LLMClient` no-ops when `_cost_sink` is None (default) — sink never called.
7. Sink that raises does NOT break `call_llm` — content still returned.
8. `call_llm_with_tools` emits one event per iteration with `call_type="tool_loop"`.
9. `PostgresCostLogger.record` round-trip + `daily_summary` aggregation — skipped if
   `POSTGRES_URL` unset (mirrors `test_db.test_postgres_database`).
10. Spike alarm: `daily_summary` over threshold triggers `_send_discord_alert` (mocked);
    under threshold does not; de-dup within a UTC day.

Tests must reset `LLMClient._cost_sink = None` in teardown to avoid cross-test leakage
(same isolation concern as the throttle/env tests).

## Backwards Compatibility

- `LLMClient` default behavior is unchanged when no sink is set (eval, tests, any existing
  caller). `call_llm` / `call_llm_with_tools` signatures and return types are unchanged.
- New table is additive; `_create_tables` is idempotent (`IF NOT EXISTS`).
- New `/api/costs` endpoint and dashboard tile are additive.
