# LLM Cost & Token Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture token usage, dollar cost, and latency for every LLM call, persist them to a Postgres `llm_cost_log` table, surface $/day + projected $/month on the dashboard, and Discord-alarm on daily-spend spikes.

**Architecture:** `LLMClient` is the single chokepoint for all model calls; after each `litellm.completion` it builds a `CostEvent` and hands it to a class-level, pluggable cost sink (default `None` → no-op for tests/eval; production sets a `PostgresCostLogger`). Pricing uses LiteLLM's model map with a shadow-price fallback for unpriced models (e.g. Gemma). Cost tracking is strictly observational — every write is best-effort and can never raise into an LLM call or a trade.

**Tech Stack:** Python 3.12, pydantic v2, LiteLLM (`get_model_info`), psycopg2 (Supabase Postgres), pytest, FastAPI, Next.js (dashboard tile).

---

## File Structure

- **Create** `src/vibe_trading/agents/cost.py` — pricing (`usage_cost` + `PRICE_OVERRIDES`), `CostEvent` model, `PostgresCostLogger` sink, `daily_summary(conn)` aggregator. One module, one responsibility: cost capture + pricing + persistence + read.
- **Modify** `src/vibe_trading/agents/client.py` — class-level `_cost_sink` + `set_cost_sink()` + `_emit_cost()`; emit after each completion in `call_llm` and `call_llm_with_tools`.
- **Modify** `src/vibe_trading/data/db.py` — `llm_cost_log` table in `PostgresDatabase._create_tables`; extend `translate_query()` for `INSERT OR IGNORE INTO llm_cost_log`.
- **Modify** `src/vibe_trading/web/main.py` — `GET /api/costs` via `daily_summary`.
- **Modify** `src/vibe_trading/runtime/scheduler.py` — wire `set_cost_sink` in `__init__`; `_check_cost_alarm()` once per tick.
- **Modify** `web/src/components/MetricsGrid.tsx` (+ `web/src/app/page.tsx`) — cost tile.
- **Create** `tests/test_cost.py`; **Modify** `tests/test_multi_provider.py` — client emit tests.

---

### Task 1: Pricing — `usage_cost` + `PRICE_OVERRIDES`

**Files:**
- Create: `src/vibe_trading/agents/cost.py`
- Create: `tests/test_cost.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cost.py`:

```python
import math
from vibe_trading.agents.cost import usage_cost, PRICE_OVERRIDES


def test_usage_cost_known_litellm_model_is_positive():
    # gemini-3.1-flash-lite IS in LiteLLM's pricing map; don't hardcode the rate
    # (it can change across litellm versions) — just assert it's priced > 0.
    cost = usage_cost("gemini/gemini-3.1-flash-lite", prompt_tokens=1000, completion_tokens=500)
    assert cost > 0.0


def test_usage_cost_override_model_uses_shadow_price():
    # gemma-4-31b-it is NOT in LiteLLM's map -> falls back to PRICE_OVERRIDES (deterministic).
    in_c, out_c = PRICE_OVERRIDES["gemma-4-31b-it"]
    expected = 1000 * in_c + 500 * out_c
    cost = usage_cost("gemini/gemma-4-31b-it", prompt_tokens=1000, completion_tokens=500)
    assert math.isclose(cost, expected, rel_tol=1e-9)


def test_usage_cost_unknown_model_returns_zero():
    cost = usage_cost("fakeprovider/does-not-exist-1.0", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.0


def test_usage_cost_never_raises_on_zero_tokens():
    assert usage_cost("gemini/gemma-4-31b-it", 0, 0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_trading.agents.cost'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/vibe_trading/agents/cost.py`:

```python
import logging
import litellm

logger = logging.getLogger(__name__)

# Shadow prices (USD per token) for models LiteLLM does not price — notably the
# free-tier Gemma models the bot runs. Keeps projected $/month meaningful ("what this
# would cost on a paid tier") instead of reading $0. Keys are substrings matched
# against the litellm-format model string. Tune as real pricing becomes known.
PRICE_OVERRIDES: dict[str, tuple[float, float]] = {
    "gemma-4-31b-it": (0.20e-6, 0.40e-6),
    "gemma-4-26b-a4b-it": (0.15e-6, 0.30e-6),
}


def usage_cost(model_str: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a call: LiteLLM pricing first, PRICE_OVERRIDES fallback, else 0.0.

    `model_str` is the litellm-format id (e.g. 'gemini/gemma-4-31b-it'). Never raises.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/cost.py tests/test_cost.py
git commit -m "feat(cost): add usage_cost pricing with litellm + shadow-price fallback"
```

---

### Task 2: `CostEvent` model + `build()`

**Files:**
- Modify: `src/vibe_trading/agents/cost.py`
- Modify: `tests/test_cost.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cost.py`:

```python
from datetime import datetime
from vibe_trading.agents.cost import CostEvent


def test_cost_event_build_populates_totals_and_cost():
    ev = CostEvent.build(
        provider="gemini", model="gemini/gemma-4-31b-it", call_type="single",
        prompt_tokens=1000, completion_tokens=500, latency_ms=1234.5,
    )
    assert ev.provider == "gemini"
    assert ev.model == "gemini/gemma-4-31b-it"
    assert ev.call_type == "single"
    assert ev.prompt_tokens == 1000
    assert ev.completion_tokens == 500
    assert ev.total_tokens == 1500
    assert ev.cost_usd > 0.0           # override-priced
    assert ev.latency_ms == 1234.5
    assert ev.call_id                  # non-empty uuid
    assert isinstance(ev.timestamp, datetime)
    assert ev.timestamp.tzinfo is None  # naive UTC, matches trades/decision_log convention


def test_cost_event_build_unique_call_ids():
    a = CostEvent.build(provider="g", model="m", call_type="single",
                        prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    b = CostEvent.build(provider="g", model="m", call_type="single",
                        prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    assert a.call_id != b.call_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py::test_cost_event_build_populates_totals_and_cost -v`
Expected: FAIL — `ImportError: cannot import name 'CostEvent'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `src/vibe_trading/agents/cost.py`:

```python
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel
```

Append to `src/vibe_trading/agents/cost.py`:

```python
def _utcnow_naive() -> datetime:
    """Naive UTC timestamp, matching the trades/decision_log storage convention so the
    day-boundary query in daily_summary compares consistently."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CostEvent(BaseModel):
    call_id: str
    timestamp: datetime
    provider: str
    model: str
    call_type: str          # "single" | "tool_loop"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float

    @classmethod
    def build(cls, *, provider: str, model: str, call_type: str,
              prompt_tokens: int, completion_tokens: int, latency_ms: float) -> "CostEvent":
        return cls(
            call_id=str(uuid4()),
            timestamp=_utcnow_naive(),
            provider=provider,
            model=model,
            call_type=call_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=usage_cost(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/cost.py tests/test_cost.py
git commit -m "feat(cost): add CostEvent model with build() factory"
```

---

### Task 3: `llm_cost_log` table + dialect translation

**Files:**
- Modify: `src/vibe_trading/data/db.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

The dialect translation is pure and unit-testable. Append to `tests/test_db.py`:

```python
from vibe_trading.data.db import translate_query


def test_translate_query_handles_llm_cost_log_insert():
    sql = "INSERT OR IGNORE INTO llm_cost_log (call_id, cost_usd) VALUES (?, ?)"
    out = translate_query(sql)
    assert "INSERT INTO llm_cost_log" in out
    assert "ON CONFLICT (call_id) DO NOTHING" in out
    assert "?" not in out  # placeholders translated to %s
    assert "%s" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py::test_translate_query_handles_llm_cost_log_insert -v`
Expected: FAIL — the assertion on `ON CONFLICT (call_id)` fails (no branch yet; `?` is translated but the conflict clause is absent).

- [ ] **Step 3: Write minimal implementation**

In `src/vibe_trading/data/db.py`, find the `translate_query()` branch for `decision_log` (around line 169):

```python
    if "INSERT OR IGNORE INTO decision_log" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO decision_log", "INSERT INTO decision_log")
        sql += " ON CONFLICT (decision_id) DO NOTHING"
```

Add a parallel branch immediately after it:

```python
    elif "INSERT OR IGNORE INTO llm_cost_log" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO llm_cost_log", "INSERT INTO llm_cost_log")
        sql += " ON CONFLICT (call_id) DO NOTHING"
```

(Place it as an `elif` in the same chain as the existing `decision_log` / `open_positions` branches.)

Then in `PostgresDatabase._create_tables` (around line 320, after the `decision_log` CREATE and before `self.conn.commit()`), add:

```python
            self.conn.execute("""
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
            """)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py::test_translate_query_handles_llm_cost_log_insert -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/data/db.py tests/test_db.py
git commit -m "feat(db): add llm_cost_log table + INSERT OR IGNORE dialect translation"
```

---

### Task 4: `PostgresCostLogger` + `daily_summary`

**Files:**
- Modify: `src/vibe_trading/agents/cost.py`
- Modify: `tests/test_cost.py`

- [ ] **Step 1: Write the failing tests**

`daily_summary` takes any object with `.execute(sql, params).fetchone()/.fetchall()` (the project's connection wrapper), so it's unit-testable with a fake connection. Append to `tests/test_cost.py`:

```python
from unittest.mock import MagicMock
from vibe_trading.agents.cost import daily_summary, PostgresCostLogger, CostEvent


class _FakeCursor:
    """Stand-in for the project's PostgresConnectionWrapper: execute() returns self,
    then fetchone()/fetchall() yield canned results queued by the test."""
    def __init__(self, scalar_row, model_rows):
        self._scalar_row = scalar_row
        self._model_rows = model_rows
        self._last = None
    def execute(self, sql, params=None):
        self._last = "by_model" if "GROUP BY" in sql else "scalar"
        return self
    def fetchone(self):
        return self._scalar_row
    def fetchall(self):
        return self._model_rows


def test_daily_summary_aggregates_and_projects():
    # scalar row: (total_cost, call_count, total_tokens); model rows: [(model, calls, cost)]
    conn = _FakeCursor(
        scalar_row=(0.0123, 47, 91234),
        model_rows=[("gemini/gemma-4-31b-it", 47, 0.0123)],
    )
    s = daily_summary(conn)
    assert abs(s["today_usd"] - 0.0123) < 1e-9
    assert s["calls"] == 47
    assert s["tokens"] == 91234
    assert abs(s["avg_cost_per_call"] - 0.0123 / 47) < 1e-9
    assert abs(s["projected_monthly_usd"] - 0.0123 * 30) < 1e-9
    assert s["by_model"][0]["model"] == "gemini/gemma-4-31b-it"


def test_daily_summary_empty_returns_zeros():
    conn = _FakeCursor(scalar_row=(None, 0, None), model_rows=[])
    s = daily_summary(conn)
    assert s["today_usd"] == 0.0
    assert s["calls"] == 0
    assert s["tokens"] == 0
    assert s["avg_cost_per_call"] == 0.0
    assert s["projected_monthly_usd"] == 0.0
    assert s["by_model"] == []


def test_postgres_cost_logger_record_is_best_effort():
    """A failing DB must NOT raise out of record() — cost logging can't break a trade."""
    boom_db = MagicMock()
    boom_db.connect.side_effect = RuntimeError("db down")
    logger_ = PostgresCostLogger(db=boom_db)
    ev = CostEvent.build(provider="g", model="m", call_type="single",
                        prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    logger_.record(ev)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py -k "daily_summary or cost_logger" -v`
Expected: FAIL — `ImportError: cannot import name 'daily_summary'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/vibe_trading/agents/cost.py`:

```python
class PostgresCostLogger:
    """Appends CostEvents to llm_cost_log. Best-effort: never raises to the caller.

    Holds its OWN PostgresDatabase instance (sharing the class-level pool) rather than
    the scheduler's, so its connect/close cycle can't alias `self.conn` on an instance
    the scheduler is concurrently using during a tick.
    """
    def __init__(self, db=None):
        from vibe_trading.data.db import PostgresDatabase
        self.pg_db = db or PostgresDatabase()

    def record(self, event: "CostEvent") -> None:
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
            try:
                self.pg_db.close()
            except Exception:
                pass


def daily_summary(conn) -> dict:
    """Aggregate today's (UTC) LLM spend from llm_cost_log. `conn` is a connection
    wrapper exposing .execute(sql, params).fetchone()/.fetchall(). Returns zeros on
    empty data; callers wrap in their own try/except for unavailable DBs."""
    today_start = _utcnow_naive().replace(hour=0, minute=0, second=0, microsecond=0)
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*), COALESCE(SUM(total_tokens), 0) "
        "FROM llm_cost_log WHERE timestamp >= ?",
        (today_start,),
    ).fetchone()
    today_usd = float(row[0] or 0.0)
    calls = int(row[1] or 0)
    tokens = int(row[2] or 0)
    model_rows = conn.execute(
        "SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0) FROM llm_cost_log "
        "WHERE timestamp >= ? GROUP BY model ORDER BY 3 DESC",
        (today_start,),
    ).fetchall()
    return {
        "today_usd": today_usd,
        "calls": calls,
        "tokens": tokens,
        "avg_cost_per_call": (today_usd / calls) if calls else 0.0,
        "projected_monthly_usd": today_usd * 30,
        "by_model": [{"model": m, "calls": int(c), "cost_usd": float(u)} for (m, c, u) in model_rows],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py -v`
Expected: all PASS (Task 1+2 tests plus the 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/cost.py tests/test_cost.py
git commit -m "feat(cost): add PostgresCostLogger sink + daily_summary aggregator"
```

---

### Task 5: Wire cost capture into `LLMClient`

**Files:**
- Modify: `src/vibe_trading/agents/client.py`
- Modify: `tests/test_multi_provider.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_provider.py`:

```python
from vibe_trading.agents.cost import CostEvent


class _CollectSink:
    def __init__(self):
        self.events = []
    def record(self, event):
        self.events.append(event)


def _mock_completion_with_usage(content='{"ok": 1}', pt=120, ct=45):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.usage = MagicMock(prompt_tokens=pt, completion_tokens=ct)
    return resp


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_call_llm_emits_cost_event_when_sink_set(mock_completion):
    mock_completion.return_value = _mock_completion_with_usage(pt=120, ct=45)
    sink = _CollectSink()
    LLMClient.set_cost_sink(sink)
    try:
        LLMClient().call_llm("gemma-4-31b-it", "sys", "usr")
    finally:
        LLMClient.set_cost_sink(None)
    assert len(sink.events) == 1
    ev = sink.events[0]
    assert isinstance(ev, CostEvent)
    assert ev.call_type == "single"
    assert ev.prompt_tokens == 120 and ev.completion_tokens == 45
    assert ev.provider == "gemini"


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_call_llm_no_sink_is_noop(mock_completion):
    mock_completion.return_value = _mock_completion_with_usage()
    LLMClient.set_cost_sink(None)  # explicit default
    # Must not raise and must return content
    out = LLMClient().call_llm("gemma-4-31b-it", "sys", "usr")
    assert out == '{"ok": 1}'


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_cost_sink_failure_does_not_break_call(mock_completion):
    mock_completion.return_value = _mock_completion_with_usage(content='{"ok": 2}')
    class _BoomSink:
        def record(self, event):
            raise RuntimeError("sink down")
    LLMClient.set_cost_sink(_BoomSink())
    try:
        out = LLMClient().call_llm("gemma-4-31b-it", "sys", "usr")
    finally:
        LLMClient.set_cost_sink(None)
    assert out == '{"ok": 2}'  # call still returns despite sink failure


@patch("litellm.completion")
@patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, clear=True)
def test_tool_loop_emits_cost_event_per_iteration(mock_completion):
    # Turn 1: a tool call; Turn 2: final content. => 2 completions => 2 cost events.
    msg1 = MagicMock()
    tc = MagicMock(); tc.id = "c1"; tc.function.name = "get_market_sentiment"; tc.function.arguments = "{}"
    msg1.tool_calls = [tc]
    msg2 = MagicMock(); msg2.tool_calls = None; msg2.content = '{"done": 1}'
    r1 = MagicMock(); r1.choices = [MagicMock(message=msg1)]; r1.usage = MagicMock(prompt_tokens=100, completion_tokens=10)
    r2 = MagicMock(); r2.choices = [MagicMock(message=msg2)]; r2.usage = MagicMock(prompt_tokens=130, completion_tokens=20)
    mock_completion.side_effect = [r1, r2]
    sink = _CollectSink()
    LLMClient.set_cost_sink(sink)
    tool_exec = MagicMock(); tool_exec.execute.return_value = "{}"
    try:
        LLMClient().call_llm_with_tools("gemma-4-31b-it", "sys", "usr", tools=[], tool_executor=tool_exec)
    finally:
        LLMClient.set_cost_sink(None)
    assert len(sink.events) == 2
    assert all(e.call_type == "tool_loop" for e in sink.events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_multi_provider.py -k "emits_cost or no_sink or cost_sink_failure or tool_loop_emits" -v`
Expected: FAIL — `AttributeError: type object 'LLMClient' has no attribute 'set_cost_sink'`.

- [ ] **Step 3: Write minimal implementation**

In `src/vibe_trading/agents/client.py`, add the import near the top (after `import litellm`):

```python
from vibe_trading.agents.cost import CostEvent
```

Add the class-level sink + helpers alongside the existing throttle fields (after `_throttle_lock = threading.Lock()`, around line 49):

```python
    _cost_sink = None  # set once at app startup; None => no-op (tests/eval)

    @classmethod
    def set_cost_sink(cls, sink) -> None:
        """Install (or clear with None) the process-wide cost sink. Mirrors the
        class-level throttle: production sets a PostgresCostLogger; tests/eval leave None."""
        cls._cost_sink = sink

    def _emit_cost(self, response, model_str: str, call_type: str, latency_ms: float) -> None:
        """Best-effort cost emit. Never raises — cost logging must not break an LLM call."""
        sink = LLMClient._cost_sink
        if sink is None:
            return
        try:
            usage = getattr(response, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            sink.record(CostEvent.build(
                provider=self.provider, model=model_str, call_type=call_type,
                prompt_tokens=pt, completion_tokens=ct, latency_ms=latency_ms,
            ))
        except Exception as e:
            logger.warning(f"cost emit failed (non-fatal): {e}")
```

In `call_llm`, replace (around lines 117-118):

```python
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
```

with:

```python
        _t0 = time.monotonic()
        response = litellm.completion(**kwargs)
        self._emit_cost(response, model_str, "single", (time.monotonic() - _t0) * 1000.0)
        return response.choices[0].message.content
```

In `call_llm_with_tools`, find the per-iteration completion (around lines 143-150):

```python
            response = litellm.completion(
                model=model_str,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
            assistant_msg = response.choices[0].message
```

Wrap with timing + emit:

```python
            _t0 = time.monotonic()
            response = litellm.completion(
                model=model_str,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
            self._emit_cost(response, model_str, "tool_loop", (time.monotonic() - _t0) * 1000.0)
            assistant_msg = response.choices[0].message
```

(`time` is already imported for the throttle.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_multi_provider.py -v`
Expected: all PASS, including the 4 new cost tests.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/client.py tests/test_multi_provider.py
git commit -m "feat(client): emit CostEvent to class-level sink after each completion"
```

---

### Task 6: `/api/costs` endpoint

**Files:**
- Modify: `src/vibe_trading/web/main.py`

This is a thin FastAPI wrapper over `daily_summary`, which is already unit-tested in Task 4. Verification is an import/route check plus the existing test suite staying green.

- [ ] **Step 1: Add the endpoint**

In `src/vibe_trading/web/main.py`, add the import near the top with the other imports:

```python
from vibe_trading.agents.cost import daily_summary
```

Add a new endpoint after the `/api/metrics` route (after line ~148, before `/api/positions`):

```python
@app.get("/api/costs")
def get_costs():
    """Today's LLM spend: $/call, $/day, projected $/month, tokens, per-model breakdown."""
    default = {
        "today_usd": 0.0, "calls": 0, "tokens": 0,
        "avg_cost_per_call": 0.0, "projected_monthly_usd": 0.0, "by_model": [],
    }
    try:
        with get_pg_conn() as conn:
            return daily_summary(conn)
    except Exception:
        return default
```

- [ ] **Step 2: Verify import + route registration**

Run: `uv run python -c "from vibe_trading.web.main import app; paths = [r.path for r in app.routes]; assert '/api/costs' in paths, paths; print('OK /api/costs registered')"`
Expected: `OK /api/costs registered`.

- [ ] **Step 3: Run the full suite (no regressions)**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/vibe_trading/web/main.py
git commit -m "feat(api): add GET /api/costs daily LLM spend summary"
```

---

### Task 7: Wire sink + spike alarm into the scheduler

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py`
- Modify: `tests/test_cost.py`

- [ ] **Step 1: Write the failing tests**

The alarm decision is testable in isolation. Append to `tests/test_cost.py`:

```python
from vibe_trading.agents.cost import should_alarm


def test_should_alarm_fires_over_threshold():
    assert should_alarm(today_usd=6.0, threshold=5.0, already_alarmed_today=False) is True


def test_should_alarm_silent_under_threshold():
    assert should_alarm(today_usd=2.0, threshold=5.0, already_alarmed_today=False) is False


def test_should_alarm_dedups_within_day():
    assert should_alarm(today_usd=6.0, threshold=5.0, already_alarmed_today=True) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py -k "should_alarm" -v`
Expected: FAIL — `ImportError: cannot import name 'should_alarm'`.

- [ ] **Step 3: Implement `should_alarm` + scheduler wiring**

Append the pure predicate to `src/vibe_trading/agents/cost.py`:

```python
def should_alarm(today_usd: float, threshold: float, already_alarmed_today: bool) -> bool:
    """True when today's spend exceeds the threshold and we haven't already alarmed today."""
    return today_usd > threshold and not already_alarmed_today
```

In `src/vibe_trading/runtime/scheduler.py`, add imports near the top:

```python
import os
from datetime import date
from vibe_trading.agents.client import LLMClient
from vibe_trading.agents.cost import PostgresCostLogger, daily_summary, should_alarm
```

(Some of these may already be imported — do not duplicate; `os` and `LLMClient` are likely present.)

In `TradingScheduler.__init__`, after `self.pg_db = PostgresDatabase()` (line ~25), wire the sink and the alarm-dedup state:

```python
        # Route every LLM call's cost into Postgres (own pooled connection, not self.pg_db).
        LLMClient.set_cost_sink(PostgresCostLogger())
        self._cost_alarmed_on: date | None = None
```

Add the alarm method to the class (e.g. just above `_send_discord_alert`):

```python
    def _check_cost_alarm(self):
        """Discord-alarm once per UTC day when LLM spend exceeds LLM_DAILY_COST_ALARM_USD."""
        threshold = float(os.getenv("LLM_DAILY_COST_ALARM_USD", "5.0"))
        try:
            self.pg_db.connect()
            summary = daily_summary(self.pg_db.conn)
        except Exception as e:
            logger.warning(f"cost alarm check skipped (non-fatal): {e}")
            return
        finally:
            try:
                self.pg_db.close()
            except Exception:
                pass

        today = datetime.utcnow().date()
        already = self._cost_alarmed_on == today
        if should_alarm(summary["today_usd"], threshold, already):
            self._cost_alarmed_on = today
            self._send_discord_alert(
                f"💸 **LLM COST ALARM:** today's spend ${summary['today_usd']:.2f} "
                f"exceeded ${threshold:.2f} ({summary['calls']} calls, "
                f"~${summary['projected_monthly_usd']:.2f}/mo projected)."
            )
```

Call it once per tick — at the start of the `sync_and_evaluate` body, inside the existing `try:` (after the execution-window log line, before fetching candles):

```python
                self._check_cost_alarm()
```

(`datetime` is already imported in scheduler.py.)

- [ ] **Step 4: Run tests + import check**

Run: `uv run pytest tests/test_cost.py -v`
Expected: all PASS.

Run: `uv run python -c "import ast; ast.parse(open('src/vibe_trading/runtime/scheduler.py').read()); print('scheduler parses OK')"`
Expected: `scheduler parses OK`.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py src/vibe_trading/agents/cost.py tests/test_cost.py
git commit -m "feat(scheduler): wire cost sink + per-tick daily-spend Discord alarm"
```

---

### Task 8: Dashboard cost tile

**Files:**
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/MetricsGrid.tsx`

Follow the existing Next.js patterns (per `web/AGENTS.md`) — mirror how `MetricsGrid` already renders metric cards and how `page.tsx` polls the API.

- [ ] **Step 1: Read the existing patterns**

Read `web/src/components/MetricsGrid.tsx` and `web/src/app/page.tsx` to match the card markup and the 30s `fetchDashboardData` poll. Note the `MetricsData` interface and how `metrics` is fetched.

- [ ] **Step 2: Fetch `/api/costs` in page.tsx**

In `web/src/app/page.tsx`, add cost state alongside the existing `metrics` state:

```tsx
const [costs, setCosts] = useState<any>(null);
```

In `fetchDashboardData`, add `/api/costs` to the `Promise.all` fetch batch and set it:

```tsx
const [metricsRes, posRes, decRes, costsRes] = await Promise.all([
  fetch("/api/metrics"),
  fetch("/api/positions"),
  fetch("/api/decisions?limit=5"),
  fetch("/api/costs"),
]);
if (metricsRes.ok) setMetrics(await metricsRes.json());
if (posRes.ok) setPositions(await posRes.json());
if (decRes.ok) setDecisions(await decRes.json());
if (costsRes.ok) setCosts(await costsRes.json());
```

Pass `costs` into `MetricsGrid`:

```tsx
<MetricsGrid metrics={metrics} costs={costs} />
```

- [ ] **Step 3: Render a cost card in MetricsGrid.tsx**

In `web/src/components/MetricsGrid.tsx`, extend the component signature to accept `costs` and add one card mirroring the existing card markup, showing **Today's LLM Spend** (`costs?.today_usd`), with **projected/mo** (`costs?.projected_monthly_usd`) and **calls** (`costs?.calls`) as the sub-line. Use the same `Card`/formatting classes the other metric cards use. Guard for `costs == null` (render `$0.00` / `—`) exactly as the file already guards `metrics == null`.

Example card body (adapt classes to match the file's existing cards):

```tsx
<Card className="bg-slate-900/40 border-slate-900/60 ...">
  <CardContent className="...">
    <p className="text-slate-500 ...">LLM Spend (today)</p>
    <p className="text-slate-100 ...">${(costs?.today_usd ?? 0).toFixed(4)}</p>
    <p className="text-slate-500 text-xs">
      ~${(costs?.projected_monthly_usd ?? 0).toFixed(2)}/mo · {costs?.calls ?? 0} calls
    </p>
  </CardContent>
</Card>
```

- [ ] **Step 4: Type-check the web project**

Run: `cd web && npx tsc --noEmit; echo "exit=$?"`
Expected: `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/page.tsx web/src/components/MetricsGrid.tsx
git commit -m "feat(web): add LLM cost tile (today + projected/mo) to dashboard"
```

---

### Task 9: README docs + full-suite verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Document the feature**

Add a short "Cost Tracking" subsection to `README.md` (near the eval/observability docs):

```markdown
## Cost Tracking

Every LLM call's tokens, dollar cost, and latency are logged to the `llm_cost_log`
Postgres table (capture happens in `LLMClient`, the single call chokepoint). Cost is
computed from LiteLLM's model pricing, with a shadow-price fallback (`PRICE_OVERRIDES`
in `agents/cost.py`) for models LiteLLM doesn't price — e.g. the free-tier Gemma models —
so projected $/month stays meaningful.

- Dashboard: a cost tile shows today's spend, projected $/month, and call count.
- API: `GET /api/costs` returns the daily summary + per-model breakdown.
- Alarm: the scheduler sends a Discord alert once per UTC day when spend exceeds
  `LLM_DAILY_COST_ALARM_USD` (default $5).

Cost tracking is observational only — a logging failure never interrupts a trade. The
hard daily-$ kill switch (blocking trades) is a separate guardrail.
```

Add to `.env.example` under the LLM config block:

```
# Daily LLM spend (USD) above which the scheduler sends a Discord cost alarm
LLM_DAILY_COST_ALARM_USD=5.0
```

- [ ] **Step 2: Full suite**

Run: `uv run pytest -v`
Expected: all PASS (existing + new `test_cost.py` + client cost tests + db translation test).

- [ ] **Step 3: Smoke-import all touched modules**

Run: `uv run python -c "from vibe_trading.agents.cost import usage_cost, CostEvent, PostgresCostLogger, daily_summary, should_alarm; from vibe_trading.agents.client import LLMClient; from vibe_trading.web.main import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md .env.example
git commit -m "docs(cost): document cost tracking + LLM_DAILY_COST_ALARM_USD"
```

---

## Spec coverage check (self-review)

- **`usage_cost` (litellm + shadow override + 0 fallback):** Task 1.
- **`CostEvent` + `build()`, naive-UTC timestamp:** Task 2.
- **`llm_cost_log` table + `INSERT OR IGNORE` translation:** Task 3.
- **`PostgresCostLogger` (own pooled instance, best-effort) + `daily_summary(conn)`:** Task 4.
- **Class-level `_cost_sink` + `set_cost_sink` + `_emit_cost`, wired into `call_llm` and `call_llm_with_tools` (per-iteration):** Task 5.
- **`GET /api/costs`:** Task 6.
- **Scheduler sink wire-up + per-tick spike alarm (`should_alarm`, once-per-UTC-day dedup, `LLM_DAILY_COST_ALARM_USD`):** Task 7.
- **Dashboard cost tile:** Task 8.
- **README + `.env.example`:** Task 9.
- **Error-handling invariant (best-effort, never raises into a call/trade):** Tasks 4 (`record`), 5 (`_emit_cost` + sink-failure test), 6 (endpoint try/except), 7 (alarm try/except).
- **Backwards compat (default no sink → unchanged behavior):** Task 5 (`test_call_llm_no_sink_is_noop`).
- **Test isolation (reset `_cost_sink`):** every client cost test resets the sink in a `finally`.

Type consistency check: `CostEvent` field names (`call_id`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd`, `latency_ms`, `call_type`) are identical across `cost.py`, the `llm_cost_log` schema (Task 3), the `PostgresCostLogger` INSERT (Task 4), and `_emit_cost` (Task 5). `daily_summary` return keys (`today_usd`, `calls`, `tokens`, `avg_cost_per_call`, `projected_monthly_usd`, `by_model`) match the `/api/costs` endpoint (Task 6), the dashboard tile (Task 8), and the alarm (Task 7).
