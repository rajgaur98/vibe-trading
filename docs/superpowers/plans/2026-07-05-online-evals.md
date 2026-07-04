# Online Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every live decision eventually gets an outcome score (deterministic) and a sampled LLM-judge score, both pushed to Langfuse and persisted to a `decision_scores` table; a weekly Discord digest makes drift visible without anyone remembering to look.

**Architecture:** A new `src/vibe_trading/eval/online.py` module runs a scoring pass: (1) for each unscored decision whose outcome is now knowable, compute a deterministic outcome score — realized PnL for decisions that became closed trades, counterfactual forward return (reusing the journal's horizon convention) for flat/rejected ones; (2) sample a capped number of scored decisions per day for a generic-rubric LLM judge (groundedness + consistency against the stored feature snapshot); (3) with `--digest`, aggregate the week vs the trailing month and send a Discord digest. Everything is strictly best-effort — a scoring failure can never block or corrupt trading.

**Tech Stack:** Python 3.12, existing `PostgresDatabase`/DuckDB `Database` wrappers, `LLMClient` + `validate_structured`-style parsing, Langfuse 4.x `create_score`, Discord webhook via a shared `monitoring.send_discord`, pytest.

**Workstream:** C of `docs/superpowers/specs/2026-07-04-ai-deepening-roadmap-design.md`.
**Depends on workstream B** (prompt registry `src/vibe_trading/agents/prompts.py`, `prompt_version=` kwarg on `LLMClient.call_llm`, `decision_log.prompt_version` column). Do not start before B is merged.

## Global Constraints

- Never-block invariant: every DB/LLM/Langfuse/Discord failure in this module is caught, logged as a warning, and skipped. The scheduler and `trade-once` must behave identically whether scoring succeeds, fails, or is absent.
- Idempotent: `decision_scores.decision_id` is the primary key; a re-run scores nothing twice (`INSERT OR IGNORE`), and the judge only fills rows where `judge_score IS NULL`.
- Counterfactual horizon reuses `journal.COUNTERFACTUAL_HORIZON_CANDLES` (default 6 × 4h = 24h) — one horizon convention across the codebase.
- Judge calls: model = `EVAL_JUDGE_MODEL` (fallback: client's model), tagged `call_type="online_judge"` in `llm_cost_log`, capped at `ONLINE_JUDGE_DAILY_CAP` (default 5) per UTC day, and skipped entirely once `LLM_DAILY_COST_CAP_USD` is reached.
- Decisions with `action = 'close'` are not scored (position-management, not an entry thesis). Decisions still open (in `open_positions`) are deferred, not skipped.
- Score names pushed to Langfuse: `outcome_score` and `online_judge_score` — exact strings, they become the Langfuse score keys.
- New table is Postgres-only (transactional state, like `decision_embeddings`).
- All tests offline (no network/LLM/Postgres). Run tests with `uv run pytest <path> -v`.

---

### Task 1: decision_scores table and dialect support

**Files:**
- Modify: `src/vibe_trading/data/db.py` (`PostgresDatabase._create_tables`, `translate_query`)
- Test: `tests/test_db.py` (extend)

**Interfaces:**
- Consumes: existing migration/translate patterns.
- Produces: Postgres table `decision_scores(decision_id VARCHAR PRIMARY KEY, scored_at TIMESTAMP, kind VARCHAR, outcome_pct DOUBLE PRECISION, outcome_score DOUBLE PRECISION, judge_score DOUBLE PRECISION, judge_note TEXT, judged_at TIMESTAMP, prompt_version VARCHAR)`; `translate_query` maps `INSERT OR IGNORE INTO decision_scores` → `ON CONFLICT (decision_id) DO NOTHING`. Tasks 3–5 and workstream D rely on these exact columns.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py`:

```python
def test_translate_query_maps_decision_scores_insert_or_ignore():
    from vibe_trading.data.db import translate_query
    sql = translate_query(
        "INSERT OR IGNORE INTO decision_scores (decision_id) VALUES (?)")
    assert "INSERT INTO decision_scores" in sql
    assert "ON CONFLICT (decision_id) DO NOTHING" in sql
    assert "?" not in sql and "%s" in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v -k decision_scores`
Expected: FAIL (`ON CONFLICT` clause absent)

- [ ] **Step 3: Write the implementation**

In `translate_query` in `src/vibe_trading/data/db.py`, add a branch after the `llm_cost_log` branch:

```python
    elif "INSERT OR IGNORE INTO decision_scores" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO decision_scores",
                          "INSERT INTO decision_scores")
        sql += " ON CONFLICT (decision_id) DO NOTHING"
```

In `PostgresDatabase._create_tables`, after the `decision_embeddings` CREATE TABLE:

```python
            # Online-eval scores: one row per scored decision (see eval/online.py).
            # outcome_* is deterministic (PnL / counterfactual forward return);
            # judge_* is the sampled generic-rubric LLM judge, filled in later.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_scores (
                    decision_id VARCHAR PRIMARY KEY,
                    scored_at TIMESTAMP,
                    kind VARCHAR,                    -- 'closed' | 'counterfactual'
                    outcome_pct DOUBLE PRECISION,    -- signed % in the decision's favor (raw move for flat)
                    outcome_score DOUBLE PRECISION,  -- [0,1]
                    judge_score DOUBLE PRECISION,    -- [0,1], NULL until sampled
                    judge_note TEXT,
                    judged_at TIMESTAMP,
                    prompt_version VARCHAR           -- copied from decision_log at scoring time
                )
            """)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/data/db.py tests/test_db.py
git commit -m "feat(online-evals): decision_scores table + dialect mapping"
```

---

### Task 2: Deterministic outcome-score math

**Files:**
- Create: `src/vibe_trading/eval/online.py`
- Test: `tests/test_online_evals.py`

**Interfaces:**
- Consumes: nothing.
- Produces: constants `DIRECTIONAL_BAND_PCT = 2.0`, `FLAT_OK_PCT = 1.0`, `FLAT_ZERO_PCT = 5.0`; `directional_score(signed_pct: float) -> float`; `flat_score(move_pct: float) -> float`. Task 3 consumes both functions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_online_evals.py
import pytest

from vibe_trading.eval.online import directional_score, flat_score


def test_directional_score_maps_band_to_unit_interval():
    assert directional_score(0.0) == pytest.approx(0.5)     # no move -> coin flip
    assert directional_score(2.0) == pytest.approx(1.0)     # +band in favor -> 1.0
    assert directional_score(-2.0) == pytest.approx(0.0)    # band against -> 0.0
    assert directional_score(1.0) == pytest.approx(0.75)    # linear in between
    assert directional_score(50.0) == 1.0                   # clamped
    assert directional_score(-50.0) == 0.0                  # clamped


def test_flat_score_rewards_small_moves():
    assert flat_score(0.0) == 1.0
    assert flat_score(1.0) == 1.0        # within FLAT_OK_PCT
    assert flat_score(-1.0) == 1.0       # sign-agnostic (uses |move|)
    assert flat_score(5.0) == 0.0        # at/after FLAT_ZERO_PCT
    assert flat_score(3.0) == pytest.approx(0.5)  # halfway on the linear ramp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_online_evals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.eval.online'`

- [ ] **Step 3: Write the implementation**

```python
# src/vibe_trading/eval/online.py
"""Online evaluation — score PRODUCTION decisions once their outcome is knowable.

Three layers (spec workstream C), all strictly best-effort:
  C1  deterministic outcome scoring  (this module's OutcomeScorer)
  C2  sampled generic-rubric LLM judge (capped per day)
  C3  weekly drift digest to Discord

A 'correct' flat is scored as 'no strong move happened' — a proxy for 'no edge
existed', not ground truth (a flat that dodged a crash scores poorly here). The
caveat is deliberate and documented; deterministic beats clever.
"""
import logging

logger = logging.getLogger(__name__)

# Directional decisions: +DIRECTIONAL_BAND_PCT% move in the decision's favor -> 1.0,
# same move against -> 0.0, linear in between, 0.5 for no move.
DIRECTIONAL_BAND_PCT = 2.0
# Flat decisions: |move| <= FLAT_OK_PCT -> 1.0, >= FLAT_ZERO_PCT -> 0.0, linear between.
FLAT_OK_PCT = 1.0
FLAT_ZERO_PCT = 5.0


def directional_score(signed_pct: float) -> float:
    """Map a signed %-move (positive = the decision's direction was profitable)
    onto [0, 1]."""
    return max(0.0, min(1.0, 0.5 + signed_pct / (2.0 * DIRECTIONAL_BAND_PCT)))


def flat_score(move_pct: float) -> float:
    """Score a flat decision by how quiet the market actually was."""
    m = abs(move_pct)
    if m <= FLAT_OK_PCT:
        return 1.0
    if m >= FLAT_ZERO_PCT:
        return 0.0
    return 1.0 - (m - FLAT_OK_PCT) / (FLAT_ZERO_PCT - FLAT_OK_PCT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_online_evals.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/online.py tests/test_online_evals.py
git commit -m "feat(online-evals): deterministic outcome-score math"
```

---

### Task 3: OutcomeScorer — resolve, persist, push to Langfuse

**Files:**
- Modify: `src/vibe_trading/eval/online.py`
- Test: `tests/test_online_evals.py`

**Interfaces:**
- Consumes: Task 1's table; Task 2's score functions; `journal.COUNTERFACTUAL_HORIZON_CANDLES` and the `_CANDLE_HOURS = 4` convention.
- Produces: `Outcome` dataclass (`kind: str`, `outcome_pct: float`, `outcome_score: float`); `resolve_outcome(action, closed_trade, is_open, entry_price, future_price) -> Optional[Outcome]` (pure); `OutcomeScorer(pg_factory=None, duck_factory=None, now_fn=None, push_fn=None, horizon_candles=None)` with `.run_pass(batch_limit: int = 200) -> int`; module helper `push_langfuse_score(trace_id, name, value, comment) -> None`. Task 4 reuses `push_langfuse_score`; Task 6 calls `OutcomeScorer().run_pass()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_online_evals.py`:

```python
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from vibe_trading.eval.online import Outcome, resolve_outcome, OutcomeScorer


def test_resolve_closed_trade_uses_realized_pnl():
    out = resolve_outcome(action="long", closed_trade=(50.0, 1000.0),  # (pnl, size)
                          is_open=False, entry_price=None, future_price=None)
    assert out.kind == "closed"
    assert out.outcome_pct == pytest.approx(5.0)          # 50/1000 * 100
    assert out.outcome_score == pytest.approx(1.0)        # +5% >> +2% band


def test_resolve_open_position_defers():
    assert resolve_outcome(action="long", closed_trade=None, is_open=True,
                           entry_price=100.0, future_price=110.0) is None


def test_resolve_counterfactual_signs_by_direction():
    # price went +10%: a rejected short would have lost
    out = resolve_outcome(action="short", closed_trade=None, is_open=False,
                          entry_price=100.0, future_price=110.0)
    assert out.kind == "counterfactual"
    assert out.outcome_pct == pytest.approx(-10.0)
    assert out.outcome_score == 0.0


def test_resolve_flat_uses_flat_score_on_raw_move():
    out = resolve_outcome(action="flat", closed_trade=None, is_open=False,
                          entry_price=100.0, future_price=100.5)
    assert out.kind == "counterfactual"
    assert out.outcome_pct == pytest.approx(0.5)
    assert out.outcome_score == 1.0


def test_resolve_missing_candles_returns_none():
    assert resolve_outcome(action="long", closed_trade=None, is_open=False,
                           entry_price=None, future_price=110.0) is None
    assert resolve_outcome(action="long", closed_trade=None, is_open=False,
                           entry_price=100.0, future_price=None) is None


def _fake_pg(rows_by_query: dict):
    """A PostgresDatabase stand-in whose conn.execute returns canned rows keyed by
    a substring of the SQL."""
    pg = MagicMock()
    def execute(sql, params=None):
        cur = MagicMock()
        for needle, rows in rows_by_query.items():
            if needle in sql:
                cur.fetchall.return_value = rows
                cur.fetchone.return_value = rows[0] if rows else None
                return cur
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        return cur
    pg.conn.execute.side_effect = execute
    return pg


def test_run_pass_scores_one_decision_and_pushes_langfuse():
    now = datetime(2026, 7, 5, 12, 0)
    old_ts = now - timedelta(hours=48)  # past the 24h horizon
    pg = _fake_pg({
        "FROM decision_log": [("dec-1", old_ts, "BTC/USDT", "long", "trace-1", "bundle-v1")],
        "FROM trades": [],
        "FROM open_positions": [],
    })
    duck = MagicMock()
    # entry close then future close
    duck.conn.execute.return_value.fetchone.side_effect = [(100.0,), (104.0,)]
    pushed = []
    scorer = OutcomeScorer(pg_factory=lambda: pg, duck_factory=lambda: duck,
                           now_fn=lambda: now,
                           push_fn=lambda *a, **k: pushed.append((a, k)))
    scored = scorer.run_pass()
    assert scored == 1
    insert_sql = [c for c in pg.conn.execute.call_args_list
                  if "INSERT OR IGNORE INTO decision_scores" in c.args[0]]
    assert insert_sql, "no decision_scores INSERT"
    params = insert_sql[0].args[1]
    assert params[0] == "dec-1"
    assert "bundle-v1" in params            # prompt_version copied over
    assert pushed and pushed[0][0][0] == "trace-1"   # trace_id
    assert pushed[0][0][1] == "outcome_score"        # Langfuse score name


def test_run_pass_never_raises(caplog):
    pg = MagicMock()
    pg.connect.side_effect = RuntimeError("db down")
    scorer = OutcomeScorer(pg_factory=lambda: pg)
    assert scorer.run_pass() == 0  # swallowed, logged, zero scored
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_online_evals.py -v`
Expected: new FAILs (`ImportError: cannot import name 'Outcome'`)

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/eval/online.py`:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from vibe_trading.journal import COUNTERFACTUAL_HORIZON_CANDLES, _CANDLE_HOURS


@dataclass
class Outcome:
    kind: str            # "closed" | "counterfactual"
    outcome_pct: float
    outcome_score: float


def resolve_outcome(action: str, closed_trade: Optional[tuple], is_open: bool,
                    entry_price: Optional[float],
                    future_price: Optional[float]) -> Optional[Outcome]:
    """Pure resolution: decide a decision's outcome from pre-fetched facts.

    closed_trade: (realized_pnl, size_usd) when the decision became a closed trade.
    is_open:      the decision's position is still open -> defer (None).
    entry/future_price: 4h closes at decision time and one horizon later (None on gaps).
    Returns None when the outcome is not yet knowable — the caller retries next pass.
    """
    if closed_trade is not None:
        pnl, size = closed_trade
        if not size:
            return None
        pct = float(pnl) / float(size) * 100.0
        return Outcome("closed", pct, directional_score(pct))
    if is_open:
        return None
    if entry_price is None or future_price is None or not entry_price:
        return None
    fwd = (float(future_price) - float(entry_price)) / float(entry_price) * 100.0
    if action == "flat":
        return Outcome("counterfactual", fwd, flat_score(fwd))
    signed = -fwd if action == "short" else fwd
    return Outcome("counterfactual", signed, directional_score(signed))


def push_langfuse_score(trace_id: Optional[str], name: str, value: float,
                        comment: str = "") -> None:
    """Attach a numeric score to the decision's Langfuse trace. Fail-soft: any
    SDK/network error is logged and swallowed."""
    if not trace_id:
        return
    try:
        from langfuse import get_client
        get_client().create_score(trace_id=trace_id, name=name, value=value,
                                  comment=comment)
    except Exception as e:
        logger.warning(f"langfuse score push failed (non-fatal): {e}")


class OutcomeScorer:
    """C1: scan unscored decisions, resolve knowable outcomes, persist + push.
    Mirrors PrecedentRetriever's injected-factory pattern for testability."""

    def __init__(self, pg_factory=None, duck_factory=None,
                 now_fn: Optional[Callable] = None, push_fn=None,
                 horizon_candles: Optional[int] = None):
        self._pg_factory = pg_factory
        self._duck_factory = duck_factory
        self._now = now_fn or (lambda: datetime.utcnow())
        self._push = push_fn or push_langfuse_score
        self.horizon_candles = horizon_candles or COUNTERFACTUAL_HORIZON_CANDLES

    def _pg(self):
        if self._pg_factory:
            return self._pg_factory()
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase()

    def _duck(self):
        if self._duck_factory:
            return self._duck_factory()
        from vibe_trading.data.db import Database
        return Database(read_only=True)

    def run_pass(self, batch_limit: int = 200) -> int:
        """Score every unscored decision whose outcome is knowable. Returns the
        number scored. Never raises."""
        try:
            return self._run_pass(batch_limit)
        except Exception as e:
            logger.warning(f"outcome scoring pass failed (non-fatal): {e}")
            return 0

    def _run_pass(self, batch_limit: int) -> int:
        pg = self._pg()
        pg.connect()
        try:
            candidates = pg.conn.execute(
                "SELECT d.decision_id, d.timestamp, d.symbol, d.action, d.trace_id, "
                "       d.prompt_version "
                "FROM decision_log d LEFT JOIN decision_scores s "
                "  ON s.decision_id = d.decision_id "
                "WHERE s.decision_id IS NULL AND d.action != 'close' "
                "ORDER BY d.timestamp ASC LIMIT ?",
                (batch_limit,),
            ).fetchall()
        finally:
            pg.close()

        scored = 0
        horizon = timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        for decision_id, ts, symbol, action, trace_id, prompt_version in candidates:
            try:
                outcome = self._resolve_for(decision_id, ts, symbol, action, horizon)
                if outcome is None:
                    continue
                self._persist(decision_id, outcome, prompt_version)
                self._push(trace_id, "outcome_score", outcome.outcome_score,
                           f"{outcome.kind}: {outcome.outcome_pct:+.2f}% ({action})")
                scored += 1
            except Exception as e:  # one bad decision must not stop the pass
                logger.warning(f"scoring {decision_id} failed (non-fatal): {e}")
        return scored

    def _resolve_for(self, decision_id, ts, symbol, action,
                     horizon: timedelta) -> Optional[Outcome]:
        pg = self._pg()
        pg.connect()
        try:
            trade = pg.conn.execute(
                "SELECT realized_pnl, size_usd FROM trades WHERE decision_id = ?",
                (decision_id,)).fetchone()
            open_row = pg.conn.execute(
                "SELECT 1 FROM open_positions WHERE decision_id = ?",
                (decision_id,)).fetchone()
        finally:
            pg.close()

        entry_price = future_price = None
        if trade is None and open_row is None:
            if ts + horizon > self._now():
                return None  # horizon not elapsed — retry next pass
            duck = self._duck()
            duck.connect()
            try:
                entry = duck.conn.execute(
                    "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                    "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                    (symbol, ts)).fetchone()
                fut = duck.conn.execute(
                    "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                    "AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                    (symbol, ts + horizon)).fetchone()
            finally:
                duck.close()
            entry_price = entry[0] if entry else None
            future_price = fut[0] if fut else None

        return resolve_outcome(
            action=action,
            closed_trade=tuple(trade) if trade is not None else None,
            is_open=open_row is not None,
            entry_price=entry_price, future_price=future_price,
        )

    def _persist(self, decision_id: str, outcome: Outcome,
                 prompt_version: Optional[str]) -> None:
        pg = self._pg()
        pg.connect()
        try:
            pg.conn.execute(
                "INSERT OR IGNORE INTO decision_scores "
                "(decision_id, scored_at, kind, outcome_pct, outcome_score, "
                " prompt_version) VALUES (?, ?, ?, ?, ?, ?)",
                (decision_id, self._now(), outcome.kind, outcome.outcome_pct,
                 outcome.outcome_score, prompt_version),
            )
        finally:
            pg.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_online_evals.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/online.py tests/test_online_evals.py
git commit -m "feat(online-evals): OutcomeScorer resolves, persists, pushes to Langfuse"
```

---

### Task 4: Sampled production judge (generic rubric, capped, cost-guarded)

**Files:**
- Modify: `src/vibe_trading/agents/prompts.py` (register `ONLINE_JUDGE_SYSTEM`)
- Regenerate: `tests/fixtures/prompt_pins.json`
- Modify: `src/vibe_trading/agents/client.py` (`call_llm` gains `call_type: str = "single"`)
- Modify: `src/vibe_trading/eval/online.py`
- Test: `tests/test_online_evals.py`, `tests/test_prompts.py` (pins pass unchanged after regen)

**Interfaces:**
- Consumes: workstream B's registry + `prompt_version=` kwarg; Task 3's `push_langfuse_score`; `daily_summary`, `should_block_trading` from `vibe_trading.agents.cost`.
- Produces: `OnlineJudgeVerdict` (pydantic: `grounded: bool`, `consistent: bool`, `justification: str`); `OnlineJudge(client=None, pg_factory=None, now_fn=None, push_fn=None, daily_cap=None)` with `.run_pass() -> int`; `prompts.ONLINE_JUDGE_SYSTEM`. `bundle_version()` is unaffected (it covers `DECISION_PROMPT_NAMES` only — that is why B scoped it).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_online_evals.py`:

```python
from vibe_trading.eval.online import OnlineJudge, OnlineJudgeVerdict


def _judge_pg(unjudged_rows, judged_today=0, spend_today=0.0):
    pg = MagicMock()
    def execute(sql, params=None):
        cur = MagicMock()
        if "judged_at >=" in sql:                      # daily cap counter
            cur.fetchone.return_value = (judged_today,)
        elif "FROM llm_cost_log" in sql:               # daily_summary spend query
            cur.fetchone.return_value = (spend_today, 0, 0, 0, 0, 0, 0)
            cur.fetchall.return_value = []
        elif "judge_score IS NULL" in sql:             # sample selection
            cur.fetchall.return_value = unjudged_rows
        else:
            cur.fetchall.return_value = []
            cur.fetchone.return_value = None
        return cur
    pg.conn.execute.side_effect = execute
    return pg


def _row():
    return ("dec-1", "long", "BTC/USDT", "went long on confluence",
            '{"close": 100.0, "rsi_14": 60.0}', "trace-1")


def test_judge_scores_a_sampled_decision():
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"grounded": true, "consistent": false, "justification": "j"}')
    pg = _judge_pg([_row()])
    pushed = []
    judge = OnlineJudge(client=client, pg_factory=lambda: pg,
                        push_fn=lambda *a, **k: pushed.append(a))
    assert judge.run_pass() == 1
    # judge call is tagged for cost attribution
    assert client.call_llm.call_args.kwargs["call_type"] == "online_judge"
    updates = [c for c in pg.conn.execute.call_args_list
               if "UPDATE decision_scores" in c.args[0]]
    assert updates and updates[0].args[1][0] == pytest.approx(0.5)  # (True+False)/2
    assert pushed[0][1] == "online_judge_score"


def test_judge_respects_daily_cap():
    client = MagicMock()
    pg = _judge_pg([_row()], judged_today=5)   # cap (default 5) already reached
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0
    client.call_llm.assert_not_called()


def test_judge_skips_when_llm_cost_cap_reached(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "10.0")
    client = MagicMock()
    pg = _judge_pg([_row()], spend_today=10.0)
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0
    client.call_llm.assert_not_called()


def test_unparseable_verdict_is_skipped_not_raised():
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = "not json"
    pg = _judge_pg([_row()])
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0  # no UPDATE, no exception
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_online_evals.py -v -k judge`
Expected: FAIL (`ImportError: cannot import name 'OnlineJudge'`)

- [ ] **Step 3: Write the implementation**

1. In `src/vibe_trading/agents/prompts.py`, register the fourth prompt (after `JUDGE_SYSTEM`):

```python
ONLINE_JUDGE_SYSTEM = PromptSpec(
    name="online_judge_system",
    version="v1",
    text="""
You are a rigorous production-quality auditor for an LLM trading system. You will
receive one trading DECISION (its action and stated reasoning) and the FEATURE
SNAPSHOT of market data the agents saw when deciding.

Evaluate exactly two properties:
1. grounded — the reasoning cites only facts present in the snapshot. Any invented
   number, indicator reading, or level not in the snapshot means grounded=false.
2. consistent — the action logically follows from the stated reasoning (e.g. a
   bearish, weak-volume rationale does not support a long entry).

Output strictly matches the OnlineJudgeVerdict JSON schema, with a one-sentence
justification.
""".strip(),
)
```

Add it to `REGISTRY`'s tuple. `DECISION_PROMPT_NAMES` stays `("analyst_system", "trader_system")`.

2. Regenerate pins:

Run: `uv run python -m vibe_trading.agents.prompts --write-pins tests/fixtures/prompt_pins.json`

3. In `src/vibe_trading/agents/client.py`, give `call_llm` a call-type tag: change the signature to

```python
    def call_llm(self, model_name, system_instruction, prompt,
                 response_schema: type = None, prompt_version: Optional[str] = None,
                 call_type: str = "single") -> str:
```

and replace the two hardcoded `"single"` literals in its `_defer_cost` / `_emit_cost` calls with `call_type`.

4. Append to `src/vibe_trading/eval/online.py`:

```python
import json
import os

from pydantic import BaseModel

from vibe_trading.agents import prompts
from vibe_trading.agents.cost import daily_summary, should_block_trading

ONLINE_JUDGE_DAILY_CAP = int(os.getenv("ONLINE_JUDGE_DAILY_CAP", "5"))


class OnlineJudgeVerdict(BaseModel):
    grounded: bool      # reasoning cites only facts present in the snapshot
    consistent: bool    # action follows from the stated reasoning
    justification: str


class OnlineJudge:
    """C2: sample scored-but-unjudged decisions and grade them with a generic
    rubric. Capped per UTC day; skipped entirely once the LLM cost cap is hit."""

    def __init__(self, client=None, pg_factory=None, now_fn=None, push_fn=None,
                 daily_cap: Optional[int] = None):
        self._client = client
        self._pg_factory = pg_factory
        self._now = now_fn or (lambda: datetime.utcnow())
        self._push = push_fn or push_langfuse_score
        self.daily_cap = daily_cap if daily_cap is not None else ONLINE_JUDGE_DAILY_CAP

    def _pg(self):
        if self._pg_factory:
            return self._pg_factory()
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase()

    def _get_client(self):
        if self._client is None:
            from vibe_trading.agents.client import LLMClient
            self._client = LLMClient()
        return self._client

    def run_pass(self) -> int:
        """Judge up to (daily_cap - already judged today) decisions. Never raises."""
        try:
            return self._run_pass()
        except Exception as e:
            logger.warning(f"online judge pass failed (non-fatal): {e}")
            return 0

    def _run_pass(self) -> int:
        today_start = self._now().replace(hour=0, minute=0, second=0, microsecond=0)
        pg = self._pg()
        pg.connect()
        try:
            # Cost-cap guard: judging is optional spend; trading's kill switch wins.
            cap = float(os.getenv("LLM_DAILY_COST_CAP_USD", "10.0"))
            if should_block_trading(daily_summary(pg.conn)["today_usd"], cap):
                logger.info("online judge skipped: LLM daily cost cap reached")
                return 0
            judged_today = pg.conn.execute(
                "SELECT COUNT(*) FROM decision_scores WHERE judged_at >= ?",
                (today_start,)).fetchone()[0]
            remaining = self.daily_cap - int(judged_today or 0)
            if remaining <= 0:
                return 0
            rows = pg.conn.execute(
                "SELECT s.decision_id, d.action, d.symbol, d.reasoning_summary, "
                "       d.agent_transcripts, d.trace_id "
                "FROM decision_scores s JOIN decision_log d "
                "  ON d.decision_id = s.decision_id "
                "WHERE s.judge_score IS NULL "
                "ORDER BY s.scored_at DESC LIMIT ?",
                (remaining,)).fetchall()
        finally:
            pg.close()

        judged = 0
        for decision_id, action, symbol, reasoning, transcripts, trace_id in rows:
            verdict = self._judge_one(action, symbol, reasoning, transcripts)
            if verdict is None:
                continue
            score = (int(verdict.grounded) + int(verdict.consistent)) / 2.0
            self._save(decision_id, score, verdict.justification)
            self._push(trace_id, "online_judge_score", score, verdict.justification)
            judged += 1
        return judged

    def _judge_one(self, action, symbol, reasoning,
                   transcripts) -> Optional[OnlineJudgeVerdict]:
        client = self._get_client()
        model = os.getenv("EVAL_JUDGE_MODEL") or client.model
        user_prompt = (
            f"DECISION:\naction: {action}\nsymbol: {symbol}\n"
            f"reasoning_summary: {reasoning}\n\n"
            f"FEATURE SNAPSHOT the agents saw:\n{transcripts}"
        )
        try:
            raw = client.call_llm(
                model_name=model,
                system_instruction=prompts.ONLINE_JUDGE_SYSTEM.text,
                prompt=user_prompt,
                response_schema=OnlineJudgeVerdict,
                prompt_version=prompts.ONLINE_JUDGE_SYSTEM.stamp,
                call_type="online_judge",
            )
            return OnlineJudgeVerdict.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"online judge call/parse failed (non-fatal): {e}")
            return None

    def _save(self, decision_id: str, score: float, note: str) -> None:
        pg = self._pg()
        pg.connect()
        try:
            pg.conn.execute(
                "UPDATE decision_scores SET judge_score = ?, judge_note = ?, "
                "judged_at = ? WHERE decision_id = ?",
                (score, note, self._now(), decision_id))
        finally:
            pg.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_online_evals.py tests/test_prompts.py -v`
Expected: all pass (pins regenerated, so `test_pins_are_current` still holds; `test_registry_contains_the_three_system_prompts` from workstream B must be updated to the four-prompt set — change its expected set to `{"analyst_system", "trader_system", "judge_system", "online_judge_system"}`)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/prompts.py src/vibe_trading/agents/client.py \
        src/vibe_trading/eval/online.py tests/fixtures/prompt_pins.json \
        tests/test_online_evals.py tests/test_prompts.py
git commit -m "feat(online-evals): sampled generic-rubric judge, capped + cost-guarded"
```

---

### Task 5: Weekly drift digest over Discord

**Files:**
- Modify: `src/vibe_trading/runtime/monitoring.py` (add `send_discord`)
- Modify: `src/vibe_trading/runtime/scheduler.py` (`_send_discord_alert` delegates)
- Modify: `src/vibe_trading/eval/online.py`
- Test: `tests/test_online_evals.py`, `tests/test_monitoring.py` (extend)

**Interfaces:**
- Consumes: Task 1's table; `llm_cost_log.schema_ok`; `decision_log.prompt_version` (workstream B).
- Produces: `monitoring.send_discord(message: str) -> None` (no-op without `DISCORD_WEBHOOK_URL`); `weekly_digest(conn, now=None) -> dict` (keys: `week_outcome_mean`, `trailing_outcome_mean`, `week_judge_mean`, `week_schema_compliance`, `week_scored`, `by_prompt_version: list[dict]`); `format_digest(d: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_monitoring.py`:

```python
def test_send_discord_posts_when_configured(monkeypatch):
    import urllib.request
    from vibe_trading.runtime import monitoring
    sent = {}
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def fake_urlopen(req):
        sent["url"] = req.full_url
        sent["data"] = req.data
        return FakeResp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monitoring.send_discord("hello")
    assert sent["url"] == "https://discord.test/hook"
    assert b"hello" in sent["data"]


def test_send_discord_noop_without_webhook(monkeypatch):
    from vibe_trading.runtime import monitoring
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monitoring.send_discord("hello")  # must not raise
```

Append to `tests/test_online_evals.py`:

```python
from vibe_trading.eval.online import weekly_digest, format_digest


def test_weekly_digest_aggregates_and_formats():
    now = datetime(2026, 7, 5)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        (0.62, 0.55, 14),    # week: outcome mean, judge mean, scored count
        (0.58,),             # trailing 4 weeks: outcome mean
        (48, 50),            # week schema: ok_true, ok_total
    ]
    conn.execute.return_value.fetchall.return_value = [
        ("analyst_system:v1@a;trader_system:v1@b", 0.62, 14),
    ]
    d = weekly_digest(conn, now=now)
    assert d["week_outcome_mean"] == pytest.approx(0.62)
    assert d["trailing_outcome_mean"] == pytest.approx(0.58)
    assert d["week_schema_compliance"] == pytest.approx(0.96)
    text = format_digest(d)
    assert "0.62" in text and "0.58" in text
    assert "analyst_system:v1@a" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_monitoring.py tests/test_online_evals.py -v -k "discord or digest"`
Expected: FAIL (missing `send_discord` / `weekly_digest`)

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/runtime/monitoring.py`:

```python
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)


def send_discord(message: str) -> None:
    """Post to the configured Discord webhook. No-op when unconfigured; never raises.
    The single Discord implementation — the scheduler delegates here."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url or "your_discord_webhook_url" in webhook_url:
        return
    data = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req):
            pass
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")
```

(Deduplicate imports against what `monitoring.py` already has.) Then in `src/vibe_trading/runtime/scheduler.py`, replace the body of `_send_discord_alert` with:

```python
    def _send_discord_alert(self, message: str):
        """Sends an alert to Discord webhook if configured."""
        monitoring.send_discord(message)
```

(`monitoring` is already imported in `cli.py`; add `from vibe_trading.runtime import monitoring` to the scheduler's imports.)

Append to `src/vibe_trading/eval/online.py`:

```python
def weekly_digest(conn, now: Optional[datetime] = None) -> dict:
    """C3: this week's quality vs the trailing four weeks, plus schema compliance,
    segmented by prompt bundle. `conn` is a connected DB wrapper (caller owns
    connect/close)."""
    now = now or datetime.utcnow()
    week_start = now - timedelta(days=7)
    trailing_start = now - timedelta(days=35)

    week = conn.execute(
        "SELECT AVG(outcome_score), AVG(judge_score), COUNT(*) "
        "FROM decision_scores WHERE scored_at >= ?", (week_start,)).fetchone()
    trailing = conn.execute(
        "SELECT AVG(outcome_score) FROM decision_scores "
        "WHERE scored_at >= ? AND scored_at < ?",
        (trailing_start, week_start)).fetchone()
    schema = conn.execute(
        "SELECT COUNT(CASE WHEN schema_ok IS TRUE THEN 1 END), "
        "       COUNT(CASE WHEN schema_ok IS NOT NULL THEN 1 END) "
        "FROM llm_cost_log WHERE timestamp >= ?", (week_start,)).fetchone()
    by_version = conn.execute(
        "SELECT prompt_version, AVG(outcome_score), COUNT(*) "
        "FROM decision_scores WHERE scored_at >= ? "
        "GROUP BY prompt_version ORDER BY 3 DESC", (week_start,)).fetchall()

    ok, total = int(schema[0] or 0), int(schema[1] or 0)
    return {
        "week_outcome_mean": float(week[0]) if week and week[0] is not None else None,
        "week_judge_mean": float(week[1]) if week and week[1] is not None else None,
        "week_scored": int(week[2] or 0) if week else 0,
        "trailing_outcome_mean": (float(trailing[0])
                                  if trailing and trailing[0] is not None else None),
        "week_schema_compliance": (ok / total) if total else None,
        "by_prompt_version": [
            {"prompt_version": pv, "outcome_mean": float(m), "count": int(c)}
            for (pv, m, c) in by_version
        ],
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def format_digest(d: dict) -> str:
    lines = [
        "📈 **WEEKLY ONLINE-EVAL DIGEST**",
        f"Outcome score (7d): **{_fmt(d['week_outcome_mean'])}** "
        f"vs trailing 4w {_fmt(d['trailing_outcome_mean'])} "
        f"({d['week_scored']} decisions scored)",
        f"Judge score (7d): {_fmt(d['week_judge_mean'])}",
        f"Schema compliance (7d): {_fmt(d['week_schema_compliance'])}",
    ]
    for row in d["by_prompt_version"]:
        lines.append(f"  · `{row['prompt_version']}`: "
                     f"{row['outcome_mean']:.2f} (n={row['count']})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_monitoring.py tests/test_online_evals.py tests/test_scheduler.py -v`
Expected: all pass (scheduler delegation covered by existing Discord tests, if any assert on `urllib`)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/monitoring.py src/vibe_trading/runtime/scheduler.py \
        src/vibe_trading/eval/online.py tests/test_monitoring.py tests/test_online_evals.py
git commit -m "feat(online-evals): weekly drift digest + shared send_discord"
```

---

### Task 6: CLI entry point and trade-once wiring

**Files:**
- Modify: `src/vibe_trading/eval/online.py` (add `run_scoring_pass` + `main`)
- Modify: `src/vibe_trading/cli.py` (`execute_trade_once`)
- Modify: `README.md`
- Test: `tests/test_online_evals.py`, `tests/test_cli_trade_once.py` (extend)

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: `run_scoring_pass(judge: bool = True) -> dict` (`{"outcomes_scored": int, "judged": int}`); `main(argv=None) -> int` supporting `--digest`, `--no-judge`, `--judge-cap N`; `python -m vibe_trading.eval.online` runnable; `execute_trade_once` invokes a best-effort pass after evaluation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_online_evals.py`:

```python
from vibe_trading.eval import online


def test_run_scoring_pass_composes_scorer_and_judge(monkeypatch):
    monkeypatch.setattr(online.OutcomeScorer, "run_pass", lambda self, batch_limit=200: 3)
    monkeypatch.setattr(online.OnlineJudge, "run_pass", lambda self: 2)
    monkeypatch.setattr(online, "PostgresCostLogger", lambda: None, raising=False)
    from vibe_trading.agents.client import LLMClient
    monkeypatch.setattr(LLMClient, "set_cost_sink", classmethod(lambda cls, s: None))
    result = online.run_scoring_pass()
    assert result == {"outcomes_scored": 3, "judged": 2}
    assert online.run_scoring_pass(judge=False) == {"outcomes_scored": 3, "judged": 0}
```

Append to `tests/test_cli_trade_once.py` (follow its existing mocking style — `TradingScheduler`, `state_sync`, `monitoring` are already patched there):

```python
def test_trade_once_runs_online_scoring_best_effort(monkeypatch, trade_once_mocks):
    """execute_trade_once triggers a scoring pass after evaluation, and a scoring
    crash does not break the trade window."""
    import vibe_trading.cli as cli
    calls = []
    def boom():
        calls.append(1)
        raise RuntimeError("scoring down")
    monkeypatch.setattr("vibe_trading.eval.online.run_scoring_pass", boom)
    cli.execute_trade_once(None)   # must not raise
    assert calls, "scoring pass was never attempted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_online_evals.py tests/test_cli_trade_once.py -v -k "scoring"`
Expected: FAIL (`run_scoring_pass` missing / not invoked)

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/eval/online.py`:

```python
def run_scoring_pass(judge: bool = True) -> dict:
    """One full online-eval pass: outcome scoring, then (optionally) the sampled
    judge. Installs the Postgres cost sink so judge calls are metered like every
    other LLM call. Never raises."""
    from vibe_trading.agents.client import LLMClient
    from vibe_trading.agents.cost import PostgresCostLogger
    try:
        LLMClient.set_cost_sink(PostgresCostLogger())
    except Exception as e:
        logger.warning(f"cost sink unavailable for scoring pass (non-fatal): {e}")
    outcomes = OutcomeScorer().run_pass()
    judged = OnlineJudge().run_pass() if judge else 0
    logger.info(f"online-eval pass: {outcomes} outcomes scored, {judged} judged")
    return {"outcomes_scored": outcomes, "judged": judged}


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="vibe-online-eval")
    parser.add_argument("--digest", action="store_true",
                        help="Also send the weekly drift digest to Discord.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Outcome scoring only (skip the LLM judge).")
    parser.add_argument("--judge-cap", type=int, default=None,
                        help="Override ONLINE_JUDGE_DAILY_CAP for this run.")
    args = parser.parse_args(argv)

    if args.judge_cap is not None:
        os.environ["ONLINE_JUDGE_DAILY_CAP"] = str(args.judge_cap)
        global ONLINE_JUDGE_DAILY_CAP
        ONLINE_JUDGE_DAILY_CAP = args.judge_cap

    result = run_scoring_pass(judge=not args.no_judge)
    print(f"outcomes scored: {result['outcomes_scored']}, "
          f"judged: {result['judged']}")

    if args.digest:
        from vibe_trading.data.db import PostgresDatabase
        from vibe_trading.runtime import monitoring
        try:
            pg = PostgresDatabase()
            pg.connect()
            try:
                digest = weekly_digest(pg.conn)
            finally:
                pg.close()
            monitoring.send_discord(format_digest(digest))
            print("digest sent")
        except Exception as e:
            logger.warning(f"digest failed (non-fatal): {e}")

    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

In `src/vibe_trading/cli.py`, extend `execute_trade_once` — after `scheduler.sync_and_evaluate()` and before `monitoring.ping_healthcheck(success=True)`:

```python
        # Online evals (best-effort): score decisions whose outcomes just became
        # knowable. A scoring failure must never fail the trade window.
        try:
            from vibe_trading.eval.online import run_scoring_pass
            run_scoring_pass()
        except Exception as e:
            logger.warning(f"online scoring pass failed (non-fatal): {e}")
```

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run pytest tests/test_online_evals.py tests/test_cli_trade_once.py -v` — expected: all pass.
Run: `uv run pytest` — expected: all pass.

- [ ] **Step 5: Add the README subsection**

```markdown
### Online evaluation

Live decisions are scored once their outcome is knowable — realized PnL for
decisions that became trades, counterfactual forward return (24h) for flat or
risk-rejected ones — and a capped daily sample gets a generic-rubric LLM judge
(groundedness + consistency vs the stored feature snapshot). Scores land in the
`decision_scores` table and on each decision's Langfuse trace
(`outcome_score`, `online_judge_score`).

    python -m vibe_trading.eval.online              # score + judge (cap: ONLINE_JUDGE_DAILY_CAP, default 5)
    python -m vibe_trading.eval.online --no-judge   # deterministic scoring only
    python -m vibe_trading.eval.online --digest     # + weekly drift digest to Discord (cron weekly)

`trade-once` runs a best-effort scoring pass automatically after each window.
Judge calls are tagged `call_type="online_judge"` in `llm_cost_log` and respect
`LLM_DAILY_COST_CAP_USD`.
```

- [ ] **Step 6: Manual verification (requires prod env)**

1. `uv run python -m vibe_trading.eval.online --no-judge` against the prod `.env` — expect `outcomes scored: N` > 0 (there are months of unscored decisions), rows in `decision_scores`, and `outcome_score` visible on the corresponding Langfuse traces. **Verify the Langfuse 4.x score API name here** — if `get_client().create_score(...)` differs in the installed SDK, adjust `push_langfuse_score` (single call site).
2. `uv run python -m vibe_trading.eval.online --judge-cap 2` — expect 2 judged rows with notes, `online_judge` rows in `llm_cost_log`.
3. `uv run python -m vibe_trading.eval.online --digest` — expect the Discord digest message.
4. Add the weekly digest cron to the Oracle VM (documented in the deploy docs): `0 6 * * 1  cd /opt/vibe-trading && uv run python -m vibe_trading.eval.online --digest`.

- [ ] **Step 7: Commit**

```bash
git add src/vibe_trading/eval/online.py src/vibe_trading/cli.py README.md \
        tests/test_online_evals.py tests/test_cli_trade_once.py
git commit -m "feat(online-evals): CLI entry + best-effort pass in trade-once"
```

---

## Self-review notes

- Spec coverage: C1 deterministic outcome scoring reusing the journal's horizon + trade join (Tasks 2–3), Langfuse push via stored `trace_id` (Task 3), idempotency via PK + `INSERT OR IGNORE` (Tasks 1, 3), C2 generic rubric with `EVAL_JUDGE_MODEL`, sample cap, `call_type="online_judge"`, cost-cap respect (Task 4), C3 digest with prompt_version segmentation over the existing webhook (Task 5), never-block invariant everywhere (`run_pass` wrappers, cli try/except).
- Deviation from spec, deliberate: `judged_at` column added beyond the spec's sketch — the daily judge cap needs a judge-time timestamp, `scored_at` records outcome-scoring time.
- Type consistency: `push_langfuse_score(trace_id, name, value, comment)` defined once (Task 3), reused by the judge (Task 4); score names `outcome_score`/`online_judge_score` used consistently; `run_scoring_pass` consumed by both CLI and `trade-once`.
- Workstream D hook: D adds `precedents_k` to `decision_log` and `decision_scores` and extends `_persist` — the INSERT here lists its columns explicitly, so D's added column defaults to NULL until D lands.
