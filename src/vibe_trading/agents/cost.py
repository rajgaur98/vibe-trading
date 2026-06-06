import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import litellm
from pydantic import BaseModel

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


def _utcnow_naive() -> datetime:
    """Naive UTC timestamp, matching the trades/decision_log storage convention so the
    day-boundary query in daily_summary compares consistently."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CostEvent(BaseModel):
    call_id: str
    timestamp: datetime
    provider: str
    model: str          # litellm-format model string
    call_type: str      # "single" | "tool_loop"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    # Prompt/context-cache token accounting (Feature 2). Defaults to 0 when the
    # provider does not report cache usage (e.g. free-tier Gemma always reports 0,
    # so cache_hit_rate stays 0 there — the plumbing is correct regardless).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Schema-compliance outcome for THIS call (Feature 1). True/False once a
    # structured response has been validated; None for calls with no schema to
    # validate (e.g. intermediate tool-call turns), so they don't skew the rate.
    schema_ok: Optional[bool] = None

    @classmethod
    def build(cls, *, provider: str, model: str, call_type: str,
              prompt_tokens: int, completion_tokens: int, latency_ms: float,
              cache_read_tokens: int = 0, cache_write_tokens: int = 0,
              schema_ok: Optional[bool] = None) -> "CostEvent":
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
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            schema_ok=schema_ok,
        )


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
                    prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms,
                    cache_read_tokens, cache_write_tokens, schema_ok)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.call_id, event.timestamp, event.provider, event.model, event.call_type,
                 event.prompt_tokens, event.completion_tokens, event.total_tokens,
                 event.cost_usd, event.latency_ms,
                 event.cache_read_tokens, event.cache_write_tokens, event.schema_ok),
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
    # Single scalar query carries the schema-compliance and cache-hit numerators/
    # denominators alongside the cost totals (positional columns 0..6).
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*), COALESCE(SUM(total_tokens), 0), "
        "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(cache_read_tokens), 0), "
        "COUNT(CASE WHEN schema_ok IS TRUE THEN 1 END), "
        "COUNT(CASE WHEN schema_ok IS NOT NULL THEN 1 END) "
        "FROM llm_cost_log WHERE timestamp >= ?",
        (today_start,),
    ).fetchone()
    today_usd = float(row[0] or 0.0)
    calls = int(row[1] or 0)
    tokens = int(row[2] or 0)
    prompt_tokens = int(row[3] or 0)
    cache_read_tokens = int(row[4] or 0)
    schema_ok_true = int(row[5] or 0)
    schema_ok_total = int(row[6] or 0)
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
        # Fraction of schema-evaluated calls that validated. 1.0 when nothing was
        # schema-evaluated (denominator 0) so unstructured-only days aren't penalized.
        "schema_compliance_rate": (schema_ok_true / schema_ok_total) if schema_ok_total else 1.0,
        # Fraction of prompt tokens served from the provider's prompt cache. 0.0 when
        # there are no prompt tokens. NOTE: free-tier Gemma reports 0 cached tokens, so
        # this reads 0.0 until run on a cache-capable provider (e.g. Anthropic) — the
        # capture/aggregation plumbing is correct regardless of provider.
        "cache_hit_rate": (cache_read_tokens / prompt_tokens) if prompt_tokens else 0.0,
        "by_model": [{"model": m, "calls": int(c), "cost_usd": float(u)} for (m, c, u) in model_rows],
    }


def should_alarm(today_usd: float, threshold: float, already_alarmed_today: bool) -> bool:
    """True when today's spend exceeds the threshold and we haven't already alarmed today."""
    return today_usd > threshold and not already_alarmed_today


def should_block_trading(today_usd: float, cap_usd: float) -> bool:
    """Named safety control: True when today's LLM spend has reached the hard daily cap,
    meaning new-entry evaluation should be blocked. `cap_usd <= 0` disables the cap
    (never blocks)."""
    if cap_usd <= 0:
        return False
    return today_usd >= cap_usd
