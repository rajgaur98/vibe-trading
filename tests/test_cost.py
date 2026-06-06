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


from unittest.mock import MagicMock
from vibe_trading.agents.cost import daily_summary, PostgresCostLogger


class _FakeCursor:
    """Stand-in for the project's PostgresConnectionWrapper: execute() returns self,
    then fetchone()/fetchall() yield canned results queued by the test."""
    def __init__(self, scalar_row, model_rows):
        self._scalar_row = scalar_row
        self._model_rows = model_rows
    def execute(self, sql, params=None):
        return self
    def fetchone(self):
        return self._scalar_row
    def fetchall(self):
        return self._model_rows


def test_daily_summary_aggregates_and_projects():
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
