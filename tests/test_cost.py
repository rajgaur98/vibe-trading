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
    # scalar_row columns (positional):
    # cost_usd, calls, total_tokens, prompt_tokens, cache_read_tokens, schema_ok_true, schema_ok_total
    conn = _FakeCursor(
        scalar_row=(0.0123, 47, 91234, 80000, 0, 47, 47),
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
    conn = _FakeCursor(scalar_row=(None, 0, None, None, None, 0, 0), model_rows=[])
    s = daily_summary(conn)
    assert s["today_usd"] == 0.0
    assert s["calls"] == 0
    assert s["tokens"] == 0
    assert s["avg_cost_per_call"] == 0.0
    assert s["projected_monthly_usd"] == 0.0
    assert s["by_model"] == []
    # No schema-evaluated calls -> compliance rate defaults to 1.0; no prompt tokens -> cache rate 0.0
    assert s["schema_compliance_rate"] == 1.0
    assert s["cache_hit_rate"] == 0.0


def test_daily_summary_schema_compliance_rate():
    """compliance = schema_ok_true / schema_ok_total (rows where schema_ok IS NOT NULL)."""
    # 8 of 10 schema-evaluated calls were compliant.
    conn = _FakeCursor(
        scalar_row=(0.05, 12, 50000, 40000, 0, 8, 10),
        model_rows=[],
    )
    s = daily_summary(conn)
    assert abs(s["schema_compliance_rate"] - 0.8) < 1e-9


def test_daily_summary_schema_compliance_rate_one_when_no_evaluated_calls():
    """Denominator 0 (no schema_ok values) -> 1.0 (don't penalize unstructured-only days)."""
    conn = _FakeCursor(
        scalar_row=(0.05, 12, 50000, 40000, 0, 0, 0),
        model_rows=[],
    )
    s = daily_summary(conn)
    assert s["schema_compliance_rate"] == 1.0


def test_daily_summary_cache_hit_rate():
    """cache_hit_rate = sum(cache_read_tokens) / sum(prompt_tokens)."""
    conn = _FakeCursor(
        scalar_row=(0.05, 12, 50000, 40000, 10000, 12, 12),
        model_rows=[],
    )
    s = daily_summary(conn)
    assert abs(s["cache_hit_rate"] - (10000 / 40000)) < 1e-9


def test_daily_summary_cache_hit_rate_zero_when_no_prompt_tokens():
    """No prompt tokens -> cache_hit_rate 0.0 (avoids div-by-zero); plumbing still correct."""
    conn = _FakeCursor(
        scalar_row=(0.0, 0, 0, 0, 0, 0, 0),
        model_rows=[],
    )
    s = daily_summary(conn)
    assert s["cache_hit_rate"] == 0.0


def test_postgres_cost_logger_record_is_best_effort():
    """A failing DB must NOT raise out of record() — cost logging can't break a trade."""
    boom_db = MagicMock()
    boom_db.connect.side_effect = RuntimeError("db down")
    logger_ = PostgresCostLogger(db=boom_db)
    ev = CostEvent.build(provider="g", model="m", call_type="single",
                        prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    logger_.record(ev)  # must not raise


from vibe_trading.agents.cost import should_alarm


def test_should_alarm_fires_over_threshold():
    assert should_alarm(today_usd=6.0, threshold=5.0, already_alarmed_today=False) is True


def test_should_alarm_silent_under_threshold():
    assert should_alarm(today_usd=2.0, threshold=5.0, already_alarmed_today=False) is False


def test_should_alarm_dedups_within_day():
    assert should_alarm(today_usd=6.0, threshold=5.0, already_alarmed_today=True) is False


from vibe_trading.agents.cost import should_block_trading


def test_should_block_trading_over_cap():
    assert should_block_trading(today_usd=10.5, cap_usd=10.0) is True


def test_should_block_trading_at_cap_blocks():
    assert should_block_trading(today_usd=10.0, cap_usd=10.0) is True


def test_should_block_trading_under_cap():
    assert should_block_trading(today_usd=4.0, cap_usd=10.0) is False


def test_should_block_trading_disabled_when_cap_zero():
    # cap <= 0 disables the kill switch entirely (never blocks)
    assert should_block_trading(today_usd=999.0, cap_usd=0.0) is False
