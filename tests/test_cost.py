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
