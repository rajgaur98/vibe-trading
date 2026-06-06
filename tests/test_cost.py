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
