import logging
from datetime import datetime, timezone
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
