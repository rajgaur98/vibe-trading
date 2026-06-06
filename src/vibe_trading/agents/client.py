import json
import os
import time
import threading
import logging
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from vibe_trading.agents.cost import CostEvent

logger = logging.getLogger(__name__)

litellm.telemetry = False

# Maps LLM_PROVIDER values to the LiteLLM prefix they require. Providers in this
# map get their `model` string namespaced as `<prefix>/<model>`. Unknown provider
# names fall through and pass `model` verbatim to LiteLLM (which may still resolve
# them via its built-in provider auto-detection for prefixed model identifiers).
_LITELLM_PROVIDER_PREFIXES = {
    "gemini": "gemini",
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
    "groq": "groq",
}


def get_litellm_model_string(provider: str, model: str) -> str:
    """Converts provider and model parameters to standard LiteLLM model identifiers."""
    prefix = _LITELLM_PROVIDER_PREFIXES.get(provider.lower())
    if prefix:
        return f"{prefix}/{model}"
    return model


# Providers that require an API key set via environment variable. Validated on
# LLMClient init so misconfiguration fails fast instead of at the first call.
_PROVIDER_API_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}


class LLMClient:
    # Class-level rate-limit gate shared across ALL instances. The analyst, trader,
    # and eval judge each construct their own LLMClient but hit the same provider
    # quota, so the minimum-interval spacing must be global, not per-instance.
    _last_call_at: float = 0.0
    _throttle_lock = threading.Lock()
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

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

        # Minimum seconds between consecutive LLM calls across all clients. Default 0
        # (no throttle — live trading is unaffected). The eval sets this to stay under
        # the provider's RPM limit (e.g. 4.5s ≈ 13 RPM, safely below Gemini's 15 RPM),
        # which avoids tripping rate limits and the long retry backoffs that follow.
        self.min_call_interval = float(os.getenv("LLM_MIN_CALL_INTERVAL_SECONDS", "0"))

        # Dynamic key validation for active provider only
        required_key = _PROVIDER_API_KEY_ENV.get(self.provider)
        if required_key and not os.getenv(required_key):
            raise ValueError(f"{required_key} environment variable is not set. Please check your .env file.")

    def _throttle(self) -> None:
        """Block until at least `min_call_interval` seconds have passed since the last
        call by any client instance. No-op when the interval is 0."""
        if self.min_call_interval <= 0:
            return
        with LLMClient._throttle_lock:
            elapsed = time.monotonic() - LLMClient._last_call_at
            wait = self.min_call_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            LLMClient._last_call_at = time.monotonic()

    @retry(
        # 5 attempts with backoff 4 → 8 → 16 → 32 → 60s ≈ 120s total max wait —
        # enough to outlast a typical 60s rate-limit bucket refill on free tiers
        # (Gemini Flash Lite, Groq Llama 70B) without giving up on transient failures.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM request failed. Retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number})"
        )
    )
    def call_llm(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        response_schema: type = None
    ) -> str:
        """
        Invokes the configured LLM provider via LiteLLM and returns the raw JSON string content.
        """
        model_str = get_litellm_model_string(self.provider, model_name)
        logger.info(f"Calling LLM provider={self.provider} model={model_str}...")
        self._throttle()

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": model_str,
            "messages": messages,
            "temperature": 0.1,
        }
        
        if response_schema:
            kwargs["response_format"] = response_schema
            
        _t0 = time.monotonic()
        response = litellm.completion(**kwargs)
        self._emit_cost(response, model_str, "single", (time.monotonic() - _t0) * 1000.0)
        return response.choices[0].message.content

    def call_llm_with_tools(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        tools: list,
        tool_executor,
        max_iterations: int = 10,
    ) -> str:
        """Multi-turn agentic loop: LLM proposes tool calls, executor runs them, results fed back.

        Returns the final `assistant.content` string once the model stops emitting tool_calls.
        Raises RuntimeError if `max_iterations` is exhausted with the model still requesting tools.
        """
        model_str = get_litellm_model_string(self.provider, model_name)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        for iteration in range(max_iterations):
            logger.info(f"Tool-use loop iteration {iteration + 1}/{max_iterations} (model={model_str})")
            self._throttle()
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
            if hasattr(assistant_msg, "model_dump"):
                messages.append(assistant_msg.model_dump())
            else:
                messages.append(dict(assistant_msg))

            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                return assistant_msg.content

            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    args_result = json.dumps({"error": f"Malformed tool arguments: {e}"})
                else:
                    logger.info(f"Executing tool: {tool_call.function.name}({args})")
                    args_result = tool_executor.execute(tool_call.function.name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", "") or "",
                    "content": args_result,
                })

        raise RuntimeError(f"Agent exceeded max tool-call iterations ({max_iterations})")
