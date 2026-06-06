import json
import os
import time
import threading
import logging
from typing import Optional

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from vibe_trading.agents.cost import CostEvent

logger = logging.getLogger(__name__)

litellm.telemetry = False


class SchemaValidationError(Exception):
    """Raised when a structured LLM response cannot be parsed/validated into the
    required Pydantic schema even after one corrective retry. A clean, typed error
    so call sites never have to catch a bare KeyError / json.JSONDecodeError /
    pydantic.ValidationError leaking out of the parse path."""


# Appended to the user prompt on the single corrective retry when the first
# structured response failed to parse/validate.
SCHEMA_CORRECTIVE_INSTRUCTION = (
    "\n\nYour previous response did not match the required schema; "
    "return ONLY valid JSON matching it (no prose, no markdown fences)."
)

# Providers whose APIs honor Anthropic-style `cache_control: {"type": "ephemeral"}`
# markers on message content blocks. LiteLLM passes the markers through for these;
# other providers (Gemini/Gemma, Groq, Ollama) ignore the structured content and
# would error on the block list, so we only emit blocks for these providers and
# send a plain string otherwise. The mechanism + metric are correct regardless;
# the marker is simply a no-op everywhere except a cache-capable provider.
_CACHE_CONTROL_PROVIDERS = {"anthropic"}

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

    @staticmethod
    def _read_cache_tokens(usage) -> tuple[int, int]:
        """Extract (cache_read, cache_write) tokens from a LiteLLM usage object.

        LiteLLM normalizes cache accounting differently across providers, so we
        check the known shapes and take the first present:
          - read:  usage.prompt_tokens_details.cached_tokens  (OpenAI/Gemini shape,
                   normalized by LiteLLM) OR usage.cache_read_input_tokens (Anthropic).
          - write: usage.cache_creation_input_tokens (Anthropic prompt-cache writes).
        Defaults to 0 when the provider reports nothing (e.g. free-tier Gemma)."""
        if usage is None:
            return 0, 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        read = cached if cached is not None else getattr(usage, "cache_read_input_tokens", 0)
        write = getattr(usage, "cache_creation_input_tokens", 0)
        return int(read or 0), int(write or 0)

    def _build_cost_event(self, response, model_str: str, call_type: str,
                          latency_ms: float, schema_ok: Optional[bool] = None) -> Optional[CostEvent]:
        """Build a CostEvent from a LiteLLM response. Returns None (and never raises)
        if usage extraction fails — cost logging must not break an LLM call."""
        try:
            usage = getattr(response, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            cache_read, cache_write = self._read_cache_tokens(usage)
            return CostEvent.build(
                provider=self.provider, model=model_str, call_type=call_type,
                prompt_tokens=pt, completion_tokens=ct, latency_ms=latency_ms,
                cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                schema_ok=schema_ok,
            )
        except Exception as e:
            logger.warning(f"cost event build failed (non-fatal): {e}")
            return None

    def _record_cost(self, event: Optional[CostEvent]) -> None:
        """The single cost sink. Every CostEvent reaches the sink through here, so
        schema_ok / cache tokens are attached before this one write. Never raises."""
        sink = LLMClient._cost_sink
        if sink is None or event is None:
            return
        try:
            sink.record(event)
        except Exception as e:
            logger.warning(f"cost emit failed (non-fatal): {e}")

    def _emit_cost(self, response, model_str: str, call_type: str, latency_ms: float,
                   schema_ok: Optional[bool] = None) -> None:
        """Build + record a cost event immediately. Used by calls that carry no schema
        to validate (so schema_ok stays None unless explicitly supplied)."""
        if LLMClient._cost_sink is None:
            return
        self._record_cost(self._build_cost_event(response, model_str, call_type, latency_ms, schema_ok))

    def _defer_cost(self, response, model_str: str, call_type: str, latency_ms: float) -> None:
        """Build a cost event but hold it instead of recording, so the schema parse
        helper can attach the validation outcome and flush it via mark_schema_outcome().
        Replaces any previously-deferred (un-flushed) event."""
        if LLMClient._cost_sink is None:
            self._deferred_cost_event = None
            return
        self._deferred_cost_event = self._build_cost_event(response, model_str, call_type, latency_ms)

    def mark_schema_outcome(self, schema_ok: bool) -> None:
        """Flush the most-recently deferred cost event with its schema-compliance
        outcome. Called by the parse helper once it knows whether the structured
        response validated. No-op if nothing was deferred (e.g. no sink installed)."""
        event = getattr(self, "_deferred_cost_event", None)
        if event is None:
            return
        event.schema_ok = schema_ok
        self._deferred_cost_event = None
        self._record_cost(event)

    def _system_message(self, system_instruction: str) -> dict:
        """Build the system message, applying Anthropic-style ephemeral cache_control
        to the (large, static) system prompt when the provider supports prompt caching.
        Non-supporting providers get a plain string so nothing changes for them."""
        if self.provider in _CACHE_CONTROL_PROVIDERS:
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_instruction,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": system_instruction}

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

        # Holds a cost event whose schema-compliance outcome is not yet known. The
        # parse helper flushes it via mark_schema_outcome(). Per-instance: the analyst,
        # trader, and judge each own their own LLMClient, so no cross-talk.
        self._deferred_cost_event: Optional[CostEvent] = None

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
            self._system_message(system_instruction),
            {"role": "user", "content": prompt},
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
        latency_ms = (time.monotonic() - _t0) * 1000.0
        if response_schema:
            # Structured call: defer the cost event so the schema parse helper can
            # attach the compliance outcome before it reaches the single sink.
            self._defer_cost(response, model_str, "single", latency_ms)
        else:
            self._emit_cost(response, model_str, "single", latency_ms)
        return response.choices[0].message.content

    def call_llm_with_tools(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        tools: list,
        tool_executor,
        max_iterations: int = 10,
        expect_schema: bool = False,
    ) -> str:
        """Multi-turn agentic loop: LLM proposes tool calls, executor runs them, results fed back.

        Returns the final `assistant.content` string once the model stops emitting tool_calls.
        Raises RuntimeError if `max_iterations` is exhausted with the model still requesting tools.

        `expect_schema`: when True, the FINAL-answer iteration's cost event is deferred so
        the schema parse helper can attach its compliance outcome (via mark_schema_outcome)
        before it reaches the single cost sink. Intermediate tool-call turns are recorded
        immediately with schema_ok=None (they have no schema to validate).
        """
        model_str = get_litellm_model_string(self.provider, model_name)
        messages = [
            self._system_message(system_instruction),
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
            latency_ms = (time.monotonic() - _t0) * 1000.0
            assistant_msg = response.choices[0].message
            if hasattr(assistant_msg, "model_dump"):
                messages.append(assistant_msg.model_dump())
            else:
                messages.append(dict(assistant_msg))

            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                # Final answer: defer its cost so the parse helper can stamp schema_ok.
                if expect_schema:
                    self._defer_cost(response, model_str, "tool_loop", latency_ms)
                else:
                    self._emit_cost(response, model_str, "tool_loop", latency_ms)
                return assistant_msg.content

            # Intermediate tool-call turn: no schema to validate -> record now.
            self._emit_cost(response, model_str, "tool_loop", latency_ms)

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


def validate_structured(client, schema, raw_output, recall, *, extract=None):
    """Parse + validate a structured LLM response into `schema` (a pydantic model),
    with exactly ONE corrective retry.

    Args:
        client:  the LLMClient that produced `raw_output`. Used to (a) stamp the
                 schema-compliance outcome onto the deferred cost event via
                 client.mark_schema_outcome(), keeping LLMClient's cost sink the
                 single place a row is written, and (b) is NOT used to re-call —
                 `recall` owns that.
        schema:  the pydantic BaseModel subclass the output must validate into.
        raw_output: the raw string from the first structured LLM call.
        recall:  callable(corrective_instruction: str) -> str. Re-invokes the LLM
                 with the corrective instruction appended and returns the new raw
                 string. Must itself defer a fresh cost event so this helper can
                 stamp the retry's outcome too.
        extract: optional callable(str) -> str run before json.loads (e.g. strip
                 markdown fences). Defaults to identity.

    Returns: a validated `schema` instance.
    Raises:  SchemaValidationError if both the first response and the corrective
             retry fail to parse/validate — never a bare KeyError / JSONDecodeError /
             pydantic ValidationError.
    """
    extract = extract or (lambda s: s)

    def _try(text: str):
        data = json.loads(extract(text))
        return schema.model_validate(data)

    # First attempt against the original response.
    try:
        model = _try(raw_output)
    except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_err:
        client.mark_schema_outcome(False)
        logger.warning(f"Structured response failed schema validation; retrying once: {first_err}")
        retry_raw = recall(SCHEMA_CORRECTIVE_INSTRUCTION)
        try:
            model = _try(retry_raw)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as second_err:
            client.mark_schema_outcome(False)
            raise SchemaValidationError(
                f"LLM response failed {schema.__name__} validation after one corrective "
                f"retry: {second_err}"
            ) from second_err
        client.mark_schema_outcome(True)
        return model

    client.mark_schema_outcome(True)
    return model
