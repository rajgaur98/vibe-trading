"""Guardrail test for the LLM client: a hard max-output-size ceiling is passed to the
provider on every call (bounds runaway cost/latency)."""
from unittest.mock import MagicMock, patch

from vibe_trading.agents.client import LLMClient


def _fake_response():
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = '{"ok": true}'
    r.choices[0].message.tool_calls = None
    r.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return r


def test_call_llm_passes_max_output_tokens(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1234")
    client = LLMClient()
    with patch("vibe_trading.agents.client.litellm.completion", return_value=_fake_response()) as comp:
        client.call_llm("some-model", "system", "prompt")
    assert comp.call_args.kwargs["max_tokens"] == 1234


def test_max_output_tokens_zero_disables_cap(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "0")
    client = LLMClient()
    with patch("vibe_trading.agents.client.litellm.completion", return_value=_fake_response()) as comp:
        client.call_llm("some-model", "system", "prompt")
    assert "max_tokens" not in comp.call_args.kwargs  # <=0 => no ceiling sent


def test_call_llm_threads_prompt_version_into_cost_event(monkeypatch):
    import litellm
    from vibe_trading.agents.client import LLMClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = []

    class Sink:
        def record(self, event):
            captured.append(event)

    class FakeMsg:
        content = "{}"
    class FakeChoice:
        message = FakeMsg()
    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    monkeypatch.setattr(litellm, "completion", lambda **kw: FakeResponse())
    LLMClient.set_cost_sink(Sink())
    try:
        client = LLMClient()
        client.call_llm(model_name="m", system_instruction="s", prompt="p",
                        prompt_version="analyst_system:v1@abcdef123456")
    finally:
        LLMClient.set_cost_sink(None)
    assert captured[0].prompt_version == "analyst_system:v1@abcdef123456"
