# Spec: Multi-Provider LLM Abstraction with LiteLLM

**Date:** 2026-05-27  
**Status:** Draft  
**Topic:** Transitioning from `GeminiClient` to a model- and provider-agnostic `LLMClient` using LiteLLM, while optimizing API key validation to only check the active provider on startup.

---

## 1. Objectives

- **Vendor Agnosticism:** Enable plug-and-play support for major LLM providers (Gemini, OpenAI, Anthropic, Ollama, etc.) without vendor lock-in.
- **Active Key Validation:** On startup, only validate and require the API key for the active `LLM_PROVIDER` configured in the `.env` file, rather than checking for all keys.
- **Structured Outputs Compatibility:** Maintain reliable Pydantic structured output mapping for Analyst and Trader components across different providers.
- **OTel & Observability:** Keep the OpenTelemetry instrumentation and Langfuse tracking operational.

---

## 2. Architecture & Design

### Configuration Variables
We will introduce two primary environment variables to control provider selection:
- `LLM_PROVIDER`: Controls the active LLM provider (options: `gemini`, `openai`, `anthropic`, `ollama`). Defaults to `gemini`.
- `LLM_MODEL`: Controls the default model to use when individual agent models are not explicitly set. Defaults to `gemini-3.1-flash-lite`.

### Unified `LLMClient`
The client class `LLMClient` (replacing [GeminiClient](file:///Users/raj/vibe-trading/src/vibe_trading/agents/client.py)) will wrap LiteLLM calls.

```python
class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        self._validate_active_provider_key()
```

#### API Key Validation Logic
On instantiation, the client validates the API key for the active provider:
- If `LLM_PROVIDER == "gemini"`: Check for `GEMINI_API_KEY` (or fallback `GOOGLE_API_KEY`). Raise `ValueError` if missing.
- If `LLM_PROVIDER == "openai"`: Check for `OPENAI_API_KEY`. Raise `ValueError` if missing.
- If `LLM_PROVIDER == "anthropic"`: Check for `ANTHROPIC_API_KEY`. Raise `ValueError` if missing.
- For other providers (e.g. `ollama`): Do not require an API key by default.

### Model Routing & Mapping
LiteLLM requires model names to follow a specific prefix syntax. We will map provider-specific models to LiteLLM paths using a helper function:
```python
def get_litellm_model_string(provider: str, model_name: str) -> str:
    provider = provider.lower()
    if provider == "gemini":
        return f"gemini/{model_name}"
    elif provider == "openai":
        return f"openai/{model_name}"
    elif provider == "anthropic":
        return f"anthropic/{model_name}"
    elif provider == "ollama":
        return f"ollama/{model_name}"
    return model_name
```

### Call Interface
The client will support system prompts, user prompts, and Pydantic schemas:
```python
def call_llm(
    self,
    model_name: str,
    system_instruction: str,
    prompt: str,
    response_schema: type[BaseModel] = None
) -> str:
    model_str = get_litellm_model_string(self.provider, model_name)
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
        
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content
```

---

## 3. Impacted Components

### 1. [client.py](file:///Users/raj/vibe-trading/src/vibe_trading/agents/client.py)
- **Change:** Replace `GeminiClient` with `LLMClient`.
- **Dependency:** Add `litellm`. Remove unused Google GenAI imports if not needed, or keep them optional.

### 2. [analyst.py](file:///Users/raj/vibe-trading/src/vibe_trading/agents/analyst.py)
- **Change:** Update `TechnicalVolumeAnalyst` constructor and references to use `LLMClient`. Change `.call_gemini()` to `.call_llm()`.

### 3. [trader.py](file:///Users/raj/vibe-trading/src/vibe_trading/agents/trader.py)
- **Change:** Update `HeadTrader` constructor and references to use `LLMClient`. Change `.call_gemini()` to `.call_llm()`.

---

## 4. Verification & Testing Plan

### Automated Unit Tests
A new test suite [test_multi_provider.py](file:///Users/raj/vibe-trading/tests/test_multi_provider.py) will cover:
1. **API Key Validation:** Assert that only the active provider's API key is validated on startup.
2. **Model Formatting:** Verify model strings are formatted correctly for LiteLLM.
3. **Mock Completion:** Call `call_llm` using a mocked `litellm.completion` to verify options are passed properly.

Run command:
```bash
uv run pytest tests/test_multi_provider.py
```
