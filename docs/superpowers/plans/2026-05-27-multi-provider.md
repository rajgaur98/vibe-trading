# Multi-Provider LLM Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the LLM client layer in vibe-trading to be model- and provider-agnostic, supporting Gemini, OpenAI, Anthropic, and Ollama using LiteLLM, with API key validation limited to the active provider.

**Architecture:** Replace the custom `GeminiClient` in `client.py` with `LLMClient` built on `litellm`. The client resolves model strings at runtime (e.g. mapping `gpt-4o-mini` to `openai/gpt-4o-mini`) and checks only the active provider's API key. Update downstream classes `TechnicalVolumeAnalyst` and `HeadTrader` to call this unified interface.

**Tech Stack:** python (>=3.12), litellm (>=1.40.0), tenacity, pytest

---

### Task 1: Add litellm Dependency

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: Write the failing test**
  
  Since this is a package installation step, we verify that `litellm` is importable.
  Create a temporary script `scratch/test_import.py` to check for `litellm` import failure.
  ```python
  # /Users/raj/vibe-trading/scratch/test_import.py
  import litellm
  print("Import success")
  ```

- [x] **Step 2: Run test to verify it fails**
  
  Run: `uv run python scratch/test_import.py`
  Expected: Failure with `ModuleNotFoundError: No module named 'litellm'`

- [x] **Step 3: Write minimal implementation**
  
  Modify `pyproject.toml` to add `litellm>=1.40.0` under project dependencies.
  
  Replace in `pyproject.toml` (lines 10-13):
  ```toml
      "duckdb>=0.10.0",
      "google-genai>=0.1.0",
      "quantstats>=0.0.62",
  ```
  With:
  ```toml
      "duckdb>=0.10.0",
      "google-genai>=0.1.0",
      "litellm>=1.40.0",
      "quantstats>=0.0.62",
  ```
  Then run `uv sync` to install dependencies.

- [x] **Step 4: Run test to verify it passes**
  
  Run: `uv run python scratch/test_import.py`
  Expected: Output `Import success`
  Remove the temporary file: `rm scratch/test_import.py`

- [x] **Step 5: Commit**
  
  ```bash
  git add pyproject.toml uv.lock
  git commit -m "chore: add litellm dependency"
  ```

---

### Task 2: Create Multi-Provider Unit Tests

**Files:**
- Create: `tests/test_multi_provider.py`

- [x] **Step 1: Write the failing test**
  
  Create the unit test file containing multi-provider client tests.
  
  ```python
  # /Users/raj/vibe-trading/tests/test_multi_provider.py
  import pytest
  from unittest.mock import patch, MagicMock
  from vibe_trading.agents.client import LLMClient, get_litellm_model_string

  def test_get_litellm_model_string():
      assert get_litellm_model_string("gemini", "gemini-3.1-flash-lite") == "gemini/gemini-3.1-flash-lite"
      assert get_litellm_model_string("openai", "gpt-4o") == "openai/gpt-4o"
      assert get_litellm_model_string("anthropic", "claude-3") == "anthropic/claude-3"
      assert get_litellm_model_string("other", "model") == "model"

  @patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
  def test_llm_client_initialization_gemini():
      client = LLMClient()
      assert client.provider == "gemini"
      assert client.model == "gemini-3.1-flash-lite"

  @patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test_openai_key"})
  def test_llm_client_initialization_openai():
      client = LLMClient()
      assert client.provider == "openai"

  @patch.dict("os.environ", {"LLM_PROVIDER": "openai"})
  @patch.dict("os.environ", {}, clear=True)
  def test_llm_client_initialization_openai_missing_key():
      with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
          LLMClient()

  @patch("litellm.completion")
  @patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
  def test_call_llm(mock_completion):
      mock_response = MagicMock()
      mock_response.choices = [MagicMock()]
      mock_response.choices[0].message.content = '{"result": "success"}'
      mock_completion.return_value = mock_response

      client = LLMClient()
      res = client.call_llm("test-model", "system prompt", "user prompt")
      assert res == '{"result": "success"}'
      mock_completion.assert_called_once()
  ```

- [x] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL with import errors or missing attributes (`LLMClient` not found)

- [x] **Step 3: Write minimal implementation**
  
  Create placeholder or empty stub functions in `src/vibe_trading/agents/client.py` so the tests can run and fail specifically on logic.
  For now, just add functions that raise `NotImplementedError`.

- [x] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL with `NotImplementedError` or assertion errors, indicating tests are loaded.

- [x] **Step 5: Commit**
  
  ```bash
  git add tests/test_multi_provider.py
  git commit -m "test: add multi-provider client tests"
  ```

---

### Task 3: Implement LLMClient

**Files:**
- Modify: `src/vibe_trading/agents/client.py`

- [x] **Step 1: Write the failing test**
  
  We already have the failing test suite in `tests/test_multi_provider.py`. Ensure we are running it.

- [x] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL (assertion or NotImplementedError).

- [x] **Step 3: Write minimal implementation**
  
  Replace the contents of `src/vibe_trading/agents/client.py` with:
  
  ```python
  import os
  import logging
  import litellm
  from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

  logger = logging.getLogger(__name__)

  litellm.telemetry = False

  def get_litellm_model_string(provider: str, model: str) -> str:
      """Converts provider and model parameters to standard LiteLLM model identifiers."""
      provider = provider.lower()
      if provider == "gemini":
          return f"gemini/{model}"
      elif provider == "openai":
          return f"openai/{model}"
      elif provider == "anthropic":
          return f"anthropic/{model}"
      elif provider == "ollama":
          return f"ollama/{model}"
      return model

  class LLMClient:
      def __init__(self):
          self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
          self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
          
          # Dynamic key validation for active provider only
          if self.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
              raise ValueError("GEMINI_API_KEY environment variable is not set. Please check your .env file.")
          elif self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
              raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
          elif self.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
              raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please check your .env file.")

      @retry(
          stop=stop_after_attempt(3),
          wait=wait_exponential(multiplier=1, min=2, max=10),
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

- [x] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: PASS

- [x] **Step 5: Commit**
  
  ```bash
  git add src/vibe_trading/agents/client.py
  git commit -m "feat: implement provider-agnostic LLMClient"
  ```

---

### Task 4: Integrate LLMClient into TechnicalVolumeAnalyst

**Files:**
- Modify: `src/vibe_trading/agents/analyst.py`

- [x] **Step 1: Write the failing test**
  
  Open `tests/test_multi_provider.py` and add a new test that instantiates `TechnicalVolumeAnalyst` and calls `.analyze()`, mocking the `LLMClient.call_llm` call.
  
  Add to `tests/test_multi_provider.py`:
  ```python
  from vibe_trading.agents.analyst import TechnicalVolumeAnalyst, AnalystOutput

  @patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
  def test_analyst_integration():
      mock_client = MagicMock()
      # Mock the raw json response matching AnalystOutput schema
      mock_client.call_llm.return_value = '{"market_bias": "bullish", "volume_confirmation": "confirmed", "thesis": "Strong breakout on high volume.", "nearest_support": 95.0, "nearest_resistance": 105.0, "confluence_score": 0.8}'
      
      analyst = TechnicalVolumeAnalyst(client=mock_client)
      snapshot = {"symbol": "BTC/USDT"}
      res = analyst.analyze(snapshot)
      
      assert isinstance(res, AnalystOutput)
      assert res.market_bias == "bullish"
      mock_client.call_llm.assert_called_once()
  ```

- [x] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL with `ImportError: cannot import name 'GeminiClient' from 'vibe_trading.agents.client'` (or similar in `analyst.py` imports)

- [x] **Step 3: Write minimal implementation**
  
  Modify `src/vibe_trading/agents/analyst.py` to use `LLMClient` instead of `GeminiClient`.
  
  Change imports:
  ```python
  from vibe_trading.agents.client import LLMClient
  ```
  
  Change class init:
  ```python
  class TechnicalVolumeAnalyst:
      def __init__(self, client: LLMClient = None):
          self.client = client or LLMClient()
          self.model = os.getenv("GEMINI_ANALYST_MODEL") or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
  ```
  
  Change execution call (under `analyze()` method):
  ```python
              raw_output = self.client.call_llm(
                  model_name=self.model,
                  system_instruction=self.system_instruction,
                  prompt=prompt,
                  response_schema=AnalystOutput
              )
  ```

- [x] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: PASS

- [x] **Step 5: Commit**
  
  ```bash
  git add src/vibe_trading/agents/analyst.py tests/test_multi_provider.py
  git commit -m "refactor: integrate LLMClient into TechnicalVolumeAnalyst"
  ```

---

### Task 5: Integrate LLMClient into HeadTrader

**Files:**
- Modify: `src/vibe_trading/agents/trader.py`

- [x] **Step 1: Write the failing test**
  
  Open `tests/test_multi_provider.py` and add a new test that instantiates `HeadTrader` and calls `.decide()`, mocking the `LLMClient.call_llm` call.
  
  Add to `tests/test_multi_provider.py`:
  ```python
  from vibe_trading.agents.trader import HeadTrader, HeadTraderOutput

  @patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_gemini_key"})
  def test_trader_integration():
      mock_client = MagicMock()
      mock_client.call_llm.return_value = '{"action": "long", "stop_loss_strategy": "1.5_atr", "take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, "hold_period_bias": "medium", "reasoning_summary": "Strong trend confirmation."}'
      
      trader = HeadTrader(client=mock_client)
      analyst_res = AnalystOutput(
          market_bias="bullish",
          volume_confirmation="confirmed",
          thesis="Strong breakout",
          nearest_support=95.0,
          nearest_resistance=105.0,
          confluence_score=0.8
      )
      
      proposal = trader.decide("BTC/USDT", analyst_res, {}, [])
      assert proposal["action"] == "long"
      mock_client.call_llm.assert_called_once()
  ```

- [x] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL with `ImportError: cannot import name 'GeminiClient' from 'vibe_trading.agents.client'` (or similar in `trader.py` imports)

- [x] **Step 3: Write minimal implementation**
  
  Modify `src/vibe_trading/agents/trader.py` to use `LLMClient` instead of `GeminiClient`.
  
  Change imports:
  ```python
  from vibe_trading.agents.client import LLMClient
  ```
  
  Change class init:
  ```python
  class HeadTrader:
      def __init__(self, client: LLMClient = None):
          self.client = client or LLMClient()
          self.model = os.getenv("GEMINI_TRADER_MODEL") or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
  ```
  
  Change execution call (under `decide()` method):
  ```python
              raw_output = self.client.call_llm(
                  model_name=self.model,
                  system_instruction=self.system_instruction,
                  prompt=prompt,
                  response_schema=HeadTraderOutput
              )
  ```

- [x] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: PASS

- [x] **Step 5: Commit**
  
  ```bash
  git add src/vibe_trading/agents/trader.py tests/test_multi_provider.py
  git commit -m "refactor: integrate LLMClient into HeadTrader"
  ```

---

### Task 6: Configure Environment Variables

**Files:**
- Modify: `.env`
- Modify: `.env.example`

- [ ] **Step 1: Write the failing test**
  
  We verify that when we initialize `LLMClient()` without setting `LLM_PROVIDER` in `.env`, it defaults correctly to gemini and raises the appropriate validation error if `GEMINI_API_KEY` is missing.
  We will also make sure the env templates contain the correct parameters. Let's write a small verification step.
  ```python
  # In tests/test_multi_provider.py:
  @patch.dict("os.environ", {}, clear=True)
  def test_default_env_load():
      # If no env vars are defined, initializing LLMClient should raise ValueError about GEMINI_API_KEY
      with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
          LLMClient()
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_multi_provider.py`
  Expected: FAIL if the test is not defined, or if `LLMClient` does not behave as expected.

- [ ] **Step 3: Write minimal implementation**
  
  Add the `test_default_env_load` test function to `tests/test_multi_provider.py`.
  
  Update `.env.example` to append the new variables:
  ```env
  # Provider-Agnostic LLM Configuration
  LLM_PROVIDER=gemini
  LLM_MODEL=gemini-3.1-flash-lite
  OPENAI_API_KEY=your_openai_key_here
  ANTHROPIC_API_KEY=your_anthropic_key_here
  ```
  
  Update `.env` to make sure it contains the active provider:
  ```env
  # Provider-Agnostic LLM Configuration
  LLM_PROVIDER=gemini
  LLM_MODEL=gemini-3.1-flash-lite
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest` (runs all tests in the suite)
  Expected: PASS

- [ ] **Step 5: Commit**
  
  ```bash
  git add .env.example tests/test_multi_provider.py
  git commit -m "config: update environment configuration templates"
  ```
