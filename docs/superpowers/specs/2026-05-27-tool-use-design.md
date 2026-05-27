# Design Spec — Analyst Agent Tool Use / Function Calling

## Problem

The `TechnicalVolumeAnalyst` currently receives a pre-computed market snapshot as a static dict.
The scheduler deterministically builds this snapshot via `FeaturePipeline.run()` and passes it
into the LLM prompt as `json.dumps(snapshot)`. The agent has no ability to request data, query
for more context, or decide what information it needs.

**Rubric gap:** "The agents themselves do not make tool calls during their reasoning loops."

**Hiring signal:** Writing tool-calling schemas, handling agent tool routing, and cleaning up
tool execution exceptions.

## Solution — Multi-Turn Agentic Tool-Use Loop

The analyst agent receives a symbol + timestamp and a set of **6 callable tools**. It runs in a
`while` loop (max 10 iterations). Each iteration:

1. LLM is called with `tools=` parameter and `tool_choice="auto"` via LiteLLM
2. If the response contains `tool_calls` → execute each tool locally via a **ToolExecutor**
   dispatcher, append results as `role: "tool"` messages, loop back to step 1
3. If the response contains **no tool calls** → the content is the final `AnalystOutput` JSON,
   loop exits
4. Safety: max iterations cap + per-tool try/except wrapping (tool errors return structured
   error messages to the LLM rather than crashing the loop)

### Data Flow

```
Scheduler                          Analyst Agent
   │                                    │
   ├─ analyst.analyze(symbol, ts) ──────►│
   │                                    ├─ LLM call 1 (tools available)
   │                                    │   ← tool_calls: [get_candles("BTC/USDT", "4h")]
   │                                    ├─ ToolExecutor dispatches → DuckDB query
   │                                    ├─ LLM call 2 (+ tool results)
   │                                    │   ← tool_calls: [get_indicators(...), get_derivatives(...)]
   │                                    ├─ ToolExecutor dispatches → TA-Lib + Binance API
   │                                    ├─ LLM call 3 (+ tool results)
   │                                    │   ← tool_calls: [get_market_sentiment()]
   │                                    ├─ ToolExecutor dispatches → Fear & Greed API
   │                                    ├─ LLM call 4 (+ tool results)
   │                                    │   ← content: AnalystOutput JSON (no tool_calls)
   │  ◄──── AnalystOutput ─────────────┘
```

## Tool Definitions

6 tools exposed to the analyst LLM, defined as OpenAI-compatible JSON schemas (LiteLLM standard):

| # | Tool Name | Data Source | Input Parameters | Returns |
|---|-----------|-------------|------------------|---------|
| 1 | `get_candles` | DuckDB | `symbol`, `timeframe` ("4h"\|"1d"), `limit` (default 20, max 50) | OHLCV rows as JSON array |
| 2 | `get_indicators` | TA-Lib on DuckDB candles | `symbol`, `timeframe` ("4h"\|"1d") | RSI(14), MACD, ADX(14), OBV, SMA(20/50/200) with regime labels |
| 3 | `get_support_resistance` | scipy on DuckDB candles | `symbol` | Support/resistance levels with proximity to current price |
| 4 | `get_candlestick_patterns` | TA-Lib on DuckDB candles | `symbol` | Active candlestick patterns (engulfing, hammer, morning star, etc.) |
| 5 | `get_derivatives` | Binance Futures (ccxt) | `symbol` | Funding rate (categorized), Open Interest |
| 6 | `get_market_sentiment` | Fear & Greed Index API | *(none)* | Current sentiment score (0-100) + classification (Extreme Fear → Extreme Greed) |

### Tool Implementation Sourcing

Tools 1-4 reuse existing `FeaturePipeline` / `Database` methods — no logic duplication.
Tool 5 reuses `DataFetcher.fetch_funding_rate_and_oi()`.
Tool 6 is new — calls the public Fear & Greed Index API (`api.alternative.me/fng/`).

## Components

### 1. `src/vibe_trading/agents/tools.py` [NEW]

Contains:

- **`ANALYST_TOOLS`**: List of 6 tool definition dicts in OpenAI function-calling JSON schema format.
- **`ToolExecutor`**: Class that holds `Database` + `DataFetcher` references, dispatches tool
  calls by name to Python handler methods, and wraps every execution in try/except.

```python
class ToolExecutor:
    def __init__(self, db: Database, fetcher: DataFetcher):
        self.db = db
        self.fetcher = fetcher
        self._dispatch = {
            "get_candles": self._get_candles,
            "get_indicators": self._get_indicators,
            "get_support_resistance": self._get_support_resistance,
            "get_candlestick_patterns": self._get_candlestick_patterns,
            "get_derivatives": self._get_derivatives,
            "get_market_sentiment": self._get_market_sentiment,
        }

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool by name. Returns JSON string (result or error)."""
        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
```

Handler methods:

- **`_get_candles(symbol, timeframe, limit=20)`**: Opens short-lived DuckDB connection, queries
  candles, returns list of dicts. Clamps `limit` to max 50.
- **`_get_indicators(symbol, timeframe="4h")`**: Queries 300 candles, runs
  `FeaturePipeline._calculate_indicators()`, returns latest indicator values + regime labels.
- **`_get_support_resistance(symbol)`**: Queries 300 4h candles, runs
  `FeaturePipeline._detect_support_resistance()` + proximity calculations, returns S/R dict.
- **`_get_candlestick_patterns(symbol)`**: Queries 30 4h candles, runs
  `FeaturePipeline._recognize_candlesticks()`, returns pattern string.
- **`_get_derivatives(symbol)`**: Delegates to `DataFetcher.fetch_funding_rate_and_oi()`.
- **`_get_market_sentiment()`**: HTTP GET to `https://api.alternative.me/fng/?limit=1`, returns
  `{"value": 72, "classification": "Greed", "timestamp": "..."}`.

### 2. `src/vibe_trading/agents/client.py` [MODIFY]

Add a new method `call_llm_with_tools()` alongside the existing `call_llm()`:

```python
def call_llm_with_tools(
    self,
    model_name: str,
    system_instruction: str,
    prompt: str,
    tools: list[dict],
    tool_executor: ToolExecutor,
    max_iterations: int = 10
) -> str:
    """Multi-turn agentic loop: LLM calls tools, executor runs them, results fed back."""
    model_str = get_litellm_model_string(self.provider, model_name)
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    for iteration in range(max_iterations):
        logger.info(f"Tool-use loop iteration {iteration + 1}/{max_iterations}")
        response = litellm.completion(
            model=model_str,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1
        )
        choice = response.choices[0]
        assistant_msg = choice.message
        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            return assistant_msg.content

        for tool_call in assistant_msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            logger.info(f"Executing tool: {tool_call.function.name}({args})")
            result = tool_executor.execute(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    raise RuntimeError(f"Agent exceeded max tool-call iterations ({max_iterations})")
```

The existing `call_llm()` method is **not modified** — the trader agent and any non-tool agents
continue using it unchanged.

### 3. `src/vibe_trading/agents/analyst.py` [MODIFY]

**Constructor**: Accept `db` and `fetcher` for tool execution context.

```python
class TechnicalVolumeAnalyst:
    def __init__(self, client: LLMClient = None, db: Database = None, fetcher: DataFetcher = None):
        self.client = client or LLMClient()
        self.tool_executor = ToolExecutor(db=db, fetcher=fetcher) if db and fetcher else None
        provider = self.client.provider
        self.model = os.getenv(f"{provider.upper()}_ANALYST_MODEL") or self.client.model
        # system_instruction updated to reference tool usage
```

**`analyze()` method**: Changed from `analyze(snapshot: dict)` to
`analyze(symbol: str, timestamp: datetime)`. Uses `call_llm_with_tools()` when tools are
available, falls back to static snapshot mode when they're not (backwards compatible for tests).

```python
def analyze(self, symbol: str, timestamp: datetime = None, snapshot: dict = None) -> AnalystOutput:
    if self.tool_executor and not snapshot:
        # Agentic tool-use path
        prompt = f"Analyze the market for {symbol} at {timestamp}. Use tools to gather data."
        raw = self.client.call_llm_with_tools(
            model_name=self.model,
            system_instruction=self.system_instruction,
            prompt=prompt,
            tools=ANALYST_TOOLS,
            tool_executor=self.tool_executor
        )
    else:
        # Legacy static snapshot path (backwards compatible)
        prompt = f"Analyze the following Market Snapshot for {symbol}:\n{json.dumps(snapshot, indent=2, default=str)}"
        raw = self.client.call_llm(
            model_name=self.model,
            system_instruction=self.system_instruction,
            prompt=prompt,
            response_schema=AnalystOutput
        )
    return AnalystOutput(**json.loads(raw))
```

**System instruction**: Updated to tell the LLM it has tools available and should use them to
gather data before producing its analysis.

### 4. `src/vibe_trading/runtime/scheduler.py` [MODIFY]

Two changes:

1. **Constructor (line 39)**: Pass `db` and `fetcher` to analyst:
   ```python
   self.analyst = TechnicalVolumeAnalyst(db=self.db, fetcher=self.fetcher)
   ```

2. **`sync_and_evaluate()` (lines 158-164)**: Remove `FeaturePipeline.run()` call, change
   analyst invocation:
   ```python
   # OLD:
   snapshot = self.pipeline.run(sym, last_ts)
   analyst_report = self.analyst.analyze(snapshot)

   # NEW:
   analyst_report = self.analyst.analyze(symbol=sym, timestamp=last_ts)
   ```

   Note: `snapshot` is still needed downstream for `decision_log` (line 178) and
   `RiskManager.evaluate_proposal()` (line 195). We keep the existing
   `self.pipeline.run(sym, last_ts)` call but move it AFTER the analyst tool-use call.
   This way the analyst drives its own data acquisition via tools, and the deterministic
   snapshot is only built for the RiskManager and audit logging:
   ```python
   analyst_report = self.analyst.analyze(symbol=sym, timestamp=last_ts)
   snapshot = self.pipeline.run(sym, last_ts)  # for RiskManager + decision_log only
   ```

### 5. `src/vibe_trading/eval/backtest.py` [MODIFY]

Update the backtest call site similarly — pass `symbol` and `timestamp` to `analyst.analyze()`.

## Error Handling

| Failure Scenario | Handling |
|------------------|----------|
| Unknown tool name | `ToolExecutor.execute()` returns `{"error": "Unknown tool: X"}` — LLM sees error, can retry or proceed |
| Tool execution exception (DB error, API timeout) | Caught in `execute()`, returns `{"error": "Tool execution failed: ..."}` — LLM continues |
| Malformed tool arguments from LLM | Caught in `json.loads()` or `**kwargs` dispatch, returned as error message |
| Max iterations exceeded (10) | `call_llm_with_tools()` raises `RuntimeError` — scheduler catches, logs, skips symbol |
| LLM returns no content and no tool_calls | Treated as empty response — loop continues (counted against max iterations) |

## Backwards Compatibility

- `analyst.analyze()` accepts both the new `(symbol, timestamp)` signature and the legacy
  `(snapshot=dict)` keyword argument. When `tool_executor` is None (no db/fetcher injected),
  it falls back to the old static path.
- `LLMClient.call_llm()` is unchanged — trader agent is unaffected.
- Existing tests using mock snapshots continue to work via the `snapshot=` parameter.

## Testing Strategy

### Unit Tests (in `tests/test_multi_provider.py`)

1. **`test_tool_executor_dispatch`** — Verify `ToolExecutor.execute()` routes to correct handler
2. **`test_tool_executor_unknown_tool`** — Verify unknown tool returns error JSON, no crash
3. **`test_tool_executor_exception_handling`** — Verify handler exception returns error JSON
4. **`test_call_llm_with_tools_single_turn`** — Mock LLM returns tool_calls once, then final answer
5. **`test_call_llm_with_tools_multi_turn`** — Mock LLM returns tool_calls twice, then final answer
6. **`test_call_llm_with_tools_max_iterations`** — Mock LLM always returns tool_calls, verify RuntimeError
7. **`test_analyst_tool_use_integration`** — Full analyst.analyze() with mocked LLM + mocked tools
8. **`test_analyst_legacy_snapshot_fallback`** — Verify old snapshot path still works
9. **`test_get_market_sentiment`** — Mock HTTP call, verify parsing

### Verification

```bash
uv run pytest tests/test_multi_provider.py -v
uv run pytest -v  # full suite
```
