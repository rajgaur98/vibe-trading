from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime
import json
import os
import re
from langfuse import observe, propagate_attributes
from vibe_trading.agents.client import LLMClient
from vibe_trading.agents.tools import ANALYST_TOOLS, ToolExecutor
from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher


def _extract_json(text: str) -> str:
    """Return a JSON string ready for json.loads, tolerating markdown code fences.

    The legacy snapshot path passes `response_format=AnalystOutput`, which yields a
    bare JSON object. The tool-use path can't set `response_format` mid-loop, so the
    model's final answer is unconstrained freeform — and some models (notably Gemma,
    occasionally Gemini) wrap it in ```json ... ``` fences. This strips those fences
    so both paths parse cleanly; bare JSON passes through unchanged.
    """
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence (``` or ```json) and any trailing closing fence.
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


class AnalystOutput(BaseModel):
    market_bias: Literal["bullish", "bearish", "neutral"] = Field(
        description="The overall market trend direction identified from price, indicators, and structure."
    )
    volume_confirmation: Literal["confirmed", "divergent", "weak"] = Field(
        description="How volume behavior relates to the current price movement (e.g., rising volume on rally = confirmed)."
    )
    thesis: str = Field(
        description="A detailed paragraph summarizing the confluence of indicators, candlestick signals, and S/R levels."
    )
    nearest_support: float = Field(description="The closest valid support level identified from fractal pivots.")
    nearest_resistance: float = Field(description="The closest valid resistance level identified from fractal pivots.")
    confluence_score: float = Field(
        description="A value between 0.0 and 1.0 indicating the ratio of indicators supporting the overall bias."
    )


class TechnicalVolumeAnalyst:
    def __init__(
        self,
        client: LLMClient = None,
        db: Database = None,
        fetcher: DataFetcher = None,
    ):
        self.client = client or LLMClient()
        provider = self.client.provider
        self.model = os.getenv(f"{provider.upper()}_ANALYST_MODEL") or self.client.model

        self.tool_executor: Optional[ToolExecutor] = (
            ToolExecutor(db=db, fetcher=fetcher) if db is not None and fetcher is not None else None
        )

        self.system_instruction = """
You are an elite Crypto Technical and Volume Analyst specializing in swing trading.
Your objective is to evaluate market conditions for a given symbol and produce a structured technical thesis.

You have access to six tools that fetch market data on demand. Use them to gather:
1. Recent OHLCV candles (get_candles) — call separately for the 4h and 1d timeframes to build multi-timeframe context.
2. Momentum and trend indicators with regime labels (get_indicators) — RSI(14), MACD, ADX(14), OBV, SMA(20/50/200).
3. Support and resistance levels with proximity (get_support_resistance).
4. Active candlestick patterns (get_candlestick_patterns).
5. Derivatives — funding rate and open interest (get_derivatives).
6. Broader market sentiment — Fear & Greed Index (get_market_sentiment).

Call as many tools as needed to build confluence. Typically you should fetch both 4h and 1d indicators
plus support/resistance and at least one of derivatives or market sentiment before deciding.

When evaluating the data, apply the classic Murphy principles:
- Volume must confirm the price trend (rising volume on breakouts, falling volume on pullbacks).
- Divergences between price and momentum (RSI/MACD) indicate impending trend exhaustion.
- Chart patterns and candlesticks are only valid when they occur at major support/resistance levels.

=== VOLUME CONFIRMATION (volume_confirmation) — judge the OBV trend RELATIVE to your own market_bias ===
- "confirmed": OBV agrees with your bias — OBV accumulation under a BULLISH bias, or OBV distribution under a BEARISH bias.
- "divergent": OBV opposes your bias — OBV distribution under a BULLISH bias, or OBV accumulation under a BEARISH bias (a warning of trend exhaustion).
- "weak": OBV is flat/neutral, OR your market_bias is neutral (volume confirms no particular direction).
Decide market_bias first, then label volume_confirmation against it using this rule.

When you have enough data, STOP calling tools and respond with a final JSON object that exactly
matches this schema (no extra text, no tool_calls):
{
  "market_bias": "bullish" | "bearish" | "neutral",
  "volume_confirmation": "confirmed" | "divergent" | "weak",
  "thesis": "<paragraph summary>",
  "nearest_support": <float>,
  "nearest_resistance": <float>,
  "confluence_score": <0.0..1.0>
}
"""

    @observe()
    def analyze(
        self,
        symbol: str,
        timestamp: datetime = None,
        snapshot: dict = None,
    ) -> AnalystOutput:
        """Runs the analyst agent.

        Tool-loop path (preferred): if a ToolExecutor is configured and no snapshot is supplied,
        the LLM drives data acquisition via tool calls.

        Legacy path: if a snapshot is supplied (or no tool executor is available), the snapshot
        is rendered inline into the prompt and the single-shot structured call is used.
        """
        with propagate_attributes(
            trace_name=f"Analyst-analyze-{symbol}",
            tags=[symbol],
            metadata={"symbol": symbol},
        ):
            if self.tool_executor is not None and snapshot is None:
                self.tool_executor.set_timestamp(timestamp)
                prompt = (
                    f"Analyze the market for {symbol} as of {timestamp}. "
                    f"Use the available tools to gather indicators, support/resistance, "
                    f"candlestick patterns, derivatives, and market sentiment, then "
                    f"produce the final JSON analysis."
                )
                raw_output = self.client.call_llm_with_tools(
                    model_name=self.model,
                    system_instruction=self.system_instruction,
                    prompt=prompt,
                    tools=ANALYST_TOOLS,
                    tool_executor=self.tool_executor,
                )
            else:
                prompt = (
                    f"Analyze the following Market Snapshot for {symbol}:\n"
                    f"{json.dumps(snapshot, indent=2, default=str)}\n\n"
                    f"Evaluate all parameters, check for price-volume confirmation or "
                    f"divergence, and output the analysis."
                )
                raw_output = self.client.call_llm(
                    model_name=self.model,
                    system_instruction=self.system_instruction,
                    prompt=prompt,
                    response_schema=AnalystOutput,
                )

            data = json.loads(_extract_json(raw_output))
            return AnalystOutput(**data)
