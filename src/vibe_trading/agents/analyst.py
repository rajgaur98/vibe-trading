from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime
import json
import os
import re
from langfuse import observe, propagate_attributes
from vibe_trading.agents.client import LLMClient, validate_structured
from vibe_trading.agents.tools import ANALYST_TOOLS, ToolExecutor
from vibe_trading.agents import prompts
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

        self.system_instruction = prompts.ANALYST_SYSTEM.text

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
            metadata={"symbol": symbol, "prompt_version": prompts.ANALYST_SYSTEM.stamp},
        ):
            if self.tool_executor is not None and snapshot is None:
                self.tool_executor.set_timestamp(timestamp)
                prompt = (
                    f"Analyze the market for {symbol} as of {timestamp}. "
                    f"Use the available tools to gather indicators, support/resistance, "
                    f"candlestick patterns, derivatives, and market sentiment, then "
                    f"produce the final JSON analysis."
                )

                def _call_tool_loop(extra: str = "") -> str:
                    return self.client.call_llm_with_tools(
                        model_name=self.model,
                        system_instruction=self.system_instruction,
                        prompt=prompt + extra,
                        tools=ANALYST_TOOLS,
                        tool_executor=self.tool_executor,
                        expect_schema=True,
                        prompt_version=prompts.ANALYST_SYSTEM.stamp,
                    )

                raw_output = _call_tool_loop()
                recall = _call_tool_loop
            else:
                prompt = (
                    f"Analyze the following Market Snapshot for {symbol}:\n"
                    f"{json.dumps(snapshot, indent=2, default=str)}\n\n"
                    f"Evaluate all parameters, check for price-volume confirmation or "
                    f"divergence, and output the analysis."
                )

                def _call_single(extra: str = "") -> str:
                    return self.client.call_llm(
                        model_name=self.model,
                        system_instruction=self.system_instruction,
                        prompt=prompt + extra,
                        response_schema=AnalystOutput,
                        prompt_version=prompts.ANALYST_SYSTEM.stamp,
                    )

                raw_output = _call_single()
                recall = _call_single

            # Validate into AnalystOutput with ONE corrective retry; records the
            # schema-compliance outcome onto the cost event. Raises SchemaValidationError
            # (never a bare KeyError/ValidationError) if both attempts fail.
            return validate_structured(
                self.client, AnalystOutput, raw_output, recall, extract=_extract_json
            )
