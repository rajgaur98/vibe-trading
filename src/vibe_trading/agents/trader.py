from pydantic import BaseModel, Field
from typing import Literal
import json
import os
from uuid import uuid4
from datetime import datetime
from decimal import Decimal
from langfuse import observe, propagate_attributes
from vibe_trading.agents.client import LLMClient
from vibe_trading.agents.analyst import AnalystOutput


class HeadTraderOutput(BaseModel):
    action: Literal["long", "short", "flat", "close"] = Field(
        description="The action to take: 'long' to enter buy, 'short' to enter sell, 'close' to exit active position, or 'flat' to do nothing."
    )
    stop_loss_strategy: Literal["1.5_atr", "2.0_atr", "swing_low", "tight_atr"] = Field(
        description="The qualitative risk model strategy to determine the stop loss price boundary."
    )
    take_profit_strategy: Literal["3.0_atr", "4.0_atr", "next_resistance", "risk_reward_multiplier"] = Field(
        description="The qualitative profit capture strategy."
    )
    risk_reward_ratio: float = Field(
        description="The target risk-to-reward ratio (e.g., 2.0 means profit target is 2x larger than stop-loss)."
    )
    hold_period_bias: Literal["short", "medium", "long"] = Field(
        description="Expected holding period: short (1-2 days), medium (3-7 days), long (weeks)."
    )
    reasoning_summary: str = Field(
        description="A clear, concise summary of the rationale behind this final decision."
    )

class HeadTrader:
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()
        provider = self.client.provider
        self.model = os.getenv(f"{provider.upper()}_TRADER_MODEL") or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

        
        self.system_instruction = """
You are the Head Trader of a systematic crypto hedge fund. 
Your job is to synthesize technical analysis, volume analysis, and historical performance metrics to make a final, highly disciplined trading decision.

You will receive:
1. An Analyst report containing bias, volume confirmation, and structural S/R zones.
2. The historical accuracy scorecard for the analyst (e.g. how often they are correct).
3. The current portfolio positions.

Your core directives:
- Keep risk parameters strict. Do not chase trades if the analyst thesis is weak or has divergence.
- Resolve conflicts: If price bias is bullish but volume is weak/divergent, lean toward 'flat'.
- Output a qualitative Stop Loss and Take Profit strategy. Do NOT compute raw prices; describe the method to place them.
- Ensure the risk_reward_ratio is mathematically sound (typically >= 1.5).

Provide your output strictly matching the Pydantic JSON schema.
"""

    @observe()
    def decide(
        self,
        symbol: str,
        analyst_output: AnalystOutput,
        scorecard: dict,
        open_positions: list
    ) -> dict:
        """Invokes the Head Trader agent to make a trade decision."""
        with propagate_attributes(
            trace_name=f"HeadTrader-decide-{symbol}",
            tags=[symbol],
            metadata={"symbol": symbol}
        ):
            prompt = f"""Make a trading decision for {symbol}.

--- Analyst Output ---
{json.dumps(analyst_output.dict(), indent=2, default=str)}

--- Historical Analyst Accuracy Scorecard ---
{json.dumps(scorecard, indent=2, default=str)}

--- Current Open Positions ---
{json.dumps(open_positions, indent=2, default=str)}

--- Rules ---
- Do not exceed maximum risk allocations.
- Chart patterns and candlesticks are only valid when they occur at major support/resistance levels.
- Always output a valid schema.
"""
            raw_output = self.client.call_llm(
                model_name=self.model,
                system_instruction=self.system_instruction,
                prompt=prompt,
                response_schema=HeadTraderOutput
            )
            
            data = json.loads(raw_output)
            
            # Hydrate the final proposal dictionary with system fields (UUID, timestamp)
            proposal = {
                "decision_id": str(uuid4()),
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": data["action"],
                "stop_loss_strategy": data["stop_loss_strategy"],
                "take_profit_strategy": data["take_profit_strategy"],
                "risk_reward_ratio": Decimal(str(data["risk_reward_ratio"])),
                "hold_period_bias": data["hold_period_bias"],
                "reasoning_summary": data["reasoning_summary"]
            }
            
            return proposal
