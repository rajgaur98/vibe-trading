"""Model benchmark matrix — run the golden-set eval suite across N models and emit
a cost-vs-quality comparison (JSON report + markdown table).

Fairness invariants:
  * EVAL_JUDGE_MODEL must be pinned; the judge is built ONCE before any model-env
    mutation so every contestant is graded by the same judge on the same provider.
  * evals/baseline.json is never read or written by a benchmark run.
"""
import logging
import threading
from typing import Optional

from pydantic import BaseModel

from vibe_trading.agents.cost import CostEvent
from vibe_trading.eval.report import SuiteReport

logger = logging.getLogger(__name__)


class InMemoryCostSink:
    """Collects CostEvents in memory for one benchmark run. Thread-safe: run_case
    workers record concurrently. Satisfies the LLMClient cost-sink protocol
    (`.record(event)`), same as PostgresCostLogger."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events: list[CostEvent] = []

    def record(self, event: CostEvent) -> None:
        with self._lock:
            self.events.append(event)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a litellm-format 'provider/model' spec into (provider, model).
    The model segment may contain further slashes (ollama tags)."""
    provider, sep, model = spec.partition("/")
    if not sep or not provider or not model:
        raise ValueError(
            f"Model spec '{spec}' must be litellm-format 'provider/model' "
            f"(e.g. 'gemini/gemma-4-31b-it')."
        )
    return provider, model


class BenchmarkEntry(BaseModel):
    model: str                     # litellm-format id, e.g. "gemini/gemma-4-31b-it"
    case_count: int
    overall_score: float
    analyst_score: float
    trader_score: float
    pass_rate: float
    schema_failures: int
    judge_errors: int
    calls: int                     # this model's calls only (judge excluded)
    total_tokens: int
    cost_usd: float                # this model's calls only
    judge_cost_usd: float          # other-model calls recorded during this run (the judge)
    mean_latency_ms: float
    score_per_dollar: Optional[float]  # None when the run cost is zero


def build_entry(model_str: str, report: SuiteReport,
                events: list[CostEvent]) -> BenchmarkEntry:
    """Aggregate one model's run into a BenchmarkEntry.

    Events are split by model string: the benchmarked model's calls versus
    everything else recorded while its run was active (i.e. the judge). Caveat,
    documented in the report: when the judge model IS the benchmarked model, its
    calls are indistinguishable and land in cost_usd.
    """
    own = [e for e in events if e.model == model_str]
    other = [e for e in events if e.model != model_str]
    cost = sum(e.cost_usd for e in own)
    return BenchmarkEntry(
        model=model_str,
        case_count=report.case_count,
        overall_score=report.overall_score,
        analyst_score=report.analyst_score,
        trader_score=report.trader_score,
        pass_rate=report.pass_rate,
        schema_failures=report.schema_failures,
        judge_errors=report.judge_errors,
        calls=len(own),
        total_tokens=sum(e.total_tokens for e in own),
        cost_usd=cost,
        judge_cost_usd=sum(e.cost_usd for e in other),
        mean_latency_ms=(sum(e.latency_ms for e in own) / len(own)) if own else 0.0,
        score_per_dollar=(report.overall_score / cost) if cost > 0 else None,
    )
