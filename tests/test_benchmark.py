import threading
from datetime import datetime

import pytest

from vibe_trading.agents.cost import CostEvent
from vibe_trading.eval.report import SuiteReport
from vibe_trading.eval.scorer import CaseScore
from vibe_trading.eval.benchmark import (
    InMemoryCostSink, parse_model_spec, build_entry, BenchmarkEntry,
)


def make_event(model: str, cost: float = 0.01, tokens: int = 1000,
               latency: float = 500.0) -> CostEvent:
    return CostEvent(
        call_id=f"call-{model}-{cost}-{latency}", timestamp=datetime(2026, 7, 5),
        provider=model.split("/")[0], model=model, call_type="tool_loop",
        prompt_tokens=tokens - 100, completion_tokens=100, total_tokens=tokens,
        cost_usd=cost, latency_ms=latency,
    )


def make_report(overall: float = 0.8) -> SuiteReport:
    score = CaseScore(case_id="001-x", schema_ok=True, field_scores=[],
                      analyst_score=overall, trader_score=overall, total_score=overall)
    return SuiteReport.from_scores([score])


def test_sink_collects_events_thread_safely():
    sink = InMemoryCostSink()
    threads = [threading.Thread(target=lambda i=i: sink.record(make_event(f"m/{i}")))
               for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(sink.events) == 20


def test_parse_model_spec_splits_on_first_slash():
    assert parse_model_spec("gemini/gemma-4-31b-it") == ("gemini", "gemma-4-31b-it")
    # model segment may itself contain a slash (e.g. ollama org/model tags)
    assert parse_model_spec("ollama/library/llama4") == ("ollama", "library/llama4")


def test_parse_model_spec_rejects_missing_provider():
    with pytest.raises(ValueError):
        parse_model_spec("gemma-4-31b-it")


def test_build_entry_separates_own_cost_from_judge_cost():
    report = make_report(overall=0.8)
    events = [
        make_event("gemini/gemma-4-31b-it", cost=0.02, latency=400.0),
        make_event("gemini/gemma-4-31b-it", cost=0.02, latency=600.0),
        make_event("gemini/gemini-3.1-flash-lite", cost=0.05),  # the judge
    ]
    entry = build_entry("gemini/gemma-4-31b-it", report, events)
    assert entry.cost_usd == pytest.approx(0.04)
    assert entry.judge_cost_usd == pytest.approx(0.05)
    assert entry.calls == 2
    assert entry.mean_latency_ms == pytest.approx(500.0)
    assert entry.score_per_dollar == pytest.approx(0.8 / 0.04)


def test_build_entry_zero_cost_gives_none_score_per_dollar():
    entry = build_entry("gemini/free-model", make_report(), [])
    assert entry.cost_usd == 0.0
    assert entry.score_per_dollar is None
    assert entry.mean_latency_ms == 0.0
