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


from pathlib import Path
import json

from vibe_trading.eval.benchmark import (
    sort_entries, render_markdown, write_benchmark_report,
)


def make_entry(model: str, overall: float, cost: float) -> BenchmarkEntry:
    return BenchmarkEntry(
        model=model, case_count=34, overall_score=overall, analyst_score=overall,
        trader_score=overall, pass_rate=0.1, schema_failures=0, judge_errors=0,
        calls=100, total_tokens=50000, cost_usd=cost, judge_cost_usd=0.01,
        mean_latency_ms=800.0,
        score_per_dollar=(overall / cost) if cost > 0 else None,
    )


def test_sort_entries_by_score_per_dollar_none_last():
    cheap_good = make_entry("g/cheap", 0.80, 0.10)     # spd 8.0
    pricey_good = make_entry("o/pricey", 0.85, 0.50)   # spd 1.7
    free_bad = make_entry("g/free", 0.60, 0.0)         # spd None
    ordered = sort_entries([free_bad, pricey_good, cheap_good])
    assert [e.model for e in ordered] == ["g/cheap", "o/pricey", "g/free"]


def test_render_markdown_has_header_and_all_models():
    md = render_markdown([make_entry("g/a", 0.8, 0.1)], judge_model="gemini-3.1-flash-lite",
                         run_at_iso="2026-07-05T00:00:00Z")
    assert "| Model |" in md
    assert "g/a" in md
    assert "gemini-3.1-flash-lite" in md   # judge disclosure
    assert "2026-07-05" in md              # run date disclosure


def test_write_benchmark_report_round_trips(tmp_path: Path):
    entries = [make_entry("g/a", 0.8, 0.1)]
    path = write_benchmark_report(entries, judge_model="j-model", reports_dir=tmp_path)
    assert path.name.startswith("benchmark-") and path.suffix == ".json"
    data = json.loads(path.read_text())
    assert data["judge_model"] == "j-model"
    assert data["entries"][0]["model"] == "g/a"
    assert data["entries"][0]["score_per_dollar"] == pytest.approx(8.0)
