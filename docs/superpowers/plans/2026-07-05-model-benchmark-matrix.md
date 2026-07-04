# Model Benchmark Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command that runs the 34-case golden-set eval across N models and emits a cost-vs-quality comparison — a JSON report plus a committed markdown table (`evals/BENCHMARK.md`) showing why the default model is the default.

**Architecture:** A new `src/vibe_trading/eval/benchmark.py` module orchestrates the *existing* suite machinery (`load_cases` / `run_case` / `score_case` / `SuiteReport`) once per model. Per-model cost is captured by installing an in-memory cost sink (`LLMClient.set_cost_sink`) for the duration of each run. The judge is constructed **once, before any model-env mutation**, so every contestant is graded by the same pinned judge on the same provider. The committed decision-quality baseline (`evals/baseline.json`) is never read or written.

**Tech Stack:** Python 3.12, pydantic v2, existing eval harness, `LLMClient` class-level cost sink, pytest.

**Workstream:** A of `docs/superpowers/specs/2026-07-04-ai-deepening-roadmap-design.md`. No dependency on workstreams B/C/D.

## Global Constraints

- `EVAL_JUDGE_MODEL` **must be set** for a benchmark run; refuse to start otherwise (exit code 2). The scorer's fallback (judge defaults to the client's model) would let each contestant grade its own homework.
- The judge's `LLMClient` is constructed under the **ambient** env before any per-model mutation — `EVAL_JUDGE_MODEL` must be a model hosted by the ambient `LLM_PROVIDER` at launch.
- `evals/baseline.json` is untouched: no `--update-baseline` plumbing, no imports of `write_baseline` / `load_baseline` / `diff_against_baseline`.
- Model specs are litellm-format `provider/model` strings (e.g. `gemini/gemma-4-31b-it`); the provider segment must be one of the keys of `_PROVIDER_API_KEY_ENV` or `ollama`.
- Per-model cost is accumulated from **in-process cost events**, never by querying `llm_cost_log` by time window (that would race with live trading on the same DB).
- All tests are offline: no network, no live LLM calls, no Postgres.
- Run tests with `uv run pytest <path> -v`.

---

### Task 1: Cost capture and per-model aggregation

**Files:**
- Create: `src/vibe_trading/eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `CostEvent` from `vibe_trading.agents.cost` (fields: `model`, `cost_usd`, `total_tokens`, `latency_ms`, …); `SuiteReport` from `vibe_trading.eval.report`.
- Produces: `InMemoryCostSink` (`.events: list[CostEvent]`, `.record(event)`); `parse_model_spec(spec: str) -> tuple[str, str]`; `BenchmarkEntry` (pydantic model, fields below); `build_entry(model_str: str, report: SuiteReport, events: list[CostEvent]) -> BenchmarkEntry`. Tasks 2 and 4 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_benchmark.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.eval.benchmark'`

- [ ] **Step 3: Write the implementation**

```python
# src/vibe_trading/eval/benchmark.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): in-memory cost sink + per-model entry aggregation"
```

---

### Task 2: Markdown table and JSON report rendering

**Files:**
- Modify: `src/vibe_trading/eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `BenchmarkEntry` from Task 1.
- Produces: `sort_entries(entries) -> list[BenchmarkEntry]`; `render_markdown(entries: list[BenchmarkEntry], judge_model: str, run_at_iso: str) -> str`; `write_benchmark_report(entries, judge_model, reports_dir: Path) -> Path` (writes `benchmark-<ts>.json`). Task 4 calls all three.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: 3 new FAILs with `ImportError: cannot import name 'sort_entries'`

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/eval/benchmark.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path


def sort_entries(entries: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
    """Best value first: score-per-dollar descending, zero-cost (None) entries last,
    ties broken by overall score descending."""
    return sorted(
        entries,
        key=lambda e: (e.score_per_dollar is None,
                       -(e.score_per_dollar or 0.0),
                       -e.overall_score),
    )


def render_markdown(entries: list[BenchmarkEntry], judge_model: str,
                    run_at_iso: str) -> str:
    """The committed evals/BENCHMARK.md content: a generated table plus the run
    conditions (judge, date, case count) so the numbers are interpretable later."""
    rows = sort_entries(entries)
    case_count = rows[0].case_count if rows else 0
    lines = [
        "# Model Benchmark",
        "",
        f"Golden-set eval ({case_count} cases, tool-loop path) run across "
        f"{len(rows)} model(s) on {run_at_iso}.",
        f"Judge (pinned for all runs): `{judge_model}`. "
        f"Generated by `python -m vibe_trading.eval.benchmark` — do not edit by hand.",
        "",
        "| Model | Overall | Analyst | Trader | Pass rate | Schema fails "
        "| Calls | Tokens | Cost (USD) | Mean latency (ms) | Score / $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in rows:
        spd = f"{e.score_per_dollar:.1f}" if e.score_per_dollar is not None else "n/a (zero cost)"
        lines.append(
            f"| `{e.model}` | {e.overall_score:.2f} | {e.analyst_score:.2f} "
            f"| {e.trader_score:.2f} | {e.pass_rate * 100:.0f}% | {e.schema_failures} "
            f"| {e.calls} | {e.total_tokens} | ${e.cost_usd:.4f} "
            f"| {e.mean_latency_ms:.0f} | {spd} |"
        )
    lines.append("")
    lines.append("Cost is per full suite run for that model's calls only; judge calls "
                 "are excluded (when the judge model is also a contestant, its judge "
                 "calls are indistinguishable and inflate that row's cost).")
    return "\n".join(lines) + "\n"


def write_benchmark_report(entries: list[BenchmarkEntry], judge_model: str,
                           reports_dir: Path) -> Path:
    """Write benchmark-<ts>.json under reports_dir (same naming style as eval reports)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc)
    path = reports_dir / ("benchmark-" + run_at.strftime("%Y-%m-%dT%H-%M-%SZ") + ".json")
    payload = {
        "run_at": run_at.isoformat(),
        "judge_model": judge_model,
        "entries": [e.model_dump() for e in sort_entries(entries)],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): markdown table + JSON report rendering"
```

---

### Task 3: Per-model environment swap

**Files:**
- Modify: `src/vibe_trading/eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `model_env(provider: str, model: str)` context manager. Task 4's `run_model` wraps each suite run in it.

Why this exists: `LLMClient.__init__` reads `LLM_PROVIDER`/`LLM_MODEL` from env, and the analyst/trader additionally honor `{PROVIDER}_ANALYST_MODEL`/`{PROVIDER}_TRADER_MODEL` overrides. A fair benchmark must force *both* agents onto exactly the specified model, then restore the user's environment.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark.py`:

```python
import os

from vibe_trading.eval.benchmark import model_env


def test_model_env_sets_and_restores(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "original-model")
    monkeypatch.setenv("OPENAI_ANALYST_MODEL", "user-analyst-override")
    with model_env("openai", "gpt-5.2-mini"):
        assert os.environ["LLM_PROVIDER"] == "openai"
        assert os.environ["LLM_MODEL"] == "gpt-5.2-mini"
        # the override would silently reroute the analyst — must be neutralized
        assert "OPENAI_ANALYST_MODEL" not in os.environ
        assert "OPENAI_TRADER_MODEL" not in os.environ
    assert os.environ["LLM_PROVIDER"] == "gemini"
    assert os.environ["LLM_MODEL"] == "original-model"
    assert os.environ["OPENAI_ANALYST_MODEL"] == "user-analyst-override"


def test_model_env_restores_on_exception(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError):
        with model_env("groq", "llama-4-70b"):
            raise RuntimeError("boom")
    assert os.environ["LLM_PROVIDER"] == "gemini"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v -k model_env`
Expected: FAIL with `ImportError: cannot import name 'model_env'`

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/eval/benchmark.py`:

```python
import contextlib
import os


@contextlib.contextmanager
def model_env(provider: str, model: str):
    """Point LLM_PROVIDER/LLM_MODEL at the benchmarked model and neutralize that
    provider's per-agent overrides, restoring the prior environment on exit.
    LLMClient instances constructed inside the block (one per run_case worker)
    pick these up; instances constructed before (the judge) are unaffected."""
    keys = [
        "LLM_PROVIDER", "LLM_MODEL",
        f"{provider.upper()}_ANALYST_MODEL", f"{provider.upper()}_TRADER_MODEL",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    os.environ.pop(f"{provider.upper()}_ANALYST_MODEL", None)
    os.environ.pop(f"{provider.upper()}_TRADER_MODEL", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): per-model env swap with override neutralization"
```

---

### Task 4: Suite runner and CLI with fairness preflights

**Files:**
- Modify: `src/vibe_trading/eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `load_cases`, `run_case` from `vibe_trading.eval.runner`; `build_judge`, `score_case`, `CaseScore` from `vibe_trading.eval.scorer`; `SuiteReport.from_scores`; `LLMClient.set_cost_sink`, `_PROVIDER_API_KEY_ENV`, `get_litellm_model_string` from `vibe_trading.agents.client`; Tasks 1–3.
- Produces: `run_model(spec, cases, judge, max_workers, analyst_path) -> tuple[SuiteReport, list[CostEvent]]`; `main(argv: Optional[list[str]] = None) -> int`; `if __name__ == "__main__"` entry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark.py`:

```python
from vibe_trading.eval import benchmark


def test_main_refuses_without_judge_model(monkeypatch, capsys):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    rc = benchmark.main(["--models", "gemini/gemma-4-31b-it"])
    assert rc == 2
    assert "EVAL_JUDGE_MODEL" in capsys.readouterr().err


def test_main_refuses_on_missing_api_key(monkeypatch, capsys):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    rc = benchmark.main(["--models", "gemini/gemma-4-31b-it,openai/gpt-5.2-mini"])
    assert rc == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_happy_path_writes_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_report = make_report(overall=0.8)
    fake_events = [make_event("gemini/gemma-4-31b-it", cost=0.02)]
    monkeypatch.setattr(benchmark, "load_cases", lambda p: ["case"] * 3)
    monkeypatch.setattr(benchmark, "build_judge", lambda: (lambda text, rubric: None))
    monkeypatch.setattr(benchmark, "run_model",
                        lambda spec, cases, judge, max_workers, analyst_path:
                        (fake_report, fake_events))

    out_md = tmp_path / "BENCHMARK.md"
    rc = benchmark.main([
        "--models", "gemini/gemma-4-31b-it",
        "--reports-dir", str(tmp_path),
        "--output-md", str(out_md),
    ])
    assert rc == 0
    assert out_md.exists()
    assert "gemma-4-31b-it" in out_md.read_text()
    assert list(tmp_path.glob("benchmark-*.json"))


def test_run_model_installs_and_clears_cost_sink(monkeypatch):
    from vibe_trading.agents.client import LLMClient
    installed = []
    monkeypatch.setattr(LLMClient, "set_cost_sink",
                        classmethod(lambda cls, s: installed.append(s)))
    monkeypatch.setattr(benchmark, "run_case",
                        lambda case, db, analyst_path: None)
    monkeypatch.setattr(benchmark, "score_case",
                        lambda result, case, judge: CaseScore(
                            case_id="c1", schema_ok=True, field_scores=[],
                            analyst_score=1.0, trader_score=1.0, total_score=1.0))
    monkeypatch.setattr(benchmark, "Database", lambda: object())
    report, events = benchmark.run_model(
        "gemini/gemma-4-31b-it", cases=[type("C", (), {"id": "c1"})()],
        judge=lambda t, r: None, max_workers=1, analyst_path="snapshot")
    assert report.case_count == 1
    # first call installs the InMemoryCostSink, last call clears it back to None
    assert isinstance(installed[0], InMemoryCostSink)
    assert installed[-1] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v -k "main or run_model"`
Expected: FAIL with `AttributeError: module ... has no attribute 'main'`

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/eval/benchmark.py`:

```python
import argparse
import concurrent.futures
import sys

from vibe_trading.agents.client import (
    LLMClient, _PROVIDER_API_KEY_ENV, get_litellm_model_string,
)
from vibe_trading.data.db import Database
from vibe_trading.eval.runner import load_cases, run_case
from vibe_trading.eval.scorer import build_judge, score_case, CaseScore


def run_model(spec: str, cases: list, judge, max_workers: int,
              analyst_path: str) -> tuple[SuiteReport, list[CostEvent]]:
    """Run the full suite once for one model spec, capturing its cost events.
    Mirrors eval.main's concurrent loop; one bad case never aborts the run."""
    provider, model = parse_model_spec(spec)
    sink = InMemoryCostSink()
    LLMClient.set_cost_sink(sink)
    try:
        with model_env(provider, model):
            def _process(case) -> CaseScore:
                try:
                    result = run_case(case, Database(), analyst_path=analyst_path)
                    return score_case(result, case, judge)
                except Exception as e:
                    logger.warning(f"Case {case.id} crashed during scoring: {e}")
                    return CaseScore(case_id=case.id, schema_ok=False, field_scores=[],
                                     analyst_score=0.0, trader_score=0.0,
                                     total_score=0.0, error=str(e))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                scores = list(pool.map(_process, cases))
    finally:
        LLMClient.set_cost_sink(None)
    return SuiteReport.from_scores(scores), sink.events


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="vibe-benchmark")
    parser.add_argument("--models", required=True,
                        help="Comma-separated litellm-format specs, e.g. "
                             "'gemini/gemini-3.1-flash-lite,openai/gpt-5.2-mini'")
    parser.add_argument("--snapshots", type=Path, default=Path("evals/snapshots"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--output-md", type=Path, default=Path("evals/BENCHMARK.md"))
    parser.add_argument("--throttle-seconds", type=float, default=4.5)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--analyst-path", choices=["snapshot", "tool-loop"],
                        default="tool-loop",
                        help="tool-loop (default) benchmarks the production path.")
    args = parser.parse_args(argv)

    specs = [s.strip() for s in args.models.split(",") if s.strip()]

    # Fairness preflight 1: the judge must be pinned, or each contestant would
    # grade its own homework (scorer falls back to the client's model).
    judge_model = os.getenv("EVAL_JUDGE_MODEL")
    if not judge_model:
        print("ERROR: EVAL_JUDGE_MODEL must be set for a benchmark run so every "
              "model is graded by the same judge. It must be a model hosted by "
              "the ambient LLM_PROVIDER.", file=sys.stderr)
        return 2

    # Fairness preflight 2: fail fast on missing API keys, before burning any calls.
    missing = []
    for spec in specs:
        provider, _ = parse_model_spec(spec)
        key_env = _PROVIDER_API_KEY_ENV.get(provider)
        if key_env and not os.getenv(key_env):
            missing.append(f"{provider}: {key_env}")
    if missing:
        print(f"ERROR: missing API key env vars: {', '.join(sorted(set(missing)))}",
              file=sys.stderr)
        return 2

    if args.throttle_seconds > 0:
        os.environ["LLM_MIN_CALL_INTERVAL_SECONDS"] = str(args.throttle_seconds)

    cases = load_cases(args.snapshots)
    if not cases:
        print(f"ERROR: no cases found in {args.snapshots}", file=sys.stderr)
        return 1

    # Build the judge ONCE, before any model_env mutation: its LLMClient captures
    # the ambient provider now, so per-model env swaps cannot reroute it.
    judge = build_judge()

    entries: list[BenchmarkEntry] = []
    for spec in specs:
        provider, model = parse_model_spec(spec)
        logger.info(f"Benchmarking {spec} over {len(cases)} cases "
                    f"({args.analyst_path} path)...")
        report, events = run_model(spec, cases, judge,
                                   args.max_workers, args.analyst_path)
        entries.append(build_entry(get_litellm_model_string(provider, model),
                                   report, events))

    run_at_iso = datetime.now(timezone.utc).isoformat()
    report_path = write_benchmark_report(entries, judge_model, args.reports_dir)
    md = render_markdown(entries, judge_model, run_at_iso)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md)
    print(md)
    print(f"JSON report: {report_path}")
    print(f"Markdown table: {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test file**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: 14 passed

- [ ] **Step 5: Run the whole suite to check for regressions**

Run: `uv run pytest`
Expected: all tests pass (285+ passed, 0 failed)

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/eval/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): suite runner + CLI with judge/API-key preflights"
```

---

### Task 5: README section and first committed benchmark

**Files:**
- Modify: `README.md` (in the eval-harness section, after the existing eval usage docs)
- Create: `evals/BENCHMARK.md` (generated by the real run)

**Interfaces:**
- Consumes: the Task 4 CLI.
- Produces: committed `evals/BENCHMARK.md`; README subsection linking to it.

- [ ] **Step 1: Add the README subsection**

Locate the eval-harness section in `README.md` (search for the heading that documents `python -m vibe_trading.eval.eval`) and append this subsection after it:

```markdown
### Model benchmark matrix

Run the full golden-set suite across several models and compare quality vs cost:

    EVAL_JUDGE_MODEL=gemini-3.1-flash-lite \
    python -m vibe_trading.eval.benchmark \
      --models "gemini/gemini-3.1-flash-lite,gemini/gemma-4-31b-it" \
      --throttle-seconds 4.5

- `EVAL_JUDGE_MODEL` is **required** (and pinned for the whole run) so every model
  is graded by the same judge — the scorer's per-run fallback would otherwise let
  each contestant grade its own homework.
- The judge must be hosted by the ambient `LLM_PROVIDER` at launch.
- Results: a timestamped JSON report under `data/reports/` and a regenerated
  [evals/BENCHMARK.md](evals/BENCHMARK.md) table (sorted by score-per-dollar).
- The committed regression baseline (`evals/baseline.json`) is never touched.
```

- [ ] **Step 2: Run the real benchmark (manual, requires API keys)**

Run:
```bash
EVAL_JUDGE_MODEL=gemini-3.1-flash-lite uv run python -m vibe_trading.eval.benchmark \
  --models "gemini/gemini-3.1-flash-lite,gemini/gemma-4-31b-it" \
  --throttle-seconds 4.5
```
Expected: table printed to stdout; `evals/BENCHMARK.md` created; one `benchmark-*.json` under `data/reports/`. Add more providers (`openai/...`, `anthropic/...`, `groq/...`) as keys are available — each provider's key is validated up front.

- [ ] **Step 3: Verify baseline untouched**

Run: `git status --short evals/baseline.json`
Expected: no output (unmodified).

- [ ] **Step 4: Commit**

```bash
git add README.md evals/BENCHMARK.md
git commit -m "docs(benchmark): README usage + first committed benchmark table"
```

---

## Self-review notes

- Spec coverage: judge-fairness invariant (Task 4 preflight + judge-before-mutation), cost from in-process events (Task 1 sink + Task 4 install/clear), baseline isolation (no baseline imports; Task 5 Step 3 verifies), throttle reuse (Task 4), markdown + JSON outputs (Task 2), README link (Task 5). Per-provider throttle override from the spec's "allow a per-model override" is deliberately dropped for YAGNI — a single `--throttle-seconds` governs all runs; revisit only if a mixed-provider run actually hits limits.
- The judge-cost separation caveat (judge model == contestant) is documented in `build_entry`'s docstring, the markdown footer, and the JSON report's per-entry `judge_cost_usd`.
