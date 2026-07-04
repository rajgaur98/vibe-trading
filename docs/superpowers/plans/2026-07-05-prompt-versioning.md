# Prompt Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM call, decision, and eval report records exactly which prompt produced it, and prompts cannot be edited without a deliberate version bump.

**Architecture:** A new in-repo registry (`src/vibe_trading/agents/prompts.py`) becomes the single home for the analyst/trader/judge system prompts, each a `PromptSpec` with a hand-bumped version and a computed content hash. A committed pins fixture plus a test makes silent edits fail CI. The stamp string (`name:vN@sha12`) is threaded through `CostEvent` → `llm_cost_log`, stamped onto `decision_log` rows (as a bundle of all prompts), attached to Langfuse trace metadata, and recorded in eval reports/baseline.

**Tech Stack:** Python 3.12, `hashlib.sha256`, pydantic v2, existing idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration lists in `db.py`, pytest.

**Workstream:** B of `docs/superpowers/specs/2026-07-04-ai-deepening-roadmap-design.md`. No dependency on A; workstream C depends on this plan being done.

## Global Constraints

- Prompt texts move **verbatim** (byte-identical) — this plan must not change any model-visible behavior, so the committed eval baseline stays valid without a re-run.
- Stamp format is exactly `"{name}:{version}@{sha}"` where `sha = sha256(text).hexdigest()[:12]` — e.g. `analyst_system:v1@a1b2c3d4e5f6`. Workstream C's digest segments by this string; do not vary the format.
- New DB columns are nullable `VARCHAR`, added via the existing idempotent migration lists (both the DuckDB list in `Database._create_tables` and the Postgres list in `PostgresDatabase._create_tables`) — never by editing the `CREATE TABLE` statements alone, since deployed tables already exist.
- No external prompt-management service; the registry lives in the repo so prompt changes are reviewable in PRs.
- `evals/baseline.json` is NOT re-seeded by this plan. The `prompt_versions` map appears in it on the next legitimate `--update-baseline`.
- All tests offline: no network, no live LLM, no Postgres. Run tests with `uv run pytest <path> -v`.

---

### Task 1: PromptSpec and the pins mechanism

**Files:**
- Create: `src/vibe_trading/agents/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PromptSpec` (frozen dataclass: `name: str`, `version: str`, `text: str`; properties `sha -> str` (12 hex chars) and `stamp -> str`); `check_pins(specs: list[PromptSpec], pins: dict) -> list[str]` (returns human-readable violations, empty when clean); `pins_payload(specs: list[PromptSpec]) -> dict`. Task 2 registers the real prompts and adds `REGISTRY`, `versions_map()`, `bundle_version()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompts.py
import hashlib

from vibe_trading.agents.prompts import PromptSpec, check_pins, pins_payload


def spec(name="p1", version="v1", text="hello") -> PromptSpec:
    return PromptSpec(name=name, version=version, text=text)


def test_sha_is_first_12_hex_of_sha256():
    s = spec(text="hello")
    assert s.sha == hashlib.sha256(b"hello").hexdigest()[:12]
    assert len(s.sha) == 12


def test_stamp_format():
    s = spec(name="analyst_system", version="v3", text="hello")
    assert s.stamp == f"analyst_system:v3@{s.sha}"


def test_pins_payload_round_trip():
    s = spec()
    pins = pins_payload([s])
    assert pins == {"p1": {"version": "v1", "sha": s.sha}}
    assert check_pins([s], pins) == []


def test_text_edit_without_bump_is_a_violation():
    pinned = pins_payload([spec(text="original")])
    violations = check_pins([spec(text="EDITED")], pinned)
    assert len(violations) == 1
    assert "bump" in violations[0]  # tells the dev to bump the version


def test_version_bump_requires_pin_regen():
    pinned = pins_payload([spec(version="v1")])
    violations = check_pins([spec(version="v2")], pinned)
    assert len(violations) == 1
    assert "regenerate" in violations[0]  # deliberate: regen pins after a bump


def test_unpinned_and_orphaned_prompts_are_violations():
    pinned = pins_payload([spec(name="old_prompt")])
    violations = check_pins([spec(name="new_prompt")], pinned)
    assert len(violations) == 2  # new_prompt unpinned + old_prompt orphaned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vibe_trading.agents.prompts'`

- [ ] **Step 3: Write the implementation**

```python
# src/vibe_trading/agents/prompts.py
"""The single home for LLM prompt text, with hand-bumped versions and computed
content hashes so every call/decision/eval can be traced to the exact prompt that
produced it — and so a prompt cannot be edited without a deliberate version bump
(enforced by tests/test_prompts.py against tests/fixtures/prompt_pins.json).

Workflow for changing a prompt:
  1. Edit the text AND bump the spec's `version` (v1 -> v2).
  2. Regenerate pins:  python -m vibe_trading.agents.prompts --write-pins tests/fixtures/prompt_pins.json
  3. Run the eval regression gate; re-seed the baseline only via --update-baseline.
"""
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    name: str          # registry key, e.g. "analyst_system"
    version: str       # hand-bumped on any semantic change, e.g. "v1"
    text: str

    @property
    def sha(self) -> str:
        """First 12 hex chars of sha256(text) — catches unbumped edits."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    @property
    def stamp(self) -> str:
        """The traceability string recorded everywhere: 'name:vN@sha12'."""
        return f"{self.name}:{self.version}@{self.sha}"


def pins_payload(specs: list) -> dict:
    """The committed pins fixture content for the given specs."""
    return {s.name: {"version": s.version, "sha": s.sha} for s in specs}


def check_pins(specs: list, pins: dict) -> list:
    """Compare the live registry against the committed pins. Returns human-readable
    violations; empty list means clean. Distinguishes 'you edited text without
    bumping' from 'you bumped — now regenerate pins deliberately'."""
    violations = []
    by_name = {s.name: s for s in specs}
    for s in specs:
        pinned = pins.get(s.name)
        if pinned is None:
            violations.append(
                f"{s.name}: not pinned — regenerate pins (--write-pins).")
        elif s.version != pinned["version"]:
            violations.append(
                f"{s.name}: version {pinned['version']} -> {s.version} — "
                f"regenerate pins to acknowledge the bump (--write-pins).")
        elif s.sha != pinned["sha"]:
            violations.append(
                f"{s.name}: text changed but version is still {s.version} — "
                f"bump the version, then regenerate pins.")
    for name in pins:
        if name not in by_name:
            violations.append(
                f"{name}: pinned but no longer registered — regenerate pins.")
    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): PromptSpec registry primitives + pins enforcement logic"
```

---

### Task 2: Move the three prompts into the registry and pin them

**Files:**
- Modify: `src/vibe_trading/agents/prompts.py`
- Modify: `src/vibe_trading/agents/analyst.py` (the `self.system_instruction = """..."""` block in `TechnicalVolumeAnalyst.__init__`)
- Modify: `src/vibe_trading/agents/trader.py` (the `self.system_instruction = """..."""` block in `HeadTrader.__init__`)
- Modify: `src/vibe_trading/eval/scorer.py` (the module-level `_JUDGE_SYSTEM = """...""".strip()`)
- Create: `tests/fixtures/prompt_pins.json` (generated)
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: Task 1's `PromptSpec`, `check_pins`, `pins_payload`.
- Produces: module constants `ANALYST_SYSTEM: PromptSpec`, `TRADER_SYSTEM: PromptSpec`, `JUDGE_SYSTEM: PromptSpec`; `REGISTRY: dict[str, PromptSpec]`; `versions_map() -> dict[str, str]` (name → stamp); `bundle_version() -> str`; `--write-pins` CLI. Tasks 4–6 and workstream C rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
import json
from pathlib import Path

from vibe_trading.agents import prompts

PINS_PATH = Path("tests/fixtures/prompt_pins.json")


def test_registry_contains_the_three_system_prompts():
    assert set(prompts.REGISTRY) == {"analyst_system", "trader_system", "judge_system"}
    for s in prompts.REGISTRY.values():
        assert s.text.strip()  # non-empty, real content


def test_registered_texts_are_the_ones_agents_use():
    from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
    from vibe_trading.agents.trader import HeadTrader
    from vibe_trading.eval import scorer
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    assert TechnicalVolumeAnalyst(db=None, fetcher=None).system_instruction \
        == prompts.ANALYST_SYSTEM.text
    assert HeadTrader().system_instruction == prompts.TRADER_SYSTEM.text
    assert scorer._JUDGE_SYSTEM == prompts.JUDGE_SYSTEM.text


def test_versions_map_and_bundle_version():
    vm = prompts.versions_map()
    assert vm["analyst_system"] == prompts.ANALYST_SYSTEM.stamp
    # bundle covers ONLY the decision-path prompts (analyst + trader), ';'-joined in
    # name order — judge prompts must not perturb the decision-level stamp
    assert prompts.bundle_version() == ";".join(
        [prompts.ANALYST_SYSTEM.stamp, prompts.TRADER_SYSTEM.stamp])


def test_pins_are_current():
    """THE enforcement test: editing any prompt text without bumping its version
    (or bumping without regenerating pins) fails here with instructions."""
    pins = json.loads(PINS_PATH.read_text())
    violations = prompts.check_pins(list(prompts.REGISTRY.values()), pins)
    assert violations == [], "\n".join(violations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 4 new FAILs (`AttributeError: ... no attribute 'REGISTRY'`)

- [ ] **Step 3: Register the prompts (verbatim moves) and add the CLI**

Append to `src/vibe_trading/agents/prompts.py`:

```python
# ---------------------------------------------------------------------------
# The registry. Texts are moved VERBATIM from their original homes — byte-
# identical, so this refactor cannot shift eval scores or live behavior:
#   analyst_system: src/vibe_trading/agents/analyst.py  (TechnicalVolumeAnalyst.__init__)
#   trader_system:  src/vibe_trading/agents/trader.py   (HeadTrader.__init__)
#   judge_system:   src/vibe_trading/eval/scorer.py     (_JUDGE_SYSTEM)
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = PromptSpec(
    name="analyst_system",
    version="v1",
    text="""
You are an elite Crypto Technical and Volume Analyst specializing in swing trading.
<... the ENTIRE existing analyst system prompt, copied byte-for-byte from
 analyst.py — from the opening triple-quote's newline through the closing
 schema example. Do NOT retype it; cut and paste the exact string literal. ...>
""",
)

TRADER_SYSTEM = PromptSpec(
    name="trader_system",
    version="v1",
    text="""
You are the Head Trader of a systematic crypto SWING-trading hedge fund.
<... the ENTIRE existing trader system prompt, copied byte-for-byte from
 trader.py, including the HOUSE METHODOLOGY block. ...>
""",
)

JUDGE_SYSTEM = PromptSpec(
    name="judge_system",
    version="v1",
    # scorer.py applies .strip() to its literal; bake the stripped form in here so
    # the registered text equals what the judge actually sends.
    text="""
You are a meticulous, code-review-style evaluator. <... byte-for-byte from
 scorer.py's _JUDGE_SYSTEM literal ...>
""".strip(),
)

REGISTRY: dict = {s.name: s for s in (ANALYST_SYSTEM, TRADER_SYSTEM, JUDGE_SYSTEM)}


def versions_map() -> dict:
    """{prompt_name: stamp} — recorded in eval reports and the baseline."""
    return {name: s.stamp for name, s in REGISTRY.items()}


# The prompts that actually shape a trading decision. Judge prompts (eval or
# online) are deliberately excluded: bumping a judge must never make decisions
# look like they ran under a different prompt state.
DECISION_PROMPT_NAMES = ("analyst_system", "trader_system")


def bundle_version() -> str:
    """Single decision-level stamp covering the decision-path prompts, ';'-joined
    in name order. Stamped onto decision_log rows."""
    return ";".join(REGISTRY[name].stamp for name in sorted(DECISION_PROMPT_NAMES))


if __name__ == "__main__":
    import argparse
    import json as _json
    from pathlib import Path as _Path

    _p = argparse.ArgumentParser(prog="vibe-prompts")
    _p.add_argument("--write-pins", type=_Path, default=None,
                    help="Write the pins fixture for the current registry state.")
    _args = _p.parse_args()
    if _args.write_pins:
        _args.write_pins.parent.mkdir(parents=True, exist_ok=True)
        _args.write_pins.write_text(
            _json.dumps(pins_payload(list(REGISTRY.values())), indent=2) + "\n")
        print(f"Pins written: {_args.write_pins}")
    else:
        for _name in sorted(REGISTRY):
            print(REGISTRY[_name].stamp)
```

Then wire the consumers (each replaces its inline literal):

In `src/vibe_trading/agents/analyst.py` — add `from vibe_trading.agents import prompts` to the imports, delete the entire `self.system_instruction = """..."""` literal in `__init__`, and replace with:

```python
        self.system_instruction = prompts.ANALYST_SYSTEM.text
```

In `src/vibe_trading/agents/trader.py` — same pattern:

```python
        self.system_instruction = prompts.TRADER_SYSTEM.text
```

In `src/vibe_trading/eval/scorer.py` — delete the `_JUDGE_SYSTEM = """...""".strip()` literal and replace with:

```python
from vibe_trading.agents import prompts

_JUDGE_SYSTEM = prompts.JUDGE_SYSTEM.text
```

- [ ] **Step 4: Generate the pins fixture**

Run: `uv run python -m vibe_trading.agents.prompts --write-pins tests/fixtures/prompt_pins.json`
Expected: `Pins written: tests/fixtures/prompt_pins.json` (file contains three entries with `version: v1` and 12-char shas)

- [ ] **Step 5: Run the prompt tests, then the whole suite**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 10 passed

Run: `uv run pytest`
Expected: all tests pass — existing trader/analyst/eval tests exercise the moved prompts; any failure means the move was not verbatim. Fix by re-copying the exact literal, not by editing tests.

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/agents/prompts.py src/vibe_trading/agents/analyst.py \
        src/vibe_trading/agents/trader.py src/vibe_trading/eval/scorer.py \
        tests/fixtures/prompt_pins.json tests/test_prompts.py
git commit -m "feat(prompts): registry is the single home for the 3 system prompts, pinned"
```

---

### Task 3: prompt_version on CostEvent, llm_cost_log, and the client

**Files:**
- Modify: `src/vibe_trading/agents/cost.py` (`CostEvent`, `CostEvent.build`, `PostgresCostLogger.record`)
- Modify: `src/vibe_trading/data/db.py` (Postgres migration list)
- Modify: `src/vibe_trading/agents/client.py` (`_build_cost_event`, `_emit_cost`, `_defer_cost`, `call_llm`, `call_llm_with_tools`)
- Test: `tests/test_cost.py`, `tests/test_client_guardrails.py` (extend)

**Interfaces:**
- Consumes: Task 2's stamps (only in tests here; agents wire them in Task 4).
- Produces: `CostEvent.prompt_version: Optional[str]` (default `None`); `CostEvent.build(..., prompt_version=None)`; `LLMClient.call_llm(..., prompt_version: Optional[str] = None)`; `LLMClient.call_llm_with_tools(..., prompt_version: Optional[str] = None)`. Task 4 and workstream C rely on these signatures.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cost.py`:

```python
def test_cost_event_carries_prompt_version():
    from vibe_trading.agents.cost import CostEvent
    e = CostEvent.build(provider="gemini", model="gemini/m", call_type="single",
                        prompt_tokens=10, completion_tokens=5, latency_ms=100.0,
                        prompt_version="analyst_system:v1@abcdef123456")
    assert e.prompt_version == "analyst_system:v1@abcdef123456"
    # default stays None so untagged calls don't fabricate a version
    e2 = CostEvent.build(provider="gemini", model="gemini/m", call_type="single",
                         prompt_tokens=10, completion_tokens=5, latency_ms=100.0)
    assert e2.prompt_version is None


def test_postgres_cost_logger_writes_prompt_version_column():
    from unittest.mock import MagicMock
    from vibe_trading.agents.cost import CostEvent, PostgresCostLogger
    fake_db = MagicMock()
    logger_ = PostgresCostLogger(db=fake_db)
    e = CostEvent.build(provider="gemini", model="gemini/m", call_type="single",
                        prompt_tokens=1, completion_tokens=1, latency_ms=1.0,
                        prompt_version="trader_system:v1@000000000000")
    logger_.record(e)
    sql = fake_db.conn.execute.call_args[0][0]
    params = fake_db.conn.execute.call_args[0][1]
    assert "prompt_version" in sql
    assert "trader_system:v1@000000000000" in params
```

Append to `tests/test_client_guardrails.py`:

```python
def test_call_llm_threads_prompt_version_into_cost_event(monkeypatch):
    import litellm
    from vibe_trading.agents.client import LLMClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = []

    class Sink:
        def record(self, event):
            captured.append(event)

    class FakeMsg:
        content = "{}"
    class FakeChoice:
        message = FakeMsg()
    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    monkeypatch.setattr(litellm, "completion", lambda **kw: FakeResponse())
    LLMClient.set_cost_sink(Sink())
    try:
        client = LLMClient()
        client.call_llm(model_name="m", system_instruction="s", prompt="p",
                        prompt_version="analyst_system:v1@abcdef123456")
    finally:
        LLMClient.set_cost_sink(None)
    assert captured[0].prompt_version == "analyst_system:v1@abcdef123456"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py tests/test_client_guardrails.py -v -k prompt_version`
Expected: FAIL (`TypeError: build() got an unexpected keyword argument 'prompt_version'` etc.)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/agents/cost.py`:

1. Add to `CostEvent` fields (after `schema_ok`):

```python
    # Stamp of the prompt that drove this call ("name:vN@sha12", see agents/prompts.py).
    # None for calls with no registered prompt (embeddings, legacy call sites).
    prompt_version: Optional[str] = None
```

2. Extend `CostEvent.build` — add `prompt_version: Optional[str] = None` to the signature and `prompt_version=prompt_version,` to the constructor call.

3. In `PostgresCostLogger.record`, extend the INSERT to 14 columns:

```python
            self.pg_db.conn.execute(
                """INSERT OR IGNORE INTO llm_cost_log
                   (call_id, timestamp, provider, model, call_type,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms,
                    cache_read_tokens, cache_write_tokens, schema_ok, prompt_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.call_id, event.timestamp, event.provider, event.model, event.call_type,
                 event.prompt_tokens, event.completion_tokens, event.total_tokens,
                 event.cost_usd, event.latency_ms,
                 event.cache_read_tokens, event.cache_write_tokens, event.schema_ok,
                 event.prompt_version),
            )
```

In `src/vibe_trading/data/db.py`, `PostgresDatabase._create_tables`: add `prompt_version VARCHAR` to the `llm_cost_log` CREATE TABLE column list, and append to the idempotent migration tuple:

```python
                "ALTER TABLE llm_cost_log ADD COLUMN IF NOT EXISTS prompt_version VARCHAR",
```

In `src/vibe_trading/agents/client.py`, thread the kwarg through the private builders and both public calls:

1. `_build_cost_event(self, response, model_str, call_type, latency_ms, schema_ok=None, prompt_version=None)` — pass `prompt_version=prompt_version` into `CostEvent.build`.
2. `_emit_cost(...)` and `_defer_cost(...)` each gain `prompt_version: Optional[str] = None` and forward it to `_build_cost_event`.
3. `call_llm(self, model_name, system_instruction, prompt, response_schema=None, prompt_version: Optional[str] = None)` — forward to the `_defer_cost` / `_emit_cost` call at the end.
4. `call_llm_with_tools(..., expect_schema=False, prompt_version: Optional[str] = None)` — forward to all three `_defer_cost` / `_emit_cost` call sites in the loop (intermediate turns and the final answer all carry the same prompt's stamp).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py tests/test_client_guardrails.py -v`
Expected: all pass, including pre-existing tests (defaults keep old call sites valid)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/cost.py src/vibe_trading/agents/client.py \
        src/vibe_trading/data/db.py tests/test_cost.py tests/test_client_guardrails.py
git commit -m "feat(prompts): thread prompt_version through CostEvent into llm_cost_log"
```

---

### Task 4: Agents and judge stamp their calls (+ Langfuse metadata)

**Files:**
- Modify: `src/vibe_trading/agents/analyst.py` (`analyze`)
- Modify: `src/vibe_trading/agents/trader.py` (`decide`)
- Modify: `src/vibe_trading/eval/scorer.py` (`build_judge`'s inner `judge`)
- Test: `tests/test_prompts.py` (extend)

**Interfaces:**
- Consumes: Task 2's `prompts.ANALYST_SYSTEM.stamp` / `TRADER_SYSTEM.stamp` / `JUDGE_SYSTEM.stamp`; Task 3's `prompt_version=` kwargs.
- Produces: every production/eval LLM call carries its stamp; Langfuse spans carry `prompt_version` in metadata.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`:

```python
def test_trader_stamps_its_calls(monkeypatch):
    import os
    from unittest.mock import MagicMock
    from vibe_trading.agents.trader import HeadTrader
    from vibe_trading.agents.analyst import AnalystOutput

    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"action": "flat", "stop_loss_strategy": "1.5_atr", '
        '"take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, '
        '"hold_period_bias": "medium", "reasoning_summary": "r"}'
    )
    trader = HeadTrader(client=client)
    analyst_output = AnalystOutput(
        market_bias="neutral", volume_confirmation="weak", thesis="t",
        nearest_support=1.0, nearest_resistance=2.0, confluence_score=0.5)
    trader.decide("BTC/USDT", analyst_output, {}, [], current_price=1.5)
    assert client.call_llm.call_args.kwargs["prompt_version"] \
        == prompts.TRADER_SYSTEM.stamp


def test_analyst_snapshot_path_stamps_its_calls(monkeypatch):
    import os
    from unittest.mock import MagicMock
    from vibe_trading.agents.analyst import TechnicalVolumeAnalyst

    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"market_bias": "neutral", "volume_confirmation": "weak", "thesis": "t", '
        '"nearest_support": 1.0, "nearest_resistance": 2.0, "confluence_score": 0.5}'
    )
    analyst = TechnicalVolumeAnalyst(client=client, db=None, fetcher=None)
    analyst.analyze("BTC/USDT", snapshot={"close": 1.5})
    assert client.call_llm.call_args.kwargs["prompt_version"] \
        == prompts.ANALYST_SYSTEM.stamp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v -k stamps`
Expected: FAIL (`prompt_version` not in `call_args.kwargs`)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/agents/trader.py`, `decide()`:

1. Extend the `propagate_attributes` metadata:

```python
            metadata={"symbol": symbol, "prompt_version": prompts.TRADER_SYSTEM.stamp}
```

2. In `_call_single`, add the kwarg to the client call:

```python
            def _call_single(extra: str = "") -> str:
                return self.client.call_llm(
                    model_name=self.model,
                    system_instruction=self.system_instruction,
                    prompt=prompt + extra,
                    response_schema=HeadTraderOutput,
                    prompt_version=prompts.TRADER_SYSTEM.stamp,
                )
```

In `src/vibe_trading/agents/analyst.py`, `analyze()` — same pattern with `prompts.ANALYST_SYSTEM.stamp`: extend the `propagate_attributes` metadata dict, and add `prompt_version=prompts.ANALYST_SYSTEM.stamp` to **both** the `call_llm_with_tools(...)` call in `_call_tool_loop` and the `call_llm(...)` call in `_call_single`.

In `src/vibe_trading/eval/scorer.py`, inside `build_judge`'s `judge` closure, add the kwarg to the `client.call_llm(...)` call:

```python
            prompt_version=prompts.JUDGE_SYSTEM.stamp,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py tests/test_trader.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/analyst.py src/vibe_trading/agents/trader.py \
        src/vibe_trading/eval/scorer.py tests/test_prompts.py
git commit -m "feat(prompts): agents + judge stamp every LLM call and Langfuse span"
```

---

### Task 5: Stamp decision_log rows with the bundle version

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py` (the `decision_log` INSERT in `sync_and_evaluate`)
- Modify: `src/vibe_trading/data/db.py` (both migration lists + both `decision_log` CREATE TABLE column lists)
- Test: `tests/test_scheduler.py` (extend), `tests/test_db.py` (extend)

**Interfaces:**
- Consumes: Task 2's `prompts.bundle_version()`.
- Produces: `decision_log.prompt_version VARCHAR` populated on every new decision. Workstream C copies it into `decision_scores` and segments the drift digest by it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
def test_duckdb_decision_log_has_prompt_version(tmp_path):
    from vibe_trading.data.db import Database
    db = Database(db_path=str(tmp_path / "t.db"))
    db.connect()
    try:
        cols = [r[1] for r in db.conn.execute(
            "PRAGMA table_info('decision_log')").fetchall()]
        assert "prompt_version" in cols
    finally:
        db.close()
```

Append to `tests/test_scheduler.py` (follow the file's existing mocking style for `TradingScheduler` — broker/fetcher/pipeline mocked, `pg_db.conn.execute` captured):

```python
def test_decision_log_insert_carries_bundle_prompt_version(scheduler_with_mocks):
    """After a tick that logs one decision, the decision_log INSERT's SQL names
    prompt_version and its params include prompts.bundle_version()."""
    from vibe_trading.agents import prompts
    scheduler, pg_execute_calls = scheduler_with_mocks  # existing fixture pattern
    scheduler.sync_and_evaluate()
    insert_calls = [c for c in pg_execute_calls
                    if "INSERT OR IGNORE INTO decision_log" in c.args[0]]
    assert insert_calls, "no decision_log INSERT captured"
    sql, params = insert_calls[0].args[0], insert_calls[0].args[1]
    assert "prompt_version" in sql
    assert prompts.bundle_version() in params
```

(If `tests/test_scheduler.py` has no reusable fixture for a mocked tick, extend its existing decision-logging test instead — the assertion content is what matters: SQL names the column, params contain the bundle stamp.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py tests/test_scheduler.py -v -k prompt_version`
Expected: FAIL (column missing / stamp not in params)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/data/db.py`:

1. Add `prompt_version VARCHAR` as the last column of **both** `decision_log` CREATE TABLE statements (DuckDB `Database._create_tables` and `PostgresDatabase._create_tables`), with the comment `-- prompts.bundle_version() at decision time`.
2. Append to **both** idempotent migration tuples:

```python
            "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS prompt_version VARCHAR",
```

In `src/vibe_trading/runtime/scheduler.py`:

1. Add the import: `from vibe_trading.agents import prompts`
2. Extend the decision-log INSERT (11 columns now):

```python
                        self.pg_db.conn.execute("""
                            INSERT OR IGNORE INTO decision_log (decision_id, timestamp, symbol, action, stop_loss_strategy, take_profit_strategy, risk_reward_ratio, reasoning_summary, agent_transcripts, trace_id, prompt_version)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (proposal["decision_id"], proposal["timestamp"], proposal["symbol"], proposal["action"],
                              proposal["stop_loss_strategy"], proposal["take_profit_strategy"], float(proposal["risk_reward_ratio"]),
                              proposal["reasoning_summary"], json.dumps(snapshot, default=str), trace_id,
                              prompts.bundle_version()))
```

(`translate_query`'s `INSERT OR IGNORE INTO decision_log` mapping is column-agnostic — no change needed there.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py tests/test_scheduler.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/data/db.py src/vibe_trading/runtime/scheduler.py \
        tests/test_db.py tests/test_scheduler.py
git commit -m "feat(prompts): stamp decision_log rows with the prompt bundle version"
```

---

### Task 6: Eval reports and baseline carry prompt_versions

**Files:**
- Modify: `src/vibe_trading/eval/report.py` (`SuiteReport`, `write_baseline`)
- Test: `tests/test_eval.py` (extend)

**Interfaces:**
- Consumes: Task 2's `versions_map()`.
- Produces: `SuiteReport.prompt_versions: dict[str, str]` (auto-populated); `write_baseline` includes `"prompt_versions"` in the snapshot. `diff_against_baseline` is untouched — an old baseline without the key still diffs cleanly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
def test_suite_report_captures_prompt_versions():
    from vibe_trading.agents import prompts
    from vibe_trading.eval.report import SuiteReport
    report = SuiteReport.from_scores([])
    assert report.prompt_versions == prompts.versions_map()


def test_write_baseline_includes_prompt_versions(tmp_path):
    import json
    from vibe_trading.eval.report import SuiteReport, write_baseline
    report = SuiteReport.from_scores([])
    path = tmp_path / "baseline.json"
    write_baseline(report, path)
    data = json.loads(path.read_text())
    assert data["prompt_versions"] == report.prompt_versions


def test_diff_tolerates_baseline_without_prompt_versions():
    from vibe_trading.eval.report import SuiteReport, diff_against_baseline
    report = SuiteReport.from_scores([])
    old_baseline = {"overall_score": 0.0, "per_case": {}}  # pre-versioning shape
    diff = diff_against_baseline(report, old_baseline)
    assert diff.is_regression is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v -k prompt_versions`
Expected: FAIL (`SuiteReport` has no field `prompt_versions`)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/eval/report.py`:

1. Imports: add `from pydantic import Field` and `from vibe_trading.agents.prompts import versions_map`.
2. Add the field to `SuiteReport` (after `per_case`):

```python
    # {prompt_name: stamp} at run time — makes every report/baseline traceable to
    # the exact prompt state it certified. Auto-captured; from_scores need not set it.
    prompt_versions: dict[str, str] = Field(default_factory=versions_map)
```

3. In `write_baseline`, add to the `snapshot` dict (after `"schema_failures"`):

```python
        "prompt_versions": report.prompt_versions,
```

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run pytest tests/test_eval.py -v` — expected: all pass.
Run: `uv run pytest` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/report.py tests/test_eval.py
git commit -m "feat(prompts): eval reports + baseline record prompt_versions"
```

---

### Task 7: Document the safe-prompt-shipping workflow

**Files:**
- Modify: `README.md` (near the eval-harness docs), `ARCHITECTURE.md` (agent-layer section)

- [ ] **Step 1: Add the README subsection**

```markdown
### Prompt versioning

All system prompts live in `src/vibe_trading/agents/prompts.py` as versioned,
content-hashed `PromptSpec`s. Every LLM call records its prompt stamp
(`name:vN@sha12`) in `llm_cost_log.prompt_version`; every decision records the
full bundle in `decision_log.prompt_version`; every eval report and the committed
baseline record the `prompt_versions` map.

Changing a prompt:

1. Edit the text in `prompts.py` **and bump its `version`** — an unbumped edit
   fails `tests/test_prompts.py`.
2. Regenerate pins:
   `python -m vibe_trading.agents.prompts --write-pins tests/fixtures/prompt_pins.json`
3. Run the eval regression gate (`python -m vibe_trading.eval.eval`); re-seed via
   `--update-baseline` only for a reviewed, intentional change.
```

- [ ] **Step 2: Add two sentences to ARCHITECTURE.md's agent-layer section**

State that prompts are registry-owned and stamped end-to-end (cost log, decision log, Langfuse metadata, eval baseline), and reference the README workflow.

- [ ] **Step 3: Commit**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs(prompts): safe prompt-shipping workflow"
```

---

## Self-review notes

- Spec coverage: registry + hand-bumped version + computed hash (Task 1–2), cannot-edit-without-bump test (Task 2 `test_pins_are_current`), `llm_cost_log`/`decision_log` columns via idempotent migrations (Tasks 3, 5), Langfuse metadata (Task 4), eval report + baseline map (Task 6), workflow docs (Task 7). Baseline is stamped-but-not-reseeded, matching the spec's backwards-compatibility section.
- Type consistency: the stamp string format is defined once (`PromptSpec.stamp`, Task 1) and consumed by name everywhere else; `bundle_version()` (Task 2) is the only value written to `decision_log`.
- The spec's "test snapshots {name: (version, sha)}" idea is realized as the generated `tests/fixtures/prompt_pins.json` + `check_pins` so the plan never has to hardcode unknowable hashes.
