# Design Spec — AI Deepening Roadmap (model benchmarking, prompt versioning, online evals, retrieval upgrade)

> **Initiative:** close the four maturity gaps between this system and a production-grade LLM
> platform: (A) model choice is unmeasured, (B) prompts are unversioned, (C) evaluation stops
> at offline regression, and (D) precedent retrieval is unindexed and unmeasured. Four
> workstreams, sequenced so each one's measurement infrastructure exists before the thing it
> is meant to measure. Each workstream is independently shippable and independently valuable.

## Problem

The system already has strong *offline* discipline: a 34-case golden set with a committed
regression baseline (`evals/baseline.json`), an LLM-as-judge scorer, per-call cost telemetry
(`llm_cost_log`), Langfuse tracing joined to decisions via `decision_log.trace_id`, and a
journal RAG that attaches real or counterfactual outcomes to past decisions. What's missing
is the layer that production LLM systems live and die by:

1. **Model choice is folklore.** The README recommends models from experience, but there is
   no measured cost-vs-quality comparison. The multi-provider client and the eval harness
   make this comparison nearly free — it just hasn't been run and published.
2. **Prompts are unversioned.** The analyst/trader system prompts are inline strings; the
   only history is git. No row in `decision_log`, `llm_cost_log`, or an eval report can be
   traced to the exact prompt text that produced it, so "did the prompt change help?" is
   unanswerable after the fact.
3. **Evaluation ends offline.** The baseline gate catches regressions before merge, but no
   live decision is ever scored. The raw material exists — 500+ decisions with trace IDs,
   closed-trade PnL, and counterfactual forward returns — and nothing consumes it.
4. **Retrieval is unindexed and unmeasured.** `journal.py` brute-force cosine-ranks
   candidates in Python, retrieval quality has no metric (recall@k unknown), and there is no
   evidence precedent injection improves decisions (the eval deliberately excludes it — see
   the trade-journal-rag spec's *Eval-safety* section).

## Workstream A — Model benchmark matrix

**Goal:** one command that runs the golden set across N models and emits a cost-vs-quality
table; a committed artifact showing *why* the default model is the default.

### Design

- **`evals/benchmark.py` [NEW]** — thin orchestrator over the existing suite runner. Takes
  `--models gemini/gemini-3.1-flash-lite,openai/gpt-5.2-mini,...` (litellm format), runs the
  full tool-loop suite once per model, collects each `SuiteReport`, and writes:
  - `data/reports/benchmark-<ts>.json` — per-model: overall/analyst/trader score, pass rate,
    schema failures, prompt/completion tokens, cost USD, mean latency.
  - `evals/BENCHMARK.md` — generated markdown table, sorted by score-per-dollar, with the
    run date and case-set hash. Linked from the README.
- **Judge fairness invariant:** `EVAL_JUDGE_MODEL` **must be pinned** for a benchmark run.
  The scorer's fallback (judge defaults to the client's model when the env var is unset)
  would let each contestant grade its own homework. `benchmark.py` refuses to start if
  `EVAL_JUDGE_MODEL` is unset.
- **Cost attribution:** each model's run cost is accumulated from the in-process cost events
  of that run (not by querying `llm_cost_log` by time window, which would race with live
  trading if run on the same DB).
- **Baseline isolation:** benchmark mode never touches `evals/baseline.json`;
  `--update-baseline` is not plumbed through.
- **Throttle:** reuse the existing `--throttle-seconds` / shared-throttle machinery; allow a
  per-model override since rate limits differ wildly across providers.

### Testing

Aggregation and table generation are pure functions over `SuiteReport` fixtures — unit-test
with canned reports, no live calls. One smoke test that the CLI arg parsing rejects a
missing judge model.

## Workstream B — Prompt versioning

**Goal:** every LLM call, decision, and eval report records which prompt produced it.

### Design

- **`src/vibe_trading/agents/prompts.py` [NEW]** — the single home for prompt text. Each
  prompt (analyst system, trader system, judge rubric preamble) becomes a named registry
  entry with a hand-bumped semantic version and a content hash computed at import:

  ```python
  @dataclass(frozen=True)
  class PromptSpec:
      name: str          # "analyst_system"
      version: str       # "v3" — bumped by hand on any semantic change
      text: str
      sha: str           # sha256(text)[:12] — computed, catches unbumped edits
  ```

  `prompt_version()` returns `"analyst_system:v3@a1b2c3d4e5f6"` for stamping. A unit test
  asserts that every registered version string changes when its text hash changes (i.e. you
  cannot edit a prompt without bumping — the test snapshots `{name: (version, sha)}`).
- **Stamping** (all via the existing idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
  migration lists in `db.py`):
  - `llm_cost_log` + `decision_log` gain a `prompt_version VARCHAR` column.
  - Eval reports and `baseline.json` gain a `prompt_versions` map — a baseline becomes
    traceable to the exact prompt state it certified.
  - Langfuse: pass `prompt_version` through the existing `propagate_attributes` metadata so
    traces are filterable by prompt.
- **Workflow this enables:** edit prompt → bump version → run eval (regression gate) →
  `--update-baseline` records the new versions → deploy → online scores (Workstream C)
  segment by `prompt_version`. That is the full safe-prompt-shipping loop.
- **Out of scope:** external prompt-management services (Langfuse prompt management is a
  later option; an in-repo registry keeps prompts reviewable in PRs, which is the point).

### Testing

Registry hash/bump invariant test; cost-event and decision-log writes carry the stamp
(extend `test_cost.py` / scheduler tests with mocks); migration adds columns (extend
`test_db.py`).

## Workstream C — Online evaluation (score production decisions)

**Goal:** every live decision eventually gets a score; drift is visible without anyone
remembering to look.

### Design

Three layers, all strictly best-effort (the invariant from the journal-RAG spec applies
verbatim: scoring can never block or corrupt a tick).

- **C1 — Deterministic outcome scoring.** **`src/vibe_trading/eval/online.py` [NEW]** with a
  CLI entry (`python -m vibe_trading.eval.online`, cron-able; also callable at the end of a
  tick). For each unscored decision whose outcome is now knowable:
  - trade closed → realized PnL % from `trades` (existing join via `decision_id`);
  - flat/rejected and past the counterfactual horizon → signed forward return, **reusing
    `journal.py`'s outcome-attachment machinery** (it already computes exactly this for
    precedents).

  Score = directional correctness in [0, 1] (did the action's sign match the realized/
  counterfactual move, scaled by magnitude bands). Persisted to a new `decision_scores`
  table (`decision_id PK, scored_at, kind, outcome_pct, outcome_score, judge_score NULLABLE`)
  and pushed to Langfuse as a numeric score on the decision's trace via the stored
  `trace_id`. The `scored_at` marker makes the job idempotent.
- **C2 — Sampled production judge.** The golden-set judge rubrics are per-case and don't
  apply to arbitrary live decisions, so C2 uses a **generic rubric**: groundedness (does the
  reasoning cite only facts present in the stored `agent_transcripts` feature snapshot?) and
  internal consistency (does the action follow from the stated bias/confluence?). Judge
  model = `EVAL_JUDGE_MODEL`; sample cap (default 5 decisions/day) bounds cost; calls are
  tagged `call_type="online_judge"` in `llm_cost_log` so they're separable from trading
  spend and count against the existing daily cost cap.
- **C3 — Drift digest.** Weekly Discord digest (reusing the existing webhook plumbing):
  mean outcome score and judge score for the week vs the trailing 4-week mean,
  schema-compliance rate from `llm_cost_log`, segmented by `prompt_version` (Workstream B).
  Pure SQL over `decision_scores` + `llm_cost_log`; no new infra.

### Testing

Outcome-score math is pure (fixture decisions + trades/candles → expected scores);
idempotency (second run scores nothing); Langfuse push mocked and asserted; generic-judge
prompt construction snapshot-tested; digest SQL tested against fixture rows.

## Workstream D — Retrieval upgrade (pgvector + retrieval evals + reranking)

**Goal:** real indexed vector search, a number for retrieval quality, and evidence for
whether precedent injection helps.

### Design

- **D1 — pgvector + HNSW.** Supabase ships the `pgvector` extension; enable it and migrate
  `decision_embeddings.embedding` from `DOUBLE PRECISION[]` to `vector(<dim>)` with an HNSW
  index (`vector_cosine_ops`). `PrecedentRetriever.retrieve()` becomes a single SQL query
  (`ORDER BY embedding <=> :query LIMIT :k`, keeping the existing older-than-horizon
  filter) instead of load-all + numpy. **Fallback:** a capability probe at startup; if the
  extension is unavailable (e.g. vanilla local Postgres), keep today's in-Python cosine
  path. At the current journal size this is a latency wash — the point is the index is in
  place before the journal grows, and the migration is done against a live table.
- **D2 — Retrieval-quality eval.** **`evals/retrieval/` [NEW]**: ~20 labeled queries (a
  setup card + hand-judged relevant/irrelevant candidate decision IDs drawn from the real
  journal), and **`evals/retrieval_eval.py`** computing recall@k and MRR for the production
  retriever configuration. Committed baseline numbers alongside the decision-quality
  baseline. This is the gate for D4: no reranker until there's a number showing headroom.
- **D3 — Precedent ablation.** The journal-RAG spec deferred eval integration (empty
  retriever in eval); this workstream answers "do precedents help?" two ways:
  - **Backtest A/B:** run the backtester with the retriever enabled vs `NoOpRetriever`
    over the same window (the backtester already replays the full decision pipeline with
    `current_timestamp` pinned; the horizon filter prevents lookahead in retrieval). Compare
    equity curves and per-decision outcome scores.
  - **Online segmentation:** record `precedents_k` (count actually injected) on
    `decision_scores` rows in C1, then compare outcome scores for decisions with vs without
    precedents. Observational, not causal — but continuous and free.
- **D4 — Two-stage retrieval (gated on D2).** ANN top-50 → rerank → top-k, only if D2 shows
  recall headroom at the final k. Reranker choice (LLM listwise vs cross-encoder) is decided
  then, by the D2 numbers — not speculated now.

### Testing

Migration round-trips a vector; SQL retrieval returns the same top-k as the in-Python path
on a fixture set (parity test); recall@k / MRR math unit-tested; capability-probe fallback
covered.

## Sequencing

**A → B → C → D.** A is standalone and an afternoon of work. B must precede C so online
scores are segmentable by prompt version from day one. D's ablation (D3) leans on C1's
outcome scores, so retrieval goes last; D1 (pgvector) has no dependencies and can be pulled
forward if convenient.

## Later (deliberately not in this initiative)

- **Reflection loop** — periodic summarization of recent outcomes into "lessons" injected
  into the trader prompt; measurable only once C exists, and it also addresses the eventual
  unbounded growth of decision history.
- **News-sentiment analyst tool + prompt-injection eval suite** — the first tool whose
  results carry *external* text, i.e. the first real injection surface; the tool and its
  adversarial test cases ship together or not at all.
- **Provider failover routing** — LiteLLM Router with a fallback chain + circuit breaking;
  today a Gemini rate-limit spike fails the tick. Failovers recorded in `llm_cost_log`.
- **SFT distillation** — fine-tune a cheaper model on accumulated tool-loop transcripts and
  judge it against the golden set; only worth doing once the transcript corpus is larger and
  the benchmark matrix (A) has established the frontier to beat.

## Out of scope entirely

Streaming (no interactive consumer on a 4h cadence), multimodal chart analysis, A/B serving
infrastructure, external prompt-management services.

## Backwards compatibility

Every workstream is additive: new files, new nullable columns via the existing idempotent
migration lists, new CLI entry points. The committed decision-quality baseline is untouched
by A (isolated), stamped-but-unchanged by B (re-committed with `prompt_versions` on the
next legitimate `--update-baseline`), and never read by C or D. `JOURNAL_RAG_ENABLED=false`
and the pgvector capability fallback preserve today's behavior at every step.
