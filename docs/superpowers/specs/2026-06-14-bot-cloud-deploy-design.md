# Sub-project 1: Bot → Cloud ($0) — Design

**Status:** Approved (brainstorm), pending implementation plan
**Date:** 2026-06-14
**Scope:** Deploy the *trading bot* to the cloud at **$0/month** as a reliable scheduled job. The dashboard re-platform (Vercel + Supabase) is a separate sub-project and is explicitly **out of scope** here.

---

## 1. Goal

Get the bot off the developer's laptop and trading the Binance Futures **demo** account 24/7, reliably, at **zero monthly cost**, with monitoring that alerts on silent failure and a one-command-ish redeploy. No real funds are involved, so there is no HA / multi-region / compliance requirement — the bar is "runs reliably, tells me when it doesn't, costs nothing."

## 2. Decisions (locked during brainstorm)

| Decision | Choice | Why |
|---|---|---|
| Process model | **Scheduled job** (`trade-once` every 4h), not an always-on process | No in-memory scheduler to silently wedge; the platform's cron is the HA, timezone-explicit scheduler. `trade-once` already exists and does exactly one reconcile + evaluate then exits. |
| Scheduler / host | **GitHub Actions scheduled workflow** | Genuinely $0 (public repo = unlimited minutes; private = 2000 free min/mo; 6 short runs/day is well under). Reuses existing CI. Secrets are free + encrypted. |
| Image | Built once in CI → pushed to **GHCR**; the cron workflow `docker run`s the published image | Keeps each scheduled run fast (no per-run TA-Lib compile). GHCR is free. |
| State cache | **Cloudflare R2** (S3-compatible, free tier, no egress) via a new `state_sync` module | Strategy "B": keep DuckDB + Parquet as-is, sync to/from a bucket around each run. Warm candle cache + durable, non-reproducible audit. R2 chosen for the free tier + zero egress; any S3-compatible store works by swapping the endpoint. |
| Monitoring | **healthchecks.io** dead-man's-switch + GitHub Actions failure notifications + existing Discord trade alerts | Closes the "silent death" gap: a *missed* tick (not just an errored one) raises an alert. |
| Secrets | **GitHub Actions Secrets** | Free, encrypted, auto-masked in logs. Replaces the plaintext `.env`. |

**Trade-off accepted:** the real-time WebSocket fill listener does not run between ticks. Exits still fire **server-side via native exchange brackets**; fills are reconciled on the next 4h `trade-once` run. For 4h swing trading this is acceptable (the dashboard reflects an exit a few hours late at worst).

## 3. Architecture

```
GitHub Actions scheduler (UTC cron "1 */4 * * *")  +  workflow_dispatch (manual)
        │  docker run ghcr.io/<owner>/vibe-trading:<tag>  trade-once   (env from Secrets)
        ▼
  ephemeral container
   ├─ state_sync.pull()   ── download vibe_trading.db ◄── Cloudflare R2
   ├─ trade-once  ── reads/writes ──► Supabase Postgres (decisions/trades/positions/embeddings/costs)
   │              ── orders + brackets ──► Binance Futures DEMO
   │              ── writes audit Parquet locally
   ├─ state_sync.push()   ── upload vibe_trading.db + new audit/*.parquet ──► Cloudflare R2
   └─ healthcheck ping ──► healthchecks.io  (on success)
```

A second workflow (CI) builds and publishes the image:

```
push to main ──► [test job: hermetic pytest]  ──pass──►  [build+push image to GHCR]
                          │ fail
                          └─► image NOT published; cron keeps running the last good image
```

## 4. Components / files

### 4.1 `src/vibe_trading/runtime/state_sync.py` (new)
S3-compatible object-storage sync, **gated by the `STATE_SYNC_BUCKET` env var** (unset ⇒ every function is a no-op, so local `live`/`trade-once` and tests are unaffected).

- Client: `boto3.client("s3", endpoint_url=STATE_SYNC_ENDPOINT, aws_access_key_id=…, aws_secret_access_key=…)`. Endpoint + creds from env (`STATE_SYNC_ENDPOINT`, `STATE_SYNC_ACCESS_KEY_ID`, `STATE_SYNC_SECRET_ACCESS_KEY`).
- `pull() -> bool`: if `STATE_SYNC_BUCKET` unset → return `False` (no-op). Else download the candle DB object (key derived from `DATABASE_PATH`, default `vibe_trading.db`) to its local path if it exists in the bucket; return `True` if a file was downloaded, `False` if the object is absent (a fresh run — `bootstrap_if_needed` will then fetch candles). Never raises: log + return `False` on any error.
- `push() -> None`: if `STATE_SYNC_BUCKET` unset → no-op. Else upload the candle DB and every `*.parquet` under the audit dir (`DEFAULT_AUDIT_DIR`, `data/audit`) to the bucket. Never raises: log a warning on error (the run's decisions are already durable in Postgres).
- Keys: candle DB at `vibe_trading.db`; audit files at `audit/<filename>.parquet` (preserve the existing audit filenames).

### 4.2 Wire sync + healthcheck into the `trade-once` CLI path
In `src/vibe_trading/cli.py`, the `trade-once` branch becomes:

```python
elif args.command == "trade-once":
    state_sync.pull()
    try:
        scheduler = TradingScheduler(args.symbols)
        scheduler.sync_and_evaluate()
        monitoring.ping_healthcheck(success=True)
    finally:
        state_sync.push()
```

Only the `trade-once` path syncs (not the always-on `live` path), so local/always-on behaviour is unchanged.

### 4.3 `src/vibe_trading/runtime/monitoring.py` (new, tiny)
`ping_healthcheck(success: bool = True) -> None`: if `HEALTHCHECK_PING_URL` unset → no-op. Else GET the URL (append `/fail` on `success=False`) with a short timeout, swallowing all errors (monitoring must never break trading). Use `requests` if already a transitive dep, else `urllib.request` (no new dependency).

### 4.4 Scheduler hardening (companion change to `src/vibe_trading/runtime/scheduler.py`)
The cloud path doesn't use APScheduler, but the local `live` path does and currently inherits its timezone implicitly. Pin it:

```python
scheduler = BlockingScheduler(timezone="UTC")
scheduler.add_job(self.sync_and_evaluate, "cron", hour="*/4", minute=1,
                  coalesce=True, misfire_grace_time=3600)
```

Factor the scheduler construction into a small `_build_scheduler()` method so it can be unit-tested for `timezone == UTC`.

### 4.5 `.github/workflows/` — two workflows
- **Extend `ci.yml`** (or add `deploy-image.yml`): a `build-image` job, gated on the existing `test` job succeeding and `push` to the default branch, that builds the Dockerfile and pushes to GHCR (`docker/build-push-action`, login via `GITHUB_TOKEN`). Tags: `latest` + the commit SHA.
- **`trade-cron.yml`** (new): triggers `schedule: cron: "1 */4 * * *"` (UTC) **and** `workflow_dispatch`. Steps: log in to GHCR, `docker run` the `latest` image with `trade-once`, passing every secret as `-e`. Concurrency group `trade-cron` with `cancel-in-progress: false` so runs never overlap.

### 4.6 `pyproject.toml`
Add `boto3` to the runtime dependencies. (`requests` only if not already transitively present; prefer `urllib` for the healthcheck to avoid a new dep.)

### 4.7 Secrets (GitHub Actions Secrets)
`GEMINI_API_KEY` (+ `GROQ_API_KEY`/`LLM_PROVIDER`/`LLM_MODEL` as needed), `POSTGRES_URL`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST`, `DISCORD_WEBHOOK_URL`, `STATE_SYNC_BUCKET`/`STATE_SYNC_ENDPOINT`/`STATE_SYNC_ACCESS_KEY_ID`/`STATE_SYNC_SECRET_ACCESS_KEY`, `HEALTHCHECK_PING_URL`. Non-secret config (`TRADING_MODE=LIVE_TESTNET`, `JOURNAL_RAG_ENABLED=true`, `LLM_DAILY_COST_CAP_USD`, `BINANCE_TESTNET_LEVERAGE`) set as workflow `env`.

### 4.8 Docs
A "Cloud deployment ($0)" section in `README.md`: the two workflows, the secret list, R2 + healthchecks.io setup, the dry-run-first rollout, and rollback.

## 5. Data flow per run

1. GitHub Actions cron fires (UTC).
2. `docker run` the GHCR image with `trade-once` + env.
3. `state_sync.pull()` downloads the warm candle DB from R2 (or no-op on a fresh bucket).
4. `trade-once` → `sync_and_evaluate()`: refresh candles, reconcile positions (Postgres ledger vs exchange), evaluate trending symbols through the decision pipeline, submit approved orders with native brackets, log every decision to Postgres + write audit Parquet.
5. On success, ping healthchecks.io.
6. `state_sync.push()` (in `finally`) uploads the updated DB + new audit Parquet to R2.
7. Container exits.

## 6. Error handling

| Failure | Behaviour |
|---|---|
| `pull()` fails / bucket empty | Log; continue. `bootstrap_if_needed` re-fetches candles (reproducible). Non-fatal. |
| `push()` fails | Log a warning + (optionally) Discord. The run's decisions are already durable in Postgres; only the warm cache + that run's audit Parquet are at risk. |
| `trade-once` raises | Container exits non-zero → GitHub Actions failure notification. `finally` still runs `push()`. healthcheck is NOT pinged → dead-man alert. |
| Scheduler outage / missed tick | No success ping within the window → healthchecks.io alerts. |
| Concurrent runs | Prevented: GH Actions concurrency group + runs (<2 min) never approach the 4h interval ⇒ single R2 writer, no race. |
| Crash mid-entry | The `-4509` rollback handles the in-call case; a hard crash between fill and bracket is caught by the next run's reconcile. Brackets are server-side regardless. |

## 7. Testing & rollout

- **Existing hermetic pytest gate** continues to block image publish.
- **`state_sync` unit tests** (mocked S3 client): unset `STATE_SYNC_BUCKET` ⇒ `pull()` returns `False` and makes no client call; bucket set + object present ⇒ downloads to the local path and returns `True`; bucket set + object absent ⇒ returns `False`; `push()` uploads the DB + each audit `*.parquet`; all functions swallow client errors without raising.
- **`monitoring.ping_healthcheck` tests**: no-op when URL unset; GETs the URL on success and the `/fail` suffix on failure; swallows network errors.
- **Scheduler hardening test**: `_build_scheduler()` returns a scheduler whose `timezone` is UTC.
- **First cloud rollout** runs with `BINANCE_TESTNET_DRY_RUN=true` (logs intended orders, places none) triggered via `workflow_dispatch`; verify a clean run end-to-end (pull → evaluate → Postgres write → push → ping), then flip `BINANCE_TESTNET_DRY_RUN=false`.

## 8. Rollback

GHCR retains prior image tags. To roll back, pin `trade-cron.yml` to the previous commit-SHA tag (or re-publish the prior commit). Re-running `trade-once` is idempotent — the reconcile + position-exists gates prevent duplicate entries.

## 9. Risks / notes

- **GH Actions cron timing is best-effort** (can lag minutes under load) — irrelevant for a settled 4h candle.
- **Scheduled workflows auto-disable after 60 days of repo inactivity** — mitigate with periodic commits or re-enable; document it.
- **Private-repo free minutes** (2000/mo) — current usage is far under; public repo is unlimited.
- **Binance demo reachability from GitHub-hosted runners** (Azure egress) — verified during the dry-run rollout.
- **Secret hygiene** — all secrets passed as env, never echoed; GH masks secret values in logs.

## 10. Out of scope (deferred)

- Dashboard re-platform to Vercel + Supabase (sub-project 2).
- Real-money production hardening (HA, secret manager beyond GH Secrets, stricter audit guarantees).
- Moving candles into Postgres (strategy "C").
