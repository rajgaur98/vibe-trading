# Bot → Cloud ($0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the trading bot to the cloud at $0/month as a GitHub Actions scheduled job that runs the GHCR image's `trade-once` every 4h, caching DuckDB/Parquet state in Cloudflare R2 and alerting on silent failure via healthchecks.io.

**Architecture:** A CI job builds the Docker image and pushes it to GHCR after the hermetic test gate passes. A scheduled workflow `docker run`s that image's `trade-once` every 4h (UTC). Inside the run, a new env-gated `state_sync` module pulls the candle DB from R2 before evaluating and pushes the DB + audit Parquet back after; a `monitoring` module pings healthchecks.io on success. All state-of-record stays in Supabase Postgres; the always-on `live` path is unchanged except a defensive UTC pin on its scheduler.

**Tech Stack:** Python 3.12, boto3 (S3-compatible → Cloudflare R2), requests, APScheduler, GitHub Actions, GHCR, healthchecks.io.

**Spec:** `docs/superpowers/specs/2026-06-14-bot-cloud-deploy-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/vibe_trading/runtime/state_sync.py` (new) | Pull/push the DuckDB candle DB + audit Parquet to an S3-compatible bucket; **no-op unless `STATE_SYNC_BUCKET` is set**. Never raises. |
| `src/vibe_trading/runtime/monitoring.py` (new) | `ping_healthcheck()` dead-man's-switch; **no-op unless `HEALTHCHECK_PING_URL` is set**. Never raises. |
| `src/vibe_trading/cli.py` (modify) | `trade-once` branch wraps evaluate with pull → evaluate → ping → (finally) flush+push, via a testable `execute_trade_once()`. |
| `src/vibe_trading/runtime/scheduler.py` (modify) | Factor `_build_scheduler()` and pin it to `timezone="UTC"` + coalesce/misfire (local `live` path hygiene). |
| `pyproject.toml` (modify) | Declare `boto3` and `requests` runtime deps. |
| `.github/workflows/ci.yml` (modify) | Add a `build-image` job (needs `test`, default-branch push only) that pushes to GHCR. |
| `.github/workflows/trade-cron.yml` (new) | Scheduled (`1 */4 * * *` UTC) + manual workflow that `docker run`s the image's `trade-once`. |
| `tests/test_state_sync.py` (new) | Unit tests for `state_sync` (mocked S3 client). |
| `tests/test_monitoring.py` (new) | Unit tests for `ping_healthcheck` (mocked requests). |
| `tests/test_cli_trade_once.py` (new) | Unit test for the `execute_trade_once` orchestration order. |
| `tests/test_scheduler.py` (modify) | Add a test that `_build_scheduler()` is UTC. |
| `README.md` (modify) | "Cloud deployment ($0)" section. |

---

### Task 1: Declare boto3 + requests dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect the current dependencies array**

Run: `grep -nA15 'dependencies = \[' pyproject.toml`
Expected: a `dependencies = [ ... ]` list under `[project]`. Note that neither `boto3` nor `requests` is present (they are only transitively available).

- [ ] **Step 2: Add the two dependencies**

Add these two lines inside the `[project] dependencies = [...]` array (keep the existing entries; match the existing quoting/indent style):

```toml
    "boto3>=1.34",
    "requests>=2.31",
```

- [ ] **Step 3: Install into the local venv**

Run: `.venv/bin/python -m pip install -e .`
Expected: installs `boto3` (and its `botocore`, `s3transfer` deps); finishes without error.

- [ ] **Step 4: Verify both import**

Run: `.venv/bin/python -c "import boto3, requests; print(boto3.__version__, requests.__version__)"`
Expected: prints two version strings, no traceback.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add boto3 + requests as explicit runtime deps"
```

---

### Task 2: `state_sync` module (object-storage cache)

**Files:**
- Create: `src/vibe_trading/runtime/state_sync.py`
- Test: `tests/test_state_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_state_sync.py`:

```python
"""Unit tests for state_sync. The boto3 S3 client is replaced via monkeypatching
state_sync._client, so no network/credentials are needed."""
import os
from unittest.mock import MagicMock

from vibe_trading.runtime import state_sync


def test_pull_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("STATE_SYNC_BUCKET", raising=False)
    called = MagicMock()
    monkeypatch.setattr(state_sync, "_client", called)
    assert state_sync.pull() is False
    called.assert_not_called()  # never even builds a client


def test_pull_downloads_db_when_present(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    client = MagicMock()
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    assert state_sync.pull() is True
    client.download_file.assert_called_once_with("vibe-state", "vibe_trading.db", str(db_path))


def test_pull_returns_false_and_swallows_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "vibe_trading.db"))
    client = MagicMock()
    client.download_file.side_effect = Exception("NoSuchKey")
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    assert state_sync.pull() is False  # absent/failed object => fresh bootstrap, no raise


def test_push_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("STATE_SYNC_BUCKET", raising=False)
    called = MagicMock()
    monkeypatch.setattr(state_sync, "_client", called)
    state_sync.push()
    called.assert_not_called()


def test_push_uploads_db_and_audit_parquet(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    db_path.write_text("duckdb")
    audit_dir = tmp_path / "audit" / "decisions"
    audit_dir.mkdir(parents=True)
    (audit_dir / "a.parquet").write_text("p1")
    (audit_dir / "b.parquet").write_text("p2")

    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    client = MagicMock()
    monkeypatch.setattr(state_sync, "_client", lambda: client)

    state_sync.push()

    uploaded = {c.args[2] for c in client.upload_file.call_args_list}  # set of object keys
    assert "vibe_trading.db" in uploaded
    assert "audit/decisions/a.parquet" in uploaded
    assert "audit/decisions/b.parquet" in uploaded


def test_push_swallows_errors(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    db_path.write_text("x")
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    client = MagicMock()
    client.upload_file.side_effect = Exception("network down")
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    state_sync.push()  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state_sync.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_trading.runtime.state_sync'`.

- [ ] **Step 3: Implement the module**

Create `src/vibe_trading/runtime/state_sync.py`:

```python
"""Object-storage cache for the DuckDB candle DB + Parquet audit corpus, so an
ephemeral scheduled `trade-once` container starts warm and never loses the
(non-reproducible) audit records. S3-compatible (Cloudflare R2 / S3 / B2 / GCS).

Entirely gated on STATE_SYNC_BUCKET: unset => every function is a no-op, so the
local `live`/`trade-once` paths and the test suite are unaffected. Never raises —
a sync failure must never break trading (decisions are durable in Postgres).
"""
import os
import glob
import logging

from vibe_trading.audit import DEFAULT_AUDIT_DIR

logger = logging.getLogger(__name__)

CANDLE_DB_KEY = "vibe_trading.db"          # object key for the candle DB
AUDIT_KEY_PREFIX = "audit/decisions/"      # object key prefix for audit parquet


def _bucket():
    return os.getenv("STATE_SYNC_BUCKET") or None


def _db_path():
    return os.getenv("DATABASE_PATH", "data/vibe_trading.db")


def _audit_decisions_dir():
    return os.path.join(os.getenv("AUDIT_DIR", DEFAULT_AUDIT_DIR), "decisions")


def _client():
    """Build an S3-compatible client from env (endpoint + creds)."""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("STATE_SYNC_ENDPOINT") or None,
        aws_access_key_id=os.getenv("STATE_SYNC_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("STATE_SYNC_SECRET_ACCESS_KEY") or None,
    )


def pull() -> bool:
    """Download the candle DB from the bucket to its local path. Returns True if a
    file was downloaded, False if sync is disabled or the object is absent/failed
    (the caller's bootstrap_if_needed will then fetch fresh candles)."""
    bucket = _bucket()
    if not bucket:
        return False
    db_path = _db_path()
    try:
        client = _client()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        client.download_file(bucket, CANDLE_DB_KEY, db_path)
        logger.info(f"state_sync: pulled {CANDLE_DB_KEY} -> {db_path}")
        return True
    except Exception as e:
        logger.warning(f"state_sync: pull skipped/failed ({e}); will bootstrap fresh candles")
        return False


def push() -> None:
    """Upload the candle DB and every audit Parquet back to the bucket. No-op when
    sync is disabled. Swallows all errors (decisions remain durable in Postgres)."""
    bucket = _bucket()
    if not bucket:
        return
    try:
        client = _client()
        db_path = _db_path()
        if os.path.exists(db_path):
            client.upload_file(db_path, bucket, CANDLE_DB_KEY)
        for path in glob.glob(os.path.join(_audit_decisions_dir(), "*.parquet")):
            client.upload_file(path, bucket, AUDIT_KEY_PREFIX + os.path.basename(path))
        logger.info("state_sync: pushed candle DB + audit parquet")
    except Exception as e:
        logger.warning(f"state_sync: push failed ({e}); decisions remain durable in Postgres")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state_sync.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/state_sync.py tests/test_state_sync.py
git commit -m "feat(deploy): R2/S3 state_sync for candle DB + audit parquet"
```

---

### Task 3: `monitoring` module (healthchecks.io dead-man's-switch)

**Files:**
- Create: `src/vibe_trading/runtime/monitoring.py`
- Test: `tests/test_monitoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring.py`:

```python
"""Unit tests for monitoring.ping_healthcheck. requests.get is monkeypatched, so
no network is used."""
from unittest.mock import MagicMock

from vibe_trading.runtime import monitoring


def test_ping_noop_when_url_unset(monkeypatch):
    monkeypatch.delenv("HEALTHCHECK_PING_URL", raising=False)
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)
    get.assert_not_called()


def test_ping_success_gets_base_url(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc")
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)
    assert get.call_args.args[0] == "https://hc.example/abc"


def test_ping_failure_appends_fail(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc/")
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=False)
    assert get.call_args.args[0] == "https://hc.example/abc/fail"


def test_ping_swallows_network_errors(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc")
    get = MagicMock(side_effect=Exception("timeout"))
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_monitoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_trading.runtime.monitoring'`.

- [ ] **Step 3: Implement the module**

Create `src/vibe_trading/runtime/monitoring.py`:

```python
"""Dead-man's-switch ping for a scheduled run. Pings HEALTHCHECK_PING_URL on a
successful run; if a run is ever missed or errors, the external monitor (e.g.
healthchecks.io) fires an alert. No-op unless the URL is configured. Never raises —
monitoring must never break trading."""
import os
import logging

import requests

logger = logging.getLogger(__name__)


def ping_healthcheck(success: bool = True) -> None:
    url = os.getenv("HEALTHCHECK_PING_URL")
    if not url:
        return
    if not success:
        url = url.rstrip("/") + "/fail"
    try:
        requests.get(url, timeout=10)
    except Exception as e:
        logger.warning(f"monitoring: healthcheck ping failed ({e})")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_monitoring.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/monitoring.py tests/test_monitoring.py
git commit -m "feat(deploy): healthchecks.io dead-man's-switch ping"
```

---

### Task 4: Wire pull → evaluate → ping → push into `trade-once`

**Files:**
- Modify: `src/vibe_trading/cli.py` (imports near top; `trade-once` branch at lines 99-110)
- Test: `tests/test_cli_trade_once.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_trade_once.py`:

```python
"""Unit test for the execute_trade_once orchestration: pull before evaluate, ping
on success, push always (finally). TradingScheduler/state_sync/monitoring/langfuse
are all monkeypatched, so nothing real runs."""
from unittest.mock import MagicMock

from vibe_trading import cli


def _patch(monkeypatch, evaluate_raises=False):
    calls = []

    def _evaluate():
        calls.append("evaluate")
        if evaluate_raises:
            raise RuntimeError("boom")

    scheduler = MagicMock()
    scheduler.sync_and_evaluate.side_effect = _evaluate
    monkeypatch.setattr(cli, "TradingScheduler", lambda symbols: scheduler)
    monkeypatch.setattr(cli.state_sync, "pull", lambda: calls.append("pull"))
    monkeypatch.setattr(cli.state_sync, "push", lambda: calls.append("push"))
    monkeypatch.setattr(cli.monitoring, "ping_healthcheck", lambda success=True: calls.append(f"ping:{success}"))
    monkeypatch.setattr(cli, "_flush_langfuse", lambda: calls.append("flush"))
    return calls


def test_execute_trade_once_happy_path_order(monkeypatch):
    calls = _patch(monkeypatch)
    cli.execute_trade_once([])
    assert calls[0] == "pull"            # warm state first
    assert "evaluate" in calls
    assert "ping:True" in calls          # pinged on success
    assert calls[-1] == "push"           # state pushed last (finally)


def test_execute_trade_once_pushes_even_on_failure(monkeypatch):
    calls = _patch(monkeypatch, evaluate_raises=True)
    try:
        cli.execute_trade_once([])
    except RuntimeError:
        pass
    assert "ping:True" not in calls      # NOT pinged => dead-man's-switch fires
    assert calls[-1] == "push"           # still pushed in finally
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_trade_once.py -q`
Expected: FAIL — `AttributeError: module 'vibe_trading.cli' has no attribute 'execute_trade_once'`.

- [ ] **Step 3: Add the imports**

In `src/vibe_trading/cli.py`, add to the import block (after the existing `from vibe_trading.runtime.scheduler import TradingScheduler` line):

```python
from vibe_trading.runtime import state_sync, monitoring
```

- [ ] **Step 4: Add `execute_trade_once` + `_flush_langfuse` and call them from the branch**

In `src/vibe_trading/cli.py`, add these two module-level functions (above `def main():`):

```python
def _flush_langfuse():
    try:
        from langfuse import get_client
        logger.info("Flushing Langfuse traces...")
        get_client().flush()
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")


def execute_trade_once(symbols):
    """One scheduled execution window: warm state from the cache, run a single
    sync+evaluate, ping the dead-man's-switch on success, and always flush traces
    and push state back (even if evaluation raised)."""
    state_sync.pull()
    try:
        scheduler = TradingScheduler(symbols)
        scheduler.sync_and_evaluate()
        monitoring.ping_healthcheck(success=True)
    finally:
        _flush_langfuse()
        state_sync.push()
```

Then replace the existing `trade-once` branch body (currently `src/vibe_trading/cli.py:99-110`):

```python
    elif args.command == "trade-once":
        logger.info(f"Triggering on-demand trading execution window for: {args.symbols}")
        scheduler = TradingScheduler(args.symbols)
        scheduler.sync_and_evaluate()
        try:
            from langfuse import get_client
            logger.info("Flushing Langfuse traces...")
            get_client().flush()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")
        logger.info("On-demand execution window completed.")
```

with:

```python
    elif args.command == "trade-once":
        logger.info(f"Triggering on-demand trading execution window for: {args.symbols}")
        execute_trade_once(args.symbols)
        logger.info("On-demand execution window completed.")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_trade_once.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/cli.py tests/test_cli_trade_once.py
git commit -m "feat(deploy): wire state_sync + healthcheck into trade-once"
```

---

### Task 5: Pin the local scheduler to UTC (companion hardening)

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py` (the `start()` method, lines ~67-86)
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scheduler.py`:

```python
def test_build_scheduler_is_utc():
    """The live-path APScheduler must be pinned to UTC, not the host's implicit tz,
    so the 4h cron aligns with the UTC candle boundaries the rest of the code assumes."""
    from vibe_trading.runtime.scheduler import TradingScheduler
    # __new__ avoids __init__'s broker/DB/network setup — we only test the factory.
    s = TradingScheduler.__new__(TradingScheduler)
    sched = s._build_scheduler()
    assert str(sched.timezone) == "UTC"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py::test_build_scheduler_is_utc -q`
Expected: FAIL — `AttributeError: 'TradingScheduler' object has no attribute '_build_scheduler'`.

- [ ] **Step 3: Add `_build_scheduler()` and use it in `start()`**

In `src/vibe_trading/runtime/scheduler.py`, add this method (next to `start`):

```python
    def _build_scheduler(self):
        """Construct the live-path scheduler, pinned to UTC and tolerant of brief
        downtime, so the 4h cron fires at 00:01/04:01/.../20:01 UTC regardless of
        the host's timezone."""
        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(self.sync_and_evaluate, "cron", hour="*/4", minute=1,
                          coalesce=True, misfire_grace_time=3600)
        return scheduler
```

Then, inside `start()`, replace these lines:

```python
        # 3. Setup recurring 4-hour scheduler
        scheduler = BlockingScheduler()
        # Schedule to run at the start of every 4h block (00:00, 04:00, 08:00, etc.)
        scheduler.add_job(self.sync_and_evaluate, 'cron', hour='*/4', minute=1)
```

with:

```python
        # 3. Setup recurring 4-hour scheduler (UTC-pinned; see _build_scheduler)
        scheduler = self._build_scheduler()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py::test_build_scheduler_is_utc -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py tests/test_scheduler.py
git commit -m "fix(scheduler): pin live BlockingScheduler to UTC + coalesce/misfire"
```

---

### Task 6: CI job to build & publish the image to GHCR

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Append the `build-image` job**

Add this job to `.github/workflows/ci.yml` (at the end, as a sibling of `test` and `eval`). It only runs on a push to the default branch and only after `test` passes, so a red test never publishes an image:

```yaml
  # ============================================================
  # Build & publish the runtime image to GHCR. Gated on the test
  # job and default-branch pushes only — a red test never ships an
  # image, so the scheduled trade-cron keeps running the last good one.
  # ============================================================
  build-image:
    name: Build & push image (GHCR)
    needs: test
    if: github.event_name == 'push' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch)
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Lowercase image name
        id: img
        run: echo "name=ghcr.io/$(echo '${{ github.repository }}' | tr '[:upper:]' '[:lower:]')" >> "$GITHUB_OUTPUT"

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ steps.img.outputs.name }}:latest
            ${{ steps.img.outputs.name }}:${{ github.sha }}
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml OK')"`
Expected: `ci.yml OK` (no exception).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build & push runtime image to GHCR on default-branch push"
```

---

### Task 7: Scheduled `trade-cron` workflow

**Files:**
- Create: `.github/workflows/trade-cron.yml`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/trade-cron.yml`. It runs every 4h (UTC) and on manual dispatch, pulls the published image, and runs `trade-once`. Secrets are exposed as the step's `env` and passed through with bare `-e NAME` (no values on the command line). `BINANCE_TESTNET_DRY_RUN` is driven by a repo **variable** so the first rollout can run dry without a code change:

```yaml
name: trade-cron

on:
  schedule:
    - cron: "1 */4 * * *"   # 00:01, 04:01, ... 20:01 UTC — just after each 4h candle close
  workflow_dispatch:

concurrency:
  group: trade-cron
  cancel-in-progress: false   # never interrupt an in-flight trading run

jobs:
  trade-once:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: read
    steps:
      - name: Lowercase image name
        id: img
        run: echo "name=ghcr.io/$(echo '${{ github.repository }}' | tr '[:upper:]' '[:lower:]')" >> "$GITHUB_OUTPUT"

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Run trade-once
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          LLM_PROVIDER: ${{ secrets.LLM_PROVIDER }}
          LLM_MODEL: ${{ secrets.LLM_MODEL }}
          POSTGRES_URL: ${{ secrets.POSTGRES_URL }}
          BINANCE_TESTNET_API_KEY: ${{ secrets.BINANCE_TESTNET_API_KEY }}
          BINANCE_TESTNET_API_SECRET: ${{ secrets.BINANCE_TESTNET_API_SECRET }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_HOST: ${{ secrets.LANGFUSE_HOST }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          STATE_SYNC_BUCKET: ${{ secrets.STATE_SYNC_BUCKET }}
          STATE_SYNC_ENDPOINT: ${{ secrets.STATE_SYNC_ENDPOINT }}
          STATE_SYNC_ACCESS_KEY_ID: ${{ secrets.STATE_SYNC_ACCESS_KEY_ID }}
          STATE_SYNC_SECRET_ACCESS_KEY: ${{ secrets.STATE_SYNC_SECRET_ACCESS_KEY }}
          HEALTHCHECK_PING_URL: ${{ secrets.HEALTHCHECK_PING_URL }}
          DRY_RUN: ${{ vars.BINANCE_TESTNET_DRY_RUN || 'false' }}
        run: |
          docker pull "${{ steps.img.outputs.name }}:latest"
          docker run --rm \
            -e TRADING_MODE=LIVE_TESTNET \
            -e JOURNAL_RAG_ENABLED=true \
            -e LLM_DAILY_COST_CAP_USD=10.0 \
            -e BINANCE_TESTNET_LEVERAGE=1 \
            -e BINANCE_TESTNET_DRY_RUN="$DRY_RUN" \
            -e GEMINI_API_KEY -e GROQ_API_KEY -e LLM_PROVIDER -e LLM_MODEL \
            -e POSTGRES_URL \
            -e BINANCE_TESTNET_API_KEY -e BINANCE_TESTNET_API_SECRET \
            -e LANGFUSE_PUBLIC_KEY -e LANGFUSE_SECRET_KEY -e LANGFUSE_HOST \
            -e DISCORD_WEBHOOK_URL \
            -e STATE_SYNC_BUCKET -e STATE_SYNC_ENDPOINT \
            -e STATE_SYNC_ACCESS_KEY_ID -e STATE_SYNC_SECRET_ACCESS_KEY \
            -e HEALTHCHECK_PING_URL \
            "${{ steps.img.outputs.name }}:latest" \
            trade-once
```

- [ ] **Step 2: Validate the YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/trade-cron.yml')); print('trade-cron.yml OK')"`
Expected: `trade-cron.yml OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/trade-cron.yml
git commit -m "ci: scheduled trade-cron workflow (4h UTC, runs GHCR image trade-once)"
```

---

### Task 8: Deploy docs + full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Cloud deployment ($0)" section**

Insert this section into `README.md` (after the existing local/docker run instructions):

````markdown
## Cloud deployment ($0)

The bot runs in the cloud as a **GitHub Actions scheduled job** — no always-on server.

**How it works**
- `ci.yml` builds the Docker image and pushes it to **GHCR** on every default-branch push (only after the hermetic `test` job passes).
- `trade-cron.yml` runs every 4h (UTC, `1 */4 * * *`) and on manual dispatch: it pulls the image and runs `trade-once`.
- Each run pulls the DuckDB candle cache from **Cloudflare R2** (S3-compatible), evaluates, writes all decisions to **Supabase Postgres**, then pushes the DB + audit Parquet back to R2.
- On success it pings **healthchecks.io**; a missed/failed run raises an alert (dead-man's-switch). Trade alerts still go to Discord.

**One-time setup**
1. Create an R2 bucket; note its S3 endpoint + access keys.
2. Create a healthchecks.io check on a 4h+grace schedule; note its ping URL.
3. In the repo, add **Settings → Secrets and variables → Actions → Secrets**:
   `GEMINI_API_KEY` (and/or `GROQ_API_KEY`/`LLM_PROVIDER`/`LLM_MODEL`), `POSTGRES_URL`,
   `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `DISCORD_WEBHOOK_URL`, `STATE_SYNC_BUCKET`,
   `STATE_SYNC_ENDPOINT`, `STATE_SYNC_ACCESS_KEY_ID`, `STATE_SYNC_SECRET_ACCESS_KEY`,
   `HEALTHCHECK_PING_URL`.
4. Add a repo **Variable** `BINANCE_TESTNET_DRY_RUN=true` for the first rollout.

**Rollout**
- Push to the default branch so the image publishes to GHCR.
- Trigger `trade-cron` manually (**Actions → trade-cron → Run workflow**) with `BINANCE_TESTNET_DRY_RUN=true`. Confirm a clean run end-to-end: pull → evaluate → Postgres writes → push → healthcheck ping (logs intended orders, places none).
- Flip the `BINANCE_TESTNET_DRY_RUN` variable to `false` to go live. The 4h schedule takes over.

**Rollback:** GHCR keeps prior tags. Pin `trade-cron.yml`'s image to a previous `:<sha>` (or re-publish an earlier commit). Re-running `trade-once` is idempotent — the reconcile + position-exists gates prevent duplicate entries.

> **Note:** GitHub disables scheduled workflows after 60 days of repo inactivity, and cron timing is best-effort (a few minutes' jitter) — both harmless for a settled 4h candle.
````

- [ ] **Step 2: Commit the docs**

```bash
git add README.md
git commit -m "docs: cloud deployment ($0) — GitHub Actions cron + R2 + healthchecks"
```

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (the prior 265 + the new `state_sync` (6), `monitoring` (4), `cli_trade_once` (2), and scheduler UTC (1) tests). No failures.

- [ ] **Step 4: Validate both workflow files one final time**

Run: `.venv/bin/python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/ci.yml', '.github/workflows/trade-cron.yml']]; print('workflows OK')"`
Expected: `workflows OK`.

- [ ] **Step 5: Restart the local stack (standing instruction)**

Run: `docker compose restart vibe-bot`
Expected: container restarts cleanly; the local `live` path now uses the UTC-pinned scheduler (the cloud path is unaffected — it's GitHub-hosted).

---

## Notes for the implementer

- **Secrets:** never echo secret env values; GitHub masks `secrets.*` in logs automatically. The `docker run -e NAME` (no `=value`) form passes values from the runner env, keeping them off the command line.
- **The cloud path is opt-in:** `state_sync` and `monitoring` are no-ops unless their env vars are set, so local `live`/`trade-once` and CI are unaffected by this change until secrets/variables exist.
- **DuckDB candles are reproducible** — if `pull()` returns False (fresh bucket), `sync_and_evaluate`'s `bootstrap_if_needed` fetches them; the run is just slower, never wrong.
- **Image name must be lowercase** for GHCR — both workflows lowercase `github.repository` before use.
