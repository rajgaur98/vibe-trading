"""Tests for the scheduler's exec-price resolution (the only new pure logic).
The full sync_and_evaluate loop is network-heavy and covered by manual verification."""
from datetime import datetime
from unittest.mock import MagicMock

from vibe_trading.runtime.scheduler import TradingScheduler


def _scheduler_without_init():
    """Build a TradingScheduler instance without running __init__ (which needs DBs/LLM)."""
    return TradingScheduler.__new__(TradingScheduler)


def test_resolve_exec_price_uses_broker_mark_when_available():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = 250.0
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 250.0


def test_resolve_exec_price_falls_back_when_mark_none():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.return_value = None
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0


def test_resolve_exec_price_falls_back_on_broker_error():
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_mark_price.side_effect = Exception("boom")
    assert sched._resolve_exec_price("SOL/USDT", fallback=100.0) == 100.0


def test_record_closed_trades_inserts_and_alerts(monkeypatch):
    sched = _scheduler_without_init()
    fake_conn = MagicMock()
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    factory = MagicMock(return_value=fake_pg)
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    trades = [{
        "trade_id": "t1", "symbol": "BTC/USDT", "action": "long",
        "entry_time": datetime(2026, 6, 1), "entry_price": 100.0,
        "close_time": datetime(2026, 6, 2), "close_price": 110.0,
        "size_usd": 1000.0, "realized_pnl": 99.6, "result": "win",
        "decision_id": "dec-42",
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1            # one INSERT
    assert fake_pg.connect.called and fake_pg.close.called  # own connection lifecycle
    assert len(alerts) == 1 and "BTC/USDT" in alerts[0]

    # the INSERT must link the trade back to the decision that opened it
    insert_call = fake_conn.execute.call_args
    insert_sql, insert_params = insert_call.args[0], insert_call.args[1]
    assert "decision_id" in insert_sql              # column present in the INSERT list
    assert "dec-42" in insert_params                # value threaded through


def test_record_closed_trades_decision_id_defaults_none(monkeypatch):
    """A closed trade lacking a decision_id (e.g. an orphan reconcile) must still
    INSERT cleanly — decision_id falls back to None rather than KeyError-ing."""
    sched = _scheduler_without_init()
    fake_conn = MagicMock()
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase",
                        MagicMock(return_value=fake_pg))
    sched._send_discord_alert = lambda msg: None

    trades = [{
        "trade_id": "t2", "symbol": "ETH/USDT", "action": "short",
        "entry_time": datetime(2026, 6, 1), "entry_price": 100.0,
        "close_time": datetime(2026, 6, 2), "close_price": 90.0,
        "size_usd": 500.0, "realized_pnl": 49.0, "result": "win",
        # NOTE: no "decision_id" key
    }]
    sched._record_closed_trades(trades)

    assert fake_conn.execute.call_count == 1
    insert_params = fake_conn.execute.call_args.args[1]
    assert None in insert_params  # decision_id defaulted to None, no KeyError


def test_record_closed_trades_empty_is_noop(monkeypatch):
    sched = _scheduler_without_init()
    factory = MagicMock()
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    sched._record_closed_trades([])
    assert factory.call_count == 0  # no connection opened
    assert alerts == []


def test_maybe_start_ws_listener_none_when_not_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    sched = _scheduler_without_init()
    assert sched._maybe_start_ws_listener() is None


def test_maybe_start_ws_listener_starts_in_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", lambda *a, **k: MagicMock())
    monkeypatch.setattr("vibe_trading.runtime.scheduler.BinanceFuturesBroker", lambda *a, **k: MagicMock())

    started = {}

    class FakeListener:
        def __init__(self, broker, record_fn):
            started["init"] = True

        def start(self):
            started["start"] = True

    monkeypatch.setattr("vibe_trading.runtime.ws_listener.UserDataStreamListener", FakeListener)

    listener = sched._maybe_start_ws_listener()
    assert isinstance(listener, FakeListener)
    assert started.get("init") and started.get("start")


def test_maybe_start_ws_listener_failopen_returns_none(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched._record_closed_trades = lambda closed: None

    def _boom(*a, **k):
        raise RuntimeError("no creds")

    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", _boom)
    # A listener-construction failure must NOT propagate (scheduler keeps running).
    assert sched._maybe_start_ws_listener() is None


# --- kill switch + spike alarm wrappers (the scheduler glue around the pure cost fns) ---

def _cost_sched(monkeypatch, today_usd, raise_summary=False):
    sched = _scheduler_without_init()
    sched.pg_db = MagicMock()
    sched._cost_blocked_on = None
    sched._cost_alarmed_on = None
    alerts = []
    sched._send_discord_alert = lambda m: alerts.append(m)
    if raise_summary:
        def _ds(conn):
            raise Exception("db down")
    else:
        def _ds(conn):
            return {"today_usd": today_usd, "calls": 1, "projected_monthly_usd": today_usd * 30}
    monkeypatch.setattr("vibe_trading.runtime.scheduler.daily_summary", _ds)
    return sched, alerts


def test_trading_blocked_when_over_cap(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "10")
    sched, alerts = _cost_sched(monkeypatch, today_usd=15.0)
    assert sched._trading_blocked_by_cost() is True
    assert any("CAP REACHED" in a for a in alerts)


def test_trading_not_blocked_under_cap(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "10")
    sched, alerts = _cost_sched(monkeypatch, today_usd=3.0)
    assert sched._trading_blocked_by_cost() is False
    assert alerts == []


def test_trading_cap_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "0")
    sched, alerts = _cost_sched(monkeypatch, today_usd=999.0)
    assert sched._trading_blocked_by_cost() is False  # cap<=0 disables the kill switch


def test_trading_blocked_fail_open_on_summary_error(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "10")
    sched, alerts = _cost_sched(monkeypatch, today_usd=0.0, raise_summary=True)
    # A spend-read error must fail OPEN (never halt trading on a logging hiccup).
    assert sched._trading_blocked_by_cost() is False


def test_check_cost_alarm_fires_once_per_day(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_ALARM_USD", "5")
    sched, alerts = _cost_sched(monkeypatch, today_usd=9.0)
    sched._check_cost_alarm()
    sched._check_cost_alarm()  # same day → must not re-alarm
    assert sum("COST ALARM" in a for a in alerts) == 1


# --- live equity snapshot (LIVE_TESTNET dashboard balance/equity/drawdown) ---

def test_snapshot_equity_persists_live_balance(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE_TESTNET")
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    sched.broker.get_balance.return_value = 9500.0
    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = (10000.0,)  # prior peak
    fake_pg = MagicMock()
    fake_pg.conn = fake_conn
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", lambda *a, **k: fake_pg)

    sched._snapshot_equity()

    insert = fake_conn.execute.call_args  # last call = the INSERT
    sql, params = insert.args[0], insert.args[1]
    assert "INSERT INTO portfolio_state" in sql
    assert 9500.0 in params           # live balance persisted
    assert 10000.0 in params          # peak = max(prior 10000, 9500)
    assert fake_pg.connect.called and fake_pg.close.called


def test_snapshot_equity_noop_when_not_testnet(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    sched = _scheduler_without_init()
    sched.broker = MagicMock()
    factory = MagicMock()
    monkeypatch.setattr("vibe_trading.runtime.scheduler.PostgresDatabase", factory)

    sched._snapshot_equity()
    assert factory.call_count == 0                 # no connection opened in PAPER
    sched.broker.get_balance.assert_not_called()


# --- journal RAG retriever wiring ---

def test_build_retriever_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("JOURNAL_RAG_ENABLED", "false")
    from vibe_trading.journal import NoOpRetriever
    sched = _scheduler_without_init()
    assert isinstance(sched._build_retriever(), NoOpRetriever)


def test_build_retriever_real_when_enabled(monkeypatch):
    monkeypatch.setenv("JOURNAL_RAG_ENABLED", "true")
    from vibe_trading.journal import PrecedentRetriever
    sched = _scheduler_without_init()
    assert isinstance(sched._build_retriever(), PrecedentRetriever)


def test_scheduler_imports_journal_for_persistence():
    # The scheduler persists embeddings via journal.persist_embedding; assert it is wired.
    import vibe_trading.runtime.scheduler as sched_mod
    assert hasattr(sched_mod.journal, "persist_embedding")
    conn = MagicMock()
    sched_mod.journal.persist_embedding(conn, "d1", "BTC/USDT", "ts", "long", 100.0, "card", [0.1])
    assert "INSERT INTO decision_embeddings" in conn.execute.call_args.args[0]


def test_build_scheduler_is_utc():
    """The live-path APScheduler must be pinned to UTC, not the host's implicit tz,
    so the 4h cron aligns with the UTC candle boundaries the rest of the code assumes."""
    from vibe_trading.runtime.scheduler import TradingScheduler
    # __new__ avoids __init__'s broker/DB/network setup — we only test the factory.
    s = TradingScheduler.__new__(TradingScheduler)
    sched = s._build_scheduler()
    assert str(sched.timezone) == "UTC"


def test_tick_runs_online_scoring_after_evaluate(monkeypatch):
    """The deployed `live` loop must score matured decisions every tick, not only the
    `trade-once` CLI path. _tick() runs sync_and_evaluate and then a scoring pass."""
    import vibe_trading.eval.online as online_mod
    from vibe_trading.runtime.scheduler import TradingScheduler

    calls = []
    sched = TradingScheduler.__new__(TradingScheduler)
    sched.sync_and_evaluate = lambda: calls.append("evaluate")
    monkeypatch.setattr(online_mod, "run_scoring_pass",
                        lambda *a, **k: (calls.append("score"), {"outcomes_scored": 0, "judged": 0})[1])

    sched._tick()

    assert calls == ["evaluate", "score"]  # evaluate first, then score


def test_build_scheduler_cron_runs_the_full_tick():
    """The 4h cron must target the full tick (evaluate + score). If it targeted
    sync_and_evaluate directly, the live deployment would silently never score."""
    from vibe_trading.runtime.scheduler import TradingScheduler
    s = TradingScheduler.__new__(TradingScheduler)
    sched = s._build_scheduler()
    jobs = sched.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].func.__name__ == "_tick"


def test_tick_scoring_failure_does_not_break_the_loop(monkeypatch):
    """A scoring failure must never fail the trade tick — evaluate already happened."""
    import vibe_trading.eval.online as online_mod
    from vibe_trading.runtime.scheduler import TradingScheduler

    calls = []
    sched = TradingScheduler.__new__(TradingScheduler)
    sched.sync_and_evaluate = lambda: calls.append("evaluate")

    def _boom(*a, **k):
        raise RuntimeError("scoring exploded")
    monkeypatch.setattr(online_mod, "run_scoring_pass", _boom)

    sched._tick()  # must not raise

    assert calls == ["evaluate"]


def test_tick_pings_dead_mans_switch_on_success(monkeypatch):
    """A healthy live tick must ping the dead-man's-switch so a *missed* tick (stalled
    or crashed scheduler) trips the external monitor. This ping was previously only in
    the trade-once path, so the `live` deployment had no silent-outage detection."""
    import vibe_trading.runtime.monitoring as monitoring
    import vibe_trading.eval.online as online_mod
    from vibe_trading.runtime.scheduler import TradingScheduler, TickOutcome

    pings = []
    sched = TradingScheduler.__new__(TradingScheduler)
    sched.sync_and_evaluate = lambda: TickOutcome()  # healthy window
    monkeypatch.setattr(online_mod, "run_scoring_pass", lambda *a, **k: {})
    monkeypatch.setattr(monitoring, "ping_healthcheck", lambda success=True: pings.append(success))

    sched._tick()

    assert pings == [True]


def test_tick_failure_pings_dead_mans_switch_fail_and_does_not_propagate(monkeypatch):
    """If the tick body raises, _tick pings the /fail endpoint and swallows the error so
    the long-running scheduler keeps ticking (and retrying) rather than dying silently."""
    import vibe_trading.runtime.monitoring as monitoring
    from vibe_trading.runtime.scheduler import TradingScheduler

    pings = []
    sched = TradingScheduler.__new__(TradingScheduler)
    def _boom():
        raise RuntimeError("tick exploded")
    sched.sync_and_evaluate = _boom
    monkeypatch.setattr(monitoring, "ping_healthcheck", lambda success=True: pings.append(success))

    sched._tick()  # must not raise

    assert pings == [False]


# --- decision_log prompt_version stamping ---

def test_decision_log_insert_carries_bundle_prompt_version(monkeypatch):
    """After a tick that logs one (flat) decision, the decision_log INSERT's SQL
    names prompt_version and its params include prompts.bundle_version()."""
    from vibe_trading.runtime.decision_pipeline import DecisionResult
    from vibe_trading.agents import prompts
    import vibe_trading.audit as audit_mod

    sched = _scheduler_without_init()
    sched.symbols = ["BTC/USDT"]

    # Fetcher: no trending-symbol lookup needed (symbols pre-set), just bootstrap/incremental no-ops.
    sched.fetcher = MagicMock()
    sched.fetcher.bootstrap_if_needed = lambda *a, **k: None
    sched.fetcher.incremental_update = lambda *a, **k: None

    # Broker: no open positions, a spot fallback mark price.
    sched.broker = MagicMock()
    sched.broker.get_open_positions.return_value = []
    sched.broker.get_mark_price.return_value = 100.0

    # DuckDB side (candle reads) — same connection object reused across connect() calls.
    fake_db_conn = MagicMock()
    fake_db_conn.execute.return_value.fetchone.return_value = (datetime(2026, 6, 1), 100.0)
    sched.db = MagicMock()
    sched.db.conn = fake_db_conn

    # Cost gates: never block/alarm.
    sched._check_cost_alarm = lambda: None
    sched._trading_blocked_by_cost = lambda: False
    sched._snapshot_equity = lambda: None
    sched._record_closed_trades = lambda closed: None
    sched._send_discord_alert = lambda msg: None

    # Decision pipeline: force a FLAT decision so the tick logs to decision_log and stops
    # (no risk-manager/broker.submit_order path needs mocking).
    proposal = {
        "decision_id": "dec-1", "timestamp": "2026-06-01T00:00:00", "symbol": "BTC/USDT",
        "action": "flat", "stop_loss_strategy": "n/a", "take_profit_strategy": "n/a",
        "risk_reward_ratio": 0.0, "reasoning_summary": "no edge",
    }
    fake_report = MagicMock()
    fake_report.model_dump.return_value = {}
    result = DecisionResult(
        symbol="BTC/USDT", status="flat", analyst_report=fake_report,
        snapshot={"close": 100.0}, proposal=proposal, trace_id="trace-1",
    )
    sched.decision_pipeline = MagicMock()
    sched.decision_pipeline.run_symbol.return_value = result

    # Postgres side — capture the decision_log INSERT.
    pg_execute_calls = []
    fake_pg_conn = MagicMock()
    fake_pg_conn.execute.side_effect = lambda *a, **k: pg_execute_calls.append(
        MagicMock(args=a, kwargs=k))
    sched.pg_db = MagicMock()
    sched.pg_db.conn = fake_pg_conn

    monkeypatch.setattr(audit_mod, "append_decision", lambda record: None)

    sched.sync_and_evaluate()

    insert_calls = [c for c in pg_execute_calls
                    if "INSERT OR IGNORE INTO decision_log" in c.args[0]]
    assert insert_calls, "no decision_log INSERT captured"
    sql, params = insert_calls[0].args[0], insert_calls[0].args[1]
    assert "prompt_version" in sql
    assert prompts.bundle_version() in params


# --- per-symbol isolation + honest tick outcome ---

class ServiceUnavailableError(Exception):
    """Stand-in for litellm.ServiceUnavailableError (NOT a RuntimeError, so the
    decision pipeline's analyst guard doesn't catch it)."""


def _loop_sched(monkeypatch, symbols, run_symbol):
    """A scheduler wired with mocks for a full sync_and_evaluate pass over `symbols`.
    `run_symbol(sym, last_ts, exec_price)` drives the decision pipeline per symbol.
    Returns (sched, alerts, decision_log_params)."""
    from vibe_trading.runtime.decision_pipeline import DecisionResult
    import vibe_trading.audit as audit_mod

    sched = _scheduler_without_init()
    sched.symbols = list(symbols)
    sched.fetcher = MagicMock()
    sched.broker = MagicMock()
    sched.broker.get_open_positions.return_value = []
    sched.broker.get_mark_price.return_value = 100.0
    fake_db_conn = MagicMock()
    fake_db_conn.execute.return_value.fetchone.return_value = (datetime(2026, 6, 1), 100.0)
    sched.db = MagicMock()
    sched.db.conn = fake_db_conn
    sched._check_cost_alarm = lambda: None
    sched._trading_blocked_by_cost = lambda: False
    sched._snapshot_equity = lambda: None
    sched._record_closed_trades = lambda closed: None
    alerts = []
    sched._send_discord_alert = lambda msg: alerts.append(msg)

    def _flat(sym):
        report = MagicMock()
        report.model_dump.return_value = {}
        proposal = {
            "decision_id": f"dec-{sym}", "timestamp": "2026-06-01T00:00:00", "symbol": sym,
            "action": "flat", "stop_loss_strategy": "n/a", "take_profit_strategy": "n/a",
            "risk_reward_ratio": 0.0, "reasoning_summary": "no edge",
        }
        return DecisionResult(symbol=sym, status="flat", analyst_report=report,
                              snapshot={"close": 100.0}, proposal=proposal, trace_id="t")

    sched.decision_pipeline = MagicMock()
    sched.decision_pipeline.run_symbol.side_effect = lambda sym, ts, px: run_symbol(sym, _flat)

    logged = []
    fake_pg_conn = MagicMock()
    fake_pg_conn.execute.side_effect = lambda sql, params=None: (
        logged.append(params) if "INSERT OR IGNORE INTO decision_log" in sql else None)
    sched.pg_db = MagicMock()
    sched.pg_db.conn = fake_pg_conn
    monkeypatch.setattr(audit_mod, "append_decision", lambda record: None)
    return sched, alerts, logged


def test_one_symbol_llm_failure_does_not_skip_remaining_symbols(monkeypatch):
    """Prod 2026-09-22: the first symbol's Gemini 503 aborted the whole tick, so NO
    symbol was evaluated. Each symbol must be isolated: log, count, move on."""
    def run_symbol(sym, flat):
        if sym == "BTC/USDT":
            raise ServiceUnavailableError("503 gemma-4-31b-it overloaded")
        return flat(sym)

    sched, alerts, logged = _loop_sched(
        monkeypatch, ["BTC/USDT", "ETH/USDT", "SOL/USDT"], run_symbol)

    outcome = sched.sync_and_evaluate()

    assert sched.decision_pipeline.run_symbol.call_count == 3
    assert [p[2] for p in logged] == ["ETH/USDT", "SOL/USDT"]  # decisions still logged
    assert outcome.attempted == 3
    assert outcome.evaluated == 2
    assert list(outcome.failures) == ["BTC/USDT"]
    assert outcome.global_error is None
    assert outcome.healthy  # partial failure still counts as a working tick


def test_symbol_failures_send_one_summary_alert(monkeypatch):
    """N failing symbols → ONE Discord summary listing them, not N alerts."""
    def run_symbol(sym, flat):
        raise ServiceUnavailableError("503")

    sched, alerts, logged = _loop_sched(
        monkeypatch, ["BTC/USDT", "ETH/USDT", "SOL/USDT"], run_symbol)

    outcome = sched.sync_and_evaluate()

    assert len(alerts) == 1
    assert all(s in alerts[0] for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT"))
    assert "ServiceUnavailableError" in alerts[0]
    assert outcome.attempted == 3 and outcome.evaluated == 0
    assert not outcome.healthy  # attempted symbols but evaluated none


def test_soft_pipeline_skips_count_as_failures(monkeypatch):
    """analyst_failed / no_snapshot produce no decision, so they must not count as
    'evaluated' (else a tick of all-skips would look healthy)."""
    from vibe_trading.runtime.decision_pipeline import DecisionResult

    def run_symbol(sym, flat):
        return DecisionResult(sym, "analyst_failed")

    sched, alerts, logged = _loop_sched(monkeypatch, ["BTC/USDT"], run_symbol)

    outcome = sched.sync_and_evaluate()

    assert outcome.attempted == 1 and outcome.evaluated == 0
    assert "BTC/USDT" in outcome.failures
    assert not outcome.healthy


def test_no_failures_sends_no_summary_alert(monkeypatch):
    sched, alerts, logged = _loop_sched(
        monkeypatch, ["BTC/USDT", "ETH/USDT"], lambda sym, flat: flat(sym))

    outcome = sched.sync_and_evaluate()

    assert alerts == []
    assert outcome.attempted == 2 and outcome.evaluated == 2 and outcome.healthy


def test_global_failure_is_reported_in_outcome(monkeypatch):
    """A window-wide failure (candle fetch, DB) still aborts the window via the outer
    guard, alerts once, and flags the outcome so the health ping goes red."""
    sched, alerts, logged = _loop_sched(
        monkeypatch, ["BTC/USDT"], lambda sym, flat: flat(sym))
    sched.fetcher.incremental_update.side_effect = RuntimeError("exchange down")

    outcome = sched.sync_and_evaluate()

    assert sched.decision_pipeline.run_symbol.call_count == 0
    assert outcome.global_error and "exchange down" in outcome.global_error
    assert not outcome.healthy
    assert len(alerts) == 1 and "SCHEDULER ERROR" in alerts[0]


def test_tick_outcome_health_rule():
    from vibe_trading.runtime.scheduler import TickOutcome
    assert TickOutcome().healthy                                   # nothing to evaluate
    assert TickOutcome(attempted=3, evaluated=1, failures={"a": "x", "b": "y"}).healthy
    assert not TickOutcome(attempted=3, evaluated=0).healthy       # evaluated nothing
    assert not TickOutcome(global_error="db down").healthy


def _tick_with_outcome(monkeypatch, outcome):
    import vibe_trading.runtime.monitoring as monitoring
    import vibe_trading.eval.online as online_mod

    calls = []
    sched = _scheduler_without_init()
    sched.sync_and_evaluate = lambda: outcome
    monkeypatch.setattr(online_mod, "run_scoring_pass",
                        lambda *a, **k: (calls.append("score"), {})[1])
    monkeypatch.setattr(monitoring, "ping_healthcheck",
                        lambda success=True: calls.append(f"ping:{success}"))
    sched._tick()
    return calls


def test_tick_pings_fail_when_no_symbol_evaluated(monkeypatch):
    """The dead-man's-switch must not stay green while the bot evaluates nothing
    (prod 2026-09-22: every tick 503'd, healthchecks.io stayed green)."""
    from vibe_trading.runtime.scheduler import TickOutcome
    calls = _tick_with_outcome(
        monkeypatch, TickOutcome(attempted=10, evaluated=0, failures={"BTC/USDT": "503"}))
    assert calls == ["score", "ping:False"]  # scoring (no LLM) still runs


def test_tick_pings_fail_on_global_error(monkeypatch):
    from vibe_trading.runtime.scheduler import TickOutcome
    calls = _tick_with_outcome(monkeypatch, TickOutcome(global_error="db down"))
    assert calls[-1] == "ping:False"


def test_tick_pings_success_on_partial_failure(monkeypatch):
    from vibe_trading.runtime.scheduler import TickOutcome
    calls = _tick_with_outcome(
        monkeypatch, TickOutcome(attempted=10, evaluated=9, failures={"BTC/USDT": "503"}))
    assert calls[-1] == "ping:True"
