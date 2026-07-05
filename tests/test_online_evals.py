import pytest

from vibe_trading.eval.online import directional_score, flat_score


def test_directional_score_maps_band_to_unit_interval():
    assert directional_score(0.0) == pytest.approx(0.5)     # no move -> coin flip
    assert directional_score(2.0) == pytest.approx(1.0)     # +band in favor -> 1.0
    assert directional_score(-2.0) == pytest.approx(0.0)    # band against -> 0.0
    assert directional_score(1.0) == pytest.approx(0.75)    # linear in between
    assert directional_score(50.0) == 1.0                   # clamped
    assert directional_score(-50.0) == 0.0                  # clamped


def test_flat_score_rewards_small_moves():
    assert flat_score(0.0) == 1.0
    assert flat_score(1.0) == 1.0        # within FLAT_OK_PCT
    assert flat_score(-1.0) == 1.0       # sign-agnostic (uses |move|)
    assert flat_score(5.0) == 0.0        # at/after FLAT_ZERO_PCT
    assert flat_score(3.0) == pytest.approx(0.5)  # halfway on the linear ramp


from datetime import datetime, timedelta
from unittest.mock import MagicMock

from vibe_trading.eval.online import Outcome, resolve_outcome, OutcomeScorer


def test_resolve_closed_trade_uses_realized_pnl():
    out = resolve_outcome(action="long", closed_trade=(50.0, 1000.0),  # (pnl, size)
                          is_open=False, entry_price=None, future_price=None)
    assert out.kind == "closed"
    assert out.outcome_pct == pytest.approx(5.0)          # 50/1000 * 100
    assert out.outcome_score == pytest.approx(1.0)        # +5% >> +2% band


def test_resolve_open_position_defers():
    assert resolve_outcome(action="long", closed_trade=None, is_open=True,
                           entry_price=100.0, future_price=110.0) is None


def test_resolve_counterfactual_signs_by_direction():
    # price went +10%: a rejected short would have lost
    out = resolve_outcome(action="short", closed_trade=None, is_open=False,
                          entry_price=100.0, future_price=110.0)
    assert out.kind == "counterfactual"
    assert out.outcome_pct == pytest.approx(-10.0)
    assert out.outcome_score == 0.0


def test_resolve_flat_uses_flat_score_on_raw_move():
    out = resolve_outcome(action="flat", closed_trade=None, is_open=False,
                          entry_price=100.0, future_price=100.5)
    assert out.kind == "counterfactual"
    assert out.outcome_pct == pytest.approx(0.5)
    assert out.outcome_score == 1.0


def test_resolve_missing_candles_returns_none():
    assert resolve_outcome(action="long", closed_trade=None, is_open=False,
                           entry_price=None, future_price=110.0) is None
    assert resolve_outcome(action="long", closed_trade=None, is_open=False,
                           entry_price=100.0, future_price=None) is None


def _fake_pg(rows_by_query: dict):
    """A PostgresDatabase stand-in whose conn.execute returns canned rows keyed by
    a substring of the SQL."""
    pg = MagicMock()
    def execute(sql, params=None):
        cur = MagicMock()
        for needle, rows in rows_by_query.items():
            if needle in sql:
                cur.fetchall.return_value = rows
                cur.fetchone.return_value = rows[0] if rows else None
                return cur
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        return cur
    pg.conn.execute.side_effect = execute
    return pg


def test_run_pass_scores_one_decision_and_pushes_langfuse():
    now = datetime(2026, 7, 5, 12, 0)
    old_ts = now - timedelta(hours=48)  # past the 24h horizon
    pg = _fake_pg({
        "FROM decision_log": [("dec-1", old_ts, "BTC/USDT", "long", "trace-1", "bundle-v1")],
        "FROM trades": [],
        "FROM open_positions": [],
    })
    duck = MagicMock()
    # entry close then future close
    duck.conn.execute.return_value.fetchone.side_effect = [(100.0,), (104.0,)]
    pushed = []
    scorer = OutcomeScorer(pg_factory=lambda: pg, duck_factory=lambda: duck,
                           now_fn=lambda: now,
                           push_fn=lambda *a, **k: pushed.append((a, k)))
    scored = scorer.run_pass()
    assert scored == 1
    insert_sql = [c for c in pg.conn.execute.call_args_list
                  if "INSERT OR IGNORE INTO decision_scores" in c.args[0]]
    assert insert_sql, "no decision_scores INSERT"
    params = insert_sql[0].args[1]
    assert params[0] == "dec-1"
    assert "bundle-v1" in params            # prompt_version copied over
    assert pushed and pushed[0][0][0] == "trace-1"   # trace_id
    assert pushed[0][0][1] == "outcome_score"        # Langfuse score name


def test_run_pass_never_raises(caplog):
    pg = MagicMock()
    pg.connect.side_effect = RuntimeError("db down")
    scorer = OutcomeScorer(pg_factory=lambda: pg)
    assert scorer.run_pass() == 0  # swallowed, logged, zero scored


from vibe_trading.eval.online import OnlineJudge, OnlineJudgeVerdict


def _judge_pg(unjudged_rows, judged_today=0, spend_today=0.0):
    pg = MagicMock()
    def execute(sql, params=None):
        cur = MagicMock()
        if "judged_at >=" in sql:                      # daily cap counter
            cur.fetchone.return_value = (judged_today,)
        elif "FROM llm_cost_log" in sql:               # daily_summary spend query
            cur.fetchone.return_value = (spend_today, 0, 0, 0, 0, 0, 0)
            cur.fetchall.return_value = []
        elif "judge_score IS NULL" in sql:             # sample selection
            cur.fetchall.return_value = unjudged_rows
        else:
            cur.fetchall.return_value = []
            cur.fetchone.return_value = None
        return cur
    pg.conn.execute.side_effect = execute
    return pg


def _row():
    return ("dec-1", "long", "BTC/USDT", "went long on confluence",
            '{"close": 100.0, "rsi_14": 60.0}', "trace-1")


def test_judge_scores_a_sampled_decision():
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"grounded": true, "consistent": false, "justification": "j"}')
    pg = _judge_pg([_row()])
    pushed = []
    judge = OnlineJudge(client=client, pg_factory=lambda: pg,
                        push_fn=lambda *a, **k: pushed.append(a))
    assert judge.run_pass() == 1
    # judge call is tagged for cost attribution
    assert client.call_llm.call_args.kwargs["call_type"] == "online_judge"
    updates = [c for c in pg.conn.execute.call_args_list
               if "UPDATE decision_scores" in c.args[0]]
    assert updates and updates[0].args[1][0] == pytest.approx(0.5)  # (True+False)/2
    assert pushed[0][1] == "online_judge_score"


def test_judge_respects_daily_cap():
    client = MagicMock()
    pg = _judge_pg([_row()], judged_today=5)   # cap (default 5) already reached
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0
    client.call_llm.assert_not_called()


def test_judge_skips_when_llm_cost_cap_reached(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_COST_CAP_USD", "10.0")
    client = MagicMock()
    pg = _judge_pg([_row()], spend_today=10.0)
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0
    client.call_llm.assert_not_called()


def test_unparseable_verdict_is_skipped_not_raised():
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = "not json"
    pg = _judge_pg([_row()])
    judge = OnlineJudge(client=client, pg_factory=lambda: pg)
    assert judge.run_pass() == 0  # no UPDATE, no exception


from vibe_trading.eval.online import weekly_digest, format_digest


def test_weekly_digest_aggregates_and_formats():
    now = datetime(2026, 7, 5)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        (0.62, 0.55, 14),    # week: outcome mean, judge mean, scored count
        (0.58,),             # trailing 4 weeks: outcome mean
        (48, 50),            # week schema: ok_true, ok_total
    ]
    conn.execute.return_value.fetchall.return_value = [
        ("analyst_system:v1@a;trader_system:v1@b", 0.62, 14),
    ]
    d = weekly_digest(conn, now=now)
    assert d["week_outcome_mean"] == pytest.approx(0.62)
    assert d["trailing_outcome_mean"] == pytest.approx(0.58)
    assert d["week_schema_compliance"] == pytest.approx(0.96)
    text = format_digest(d)
    assert "0.62" in text and "0.58" in text
    assert "analyst_system:v1@a" in text
