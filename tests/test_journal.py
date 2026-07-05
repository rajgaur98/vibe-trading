from types import SimpleNamespace

import pytest

from vibe_trading.journal import (
    build_setup_card, cosine_topk, NoOpRetriever, RetrievalResult, Precedent,
)


def _analyst(**over):
    base = dict(market_bias="bullish", volume_confirmation="confirmed", confluence_score=0.8,
                thesis="uptrend holding above support")
    base.update(over)
    return SimpleNamespace(**base)


def _snapshot(**over):
    base = dict(rsi_regime="overbought", macd_regime="bullish", adx_regime="strong_trend",
                obv_trend="rising", support_proximity="near", resistance_proximity="immediate_contact",
                candlestick_pattern="none", funding_rate="positive", open_interest_trend="rising",
                close=100.0)
    base.update(over)
    return base


def test_build_setup_card_deterministic_and_includes_fields():
    card = build_setup_card(_analyst(), _snapshot())
    assert "bias=bullish" in card and "rsi=overbought" in card and "adx=strong_trend" in card
    assert "thesis=uptrend holding above support" in card
    assert build_setup_card(_analyst(), _snapshot()) == card


def test_build_setup_card_tolerates_missing_snapshot_keys():
    card = build_setup_card(_analyst(), {})
    assert "rsi=None" in card  # missing keys render as None, never raise


def test_cosine_topk_ranks_by_similarity():
    q = [1.0, 0.0]
    cands = [("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [0.7, 0.7])]
    top = cosine_topk(q, cands, k=2)
    assert [k for k, _ in top] == ["a", "c"]
    assert top[0][1] > top[1][1]


def test_cosine_topk_empty_and_zero_vectors():
    assert cosine_topk([1.0, 0.0], [], k=3) == []
    assert cosine_topk(None, [("a", [1.0])], k=3) == []
    assert cosine_topk([0.0, 0.0], [("a", [1.0, 1.0])], k=3) == []


def test_noop_retriever_returns_empty():
    r = NoOpRetriever().retrieve_for("any setup")
    assert isinstance(r, RetrievalResult)
    assert r.embedding is None and r.precedents == []


from unittest.mock import MagicMock, patch
from vibe_trading.journal import embed, persist_embedding


def test_embed_returns_vector():
    fake = MagicMock()
    fake.data = [{"embedding": [0.1, 0.2, 0.3]}]
    with patch("vibe_trading.journal.litellm.embedding", return_value=fake) as emb:
        out = embed("setup card text")
    assert out == [0.1, 0.2, 0.3]
    assert emb.call_args.kwargs["model"] == "gemini/gemini-embedding-001"


def test_embed_returns_none_on_error():
    with patch("vibe_trading.journal.litellm.embedding", side_effect=Exception("rate limited")):
        assert embed("x") is None


def test_persist_embedding_inserts():
    conn = MagicMock()
    persist_embedding(conn, "d1", "BTC/USDT", "2026-06-01", "long", 100.0, "card", [0.1, 0.2])
    sql, params = conn.execute.call_args.args
    assert "INSERT INTO decision_embeddings" in sql
    assert params[0] == "d1" and params[-1] == [0.1, 0.2]


def test_persist_embedding_noop_on_none():
    conn = MagicMock()
    persist_embedding(conn, "d1", "BTC/USDT", "2026-06-01", "long", 100.0, "card", None)
    conn.execute.assert_not_called()


from datetime import datetime
from vibe_trading.journal import PrecedentRetriever


def _retriever_with_candidates(rows, attach=None):
    r = PrecedentRetriever(k=2, horizon_candles=6, embed_fn=lambda t: [1.0, 0.0])
    r._load_candidates = lambda cutoff: rows
    if attach is not None:
        r._attach_outcome = attach
    return r


def test_retrieve_ranks_and_limits_to_k():
    rows = [
        ("d1", "BTC/USDT", datetime(2026, 6, 1), "long", 100.0, [1.0, 0.0]),
        ("d2", "ETH/USDT", datetime(2026, 6, 1), "short", 50.0, [0.0, 1.0]),
        ("d3", "SOL/USDT", datetime(2026, 6, 1), "long", 20.0, [0.7, 0.7]),
    ]
    r = _retriever_with_candidates(
        rows, attach=lambda row, score: Precedent(row[1], row[3], "x", score, "closed", 1.0, "ok")
    )
    out = r.retrieve([1.0, 0.0])
    assert [p.symbol for p in out] == ["BTC/USDT", "SOL/USDT"]


def test_retrieve_for_embeds_then_retrieves():
    r = PrecedentRetriever(k=2, embed_fn=lambda t: [1.0, 0.0])
    r._load_candidates = lambda cutoff: []
    res = r.retrieve_for("setup card")
    assert res.embedding == [1.0, 0.0] and res.precedents == []


def test_retrieve_for_embed_failure_returns_empty():
    r = PrecedentRetriever(embed_fn=lambda t: None)
    res = r.retrieve_for("setup card")
    assert res.embedding is None and res.precedents == []


def test_retrieve_recency_cutoff_passed_to_loader():
    captured = {}
    r = PrecedentRetriever(k=2, horizon_candles=6, embed_fn=lambda t: [1.0, 0.0],
                           now_fn=lambda: datetime(2026, 6, 2, 0, 0, 0))
    def _loader(cutoff):
        captured["cutoff"] = cutoff
        return []
    r._load_candidates = _loader
    r.retrieve([1.0, 0.0])
    assert captured["cutoff"] == datetime(2026, 6, 1, 0, 0, 0)


def _row(action="long", symbol="BTC/USDT", entry=100.0):
    return ("d1", symbol, datetime(2026, 6, 1), action, entry, [1.0, 0.0])


def test_attach_outcome_closed_trade():
    r = PrecedentRetriever()
    pg = MagicMock()
    pg.conn.execute.return_value.fetchone.return_value = ("win", 23.0, 1000.0)
    r._pg = lambda: pg
    p = r._attach_outcome(_row(action="long"), score=0.9)
    assert p.kind == "closed"
    assert round(p.outcome_pct, 2) == 2.3
    assert "win" in p.outcome_label and "+2.3%" in p.outcome_label


def test_attach_outcome_counterfactual_flat():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = (110.0,)
    r._pg = lambda: pg
    r._duck = lambda: duck
    p = r._attach_outcome(_row(action="flat", entry=100.0), score=0.8)
    assert p.kind == "counterfactual"
    assert round(p.outcome_pct, 2) == 10.0
    assert "skipped" in p.outcome_label


def test_attach_outcome_counterfactual_rejected_short_signs_by_action():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = (90.0,)
    r._pg = lambda: pg
    r._duck = lambda: duck
    p = r._attach_outcome(_row(action="short", entry=100.0), score=0.7)
    assert round(p.outcome_pct, 2) == 10.0  # short profits when price falls 10%
    assert "short" in p.outcome_label


def test_attach_outcome_counterfactual_missing_future_candle_drops():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = None
    r._pg = lambda: pg
    r._duck = lambda: duck
    assert r._attach_outcome(_row(action="flat"), score=0.5) is None


def test_pgvector_topk_sql_halfvec_matches_index_expression():
    from vibe_trading.journal import pgvector_topk_sql
    sql = pgvector_topk_sql(halfvec=True, dim=3072)
    # must be byte-identical to the index expression or the HNSW index is skipped
    assert "embedding::halfvec(3072) <=> ?::halfvec(3072)" in sql
    assert "1 - (embedding::halfvec(3072) <=> ?::halfvec(3072)) AS similarity" in sql
    assert "WHERE timestamp < ?" in sql and "LIMIT ?" in sql


def test_pgvector_topk_sql_plain_vector_without_halfvec():
    from vibe_trading.journal import pgvector_topk_sql
    sql = pgvector_topk_sql(halfvec=False, dim=3072)
    assert "halfvec" not in sql
    assert "embedding <=> ?" in sql


def test_retriever_uses_sql_path_when_pgvector_active():
    from unittest.mock import MagicMock
    from datetime import datetime
    from vibe_trading.journal import PrecedentRetriever

    pg = MagicMock()
    pg.conn.execute.return_value.fetchall.return_value = [
        ("dec-1", "BTC/USDT", datetime(2026, 6, 1), "long", 100.0, 0.93),
    ]
    pg.conn.execute.return_value.fetchone.return_value = ("win", 50.0, 1000.0)
    r = PrecedentRetriever(pg_factory=lambda: pg, use_pgvector=True,
                           now_fn=lambda: datetime(2026, 7, 5))
    precedents = r.retrieve([0.1] * 4)
    assert len(precedents) == 1
    assert precedents[0].similarity == pytest.approx(0.93)
    assert precedents[0].kind == "closed"
    ranking_sql = [c.args[0] for c in pg.conn.execute.call_args_list
                   if "ORDER BY" in c.args[0]]
    assert ranking_sql and "<=>" in ranking_sql[0]   # ranking happened in SQL


def test_retriever_fallback_path_unchanged_when_pgvector_off():
    from unittest.mock import MagicMock
    from datetime import datetime
    from vibe_trading.journal import PrecedentRetriever

    pg = MagicMock()
    pg.conn.execute.return_value.fetchall.return_value = []
    r = PrecedentRetriever(pg_factory=lambda: pg, use_pgvector=False,
                           now_fn=lambda: datetime(2026, 7, 5))
    assert r.retrieve([0.1] * 4) == []
    sql = pg.conn.execute.call_args_list[0].args[0]
    assert "<=>" not in sql                 # candidates loaded, ranked in Python
    assert "SELECT decision_id, symbol" in sql


def test_replay_journal_no_lookahead_and_outcomes():
    from datetime import datetime, timedelta
    from vibe_trading.journal import ReplayJournal

    fixed = [0.9, 0.1, 0.0]
    journal_ = ReplayJournal(
        k=4, horizon_candles=6, embed_fn=lambda text: fixed,
        candle_close_fn=lambda sym, target, not_after: 104.0)
    t0 = datetime(2026, 6, 1, 0, 0)

    # decision older than the horizon relative to the replay clock -> retrievable
    journal_.record_decision("dec-old", "BTC/USDT", t0, "long", 100.0, fixed)
    # decision INSIDE the horizon -> must NOT be retrievable (outcome unknown yet)
    journal_.record_decision("dec-new", "BTC/USDT", t0 + timedelta(hours=30),
                             "long", 100.0, fixed)
    journal_.current_ts = t0 + timedelta(hours=36)   # horizon = 24h

    result = journal_.retrieve_for("whatever")
    ids = [p.outcome_label for p in result.precedents]
    assert len(result.precedents) == 1               # only dec-old
    p = result.precedents[0]
    assert p.kind == "counterfactual"
    assert p.outcome_pct == pytest.approx(4.0)       # (104-100)/100, long

    # once its trade closes, the precedent switches to the real outcome
    journal_.record_closed_trade({"decision_id": "dec-old", "result": "win",
                                  "realized_pnl": 80.0, "size_usd": 1000.0})
    p2 = journal_.retrieve_for("whatever").precedents[0]
    assert p2.kind == "closed"
    assert p2.outcome_pct == pytest.approx(8.0)


def test_replay_journal_empty_and_failed_embed_degrade():
    from datetime import datetime
    from vibe_trading.journal import ReplayJournal
    journal_ = ReplayJournal(embed_fn=lambda text: None)
    journal_.current_ts = datetime(2026, 6, 2)
    result = journal_.retrieve_for("x")
    assert result.embedding is None and result.precedents == []
