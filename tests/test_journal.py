from types import SimpleNamespace

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
