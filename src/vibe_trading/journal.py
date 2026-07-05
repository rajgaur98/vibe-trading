import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import litellm

from vibe_trading.data.db import adapt_embedding, EMBEDDING_DIM

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
PRECEDENT_K = int(os.getenv("JOURNAL_PRECEDENT_K", "4"))
COUNTERFACTUAL_HORIZON_CANDLES = int(os.getenv("JOURNAL_COUNTERFACTUAL_HORIZON_CANDLES", "6"))
_CANDLE_HOURS = 4  # the bot trades the 4h timeframe


@dataclass
class Precedent:
    symbol: str
    action: str
    when: str          # ISO date of the past decision
    similarity: float
    kind: str          # "closed" | "counterfactual"
    outcome_pct: float
    outcome_label: str


@dataclass
class RetrievalResult:
    embedding: Optional[list]
    precedents: list   # list[Precedent]


def build_setup_card(analyst_report, snapshot: dict) -> str:
    """Compact, deterministic text representation of a setup — what gets embedded.
    Combines the analyst's qualitative read with the snapshot's regime labels."""
    a = analyst_report
    s = snapshot or {}
    return (
        f"bias={a.market_bias}; volume={a.volume_confirmation}; confluence={a.confluence_score}; "
        f"rsi={s.get('rsi_regime')}; macd={s.get('macd_regime')}; adx={s.get('adx_regime')}; "
        f"obv={s.get('obv_trend')}; support_prox={s.get('support_proximity')}; "
        f"resistance_prox={s.get('resistance_proximity')}; pattern={s.get('candlestick_pattern')}; "
        f"funding={s.get('funding_rate')}; oi={s.get('open_interest_trend')}; "
        f"thesis={a.thesis}"
    )


def cosine_topk(query, candidates, k):
    """`candidates`: list of (key, vector). Returns the top-k [(key, score)] by cosine
    similarity, descending. Skips zero/length-mismatched vectors; [] on empty/zero query."""
    import numpy as np
    if query is None or not candidates:
        return []
    q = np.asarray(query, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    scored = []
    for key, vec in candidates:
        v = np.asarray(vec, dtype=float)
        vn = np.linalg.norm(v)
        if vn == 0 or v.shape != q.shape:
            continue
        scored.append((key, float(q.dot(v) / (qn * vn))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def pgvector_topk_sql(halfvec: bool, dim: int) -> str:
    """Top-k cosine retrieval in SQL. With halfvec=True the cast expression is
    byte-identical to the HNSW expression index (db.pgvector_migration_statements)
    so the planner can use it; without halfvec it is a correct (unindexed) scan
    on the vector column. Params: (query, cutoff, query, k)."""
    if halfvec:
        col = f"embedding::halfvec({dim})"
        q = f"?::halfvec({dim})"
    else:
        col, q = "embedding", "?"
    return (
        "SELECT decision_id, symbol, timestamp, action, entry_price, "
        f"1 - ({col} <=> {q}) AS similarity "
        "FROM decision_embeddings WHERE timestamp < ? "
        f"ORDER BY {col} <=> {q} LIMIT ?"
    )


def embed(text: str, model: str = None) -> Optional[list]:
    """Embed `text` via LiteLLM (Gemini by default). Returns the vector, or None on any
    error (rate limit, network, bad model) — retrieval then degrades to no precedents."""
    try:
        resp = litellm.embedding(model=model or EMBEDDING_MODEL, input=[text])
        return list(resp.data[0]["embedding"])
    except Exception as e:
        logger.warning(f"journal embed failed (non-fatal): {e}")
        return None


def persist_embedding(conn, decision_id, symbol, timestamp, action, entry_price,
                      setup_text, embedding) -> None:
    """Append this decision's setup embedding to decision_embeddings, keyed by decision_id,
    so it becomes a future precedent once its outcome lands. `conn` is a connected DB
    wrapper (caller owns connect/close). No-op when embedding is None; never raises."""
    if embedding is None:
        return
    try:
        conn.execute(
            "INSERT INTO decision_embeddings "
            "(decision_id, symbol, timestamp, action, entry_price, setup_text, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (decision_id, symbol, timestamp, action, entry_price, setup_text,
             adapt_embedding(embedding)),
        )
    except Exception as e:
        logger.error(f"journal persist_embedding failed (non-fatal): {e}")


class NoOpRetriever:
    """Used by the eval and whenever JOURNAL_RAG_ENABLED is false — the trader then
    behaves exactly as before (no precedents, no embedding)."""
    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        return RetrievalResult(None, [])


class PrecedentRetriever:
    """Semantic retrieval over the decision journal: embed the current setup, cosine-rank
    past decisions (older than the counterfactual horizon, so their outcome is known),
    take top-k, attach each outcome. All DB access is fail-soft."""

    def __init__(self, k: int = PRECEDENT_K, horizon_candles: int = COUNTERFACTUAL_HORIZON_CANDLES,
                 pg_factory=None, duck_factory=None, embed_fn=embed, now_fn=None,
                 use_pgvector: Optional[bool] = None):
        self.k = k
        self.horizon_candles = horizon_candles
        self._embed = embed_fn
        self._now = now_fn or (lambda: datetime.utcnow())
        self._pg_factory = pg_factory
        self._duck_factory = duck_factory
        self.use_pgvector = use_pgvector

    def _pg(self):
        if self._pg_factory:
            return self._pg_factory()
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase()

    def _duck(self):
        if self._duck_factory:
            return self._duck_factory()
        from vibe_trading.data.db import Database
        return Database(read_only=True)

    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        emb = self._embed(setup_text)
        if emb is None:
            return RetrievalResult(None, [])
        try:
            precedents = self.retrieve(emb)
        except Exception as e:
            logger.error(f"journal retrieve failed (non-fatal): {e}")
            precedents = []
        return RetrievalResult(emb, precedents)

    def retrieve(self, embedding) -> list:
        cutoff = self._now() - timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        if self._pgvector_active():
            ranked = self._load_topk_pgvector(embedding, cutoff)
        else:
            rows = self._load_candidates(cutoff)         # [(id, symbol, ts, action, entry, vector)]
            ranked = cosine_topk(embedding, [(row, row[5]) for row in rows], self.k)
        out = []
        for row, score in ranked:
            p = self._attach_outcome(row, score)
            if p is not None:
                out.append(p)
        return out

    def _pgvector_active(self) -> bool:
        if self.use_pgvector is not None:
            return self.use_pgvector
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase.pgvector_enabled

    def _load_topk_pgvector(self, embedding, cutoff):
        """[(row, similarity)] where row matches _load_candidates' shape (vector
        slot None — _attach_outcome never reads it)."""
        from vibe_trading.data.db import PostgresDatabase
        sql = pgvector_topk_sql(PostgresDatabase.pgvector_halfvec, EMBEDDING_DIM)
        q = adapt_embedding(embedding)
        pg = self._pg()
        pg.connect()
        try:
            rows = pg.conn.execute(sql, (q, cutoff, q, self.k)).fetchall()
        finally:
            pg.close()
        return [((r[0], r[1], r[2], r[3], r[4], None), float(r[5])) for r in rows]

    def _load_candidates(self, cutoff):
        pg = self._pg()
        pg.connect()
        try:
            rows = pg.conn.execute(
                "SELECT decision_id, symbol, timestamp, action, entry_price, embedding "
                "FROM decision_embeddings WHERE timestamp < ?",
                (cutoff,),
            ).fetchall()
            return [tuple(r) for r in rows]
        finally:
            pg.close()

    def _attach_outcome(self, row, score):
        decision_id, symbol, ts, action, entry_price, _vec = row
        when = ts.date().isoformat() if hasattr(ts, "date") else str(ts)

        # 1) Did this decision become a closed trade? -> real outcome.
        pg = self._pg()
        pg.connect()
        try:
            trade = pg.conn.execute(
                "SELECT result, realized_pnl, size_usd FROM trades WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        finally:
            pg.close()
        if trade is not None:
            result, realized_pnl, size_usd = trade
            pct = (float(realized_pnl) / float(size_usd) * 100.0) if size_usd else 0.0
            return Precedent(symbol, action, when, score, "closed", pct,
                             f"traded {action} -> {result} {pct:+.1f}%")

        # 2) Otherwise (FLAT / risk-rejected) -> counterfactual forward return from candles.
        if not entry_price:
            return None
        cutoff = ts + timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        duck = self._duck()
        duck.connect()
        try:
            fut = duck.conn.execute(
                "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                "AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                (symbol, cutoff),
            ).fetchone()
        finally:
            duck.close()
        if not fut or fut[0] is None:
            return None  # horizon not elapsed / data gap -> drop, never fabricate
        fwd = (float(fut[0]) - float(entry_price)) / float(entry_price) * 100.0
        # sign by the proposed direction: a long profits on an up-move, a short on a down-move
        signed = -fwd if action == "short" else fwd
        if action == "flat":
            label = f"skipped (flat); price moved {fwd:+.1f}% over {self.horizon_candles * _CANDLE_HOURS}h"
        else:
            label = f"{action} skipped (risk veto); would have {signed:+.1f}%"
        return Precedent(symbol, action, when, score, "counterfactual", signed, label)


class ReplayJournal:
    """Backtest-time journal RAG (spec D3): accumulates decisions during a
    chronological replay and retrieves precedents with NO lookahead —
      * a decision becomes a candidate only once it is older than the
        counterfactual horizon relative to the replay clock (current_ts), and
      * its counterfactual candle is bounded by not_after=current_ts, so a data
        gap can never leak a future price into the replay past.
    In-memory only; nothing is persisted."""

    def __init__(self, k: int = PRECEDENT_K,
                 horizon_candles: int = COUNTERFACTUAL_HORIZON_CANDLES,
                 embed_fn=embed, candle_close_fn=None):
        self.k = k
        self.horizon_candles = horizon_candles
        self._embed = embed_fn
        # (symbol, target_ts, not_after_ts) -> first 4h close in [target, not_after]
        self._candle_close = candle_close_fn
        self.current_ts = None            # set by the engine each replay step
        self._records: list = []          # dicts, insertion-ordered (chronological)
        self._trades: dict = {}           # decision_id -> (result, pnl, size)

    def record_decision(self, decision_id, symbol, ts, action, entry_price,
                        embedding) -> None:
        if embedding is None:
            return
        self._records.append({"decision_id": decision_id, "symbol": symbol,
                              "ts": ts, "action": action,
                              "entry_price": entry_price, "embedding": embedding})

    def record_closed_trade(self, trade: dict) -> None:
        decision_id = trade.get("decision_id")
        if decision_id:
            self._trades[decision_id] = (trade["result"], trade["realized_pnl"],
                                         trade["size_usd"])

    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        emb = self._embed(setup_text)
        if emb is None or self.current_ts is None:
            return RetrievalResult(emb, [])
        horizon = timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        cutoff = self.current_ts - horizon
        candidates = [(r, r["embedding"]) for r in self._records if r["ts"] < cutoff]
        ranked = cosine_topk(emb, candidates, self.k)
        out = []
        for rec, score in ranked:
            p = self._attach(rec, score, horizon)
            if p is not None:
                out.append(p)
        return RetrievalResult(emb, out)

    def _attach(self, rec: dict, score: float, horizon) -> Optional[Precedent]:
        when = rec["ts"].date().isoformat() if hasattr(rec["ts"], "date") else str(rec["ts"])
        action = rec["action"]
        trade = self._trades.get(rec["decision_id"])
        if trade is not None:
            result, pnl, size = trade
            pct = (float(pnl) / float(size) * 100.0) if size else 0.0
            return Precedent(rec["symbol"], action, when, score, "closed", pct,
                             f"traded {action} -> {result} {pct:+.1f}%")
        if not rec["entry_price"] or self._candle_close is None:
            return None
        fut = self._candle_close(rec["symbol"], rec["ts"] + horizon, self.current_ts)
        if fut is None:
            return None  # gap — drop, never fabricate (matches PrecedentRetriever)
        fwd = (float(fut) - float(rec["entry_price"])) / float(rec["entry_price"]) * 100.0
        signed = -fwd if action == "short" else fwd
        if action == "flat":
            label = (f"skipped (flat); price moved {fwd:+.1f}% over "
                     f"{self.horizon_candles * _CANDLE_HOURS}h")
        else:
            label = f"{action} skipped (risk veto); would have {signed:+.1f}%"
        return Precedent(rec["symbol"], action, when, score, "counterfactual",
                         signed, label)
