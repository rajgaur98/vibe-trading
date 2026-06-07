import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

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


class NoOpRetriever:
    """Used by the eval and whenever JOURNAL_RAG_ENABLED is false — the trader then
    behaves exactly as before (no precedents, no embedding)."""
    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        return RetrievalResult(None, [])
