"""Online evaluation — score PRODUCTION decisions once their outcome is knowable.

Three layers (spec workstream C), all strictly best-effort:
  C1  deterministic outcome scoring  (this module's OutcomeScorer)
  C2  sampled generic-rubric LLM judge (capped per day)
  C3  weekly drift digest to Discord

A 'correct' flat is scored as 'no strong move happened' — a proxy for 'no edge
existed', not ground truth (a flat that dodged a crash scores poorly here). The
caveat is deliberate and documented; deterministic beats clever.
"""
import logging

logger = logging.getLogger(__name__)

# Directional decisions: +DIRECTIONAL_BAND_PCT% move in the decision's favor -> 1.0,
# same move against -> 0.0, linear in between, 0.5 for no move.
DIRECTIONAL_BAND_PCT = 2.0
# Flat decisions: |move| <= FLAT_OK_PCT -> 1.0, >= FLAT_ZERO_PCT -> 0.0, linear between.
FLAT_OK_PCT = 1.0
FLAT_ZERO_PCT = 5.0


def directional_score(signed_pct: float) -> float:
    """Map a signed %-move (positive = the decision's direction was profitable)
    onto [0, 1]."""
    return max(0.0, min(1.0, 0.5 + signed_pct / (2.0 * DIRECTIONAL_BAND_PCT)))


def flat_score(move_pct: float) -> float:
    """Score a flat decision by how quiet the market actually was."""
    m = abs(move_pct)
    if m <= FLAT_OK_PCT:
        return 1.0
    if m >= FLAT_ZERO_PCT:
        return 0.0
    return 1.0 - (m - FLAT_OK_PCT) / (FLAT_ZERO_PCT - FLAT_OK_PCT)


from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from vibe_trading.journal import COUNTERFACTUAL_HORIZON_CANDLES, _CANDLE_HOURS


@dataclass
class Outcome:
    kind: str            # "closed" | "counterfactual"
    outcome_pct: float
    outcome_score: float


def resolve_outcome(action: str, closed_trade: Optional[tuple], is_open: bool,
                    entry_price: Optional[float],
                    future_price: Optional[float]) -> Optional[Outcome]:
    """Pure resolution: decide a decision's outcome from pre-fetched facts.

    closed_trade: (realized_pnl, size_usd) when the decision became a closed trade.
    is_open:      the decision's position is still open -> defer (None).
    entry/future_price: 4h closes at decision time and one horizon later (None on gaps).
    Returns None when the outcome is not yet knowable — the caller retries next pass.
    """
    if closed_trade is not None:
        pnl, size = closed_trade
        if not size:
            return None
        pct = float(pnl) / float(size) * 100.0
        return Outcome("closed", pct, directional_score(pct))
    if is_open:
        return None
    if entry_price is None or future_price is None or not entry_price:
        return None
    fwd = (float(future_price) - float(entry_price)) / float(entry_price) * 100.0
    if action == "flat":
        return Outcome("counterfactual", fwd, flat_score(fwd))
    signed = -fwd if action == "short" else fwd
    return Outcome("counterfactual", signed, directional_score(signed))


def push_langfuse_score(trace_id: Optional[str], name: str, value: float,
                        comment: str = "") -> None:
    """Attach a numeric score to the decision's Langfuse trace. Fail-soft: any
    SDK/network error is logged and swallowed."""
    if not trace_id:
        return
    try:
        from langfuse import get_client
        get_client().create_score(trace_id=trace_id, name=name, value=value,
                                  comment=comment)
    except Exception as e:
        logger.warning(f"langfuse score push failed (non-fatal): {e}")


class OutcomeScorer:
    """C1: scan unscored decisions, resolve knowable outcomes, persist + push.
    Mirrors PrecedentRetriever's injected-factory pattern for testability."""

    def __init__(self, pg_factory=None, duck_factory=None,
                 now_fn: Optional[Callable] = None, push_fn=None,
                 horizon_candles: Optional[int] = None):
        self._pg_factory = pg_factory
        self._duck_factory = duck_factory
        self._now = now_fn or (lambda: datetime.utcnow())
        self._push = push_fn or push_langfuse_score
        self.horizon_candles = horizon_candles or COUNTERFACTUAL_HORIZON_CANDLES

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

    def run_pass(self, batch_limit: int = 200) -> int:
        """Score every unscored decision whose outcome is knowable. Returns the
        number scored. Never raises."""
        try:
            return self._run_pass(batch_limit)
        except Exception as e:
            logger.warning(f"outcome scoring pass failed (non-fatal): {e}")
            return 0

    def _run_pass(self, batch_limit: int) -> int:
        pg = self._pg()
        pg.connect()
        try:
            candidates = pg.conn.execute(
                "SELECT d.decision_id, d.timestamp, d.symbol, d.action, d.trace_id, "
                "       d.prompt_version "
                "FROM decision_log d LEFT JOIN decision_scores s "
                "  ON s.decision_id = d.decision_id "
                "WHERE s.decision_id IS NULL AND d.action != 'close' "
                "ORDER BY d.timestamp ASC LIMIT ?",
                (batch_limit,),
            ).fetchall()
        finally:
            pg.close()

        scored = 0
        horizon = timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        for decision_id, ts, symbol, action, trace_id, prompt_version in candidates:
            try:
                outcome = self._resolve_for(decision_id, ts, symbol, action, horizon)
                if outcome is None:
                    continue
                self._persist(decision_id, outcome, prompt_version)
                self._push(trace_id, "outcome_score", outcome.outcome_score,
                           f"{outcome.kind}: {outcome.outcome_pct:+.2f}% ({action})")
                scored += 1
            except Exception as e:  # one bad decision must not stop the pass
                logger.warning(f"scoring {decision_id} failed (non-fatal): {e}")
        return scored

    def _resolve_for(self, decision_id, ts, symbol, action,
                     horizon: timedelta) -> Optional[Outcome]:
        pg = self._pg()
        pg.connect()
        try:
            trade = pg.conn.execute(
                "SELECT realized_pnl, size_usd FROM trades WHERE decision_id = ?",
                (decision_id,)).fetchone()
            open_row = pg.conn.execute(
                "SELECT 1 FROM open_positions WHERE decision_id = ?",
                (decision_id,)).fetchone()
        finally:
            pg.close()

        entry_price = future_price = None
        if trade is None and open_row is None:
            if ts + horizon > self._now():
                return None  # horizon not elapsed — retry next pass
            duck = self._duck()
            duck.connect()
            try:
                entry = duck.conn.execute(
                    "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                    "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                    (symbol, ts)).fetchone()
                fut = duck.conn.execute(
                    "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                    "AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                    (symbol, ts + horizon)).fetchone()
            finally:
                duck.close()
            entry_price = entry[0] if entry else None
            future_price = fut[0] if fut else None

        return resolve_outcome(
            action=action,
            closed_trade=tuple(trade) if trade is not None else None,
            is_open=open_row is not None,
            entry_price=entry_price, future_price=future_price,
        )

    def _persist(self, decision_id: str, outcome: Outcome,
                 prompt_version: Optional[str]) -> None:
        pg = self._pg()
        pg.connect()
        try:
            pg.conn.execute(
                "INSERT OR IGNORE INTO decision_scores "
                "(decision_id, scored_at, kind, outcome_pct, outcome_score, "
                " prompt_version) VALUES (?, ?, ?, ?, ?, ?)",
                (decision_id, self._now(), outcome.kind, outcome.outcome_pct,
                 outcome.outcome_score, prompt_version),
            )
        finally:
            pg.close()
