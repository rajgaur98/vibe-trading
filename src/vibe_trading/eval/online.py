"""Online evaluation — score PRODUCTION decisions once their outcome is knowable.

Three layers (spec workstream C), all strictly best-effort:
  C1  deterministic outcome scoring  (this module's OutcomeScorer)
  C2  sampled generic-rubric LLM judge (capped per day)
  C3  weekly drift digest to Discord

A 'correct' flat is scored as 'no strong move happened' — a proxy for 'no edge
existed', not ground truth (a flat that dodged a crash scores poorly here). The
caveat is deliberate and documented; deterministic beats clever.
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from pydantic import BaseModel

from vibe_trading.agents import prompts
from vibe_trading.agents.cost import daily_summary, should_block_trading
from vibe_trading.journal import COUNTERFACTUAL_HORIZON_CANDLES, _CANDLE_HOURS

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
                "       d.prompt_version, d.precedents_k "
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
        for decision_id, ts, symbol, action, trace_id, prompt_version, precedents_k in candidates:
            try:
                outcome = self._resolve_for(decision_id, ts, symbol, action, horizon)
                if outcome is None:
                    continue
                self._persist(decision_id, outcome, prompt_version, precedents_k)
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
                 prompt_version: Optional[str], precedents_k: Optional[int] = None) -> None:
        pg = self._pg()
        pg.connect()
        try:
            pg.conn.execute(
                "INSERT OR IGNORE INTO decision_scores "
                "(decision_id, scored_at, kind, outcome_pct, outcome_score, "
                " prompt_version, precedents_k) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (decision_id, self._now(), outcome.kind, outcome.outcome_pct,
                 outcome.outcome_score, prompt_version, precedents_k),
            )
        finally:
            pg.close()


ONLINE_JUDGE_DAILY_CAP = int(os.getenv("ONLINE_JUDGE_DAILY_CAP", "5"))


class OnlineJudgeVerdict(BaseModel):
    grounded: bool      # reasoning cites only facts present in the snapshot
    consistent: bool    # action follows from the stated reasoning
    justification: str


class OnlineJudge:
    """C2: sample scored-but-unjudged decisions and grade them with a generic
    rubric. Capped per UTC day; skipped entirely once the LLM cost cap is hit."""

    def __init__(self, client=None, pg_factory=None, now_fn=None, push_fn=None,
                 daily_cap: Optional[int] = None):
        self._client = client
        self._pg_factory = pg_factory
        self._now = now_fn or (lambda: datetime.utcnow())
        self._push = push_fn or push_langfuse_score
        self.daily_cap = daily_cap if daily_cap is not None else ONLINE_JUDGE_DAILY_CAP

    def _pg(self):
        if self._pg_factory:
            return self._pg_factory()
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase()

    def _get_client(self):
        if self._client is None:
            from vibe_trading.agents.client import LLMClient
            self._client = LLMClient()
        return self._client

    def run_pass(self) -> int:
        """Judge up to (daily_cap - already judged today) decisions. Never raises."""
        try:
            return self._run_pass()
        except Exception as e:
            logger.warning(f"online judge pass failed (non-fatal): {e}")
            return 0

    def _run_pass(self) -> int:
        today_start = self._now().replace(hour=0, minute=0, second=0, microsecond=0)
        pg = self._pg()
        pg.connect()
        try:
            # Cost-cap guard: judging is optional spend; trading's kill switch wins.
            cap = float(os.getenv("LLM_DAILY_COST_CAP_USD", "10.0"))
            if should_block_trading(daily_summary(pg.conn)["today_usd"], cap):
                logger.info("online judge skipped: LLM daily cost cap reached")
                return 0
            judged_today = pg.conn.execute(
                "SELECT COUNT(*) FROM decision_scores WHERE judged_at >= ?",
                (today_start,)).fetchone()[0]
            remaining = self.daily_cap - int(judged_today or 0)
            if remaining <= 0:
                return 0
            rows = pg.conn.execute(
                "SELECT s.decision_id, d.action, d.symbol, d.reasoning_summary, "
                "       d.agent_transcripts, d.trace_id "
                "FROM decision_scores s JOIN decision_log d "
                "  ON d.decision_id = s.decision_id "
                "WHERE s.judge_score IS NULL "
                "ORDER BY s.scored_at DESC LIMIT ?",
                (remaining,)).fetchall()
        finally:
            pg.close()

        judged = 0
        for decision_id, action, symbol, reasoning, transcripts, trace_id in rows:
            verdict = self._judge_one(action, symbol, reasoning, transcripts)
            if verdict is None:
                continue
            score = (int(verdict.grounded) + int(verdict.consistent)) / 2.0
            self._save(decision_id, score, verdict.justification)
            self._push(trace_id, "online_judge_score", score, verdict.justification)
            judged += 1
        return judged

    def _judge_one(self, action, symbol, reasoning,
                   transcripts) -> Optional[OnlineJudgeVerdict]:
        client = self._get_client()
        model = os.getenv("EVAL_JUDGE_MODEL") or client.model
        user_prompt = (
            f"DECISION:\naction: {action}\nsymbol: {symbol}\n"
            f"reasoning_summary: {reasoning}\n\n"
            f"FEATURE SNAPSHOT the agents saw:\n{transcripts}"
        )
        try:
            raw = client.call_llm(
                model_name=model,
                system_instruction=prompts.ONLINE_JUDGE_SYSTEM.text,
                prompt=user_prompt,
                response_schema=OnlineJudgeVerdict,
                prompt_version=prompts.ONLINE_JUDGE_SYSTEM.stamp,
                call_type="online_judge",
            )
            return OnlineJudgeVerdict.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"online judge call/parse failed (non-fatal): {e}")
            return None

    def _save(self, decision_id: str, score: float, note: str) -> None:
        pg = self._pg()
        pg.connect()
        try:
            pg.conn.execute(
                "UPDATE decision_scores SET judge_score = ?, judge_note = ?, "
                "judged_at = ? WHERE decision_id = ?",
                (score, note, self._now(), decision_id))
        finally:
            pg.close()


def weekly_digest(conn, now: Optional[datetime] = None) -> dict:
    """C3: this week's quality vs the trailing four weeks, plus schema compliance,
    segmented by prompt bundle. `conn` is a connected DB wrapper (caller owns
    connect/close)."""
    now = now or datetime.utcnow()
    week_start = now - timedelta(days=7)
    trailing_start = now - timedelta(days=35)

    week = conn.execute(
        "SELECT AVG(outcome_score), AVG(judge_score), COUNT(*) "
        "FROM decision_scores WHERE scored_at >= ?", (week_start,)).fetchone()
    trailing = conn.execute(
        "SELECT AVG(outcome_score) FROM decision_scores "
        "WHERE scored_at >= ? AND scored_at < ?",
        (trailing_start, week_start)).fetchone()
    schema = conn.execute(
        "SELECT COUNT(CASE WHEN schema_ok IS TRUE THEN 1 END), "
        "       COUNT(CASE WHEN schema_ok IS NOT NULL THEN 1 END) "
        "FROM llm_cost_log WHERE timestamp >= ?", (week_start,)).fetchone()
    by_version = conn.execute(
        "SELECT prompt_version, AVG(outcome_score), COUNT(*) "
        "FROM decision_scores WHERE scored_at >= ? "
        "GROUP BY prompt_version ORDER BY 3 DESC", (week_start,)).fetchall()

    ok, total = int(schema[0] or 0), int(schema[1] or 0)
    return {
        "week_outcome_mean": float(week[0]) if week and week[0] is not None else None,
        "week_judge_mean": float(week[1]) if week and week[1] is not None else None,
        "week_scored": int(week[2] or 0) if week else 0,
        "trailing_outcome_mean": (float(trailing[0])
                                  if trailing and trailing[0] is not None else None),
        "week_schema_compliance": (ok / total) if total else None,
        "by_prompt_version": [
            {"prompt_version": pv, "outcome_mean": float(m), "count": int(c)}
            for (pv, m, c) in by_version
        ],
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def format_digest(d: dict) -> str:
    lines = [
        "📈 **WEEKLY ONLINE-EVAL DIGEST**",
        f"Outcome score (7d): **{_fmt(d['week_outcome_mean'])}** "
        f"vs trailing 4w {_fmt(d['trailing_outcome_mean'])} "
        f"({d['week_scored']} decisions scored)",
        f"Judge score (7d): {_fmt(d['week_judge_mean'])}",
        f"Schema compliance (7d): {_fmt(d['week_schema_compliance'])}",
    ]
    for row in d["by_prompt_version"]:
        lines.append(f"  · `{row['prompt_version']}`: "
                     f"{row['outcome_mean']:.2f} (n={row['count']})")
    return "\n".join(lines)


def run_scoring_pass(judge: bool = True) -> dict:
    """One full online-eval pass: outcome scoring, then (optionally) the sampled
    judge. Installs the Postgres cost sink so judge calls are metered like every
    other LLM call. Never raises."""
    from vibe_trading.agents.client import LLMClient
    from vibe_trading.agents.cost import PostgresCostLogger
    try:
        LLMClient.set_cost_sink(PostgresCostLogger())
    except Exception as e:
        logger.warning(f"cost sink unavailable for scoring pass (non-fatal): {e}")
    outcomes = OutcomeScorer().run_pass()
    judged = OnlineJudge().run_pass() if judge else 0
    logger.info(f"online-eval pass: {outcomes} outcomes scored, {judged} judged")
    return {"outcomes_scored": outcomes, "judged": judged}


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="vibe-online-eval")
    parser.add_argument("--digest", action="store_true",
                        help="Also send the weekly drift digest to Discord.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Outcome scoring only (skip the LLM judge).")
    parser.add_argument("--judge-cap", type=int, default=None,
                        help="Override ONLINE_JUDGE_DAILY_CAP for this run.")
    args = parser.parse_args(argv)

    if args.judge_cap is not None:
        os.environ["ONLINE_JUDGE_DAILY_CAP"] = str(args.judge_cap)
        global ONLINE_JUDGE_DAILY_CAP
        ONLINE_JUDGE_DAILY_CAP = args.judge_cap

    result = run_scoring_pass(judge=not args.no_judge)
    print(f"outcomes scored: {result['outcomes_scored']}, "
          f"judged: {result['judged']}")

    if args.digest:
        from vibe_trading.data.db import PostgresDatabase
        from vibe_trading.runtime import monitoring
        try:
            pg = PostgresDatabase()
            pg.connect()
            try:
                digest = weekly_digest(pg.conn)
            finally:
                pg.close()
            monitoring.send_discord(format_digest(digest))
            print("digest sent")
        except Exception as e:
            logger.warning(f"digest failed (non-fatal): {e}")

    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
