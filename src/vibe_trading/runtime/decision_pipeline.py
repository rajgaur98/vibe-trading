"""DecisionPipeline — the per-symbol agent decision graph, extracted from the
scheduler so the orchestration is a small, dependency-light, unit-testable unit.

Graph:  analyst.analyze ─▶ feature snapshot ─▶ trader.decide ─▶ [risk.evaluate]

It is PURE orchestration: `run_symbol` produces a `DecisionResult` describing what
the agents decided; it performs NO side effects. The caller (TradingScheduler) owns
all I/O — persisting the decision, the Parquet audit record, submitting the order,
and Discord alerts — driven off the returned result's `status`.
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Outcome of running the decision graph for one symbol.

    status is one of:
      - "analyst_failed" : the analyst tool-loop raised (symbol skipped upstream)
      - "no_snapshot"    : feature snapshot was empty (insufficient history)
      - "flat"           : trader chose no trade (logged + audited, no execution)
      - "approved"       : risk manager approved → caller should submit the order
      - "rejected"       : risk manager vetoed → caller should alert only
    """
    symbol: str
    status: str
    analyst_report: Any = None
    snapshot: Optional[dict] = None
    proposal: Optional[dict] = None
    trace_id: Optional[str] = None
    risk_result: Optional[dict] = None


class DecisionPipeline:
    def __init__(self, analyst, trader, risk_manager, feature_pipeline, broker,
                 scorecard: dict, trace_id_fn: Optional[Callable[[], Optional[str]]] = None):
        self.analyst = analyst
        self.trader = trader
        self.risk_manager = risk_manager
        self.feature_pipeline = feature_pipeline
        self.broker = broker
        self.scorecard = scorecard
        # Captures the current observability trace id at decision time (None if unavailable).
        self._trace_id_fn = trace_id_fn or (lambda: None)

    def run_symbol(self, symbol: str, last_ts, exec_price: float) -> DecisionResult:
        """Run the analyst → trader → risk graph for one symbol. Never executes orders."""
        # Stage 1 — Analyst (tool-use loop). A RuntimeError means the loop blew its
        # iteration budget; skip the symbol rather than abort the whole tick.
        try:
            analyst_report = self.analyst.analyze(symbol=symbol, timestamp=last_ts)
        except RuntimeError as e:
            logger.error(f"Analyst tool-loop failed for {symbol}: {e}. Skipping symbol.")
            return DecisionResult(symbol, "analyst_failed")

        # Stage 2 — Deterministic feature snapshot (RiskManager + audit inputs).
        snapshot = self.feature_pipeline.run(symbol, last_ts)
        if not snapshot:
            return DecisionResult(symbol, "no_snapshot", analyst_report=analyst_report)

        # Stage 3 — Head Trader decision.
        open_positions = self.broker.get_open_positions()
        proposal = self.trader.decide(
            symbol, analyst_report, self.scorecard, open_positions, current_price=exec_price
        )
        trace_id = self._trace_id_fn()

        if proposal["action"] == "flat":
            return DecisionResult(symbol, "flat", analyst_report=analyst_report,
                                  snapshot=snapshot, proposal=proposal, trace_id=trace_id)

        # Stage 4 — Risk Manager (deterministic sizing / veto). Balance is fetched here
        # (only for non-flat proposals) to preserve the original lazy-fetch behavior.
        df_4h = self.feature_pipeline._get_candles(symbol, "4h", last_ts, limit=30)
        risk_result = self.risk_manager.evaluate_proposal(
            proposal=proposal,
            current_price=exec_price,
            df_4h=df_4h,
            account_balance=self.broker.get_balance(),
            peak_balance=self.broker.peak_balance,
            open_positions=open_positions,
            snapshot=snapshot,
        )
        status = "approved" if risk_result["approved"] else "rejected"
        return DecisionResult(symbol, status, analyst_report=analyst_report, snapshot=snapshot,
                              proposal=proposal, trace_id=trace_id, risk_result=risk_result)
