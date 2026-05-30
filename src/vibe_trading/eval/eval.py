import argparse
import concurrent.futures
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from vibe_trading.data.db import Database
from vibe_trading.eval.runner import load_cases, run_case
from vibe_trading.eval.scorer import build_judge, score_case, CaseScore
from vibe_trading.eval.report import (
    SuiteReport, write_report, load_baseline, write_baseline,
    diff_against_baseline, print_summary,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Emit INFO-level progress to stderr; quiet the chatty LLM/HTTP libraries.

    Without this the eval ran silently for many minutes — no way to see which case it
    was on. Now each case logs a `[n/total]` line as it completes.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="vibe-eval")
    parser.add_argument("--snapshots", type=Path, default=Path("evals/snapshots"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline.json with current run scores")
    parser.add_argument("--throttle-seconds", type=float, default=4.5,
                        help="Minimum seconds between LLM calls (per CALL, shared across all "
                             "agents+judge) to stay under the provider's RPM limit. Default 4.5 "
                             "≈ 13 RPM, under Gemini's 15. With parallel workers this is the real "
                             "governor: it spaces call STARTS while slow calls overlap.")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Number of cases to score concurrently (default 6). Combined with a "
                             "slow model (e.g. Gemma 4 31B ~19s/call) this overlaps the latency so "
                             "wall-clock is bounded by the throttle, not by calls x latency.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-case progress logging.")
    args = parser.parse_args(argv)

    if not args.quiet:
        _configure_logging()

    # Apply the per-call rate limit globally before any LLMClient is constructed, so the
    # analyst, trader, and judge clients all share one minimum-interval gate. Each case
    # fires ~4 calls, so pacing per-call (not per-case) is what actually bounds the RPM.
    if args.throttle_seconds > 0:
        os.environ["LLM_MIN_CALL_INTERVAL_SECONDS"] = str(args.throttle_seconds)

    try:
        cases = load_cases(args.snapshots)
    except ValueError as e:
        print(f"ERROR loading cases: {e}", file=sys.stderr)
        return 1

    if not cases:
        print(f"ERROR: no cases found in {args.snapshots}", file=sys.stderr)
        return 1

    judge = build_judge()
    total = len(cases)

    def _process(indexed_case: tuple[int, object]) -> tuple[int, CaseScore]:
        idx, case = indexed_case
        # Each task gets its OWN Database — DuckDB connections are not safe to share
        # across threads, and run_case opens/closes short-lived connections internally.
        try:
            result = run_case(case, Database())
            score = score_case(result, case, judge)
        except Exception as e:  # defensive: one bad case must not abort the whole run
            logger.warning(f"Case {case.id} crashed during scoring: {e}")
            score = CaseScore(
                case_id=case.id, schema_ok=False, field_scores=[],
                analyst_score=0.0, trader_score=0.0, total_score=0.0, error=str(e),
            )
        return idx, score

    # Score cases concurrently. The LLMClient throttle (shared, class-level) bounds the
    # global RPM regardless of worker count, so more workers only fill the pipeline up to
    # the throttle ceiling — they never exceed it.
    results: list[Optional[CaseScore]] = [None] * total
    logger.info(f"Scoring {total} cases | {args.max_workers} workers | "
                f"min {args.throttle_seconds}s between LLM calls")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_process, (i, c)) for i, c in enumerate(cases)]
        for fut in concurrent.futures.as_completed(futures):
            idx, score = fut.result()
            results[idx] = score
            done += 1
            logger.info(f"[{done}/{total}] {score.case_id}: total={score.total_score:.2f} "
                        f"analyst={score.analyst_score:.2f} trader={score.trader_score:.2f}")

    case_scores = [s for s in results if s is not None]
    report = SuiteReport.from_scores(case_scores)
    report_path = write_report(report, args.reports_dir)
    logger.info(f"Wrote report to {report_path}")

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"No baseline at {args.baseline}. "
              f"Run with --update-baseline to seed it.")
        print_summary(report, diff=None)
        if args.update_baseline:
            write_baseline(report, args.baseline)
            print(f"Baseline written: {args.baseline}")
        return 0

    diff = diff_against_baseline(report, baseline)
    print_summary(report, diff)

    if args.update_baseline:
        write_baseline(report, args.baseline)
        print(f"Baseline updated: {args.baseline}")

    return 1 if diff.is_regression else 0


if __name__ == "__main__":
    sys.exit(main())
