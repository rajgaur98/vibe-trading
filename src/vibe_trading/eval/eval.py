import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from vibe_trading.data.db import Database
from vibe_trading.eval.runner import load_cases, run_case
from vibe_trading.eval.scorer import build_judge, score_case
from vibe_trading.eval.report import (
    SuiteReport, write_report, load_baseline, write_baseline,
    diff_against_baseline, print_summary,
)

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="vibe-eval")
    parser.add_argument("--snapshots", type=Path, default=Path("evals/snapshots"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline.json with current run scores")
    parser.add_argument("--throttle-seconds", type=float, default=3.0,
                        help="Sleep N seconds between cases to stay under LLM-provider rate limits (default: 3.0)")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.snapshots)
    except ValueError as e:
        print(f"ERROR loading cases: {e}", file=sys.stderr)
        return 1

    if not cases:
        print(f"ERROR: no cases found in {args.snapshots}", file=sys.stderr)
        return 1

    db = Database()
    judge = build_judge()

    case_scores = []
    for idx, case in enumerate(cases):
        if idx > 0 and args.throttle_seconds > 0:
            time.sleep(args.throttle_seconds)
        logger.info(f"Running case {idx + 1}/{len(cases)}: {case.id}")
        result = run_case(case, db)
        score = score_case(result, case, judge)
        case_scores.append(score)

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
