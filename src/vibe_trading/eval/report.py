import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from vibe_trading.eval.scorer import CaseScore


class SuiteReport(BaseModel):
    run_id: str
    run_at: datetime
    case_count: int
    overall_score: float
    analyst_score: float
    trader_score: float
    pass_rate: float
    schema_failures: int
    judge_errors: int
    per_case: dict[str, CaseScore]

    @classmethod
    def from_scores(cls, scores: list[CaseScore]) -> "SuiteReport":
        if not scores:
            run_at = datetime.now(timezone.utc)
            return cls(
                run_id=cls._run_id(run_at), run_at=run_at,
                case_count=0, overall_score=0.0, analyst_score=0.0, trader_score=0.0,
                pass_rate=0.0, schema_failures=0, judge_errors=0, per_case={},
            )

        run_at = datetime.now(timezone.utc)
        overall = sum(s.total_score for s in scores) / len(scores)
        analyst = sum(s.analyst_score for s in scores) / len(scores)
        trader = sum(s.trader_score for s in scores) / len(scores)
        schema_failures = sum(1 for s in scores if not s.schema_ok)
        pass_rate = sum(1 for s in scores if all(fs.passed for fs in s.field_scores)) / len(scores)
        judge_errors = sum(
            1 for s in scores
            for fs in s.field_scores
            if "judge_error" in fs.note
        )

        return cls(
            run_id=cls._run_id(run_at), run_at=run_at,
            case_count=len(scores), overall_score=overall,
            analyst_score=analyst, trader_score=trader, pass_rate=pass_rate,
            schema_failures=schema_failures, judge_errors=judge_errors,
            per_case={s.case_id: s for s in scores},
        )

    @staticmethod
    def _run_id(ts: datetime) -> str:
        return "eval-" + ts.strftime("%Y-%m-%dT%H-%M-%SZ")


def write_report(report: SuiteReport, reports_dir: Path) -> Path:
    """Write a timestamped JSON report under reports_dir. Returns the path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / (report.run_id + ".json")
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    return path


REGRESSION_THRESHOLDS = {
    "overall": 0.02,
    "per_case": 0.05,
}


class DiffResult(BaseModel):
    overall_delta: float
    per_case_regressions: list[str]
    per_case_improvements: list[str]
    new_schema_failures: list[str]
    is_regression: bool


def load_baseline(baseline_path: Path) -> Optional[dict]:
    """Load the baseline JSON. Returns None if file is missing or malformed."""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_baseline(report: SuiteReport, baseline_path: Path) -> None:
    """Overwrite baseline.json with the current report's scores."""
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "baseline_committed_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": report.run_id,
        "overall_score": report.overall_score,
        "analyst_score": report.analyst_score,
        "trader_score": report.trader_score,
        "pass_rate": report.pass_rate,
        "schema_failures": report.schema_failures,
        "per_case": {
            cid: {
                "total_score": cs.total_score,
                "analyst_score": cs.analyst_score,
                "trader_score": cs.trader_score,
                "schema_ok": cs.schema_ok,
            }
            for cid, cs in report.per_case.items()
        },
    }
    baseline_path.write_text(json.dumps(snapshot, indent=2, default=str))


def diff_against_baseline(report: SuiteReport, baseline: dict) -> DiffResult:
    """Compute regression status against a loaded baseline dict."""
    overall_delta = report.overall_score - baseline.get("overall_score", 0.0)

    per_case_regressions: list[str] = []
    per_case_improvements: list[str] = []
    new_schema_failures: list[str] = []

    baseline_cases = baseline.get("per_case", {})
    for case_id, current in report.per_case.items():
        prior = baseline_cases.get(case_id)
        if prior is None:
            continue  # new case not in baseline -> not graded against history

        delta = current.total_score - prior.get("total_score", 0.0)
        if delta < -REGRESSION_THRESHOLDS["per_case"]:
            per_case_regressions.append(case_id)
        elif delta > REGRESSION_THRESHOLDS["per_case"]:
            per_case_improvements.append(case_id)

        # Schema regression: case parsed in baseline but failed now
        if prior.get("schema_ok") is True and current.schema_ok is False:
            new_schema_failures.append(case_id)

    is_regression = (
        overall_delta < -REGRESSION_THRESHOLDS["overall"]
        or len(per_case_regressions) > 0
        or len(new_schema_failures) > 0
    )

    return DiffResult(
        overall_delta=overall_delta,
        per_case_regressions=per_case_regressions,
        per_case_improvements=per_case_improvements,
        new_schema_failures=new_schema_failures,
        is_regression=is_regression,
    )


def print_summary(report: SuiteReport, diff: Optional[DiffResult]) -> None:
    """Print a human-readable summary to stdout."""
    print(f"Eval run: {report.run_id}   ({report.case_count} cases)")
    print("─" * 60)
    if diff is None:
        print(f"Overall score:    {report.overall_score:.2f}   (no baseline)")
        print(f"  Analyst:        {report.analyst_score:.2f}")
        print(f"  Trader:         {report.trader_score:.2f}")
    else:
        arrow = "▲" if diff.overall_delta > 0 else ("▼" if diff.overall_delta < 0 else "=")
        print(f"Overall score:    {report.overall_score:.2f}   "
              f"(delta {diff.overall_delta:+.2f} {arrow})")
        print(f"  Analyst:        {report.analyst_score:.2f}")
        print(f"  Trader:         {report.trader_score:.2f}")
    print(f"Pass rate:        {report.pass_rate * 100:.0f}%")
    print(f"Schema failures:  {report.schema_failures}")
    if report.judge_errors > 0:
        print(f"Judge errors:     {report.judge_errors}")
    print()

    if diff is not None:
        if diff.per_case_regressions:
            print("Per-case regressions (>5% drop):")
            for case_id in diff.per_case_regressions:
                print(f"  {case_id}  ▼")
            print()
        if diff.new_schema_failures:
            print("New schema failures:")
            for case_id in diff.new_schema_failures:
                print(f"  {case_id}")
            print()
        if diff.per_case_improvements:
            print("Per-case improvements (>5% gain):")
            for case_id in diff.per_case_improvements:
                print(f"  {case_id}  ▲")
            print()

    exit_code = 1 if (diff is not None and diff.is_regression) else 0
    status = "regression" if exit_code == 1 else "no regressions"
    print(f"Exit: {exit_code} ({status})")
