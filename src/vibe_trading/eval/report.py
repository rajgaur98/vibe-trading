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
