"""
Analysis agent — computes summary statistics on the Query agent's output.
No LLM calls; pure computation so it stays fast and deterministic.
"""

import pandas as pd

from utils.metrics import QueryTimer
from utils.logger import get_logger

logger = get_logger("analysis_agent")


def run_analysis(results: pd.DataFrame, total_record_count: int | None = None,
                 filter_applied: bool = False, warnings: list[str] | None = None,
                 queried_columns: list[str] | None = None) -> dict:
    """
    total_record_count/filter_applied let the summary state a real,
    computed "N of M records matched" figure instead of leaving the
    Report agent to estimate one on its own from a bare describe() —
    which is where invented-sounding numbers ("roughly half", "about
    15,000") came from: nothing in the old payload actually told the
    model how many rows matched, so any specific count it gave was a
    guess dressed up as a fact.
    """
    with QueryTimer() as timer:
        numeric_df = results.select_dtypes(include="number")
        summary = {
            "record_count": len(results),
            "total_record_count": total_record_count if total_record_count is not None else len(results),
            "filter_applied": filter_applied,
            "columns": list(results.columns),
            "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {},
            "warnings": warnings or [],
        }
        if filter_applied and summary["total_record_count"]:
            summary["match_rate_pct"] = round(
                100 * summary["record_count"] / summary["total_record_count"], 2)

        if queried_columns:
            summary["queried_columns"] = queried_columns
            summary["queried_column_stats"] = {
                c: results[c].describe().to_dict()
                for c in queried_columns
                if c in numeric_df.columns
            }

    summary["query_execution_time_ms"] = timer.elapsed_ms
    logger.info(f"Analysis complete: {summary['record_count']} records, "
                f"{summary['query_execution_time_ms']} ms")
    return summary
