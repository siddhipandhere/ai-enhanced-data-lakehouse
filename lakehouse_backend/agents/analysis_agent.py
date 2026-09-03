"""
Analysis agent — computes summary statistics on the Query agent's output.
No LLM calls; pure computation so it stays fast and deterministic.
"""

import pandas as pd

from utils.metrics import QueryTimer
from utils.logger import get_logger

logger = get_logger("analysis_agent")


def run_analysis(results: pd.DataFrame) -> dict:
    with QueryTimer() as timer:
        numeric_df = results.select_dtypes(include="number")
        summary = {
            "record_count": len(results),
            "columns": list(results.columns),
            "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {},
        }

    summary["query_execution_time_ms"] = timer.elapsed_ms
    logger.info(f"Analysis complete: {summary['record_count']} records, "
                f"{summary['query_execution_time_ms']} ms")
    return summary
