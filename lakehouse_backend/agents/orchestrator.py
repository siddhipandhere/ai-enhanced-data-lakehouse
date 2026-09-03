"""
Orchestrator agent — full implementation. Builds a schema-aware query
plan (first LLM call, via query_planner.py) then coordinates
Query -> Analysis -> Report.

This replaces the earlier fixed sql/semantic/both classifier with a
planner that reads the actual dataset's columns at query time, which
is what lets the same code run against any dataset/domain without
hardcoded column names.
"""

import pandas as pd

from agents.query_planner import build_plan
from agents.query_agent import run_query
from agents.analysis_agent import run_analysis
from agents.report_agent import generate_report
from utils.logger import get_logger

logger = get_logger("orchestrator")


def handle_request(user_request: str, gold_df: pd.DataFrame, table_name: str | None = None) -> dict:
    """
    Main entry point. Returns the query plan, raw results, analysis
    stats, and a plain-language summary ready for display.

    table_name is passed through to the query agent so that, if the plan
    calls for semantic search, the shared FAISS index's staleness check
    can correctly tell this dataset apart from any other with the same
    row count and text column (see query_agent.py for why this matters).
    """
    plan = build_plan(user_request, gold_df)
    logger.info(f"Request planned as '{plan['mode']}': {user_request}")

    results = run_query(user_request, gold_df, plan, table_name=table_name)
    analysis = run_analysis(results)
    summary = generate_report(user_request, analysis)

    return {
        "plan": plan,
        "results": results,
        "analysis": analysis,
        "summary": summary,
    }