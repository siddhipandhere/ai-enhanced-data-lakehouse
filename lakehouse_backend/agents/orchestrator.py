"""
Orchestrator agent — full implementation. Builds a schema-aware query
plan (first LLM call, via query_planner.py) then coordinates
Query -> Analysis -> Report.

This replaces the earlier fixed sql/semantic/both classifier with a
planner that reads the actual dataset's columns at query time, which
is what lets the same code run against any dataset/domain without
hardcoded column names.
"""

import re

import pandas as pd

from agents.query_planner import build_plan
from agents.query_agent import run_query
from agents.analysis_agent import run_analysis
from agents.report_agent import generate_report
from utils.logger import get_logger

logger = get_logger("orchestrator")

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _referenced_columns(plan: dict, df: pd.DataFrame) -> list[str]:
    """
    Which real columns this plan actually asked about, e.g. ["discount_pct"]
    for a filter_query of "discount_pct > 50". A Gold table can end up
    with several similarly-named numeric columns (a real "discount_pct"
    next to an unrelated "description_pct" derived from fabric-composition
    text, say), and analysis_agent's describe() dump previously included
    all of them with none flagged as "the one the question was actually
    about" -- leaving the Report agent to guess which numbers to lead
    with, and it doesn't always guess right. Pointing directly at the
    columns the plan touched removes that guess.
    """
    valid = set(df.columns)
    found: list[str] = []

    if plan.get("filter_query"):
        found += [tok for tok in _IDENTIFIER.findall(
            plan["filter_query"]) if tok in valid]
    if plan.get("agg_column") in valid:
        found.append(plan["agg_column"])
    for c in (plan.get("group_by") or []):
        if c in valid:
            found.append(c)

    seen = set()
    return [c for c in found if not (c in seen or seen.add(c))]


def handle_request(user_request: str, gold_df: pd.DataFrame, table_name: str | None = None,
                   text_column: str | None = None) -> dict:
    """
    Main entry point. Returns the query plan, raw results, analysis
    stats, and a plain-language summary ready for display.

    table_name is passed through to the query agent so that, if the plan
    calls for semantic search, the shared FAISS index's staleness check
    can correctly tell this dataset apart from any other with the same
    row count and text column (see query_agent.py for why this matters).
    """
    plan = build_plan(user_request, gold_df)
    # The pipeline already picked (by content, not name) and indexed the
    # dataset's real free-text column. If the LLM picks a different one
    # for semantic mode, honouring it means re-embedding the whole table
    # inside this request (minutes on a laptop -> looks like a hang), and
    # often over a worse column (e.g. a mislabelled "description" that
    # really holds "69% off" badges). Use the indexed column instead.
    if text_column and plan["mode"] in {"semantic", "both"} and plan["semantic_text_column"] != text_column:
        logger.info(f"Using indexed text column '{text_column}' instead of planner's "
                    f"'{plan['semantic_text_column']}' for semantic search")
        plan = {**plan, "semantic_text_column": text_column}
    logger.info(f"Request planned as '{plan['mode']}': {user_request}")

    results, warnings = run_query(
        user_request, gold_df, plan, table_name=table_name)
    # "Filter successfully applied" specifically means a filter_query was
    # requested AND it didn't fail (a failure is recorded in `warnings`
    # instead) -- that's what makes match_rate_pct in the analysis a real,
    # trustworthy number rather than one computed on accidentally-
    # unfiltered data.
    filter_applied = bool(plan["filter_query"]) and not any(
        "filter" in w.lower() for w in warnings)
    queried_columns = _referenced_columns(plan, gold_df)
    analysis = run_analysis(results, total_record_count=len(gold_df), filter_applied=filter_applied,
                            warnings=warnings, queried_columns=queried_columns,
                            matched_rows=results.attrs.get("matched_rows"),
                            aggregated=results.attrs.get("aggregated", False),
                            passage_column=plan.get("semantic_text_column")
                            if plan["mode"] in {"semantic", "both"} else None)
    summary = generate_report(user_request, analysis)

    return {
        "plan": plan,
        "results": results,
        "analysis": analysis,
        "summary": summary,
    }
