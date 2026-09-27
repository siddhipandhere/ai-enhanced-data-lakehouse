"""
Query agent — executes a validated plan from query_planner.py.

No guessing happens here. By the time a plan reaches this module it has
already been checked against the real schema, so every operation below
is either a no-op (mode/columns are null) or a safe, schema-valid call.
"""

import pandas as pd

import config
from sql import query_engine
from embeddings.vector_store import build_index_if_needed
from utils.logger import get_logger

logger = get_logger("query_agent")


def execute_plan(plan: dict, df: pd.DataFrame, table_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (result, warnings). Every step here used to catch its own
    exception and silently fall through to unfiltered/partial data —
    which meant a failed filter and a *successful* filter that simply
    matched everything looked identical by the time the Report agent
    saw the result, and the LLM would go on to describe the fallback
    data as if the requested condition had actually been applied.
    Collecting warnings here lets the Analysis/Report agents state
    plainly when that happened instead of quietly presenting fallback
    data as the real answer.
    """
    if df.empty:
        df.attrs["matched_rows"], df.attrs["aggregated"] = 0, False
        return df, []

    result = df
    warnings: list[str] = []

    if plan["filter_query"]:
        try:
            result = query_engine.filter_query(result, plan["filter_query"])
        except Exception as e:
            logger.warning(
                f"Filter execution failed ({e}), continuing with unfiltered data")
            warnings.append(
                f"The filter '{plan['filter_query']}' could not be applied ({e}); "
                f"the figures below are for the FULL unfiltered dataset, not the requested subset."
            )

    matched_rows = len(result)  # rows that passed the filter, BEFORE grouping
    aggregated = False
    if plan["group_by"] and plan["agg_column"] and plan["agg_func"]:
        try:
            result = query_engine.aggregate_query(
                result, group_by=plan["group_by"],
                agg_spec={plan["agg_column"]: plan["agg_func"]},
                sort_desc=True,  # biggest group first: "which X has the most..." reads top-down
            )
            aggregated = True
            # "count of id per mailbox" leaves the counts in a column still
            # called "id", which everything downstream treats as a row
            # number (and hides) -- so the report never saw the counts.
            if plan["agg_column"] == config.JOIN_KEY and config.JOIN_KEY in result.columns:
                result = result.rename(columns={config.JOIN_KEY: "record_count"})
        except Exception as e:
            logger.warning(
                f"Aggregation failed ({e}), returning pre-aggregation result")
            warnings.append(
                f"Grouping/aggregation by {plan['group_by']} failed ({e}); showing row-level data instead.")

    if plan["mode"] in {"semantic", "both"} and plan["semantic_text_column"]:
        try:
            # table_name MUST be passed: each dataset has its own index
            # folder (embeddings/vector_store.py). With the default None,
            # every dataset would share one "_default" slot again.
            build_index_if_needed(
                df, text_column=plan["semantic_text_column"], table_name=table_name)
            result = query_engine.semantic_query(
                result, plan.get("_user_request", ""), top_k=plan["top_k"],
                table_name=table_name, full_row_count=len(df))
        except Exception as e:
            logger.warning(
                f"Semantic search failed ({e}), returning structured result only")
            warnings.append(
                f"Semantic search failed ({e}); showing structured results only, not ranked by meaning.")

    result.attrs["matched_rows"] = matched_rows
    result.attrs["aggregated"] = aggregated
    return result, warnings


def run_query(user_request: str, df: pd.DataFrame, plan: dict, table_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Convenience wrapper — attaches the raw request text (needed for the
    semantic search embedding step) and executes the plan. Returns
    (result, warnings); see execute_plan()."""
    plan = {**plan, "_user_request": user_request}
    return execute_plan(plan, df, table_name=table_name)
