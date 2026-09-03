"""
Query agent — executes a validated plan from query_planner.py.

No guessing happens here. By the time a plan reaches this module it has
already been checked against the real schema, so every operation below
is either a no-op (mode/columns are null) or a safe, schema-valid call.
"""

import pandas as pd

from sql import query_engine
from embeddings.vector_store import build_index_if_needed
from utils.logger import get_logger

logger = get_logger("query_agent")


def execute_plan(plan: dict, df: pd.DataFrame, table_name: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    result = df

    if plan["filter_query"]:
        try:
            result = query_engine.filter_query(result, plan["filter_query"])
        except Exception as e:
            logger.warning(f"Filter execution failed ({e}), continuing with unfiltered data")

    if plan["group_by"] and plan["agg_column"] and plan["agg_func"]:
        try:
            result = query_engine.aggregate_query(
                result, group_by=plan["group_by"],
                agg_spec={plan["agg_column"]: plan["agg_func"]},
            )
        except Exception as e:
            logger.warning(f"Aggregation failed ({e}), returning pre-aggregation result")

    if plan["mode"] in {"semantic", "both"} and plan["semantic_text_column"]:
        try:
            # table_name MUST be passed here, not left as the default None.
            # The shared FAISS index's staleness check compares table_name
            # to decide whether to rebuild; if every caller left it None,
            # two different datasets with the same row_count and the same
            # chosen text_column would silently reuse each other's index
            # (None != None is False, so no rebuild fires) and return
            # search results embedded from the wrong dataset entirely,
            # with no error to signal it.
            build_index_if_needed(df, text_column=plan["semantic_text_column"], table_name=table_name)
            result = query_engine.semantic_query(result, plan.get("_user_request", ""), top_k=plan["top_k"])
        except Exception as e:
            logger.warning(f"Semantic search failed ({e}), returning structured result only")

    return result


def run_query(user_request: str, df: pd.DataFrame, plan: dict, table_name: str | None = None) -> pd.DataFrame:
    """Convenience wrapper — attaches the raw request text (needed for the
    semantic search embedding step) and executes the plan."""
    plan = {**plan, "_user_request": user_request}
    return execute_plan(plan, df, table_name=table_name)