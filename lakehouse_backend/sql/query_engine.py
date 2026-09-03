"""
Unified query engine — the interface the Query agent calls, hiding
whether a request resolves to a pandas filter, an aggregation, or a
FAISS semantic search.
"""

import pandas as pd

from medallion.gold import run_sql_query, aggregate
from embeddings.vector_store import semantic_search, index_exists
import config


def filter_query(df: pd.DataFrame, pandas_query_str: str) -> pd.DataFrame:
    return run_sql_query(df, pandas_query_str)


def aggregate_query(df: pd.DataFrame, group_by: list[str], agg_spec: dict[str, str]) -> pd.DataFrame:
    return aggregate(df, group_by, agg_spec)


def to_json_safe_records(df: pd.DataFrame) -> list[dict]:
    """`df.to_dict(orient="records")` on a frame with missing values leaves
    real pandas NaN floats in the output. Starlette's JSONResponse calls
    json.dumps(..., allow_nan=False), so any NaN reaching it raises
    `ValueError: Out of range float values are not JSON compliant: nan` --
    and since most real-world data has at least some nulls, this hits on
    ordinary queries, not just edge cases.

    `df.where(pd.notnull(df), None)` looks like the fix but silently does
    nothing on float64 columns: a numpy float64 array has no way to store
    a real `None`, so pandas coerces it right back to NaN. Casting to
    `object` dtype first gives None somewhere to actually live.
    """
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def semantic_query(df: pd.DataFrame, query_text: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
    """Runs FAISS semantic search and joins results back to the Gold table."""
    if not index_exists():
        raise RuntimeError("No vector index built yet. Run vector_store.build_index() first.")
    ranked = semantic_search(query_text, top_k=top_k)
    return ranked.merge(df, on=config.JOIN_KEY, how="left")