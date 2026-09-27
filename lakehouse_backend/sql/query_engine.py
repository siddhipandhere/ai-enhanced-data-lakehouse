"""
Unified query engine — the interface the Query agent calls, hiding
whether a request resolves to a pandas filter, an aggregation, or a
FAISS semantic search.
"""

import pandas as pd

from medallion.gold import run_sql_query, aggregate
from embeddings.vector_store import semantic_search, index_exists
import config
from utils.media import image_column, thumbnail_data_uri


def filter_query(df: pd.DataFrame, pandas_query_str: str) -> pd.DataFrame:
    return run_sql_query(df, pandas_query_str)


def aggregate_query(df: pd.DataFrame, group_by: list[str], agg_spec: dict[str, str],
                    sort_desc: bool = False) -> pd.DataFrame:
    return aggregate(df, group_by, agg_spec, sort_desc=sort_desc)


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
    import numpy as np

    df = df.replace([np.inf, -np.inf], np.nan)
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    # Image uploads keep raw file bytes in a 'content' column; FastAPI
    # tries to utf-8-decode bytes and 500s on binary data. Pictures are sent
    # as a small JPEG thumbnail (data URI) the dashboard shows inline; any
    # other binary value gets a size placeholder.
    for row in records:
        for k, v in row.items():
            if isinstance(v, (bytes, bytearray, memoryview)):
                row[k] = thumbnail_data_uri(v) or f"<{len(v)} bytes>"
    return records


def image_columns_first(df: pd.DataFrame) -> pd.DataFrame:
    """For an image collection, move the picture and its file name to the
    front of a result so the thumbnail is the first thing you see."""
    img = image_column(df)
    if not img:
        return df
    front = [c for c in ("similarity_score", img, "original_name") if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def semantic_query(df: pd.DataFrame, query_text: str, top_k: int = config.DEFAULT_TOP_K,
                   table_name: str | None = None, full_row_count: int | None = None) -> pd.DataFrame:
    """
    Runs FAISS semantic search over table_name's index and joins the hits
    back onto `df`.

    `df` may already be FILTERED (mode "both": e.g. "discount_pct > 50",
    then rank by meaning). The old version searched only the top_k best
    matches from the WHOLE index and LEFT-joined them onto the filtered
    frame, so (a) hits outside the filter came back as rows of empty/NaN
    columns and (b) the filter was effectively ignored. Now: search a
    wider candidate pool when df is a subset, INNER-join so only rows
    that passed the filter survive, then keep the best top_k.
    """
    if not index_exists(table_name):
        raise RuntimeError("No vector index built yet for this dataset.")
    if config.JOIN_KEY not in df.columns:
        raise RuntimeError(f"Result has no '{config.JOIN_KEY}' column to join search hits onto "
                           f"(it was probably aggregated first).")
    is_subset = full_row_count is not None and len(df) < full_row_count
    pool = top_k if not is_subset else max(top_k * 50, 500)
    ranked = semantic_search(query_text, top_k=pool, table_name=table_name)
    merged = ranked.merge(df, on=config.JOIN_KEY, how="inner")
    return merged.head(top_k).reset_index(drop=True)