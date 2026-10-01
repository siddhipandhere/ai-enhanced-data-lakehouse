"""
Gold layer (report Algorithms 4-5: large-scale processing + SQL analytics,
implemented with pandas per the project's laptop-demo scope).

Produces the final analytics-ready table: cleaned, joined, aggregated,
and cached for fast repeated queries.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from pipeline import registry
from utils.logger import get_logger

logger = get_logger("gold")

_QUERY_CACHE: dict[str, pd.DataFrame] = {}

COMBINED_TABLE_SENTINEL = "__all__"


def find_combinable_datasets(uploaded_by: str) -> list[dict]:
    """
    Returns the largest group of the user's Gold-ready datasets that
    share an EXACT matching column set (same names, same count), or []
    if fewer than 2 such datasets exist. Ties are broken by total row
    count.
    """
    ready = [d for d in registry.list_datasets(
        uploaded_by=uploaded_by) if d.get("gold_status") == "ready"]
    groups: dict[tuple, list[dict]] = {}
    for d in ready:
        signature = tuple(sorted(d.get("columns") or []))
        if not signature:
            continue
        groups.setdefault(signature, []).append(d)

    candidates = [g for g in groups.values() if len(g) >= 2]
    if not candidates:
        return []

    candidates.sort(key=lambda g: sum(
        d.get("record_count") or 0 for d in g), reverse=True)
    return candidates[0]


def describe_combine_status(uploaded_by: str) -> dict:
    """Explains, in UI-displayable terms, whether the combined virtual
    table is available and why/why not."""
    ready = [d for d in registry.list_datasets(
        uploaded_by=uploaded_by) if d.get("gold_status") == "ready"]
    group = find_combinable_datasets(uploaded_by)

    if group:
        return {
            "eligible": True,
            "ready_dataset_count": len(ready),
            "combined_dataset_count": len(group),
            "combined_names": [d.get("original_name") or d.get("table_name") for d in group],
            "reason": None,
        }

    if len(ready) < 2:
        reason = "Upload at least 2 datasets and let them finish the Gold stage to unlock combining."
    else:
        reason = (f"You have {len(ready)} ready datasets, but no two of them share an identical "
                  f"set of column names, so there's nothing eligible to combine yet — combining "
                  f"only activates for datasets with matching schemas.")

    return {
        "eligible": False,
        "ready_dataset_count": len(ready),
        "combined_dataset_count": 0,
        "combined_names": [],
        "reason": reason,
    }


def load_combined_gold_table(uploaded_by: str) -> pd.DataFrame:
    """
    Loads and concatenates every dataset in the largest exact-schema-
    match group for this user. Adds a '_source_dataset' column so any
    result row can be traced back to its upload.
    """
    group = find_combinable_datasets(uploaded_by)
    if not group:
        raise ValueError(
            "No two ready datasets share a matching schema to combine.")

    frames = []
    for entry in group:
        df = load_gold_table(entry["table_name"])
        df = df.copy()
        df["_source_dataset"] = entry.get(
            "original_name") or entry["table_name"]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Renumber the join key so ids are unique across the combined rows
    # (otherwise semantic search's join-back could match the wrong rows).
    combined[config.JOIN_KEY] = range(len(combined))
    logger.info(
        f"Combined {len(group)} datasets into one table: {len(combined)} total rows")
    return combined


def combined_text_column(uploaded_by: str) -> str | None:
    """The combined table is only eligible for semantic search if EVERY
    dataset in the group is search-ready AND they agree on text_column."""
    group = find_combinable_datasets(uploaded_by)
    if not group:
        return None
    text_columns = {d.get("text_column") for d in group}
    if len(text_columns) != 1 or None in text_columns:
        return None
    if not all(d.get("vector_status") == "ready" for d in group):
        return None
    return text_columns.pop()


def resolve_dataset(table_name: str, uploaded_by: str) -> pd.DataFrame:
    """Single entry point routes should call instead of load_gold_table
    directly: also handles the COMBINED_TABLE_SENTINEL value."""
    if table_name == COMBINED_TABLE_SENTINEL:
        return load_combined_gold_table(uploaded_by)
    return load_gold_table(table_name)


def vector_index_key(table_name: str, uploaded_by: str) -> str:
    """Which vector-index slot a table uses. Every user's combined view
    shares the same '__all__' sentinel name, so it's made per-user here --
    otherwise two users with same-sized combined tables could be served
    each other's search index."""
    if table_name == COMBINED_TABLE_SENTINEL:
        return f"{COMBINED_TABLE_SENTINEL}{uploaded_by}"
    return table_name


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def build_gold_table(silver_df: pd.DataFrame, table_name: str = "main") -> Path:
    """
    Repartitions/cleans a Silver DataFrame into the analytics-ready Gold
    layer and writes it as Parquet for efficient repeated reads.
    """
    df = silver_df.copy()
    df = df.reset_index(drop=True)

    dest_path = config.GOLD_DIR / f"{table_name}.parquet"
    _atomic_write_parquet(df, dest_path)

    original_size = df.memory_usage(deep=True).sum()
    stored_size = dest_path.stat().st_size
    storage_efficiency = round(
        original_size / stored_size, 2) if stored_size else None

    logger.info(f"Built Gold table '{table_name}': {len(df)} records, "
                f"storage efficiency {storage_efficiency}x")
    return dest_path


def load_gold_table(table_name: str = "main") -> pd.DataFrame:
    path = config.GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No Gold table named '{table_name}' at {path}")
    return pd.read_parquet(path)


def run_sql_query(df: pd.DataFrame, query: str, cache_key: str | None = None) -> pd.DataFrame:
    """
    Executes a pandas-`query()`-style filter string against the Gold
    table (the report's SQL Analytics Layer, Algorithm 5).

    Example: run_sql_query(df, "region == 'North' and revenue > 1000")
    """
    if cache_key and cache_key in _QUERY_CACHE:
        logger.info(f"Cache hit for query key '{cache_key}'")
        return _QUERY_CACHE[cache_key]

    # Empty local/global dicts: pandas' "@name" syntax resolves names from
    # the CALLER's scope, which here includes this module's globals (os,
    # config, registry...). Without this, "@os.system('...')" typed into
    # SQL Explorer executed a shell command. Callers should also run
    # agents.query_planner.check_filter_query() first -- this is the
    # second, independent layer.
    kwargs = {"local_dict": {}, "global_dict": {}}
    if ".str." in query or ".dt." in query:
        # numexpr (pandas' default engine when installed) can't evaluate
        # .str/.dt accessors; the python engine can.
        kwargs["engine"] = "python"
    result = df.query(query, **kwargs)

    if cache_key:
        _QUERY_CACHE[cache_key] = result

    return result


def aggregate(df: pd.DataFrame, group_by: list[str], agg_spec: dict[str, str],
              sort_desc: bool = False) -> pd.DataFrame:
    """
    e.g. aggregate(df, group_by=["region"], agg_spec={"revenue": "sum", "order_id": "count"})

    If the aggregated column is also a group-by column, the output column
    is renamed "<col>_<func>" so the group labels aren't overwritten.

    sort_desc=True orders the result by the aggregated value, largest
    first -- needed for "top N" views, which otherwise got the first N
    groups ALPHABETICALLY, not the N biggest.
    """
    overlap = [c for c in agg_spec if c in group_by]
    if overlap:
        col, func = overlap[0], agg_spec[overlap[0]]
        out_name = f"{col}_{func}"
        result = df.groupby(group_by, dropna=False).agg(
            **{out_name: (col, func)}).reset_index()
        sort_col = out_name
    else:
        result = df.groupby(group_by).agg(agg_spec).reset_index()
        sort_col = next(iter(agg_spec))
    if sort_desc and sort_col in result.columns:
        result = result.sort_values(
            sort_col, ascending=False, kind="stable").reset_index(drop=True)
    return result

