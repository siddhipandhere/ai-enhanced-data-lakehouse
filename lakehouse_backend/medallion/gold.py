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
    if fewer than 2 such datasets exist.

    Deliberately exact-match only: combining datasets with different
    schemas would silently misalign columns or fill most of the result
    with nulls, which is worse than not offering a combined view at
    all. Ties (multiple groups of the same size) are broken by total
    row count, so the combination with the most actual data wins.
    """
    ready = [d for d in registry.list_datasets(uploaded_by=uploaded_by) if d.get("gold_status") == "ready"]
    groups: dict[tuple, list[dict]] = {}
    for d in ready:
        signature = tuple(sorted(d.get("columns") or []))
        if not signature:
            continue
        groups.setdefault(signature, []).append(d)

    candidates = [g for g in groups.values() if len(g) >= 2]
    if not candidates:
        return []

    candidates.sort(key=lambda g: sum(d.get("record_count") or 0 for d in g), reverse=True)
    return candidates[0]


def load_combined_gold_table(uploaded_by: str) -> pd.DataFrame:
    """
    Loads and concatenates every dataset in the largest exact-schema-
    match group for this user (see find_combinable_datasets). Adds a
    '_source_dataset' column so any result row can still be traced back
    to which original upload it came from. Raises ValueError if there's
    nothing eligible to combine.
    """
    group = find_combinable_datasets(uploaded_by)
    if not group:
        raise ValueError("No two ready datasets share a matching schema to combine.")

    frames = []
    for entry in group:
        df = load_gold_table(entry["table_name"])
        df = df.copy()
        df["_source_dataset"] = entry.get("original_name") or entry["table_name"]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Renumber the join-key column to guarantee uniqueness across the
    # combined rows. Datasets being combined here are near-certain to
    # have overlapping id ranges (e.g. two separate uploads of the same
    # source file both numbering rows 0..N-1) -- left as-is, that would
    # make semantic search's join-back step (which merges FAISS results
    # onto this table via config.JOIN_KEY) silently match the wrong
    # rows once ids collide across sources. Any other id-like column
    # (e.g. a UUID-style '_id') is untouched and still usable for display.
    combined[config.JOIN_KEY] = range(len(combined))
    logger.info(f"Combined {len(group)} datasets into one table: {len(combined)} total rows")
    return combined


def combined_text_column(uploaded_by: str) -> str | None:
    """The combined virtual table is only eligible for semantic search
    if EVERY dataset in the combinable group is itself search-ready AND
    they all agree on the same text_column. Matching schemas (required
    to combine at all) doesn't guarantee matching text_column choice --
    that's picked independently per upload -- so this is a stricter,
    separate check from find_combinable_datasets()."""
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
    directly: transparently handles the COMBINED_TABLE_SENTINEL value
    in addition to normal single-dataset table names."""
    if table_name == COMBINED_TABLE_SENTINEL:
        return load_combined_gold_table(uploaded_by)
    return load_gold_table(table_name)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def build_gold_table(silver_df: pd.DataFrame, table_name: str = "main") -> Path:
    """
    Repartitions/cleans a Silver DataFrame into the analytics-ready Gold
    layer and writes it as Parquet for efficient repeated reads
    (Storage Efficiency metric from the report, Section 3.3).
    """
    df = silver_df.copy()
    df = df.reset_index(drop=True)

    dest_path = config.GOLD_DIR / f"{table_name}.parquet"
    _atomic_write_parquet(df, dest_path)

    original_size = df.memory_usage(deep=True).sum()
    stored_size = dest_path.stat().st_size
    storage_efficiency = round(original_size / stored_size, 2) if stored_size else None

    logger.info(f"Built Gold table '{table_name}': {len(df)} records, "
                f"storage efficiency {storage_efficiency}x")
    return dest_path


def load_gold_table(table_name: str = "main") -> pd.DataFrame:
    path = config.GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No Gold table named '{table_name}' at {path}")
    return pd.read_parquet(path)


def run_sql_query(df: pd.DataFrame, query: str, cache_key: str | None = None) -> pd.DataFrame:
    """
    Executes a pandas-`query()`-style filter/aggregation string against
    the Gold table. This mirrors the report's SQL Analytics Layer
    (Algorithm 5) without requiring a Spark cluster.

    Example: run_sql_query(df, "region == 'North' and revenue > 1000")

    For grouped aggregations, prefer aggregate() below — pandas.query()
    only supports row-level filtering.
    """
    if cache_key and cache_key in _QUERY_CACHE:
        logger.info(f"Cache hit for query key '{cache_key}'")
        return _QUERY_CACHE[cache_key]

    result = df.query(query)

    if cache_key:
        _QUERY_CACHE[cache_key] = result

    return result


def aggregate(df: pd.DataFrame, group_by: list[str], agg_spec: dict[str, str]) -> pd.DataFrame:
    """
    e.g. aggregate(df, group_by=["region"], agg_spec={"revenue": "sum", "order_id": "count"})
    """
    return df.groupby(group_by).agg(agg_spec).reset_index()


def clear_cache() -> None:
    _QUERY_CACHE.clear()