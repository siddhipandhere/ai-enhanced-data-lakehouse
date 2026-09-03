"""
Auto-promotion pipeline: Bronze -> Silver -> Gold -> (optional) vector
index, triggered right after upload instead of requiring a manual
example_run.py-style script.

The medallion/embeddings modules were built assuming a single
"currently loaded" dataset (report Algorithm 4-8 scope). This app lets
someone upload many datasets of different shapes at once, so this
module adds the two small things needed to make that safe:

1. A join key. Silver/Gold/the vector store all key off
   ``config.JOIN_KEY`` ("id"). Real uploads frequently don't have an
   "id" column (a raw product JSON export, a folder of PDFs, etc.), so
   ``_ensure_id_column`` adds a positional one when it's missing,
   instead of every downstream join silently breaking.
2. A text column to search on. Semantic Search needs *some* free-text
   column to embed. There's no user-authored schema to read that from,
   so ``_pick_text_column`` applies a simple heuristic (prefer columns
   whose name suggests free text; skip binary/short columns) and skips
   building an index entirely when nothing suitable is found — Reports
   and SQL Explorer still work against that dataset either way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from embeddings.vector_store import build_index_if_needed
from medallion.gold import build_gold_table
from medallion.silver import clean_and_promote, load_silver_as_pandas
from pipeline import registry
from utils.logger import get_logger

logger = get_logger("orchestration")

_TEXT_COLUMN_KEYWORDS = (
    "text", "description", "desc", "title", "name", "summary",
    "content", "comment", "review", "body", "caption",
)


def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if config.JOIN_KEY in df.columns:
        return df
    df = df.reset_index(drop=True).copy()
    df.insert(0, config.JOIN_KEY, range(len(df)))
    return df


def _pick_text_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for c in df.columns:
        if c == config.JOIN_KEY:
            continue
        # pandas 3.x gives plain string columns a dedicated 'str' dtype
        # instead of 'object' by default. is_string_dtype() covers both
        # that and the legacy object-dtype case, so this still works
        # whether the column came from pandas <3.0 or >=3.0. The old
        # `df[c].dtype != object` check silently excluded every ordinary
        # string column under pandas 3.x, so text_column was always None
        # and vector indexing was skipped for every dataset regardless
        # of whether it actually had good free-text content.
        if not pd.api.types.is_string_dtype(df[c]):
            continue
        sample = df[c].dropna().head(5).tolist()
        if not sample or not all(isinstance(v, str) for v in sample):
            continue  # skip binary/bytes columns (e.g. image content)
        avg_len = sum(len(v) for v in sample) / len(sample)
        if avg_len < 3:
            continue
        priority = next((i for i, kw in enumerate(_TEXT_COLUMN_KEYWORDS) if kw in c.lower()), len(_TEXT_COLUMN_KEYWORDS))
        candidates.append((priority, -avg_len, c))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def process_dataset(bronze_table: str, bronze_path: Path, category: str,
                     original_name: str, uploaded_by: str, record_count: int) -> None:
    """
    Drives one freshly-ingested Bronze table through Silver -> Gold ->
    vector index, recording status/errors in the registry at every
    stage. Never raises — a failure at any stage is recorded and the
    dataset simply stops at whatever layer it reached; it doesn't take
    the rest of the upload batch down with it.
    """
    registry.upsert_dataset(
        bronze_table,
        original_name=original_name,
        category=category,
        uploaded_by=uploaded_by,
        bronze_status="ready",
        record_count=record_count,
        silver_status="running",
    )

    try:
        silver_path = clean_and_promote(bronze_path, table_name=bronze_table)
    except Exception as e:
        logger.error(f"[{bronze_table}] Silver promotion failed: {e}")
        registry.upsert_dataset(bronze_table, silver_status="failed", error=str(e))
        return
    registry.upsert_dataset(bronze_table, silver_status="ready")

    registry.upsert_dataset(bronze_table, gold_status="running")
    try:
        silver_df = load_silver_as_pandas(silver_path)
        silver_df = _ensure_id_column(silver_df)
        build_gold_table(silver_df, table_name=bronze_table)
    except Exception as e:
        logger.error(f"[{bronze_table}] Gold build failed: {e}")
        registry.upsert_dataset(bronze_table, gold_status="failed", error=str(e))
        return
    registry.upsert_dataset(
        bronze_table,
        gold_status="ready",
        columns=list(silver_df.columns),
        record_count=len(silver_df),
    )

    text_column = _pick_text_column(silver_df)
    registry.upsert_dataset(bronze_table, text_column=text_column)

    if not text_column or silver_df.empty:
        registry.upsert_dataset(bronze_table, vector_status="skipped")
        return

    registry.upsert_dataset(bronze_table, vector_status="running")
    try:
        build_index_if_needed(
            silver_df, text_column=text_column,
            id_column=config.JOIN_KEY, table_name=bronze_table,
        )
    except Exception as e:
        logger.error(f"[{bronze_table}] Vector index build failed: {e}")
        registry.upsert_dataset(bronze_table, vector_status="failed", error=str(e))
        return
    registry.upsert_dataset(bronze_table, vector_status="ready")
    logger.info(f"[{bronze_table}] Pipeline complete: Bronze -> Silver -> Gold -> Vector index")