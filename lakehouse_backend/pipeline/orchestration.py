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


_MEANINGFUL_TEXT_LEN = 20  # below this, a string column reads as a short
# label/code/badge, not free-form prose worth
# semantically searching over


def _pick_text_column(df: pd.DataFrame) -> str | None:
    """
    Picks the column to embed for semantic search.

    This used to rank primarily by column NAME (does it contain "text",
    "description", "title", ...?) with actual content length only as a
    tie-breaker. That trusts the source data's column names to be
    accurate -- which a real-world export can't be relied on for. A
    real Flipkart export processed through this app had its "discount"
    and "description" columns swapped: "description" held short values
    like "69% off" (a discount badge) while "discount" held the actual
    long-form product description. Name-first ranking picked
    "description" every time purely because of its name, and the
    resulting semantic search embedded thousands of near-duplicate
    "NN% off" strings instead of real product text -- technically
    "working" (no error, an index got built) but semantically useless,
    with every result an near-arbitrary tie at a very low similarity
    score.

    Ranking the qualifying (long-enough) columns by actual sampled
    length FIRST, with the column-name keywords only as a tie-breaker,
    means a mislabeled column can't win just because of its name -- the
    real prose wins because it *is* long, regardless of what its column
    happens to be called. A column of serialized structure (a list of
    image URLs, a list of {"key": "value"} spec dicts, a single bare
    URL) can also be "long" without being prose at all, so those are
    filtered out by shape before length is even considered -- otherwise
    "longest average string" would just as happily pick a JSON blob as
    a real description.
    """
    candidates = []
    fallback_candidates = []
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
        sample = df[c].dropna().head(30).tolist()
        if not sample or not all(isinstance(v, str) for v in sample):
            continue  # skip binary/bytes columns (e.g. image content)
        # Structured/serialized values -- a Python-repr list or dict, or
        # a bare URL -- read as long "text" by character count but carry
        # no natural-language meaning to embed. A real Flipkart export's
        # "images" (list of URLs) and "product_details" (list of spec
        # dicts) columns both look like this.
        structured = sum(1 for v in sample if v.strip()[:1] in (
            "[", "{") or v.strip().lower().startswith(("http://", "https://")))
        if structured / len(sample) > 0.5:
            continue
        avg_len = sum(len(v) for v in sample) / len(sample)
        if avg_len < 3:
            continue
        priority = next((i for i, kw in enumerate(
            _TEXT_COLUMN_KEYWORDS) if kw in c.lower()), len(_TEXT_COLUMN_KEYWORDS))
        fallback_candidates.append((priority, -avg_len, c))
        if avg_len >= _MEANINGFUL_TEXT_LEN:
            candidates.append((-avg_len, priority, c))

    # Prefer genuine long-form text (ranked by how much of it there
    # actually is); only fall back to ranking short columns by name
    # (the original behavior) if nothing in the dataset clears the
    # length bar at all.
    if candidates:
        candidates.sort()
        return candidates[0][2]
    if fallback_candidates:
        fallback_candidates.sort()
        return fallback_candidates[0][2]
    return None


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
        registry.upsert_dataset(
            bronze_table, silver_status="failed", error=str(e))
        return
    registry.upsert_dataset(bronze_table, silver_status="ready")

    registry.upsert_dataset(bronze_table, gold_status="running")
    try:
        silver_df = load_silver_as_pandas(silver_path)
        silver_df = _ensure_id_column(silver_df)
        build_gold_table(silver_df, table_name=bronze_table)
    except Exception as e:
        logger.error(f"[{bronze_table}] Gold build failed: {e}")
        registry.upsert_dataset(
            bronze_table, gold_status="failed", error=str(e))
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
        registry.upsert_dataset(
            bronze_table, vector_status="failed", error=str(e))
        return
    registry.upsert_dataset(bronze_table, vector_status="ready")
    logger.info(
        f"[{bronze_table}] Pipeline complete: Bronze -> Silver -> Gold -> Vector index")
