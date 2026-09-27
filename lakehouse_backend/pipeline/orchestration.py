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

import re
from pathlib import Path

import pandas as pd

import config
from embeddings.vector_store import build_index_if_needed, has_index
from medallion.gold import build_gold_table, load_gold_table
from medallion.refine import refine
from medallion.silver import clean_and_promote, load_silver_as_pandas
from pipeline import registry
from utils.logger import get_logger
from utils.media import image_column

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


_FILE_IDENTITY_COLUMNS = {"source_file", "original_name", "path", "source_path", "message_id",
                          "image_format", "color_mode"}

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
        # File-identity columns (added by PDF/image ingestion) are names, not
        # content: indexing them made "semantic search" over a scanned PDF or
        # an image just match file names. With no real text, skip the index.
        if c == config.JOIN_KEY or c in _FILE_IDENTITY_COLUMNS:
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
            "[", "{") or v.strip().lower().startswith(("http://", "https://", "file:", "dbfs:", "s3:"))
            or re.match(r"^[A-Za-z]:[\\/]", v.strip())  # bare URLs and file paths aren't prose
            or re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", v.strip()))  # nor are email addresses
        if structured / len(sample) > 0.5:
            continue
        # Dates, amounts and codes ("2001-05-01 00:00:00", "$1,200.50",
        # "12/31") are strings of digits and separators -- nothing to
        # search by meaning. Several Enron sheets indexed a date column.
        numeric_like = sum(1 for v in sample if re.fullmatch(r"[\d\s:/.,+\-$%()]*", v.strip()))
        if numeric_like / len(sample) > 0.5:
            continue
        avg_len = sum(len(v) for v in sample) / len(sample)
        if avg_len < 3:
            continue
        priority = next((i for i, kw in enumerate(
            _TEXT_COLUMN_KEYWORDS) if kw in c.lower()), len(_TEXT_COLUMN_KEYWORDS))
        if priority < len(_TEXT_COLUMN_KEYWORDS):
            # Short columns only qualify by name ("name", "title", ...): a
            # short column called "month" or "bid_2" is not search text.
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


IMAGE_SEARCH_NOTE = ("Built an image search index (CLIP): in Semantic Search, describe what's in a "
                     "picture (e.g. 'an office tower', 'a stock price chart') to find it")


def _pick_search_column(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """(column, kind): free text if the table has any, otherwise the image
    bytes of an image collection (searched with CLIP), otherwise nothing."""
    text_column = _pick_text_column(df)
    if text_column:
        return text_column, "text"
    img = image_column(df)
    if img:
        return img, "image"
    return None, None


def _add_note(table_name: str, note: str) -> None:
    notes = list((registry.get_dataset(table_name) or {}).get("refine_notes") or [])
    if note not in notes:
        registry.upsert_dataset(table_name, refine_notes=notes + [note])


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
        create=True,
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
        # Structural refinement (medallion/refine.py): parse raw email text,
        # treat "NaN"/"N/A" text as missing, drop TOTAL rows, de-duplicate
        # emails. The notes are shown on the Pipeline page.
        silver_df, refine_notes = refine(silver_df)
        registry.upsert_dataset(bronze_table, refine_notes=refine_notes)
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

    text_column, search_kind = _pick_search_column(silver_df)
    registry.upsert_dataset(bronze_table, text_column=text_column, search_kind=search_kind)

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
    registry.upsert_dataset(bronze_table, vector_status="ready", error=None)
    if search_kind == "image":
        _add_note(bronze_table, IMAGE_SEARCH_NOTE)
    logger.info(
        f"[{bronze_table}] Pipeline complete: Bronze -> Silver -> Gold -> Vector index")


def mark_running_stages_failed(table_name: str, error: str) -> None:
    """Safety net for an unexpected exception escaping process_dataset():
    without it the stage that was running stays "running" forever."""
    entry = registry.get_dataset(table_name) or {}
    updates = {st: "failed" for st in registry.STAGES if entry.get(st) == "running"}
    registry.upsert_dataset(table_name, error=error, **updates)


def retry_dataset(table_name: str) -> str:
    """
    Re-runs whatever didn't finish for one dataset (used by the Pipeline
    page's Retry button, POST /pipeline/{table}/retry):
      - Gold ready, only the vector index missing/failed -> rebuild just
        the index from the Gold table (no Spark needed);
      - otherwise, if the Bronze Delta table is still on disk -> re-run
        Silver -> Gold -> index from Bronze.
    Returns a short description of what was started. Raises ValueError
    when there's nothing to retry from (Bronze data is gone).
    """
    entry = registry.get_dataset(table_name)
    if not entry:
        raise ValueError("Dataset not found")

    if entry.get("gold_status") == "ready":
        gold_df = load_gold_table(table_name)
        text_column, search_kind = _pick_search_column(gold_df)
        if entry.get("text_column") in gold_df.columns:
            text_column = entry["text_column"]
            search_kind = "image" if text_column == image_column(gold_df) else "text"
        if not text_column:
            registry.upsert_dataset(table_name, text_column=None, search_kind=None,
                                    vector_status="skipped", error=None)
            return "no text column; vector index skipped"
        registry.upsert_dataset(table_name, text_column=text_column, search_kind=search_kind,
                                vector_status="running", error=None)
        try:
            build_index_if_needed(gold_df, text_column=text_column,
                                  id_column=config.JOIN_KEY, table_name=table_name)
        except Exception as e:
            logger.error(f"[{table_name}] Vector index retry failed: {e}")
            registry.upsert_dataset(table_name, vector_status="failed", error=str(e))
            return "vector index retry failed"
        registry.upsert_dataset(table_name, vector_status="ready")
        if search_kind == "image":
            _add_note(table_name, IMAGE_SEARCH_NOTE)
        return "vector index rebuilt"

    bronze_path = config.BRONZE_DIR / table_name
    if not (bronze_path / "_delta_log").exists():
        raise ValueError("The Bronze data for this dataset no longer exists - delete it and upload the file again.")
    registry.upsert_dataset(table_name, error=None, silver_status="pending",
                            gold_status="pending", vector_status="pending")
    process_dataset(
        bronze_table=table_name, bronze_path=bronze_path,
        category=entry.get("category"), original_name=entry.get("original_name") or table_name,
        uploaded_by=entry.get("uploaded_by"), record_count=entry.get("record_count") or 0,
    )
    return "pipeline re-run from Bronze"


def verify_vector_indexes() -> list[str]:
    """
    Startup check: a dataset marked vector_status="ready" must actually
    have its own index on disk. Under the old single-slot design every
    new upload overwrote the previous dataset's index while its status
    stayed "ready"; the next search on it then re-embedded the whole
    table inside the HTTP request (minutes -> looks like a hang). Such
    datasets are flagged as failed so the Pipeline page offers Retry,
    which rebuilds the index in the background instead.
    """
    flagged = []
    for e in registry.list_datasets():
        name = e.get("table_name")
        if e.get("vector_status") != "ready" or not name:
            continue
        if not has_index(name):
            registry.upsert_dataset(
                name, vector_status="failed",
                error="Search index missing (overwritten by an older version of the app). Click Retry to rebuild it.")
            flagged.append(name)
    if flagged:
        logger.warning(f"Datasets whose search index is missing (Retry to rebuild): {flagged}")
    return flagged
