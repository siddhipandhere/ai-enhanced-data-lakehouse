"""
Vector database layer using FAISS (report Algorithms 7-8: vector storage
and semantic search).

One index PER DATASET, stored at data/vector_store/<table_name>/.

The previous version had a single shared slot (one index.faiss / one
id_map.csv / one meta.json for the whole app). With several datasets
that meant:
  - every upload's index build overwrote the previous dataset's index,
    while the registry still showed that older dataset as
    vector_status="ready";
  - the first search (or semantic report) on any dataset other than the
    most recent one silently triggered a FULL re-embed of 30,000 rows
    inside the HTTP request -- several minutes on a laptop CPU, which
    looks exactly like a search/report that hangs forever;
  - switching back and forth between two datasets re-embedded each time.
Per-dataset folders make "ready" actually mean ready, and a search never
rebuilds unless that dataset's own data changed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import config
from embeddings.embedder import embed_clip_text, embed_images, embed_query, embed_texts
from utils.media import is_bytes_column, open_image
from utils.logger import get_logger

logger = get_logger("vector_store")

VECTOR_ROOT = config.VECTOR_STORE_DIR
_DEFAULT_KEY = "_default"

# Legacy single-slot files (pre per-dataset layout). Kept so an index
# built by the old code can be adopted instead of re-embedded.
_LEGACY_INDEX = VECTOR_ROOT / "index.faiss"
_LEGACY_ID_MAP = VECTOR_ROOT / "id_map.csv"
_LEGACY_META = VECTOR_ROOT / "meta.json"


def _safe_key(table_name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", table_name or _DEFAULT_KEY)


def index_dir(table_name: str | None) -> Path:
    return VECTOR_ROOT / _safe_key(table_name)


def _paths(table_name: str | None) -> tuple[Path, Path, Path]:
    d = index_dir(table_name)
    return d / "index.faiss", d / "id_map.csv", d / "meta.json"


def _read_meta(table_name: str | None) -> dict | None:
    _, _, meta_path = _paths(table_name)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _adopt_legacy_index(table_name: str | None) -> None:
    """Moves an old single-slot index into this table's folder if it was
    built for this table. Saves a full re-embed after upgrading."""
    if not (_LEGACY_META.exists() and _LEGACY_INDEX.exists() and _LEGACY_ID_MAP.exists()):
        return
    try:
        with open(_LEGACY_META) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if meta.get("table_name") != table_name:
        return
    dest = index_dir(table_name)
    dest.mkdir(parents=True, exist_ok=True)
    index_path, id_map_path, meta_path = _paths(table_name)
    shutil.move(str(_LEGACY_INDEX), str(index_path))
    shutil.move(str(_LEGACY_ID_MAP), str(id_map_path))
    shutil.move(str(_LEGACY_META), str(meta_path))  # meta last = "complete"
    logger.info(f"Adopted legacy vector index for '{table_name}' into {dest}")


def build_index(df: pd.DataFrame, text_column: str, id_column: str = config.JOIN_KEY,
                table_name: str | None = None) -> None:
    """
    Embeds every row's text, builds a FAISS index, and saves index + id
    map + meta into this table's own folder.

    Files are written to a temp folder first and meta.json is moved in
    LAST, so an interrupted build (server stopped mid-way) never leaves
    a half-written index that looks valid -- no meta means "not built".
    """
    kind = _kind_for(df, text_column)
    if kind == "image":
        # Image collection: embed the pictures themselves with CLIP, so the
        # index can be searched by typing a description.
        vectors = embed_images([open_image(b) for b in df[text_column].tolist()])
        model_name = config.CLIP_MODEL_NAME
    else:
        # .fillna('') must run BEFORE .astype(str): under pandas 3.x's 'str'
        # dtype, astype(str) leaves real NaN floats that crash the encoder.
        texts = df[text_column].fillna("").astype(str).tolist()
        vectors = embed_texts(texts)
        model_name = config.EMBEDDING_MODEL_NAME
    dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[1] else config.EMBEDDING_DIM

    index = faiss.IndexFlatIP(dim)  # normalized vectors -> cosine
    index.add(vectors)

    final_dir = index_dir(table_name)
    tmp_dir = VECTOR_ROOT / f"_tmp_{_safe_key(table_name)}_{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(tmp_dir / "index.faiss"))
        df[[id_column]].reset_index(drop=True).to_csv(tmp_dir / "id_map.csv", index=False)
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump({
                "table_name": table_name,
                "text_column": text_column,
                "id_column": id_column,
                "row_count": len(df),
                "kind": kind,
                "model": model_name,
                "dim": dim,
            }, f)

        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "meta.json").unlink(missing_ok=True)  # invalidate old index first
        for name in ("index.faiss", "id_map.csv", "meta.json"):
            os.replace(tmp_dir / name, final_dir / name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Built FAISS {kind} index: {index.ntotal} vectors, dim {dim}, "
                f"table='{table_name}', column='{text_column}'")


def _kind_for(df: pd.DataFrame, column: str) -> str:
    return "image" if column in df.columns and is_bytes_column(df[column]) else "text"


def has_index(table_name: str | None) -> bool:
    """True if this table has an index on disk (adopting a legacy
    single-slot index for it first, if that's where it lives)."""
    if not index_exists(table_name):
        _adopt_legacy_index(table_name)
    return index_exists(table_name)


def is_index_current(table_name: str | None, text_column: str, row_count: int,
                     kind: str = "text") -> bool:
    meta = _read_meta(table_name)
    return bool(meta
                and index_exists(table_name)
                and meta.get("table_name") == table_name
                and meta.get("text_column") == text_column
                and meta.get("row_count") == row_count
                and (meta.get("kind") or "text") == kind)


def build_index_if_needed(df: pd.DataFrame, text_column: str, id_column: str = config.JOIN_KEY,
                          table_name: str | None = None) -> bool:
    """Builds this table's index only if it's missing or stale (different
    text column / row count). Returns True if a (slow) build happened."""
    if not index_exists(table_name):
        _adopt_legacy_index(table_name)
    if is_index_current(table_name, text_column, len(df), kind=_kind_for(df, text_column)):
        return False
    logger.info(f"Vector index for '{table_name}' missing or stale, building")
    build_index(df, text_column, id_column, table_name)
    return True


def _load_index(table_name: str | None) -> tuple[faiss.Index, pd.DataFrame, dict]:
    index_path, id_map_path, _ = _paths(table_name)
    meta = _read_meta(table_name)
    if not meta or not index_path.exists() or not id_map_path.exists():
        raise FileNotFoundError(f"Vector index for '{table_name}' not found. Call build_index() first.")
    index = faiss.read_index(str(index_path))
    id_map = pd.read_csv(id_map_path)
    return index, id_map, meta


def semantic_search(query_text: str, top_k: int = config.DEFAULT_TOP_K,
                    table_name: str | None = None) -> pd.DataFrame:
    """Returns [id, similarity_score] ranked by relevance to the query."""
    index, id_map, meta = _load_index(table_name)
    id_col = meta.get("id_column") or config.JOIN_KEY
    top_k = max(1, min(int(top_k), index.ntotal)) if index.ntotal else 0
    if top_k == 0:
        return pd.DataFrame({config.JOIN_KEY: [], "similarity_score": []})

    # An image index was built with CLIP, so the query must be embedded by
    # CLIP's text encoder to land in the same vector space as the pictures.
    embed = embed_clip_text if (meta.get("kind") or "text") == "image" else embed_query
    query_vec = embed(query_text).reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    scores, indices = scores[0], indices[0]

    valid = indices >= 0
    result_ids = id_map.iloc[indices[valid]][id_col].values
    return pd.DataFrame({config.JOIN_KEY: result_ids, "similarity_score": scores[valid]}) \
        .sort_values("similarity_score", ascending=False) \
        .reset_index(drop=True)


def index_exists(table_name: str | None = None) -> bool:
    index_path, id_map_path, meta_path = _paths(table_name)
    return index_path.exists() and id_map_path.exists() and meta_path.exists()


def index_size_bytes(table_name: str | None) -> int:
    d = index_dir(table_name)
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def delete_index(table_name: str | None) -> None:
    shutil.rmtree(index_dir(table_name), ignore_errors=True)
    # Also clear a legacy single-slot index that belonged to this table.
    try:
        if _LEGACY_META.exists():
            with open(_LEGACY_META) as f:
                if json.load(f).get("table_name") == table_name:
                    for p in (_LEGACY_INDEX, _LEGACY_ID_MAP, _LEGACY_META):
                        p.unlink(missing_ok=True)
    except (json.JSONDecodeError, OSError):
        pass
