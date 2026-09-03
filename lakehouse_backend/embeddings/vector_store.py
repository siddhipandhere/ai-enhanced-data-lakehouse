"""
Vector database layer using FAISS (report Algorithms 7-8: vector storage
and semantic search).
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import config
from embeddings.embedder import embed_texts, embed_query
from utils.logger import get_logger

logger = get_logger("vector_store")

INDEX_PATH = config.VECTOR_STORE_DIR / "index.faiss"
ID_MAP_PATH = config.VECTOR_STORE_DIR / "id_map.csv"
META_PATH = config.VECTOR_STORE_DIR / "meta.json"


def build_index(df: pd.DataFrame, text_column: str, id_column: str = config.JOIN_KEY,
                 table_name: str | None = None) -> None:
    """
    Generates embeddings for every row's text field, builds a FAISS
    index over them, and saves both the index and an id map (FAISS row
    position -> original record id) so search results can be joined
    back to the Gold table. Also records which table/text/id column were
    used, so a query against a different dataset or column triggers a
    rebuild instead of silently searching stale data.
    """
    # .fillna('') must run BEFORE .astype(str), not after. Under pandas
    # 3.x's dedicated 'str' dtype, .astype(str) alone is a no-op for
    # missing values -- it leaves real float NaN objects sitting in the
    # resulting list instead of stringifying them to 'nan' (this is a
    # behavioral change from pandas <3.0, where object-dtype columns did
    # get NaN stringified by astype(str)). Any column with missing
    # values then crashes the embedding model with "Unsupported input
    # type: float" the moment indexing reaches one of those rows.
    texts = df[text_column].fillna("").astype(str).tolist()
    vectors = embed_texts(texts)

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)  # inner product on normalized vectors = cosine
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    df[[id_column]].reset_index(drop=True).to_csv(ID_MAP_PATH, index=False)
    with open(META_PATH, "w") as f:
        json.dump({
            "table_name": table_name,
            "text_column": text_column,
            "id_column": id_column,
            "row_count": len(df),
        }, f)

    logger.info(f"Built FAISS index: {index.ntotal} vectors, dim {config.EMBEDDING_DIM}, "
                f"table='{table_name}', text_column='{text_column}'")


def build_index_if_needed(df: pd.DataFrame, text_column: str, id_column: str = config.JOIN_KEY,
                           table_name: str | None = None) -> None:
    """
    Builds the index only if it doesn't exist yet, or was built against a
    different dataset / text column / a differently-sized dataset. Since
    this app can have several Gold tables at once, ``table_name`` is
    checked first — otherwise switching datasets with the same column
    name and row count would silently reuse the wrong index. This is
    what lets the query agent (or the Semantic Search page) request
    search on whatever dataset/column is currently selected without the
    caller having to manage index lifecycle manually.
    """
    if not index_exists():
        build_index(df, text_column, id_column, table_name)
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    if (meta.get("table_name") != table_name
            or meta.get("text_column") != text_column
            or meta.get("row_count") != len(df)):
        logger.info("Vector index is stale for this dataset/column, rebuilding")
        build_index(df, text_column, id_column, table_name)


def _load_index() -> tuple[faiss.Index, pd.DataFrame]:
    if not INDEX_PATH.exists() or not ID_MAP_PATH.exists():
        raise FileNotFoundError("Vector index not found. Call build_index() first.")
    index = faiss.read_index(str(INDEX_PATH))
    id_map = pd.read_csv(ID_MAP_PATH)
    return index, id_map


def semantic_search(query_text: str, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
    """
    Returns a DataFrame of [id, similarity_score] ranked by relevance
    to the query, ready to be joined back against the Gold table.
    """
    index, id_map = _load_index()
    query_vec = embed_query(query_text).reshape(1, -1)

    scores, indices = index.search(query_vec, top_k)
    scores, indices = scores[0], indices[0]

    valid = indices >= 0
    result_ids = id_map.iloc[indices[valid]][config.JOIN_KEY].values
    result_scores = scores[valid]

    return pd.DataFrame({config.JOIN_KEY: result_ids, "similarity_score": result_scores}) \
        .sort_values("similarity_score", ascending=False) \
        .reset_index(drop=True)


def index_exists() -> bool:
    return INDEX_PATH.exists() and ID_MAP_PATH.exists()