"""Semantic Search endpoint — FAISS search over a chosen dataset."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import config
from auth.dependencies import get_current_user
from auth.models import User
from embeddings.vector_store import build_index_if_needed
from medallion.gold import (COMBINED_TABLE_SENTINEL, combined_text_column, resolve_dataset,
                           vector_index_key)
from pipeline import registry
from sql import query_engine

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    table_name: str
    query: str
    top_k: int = config.DEFAULT_TOP_K


@router.get("/tables")
def list_searchable_tables(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    tables = [
        {"table_name": e["table_name"], "original_name": e.get(
            "original_name"), "text_column": e.get("text_column")}
        for e in entries if e.get("vector_status") == "ready"
    ]

    text_column = combined_text_column(current_user.username)
    if text_column:
        tables.insert(0, {
            "table_name": COMBINED_TABLE_SENTINEL,
            "original_name": "All datasets (combined)",
            "text_column": text_column,
        })

    return {"tables": tables}


@router.post("/query")
def semantic_search(req: SearchRequest, current_user: User = Depends(get_current_user)):
    if req.table_name == COMBINED_TABLE_SENTINEL:
        text_column = combined_text_column(current_user.username)
        if not text_column:
            raise HTTPException(
                status_code=404, detail="No datasets are currently eligible for combined semantic search")
    else:
        entry = registry.get_dataset(req.table_name)
        if not entry or entry.get("uploaded_by") != current_user.username:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not entry.get("text_column"):
            raise HTTPException(
                status_code=422, detail="This dataset has no free-text column suitable for semantic search")
        if entry.get("vector_status") != "ready":
            raise HTTPException(
                status_code=409, detail=f"Search index not ready yet (vector_status={entry.get('vector_status')})")
        text_column = entry["text_column"]

    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Search text is empty")
    top_k = max(1, min(req.top_k, 100))

    df = resolve_dataset(req.table_name, current_user.username)
    index_key = vector_index_key(req.table_name, current_user.username)
    # Each dataset has its own index now, so this is a cheap no-op unless
    # the data changed (the first combined-view search does build one).
    build_index_if_needed(
        df, text_column=text_column,
        id_column=config.JOIN_KEY, table_name=index_key,
    )

    try:
        results = query_engine.semantic_query(df, req.query, top_k=top_k, table_name=index_key)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    results = query_engine.image_columns_first(results)
    registry.record_query(current_user.username)
    return {
        "table_name": req.table_name,
        "text_column": text_column,
        "columns": list(results.columns),
        "rows": query_engine.to_json_safe_records(results),
    }
