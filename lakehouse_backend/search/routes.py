"""Semantic Search endpoint — FAISS search over a chosen dataset."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import config
from auth.dependencies import get_current_user
from auth.models import User
from embeddings.vector_store import build_index_if_needed
from medallion.gold import COMBINED_TABLE_SENTINEL, combined_text_column, resolve_dataset
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

    df = resolve_dataset(req.table_name, current_user.username)
    # Cheap no-op if the index already matches this table/column/size.
    build_index_if_needed(
        df, text_column=text_column,
        id_column=config.JOIN_KEY, table_name=req.table_name,
    )

    try:
        results = query_engine.semantic_query(df, req.query, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    registry.record_query(current_user.username)
    return {
        "table_name": req.table_name,
        "text_column": text_column,
        "columns": list(results.columns),
        "rows": query_engine.to_json_safe_records(results),
    }
