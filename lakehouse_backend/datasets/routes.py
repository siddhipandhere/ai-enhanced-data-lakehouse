"""Dataset listing + preview endpoints, backed by pipeline.registry."""

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import get_current_user
from auth.models import User
from medallion.gold import COMBINED_TABLE_SENTINEL, combined_text_column, find_combinable_datasets, resolve_dataset
from pipeline import registry
from sql import query_engine

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _overall_status(entry: dict) -> str:
    if entry.get("gold_status") == "ready":
        return "ready"
    if "failed" in (entry.get("bronze_status"), entry.get("silver_status"), entry.get("gold_status")):
        return "failed"
    return "processing"


def _to_summary(entry: dict) -> dict:
    return {
        "table_name": entry.get("table_name"),
        "original_name": entry.get("original_name"),
        "category": entry.get("category"),
        "record_count": entry.get("record_count", 0),
        "status": _overall_status(entry),
        "bronze_status": entry.get("bronze_status"),
        "silver_status": entry.get("silver_status"),
        "gold_status": entry.get("gold_status"),
        "vector_status": entry.get("vector_status"),
        "text_column": entry.get("text_column"),
        "columns": entry.get("columns", []),
        "error": entry.get("error"),
        "created_at": entry.get("created_at"),
    }


def _combined_summary(uploaded_by: str) -> dict | None:
    """A synthetic dataset-list entry for the combined virtual table.
    Fabricates the same status fields real entries have (gold_status,
    vector_status, etc.) so the frontend's existing filtering logic --
    "show in the dataset dropdowns if gold_status/vector_status is
    ready" -- picks this up with no frontend changes needed at all."""
    group = find_combinable_datasets(uploaded_by)
    if not group:
        return None
    text_column = combined_text_column(uploaded_by)
    return {
        "table_name": COMBINED_TABLE_SENTINEL,
        "original_name": f"All datasets (combined) — {len(group)} sources",
        "category": "combined",
        "record_count": sum(d.get("record_count") or 0 for d in group),
        "status": "ready",
        "bronze_status": "ready",
        "silver_status": "ready",
        "gold_status": "ready",
        "vector_status": "ready" if text_column else "skipped",
        "text_column": text_column,
        "columns": group[0].get("columns", []),
        "error": None,
        "created_at": None,
    }


@router.get("")
def list_datasets(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    summaries = [_to_summary(e) for e in entries]
    combined = _combined_summary(current_user.username)
    if combined:
        summaries.insert(0, combined)
    return {"datasets": summaries}


@router.get("/{table_name}/preview")
def preview_dataset(table_name: str, limit: int = 20, current_user: User = Depends(get_current_user)):
    if table_name == COMBINED_TABLE_SENTINEL:
        if not find_combinable_datasets(current_user.username):
            raise HTTPException(status_code=404, detail="No datasets are currently eligible to combine")
    else:
        entry = registry.get_dataset(table_name)
        if not entry or entry.get("uploaded_by") != current_user.username:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if entry.get("gold_status") != "ready":
            raise HTTPException(status_code=409, detail=f"Dataset not ready yet (gold_status={entry.get('gold_status')})")

    df = resolve_dataset(table_name, current_user.username)
    preview = df.head(max(1, min(limit, 200)))
    return {
        "table_name": table_name,
        "columns": list(preview.columns),
        "rows": query_engine.to_json_safe_records(preview),
        "total_records": len(df),
    }


@router.delete("/{table_name}")
def delete_dataset(table_name: str, current_user: User = Depends(get_current_user)):
    entry = registry.get_dataset(table_name)
    if not entry or entry.get("uploaded_by") != current_user.username:
        raise HTTPException(status_code=404, detail="Dataset not found")
    registry.delete_dataset(table_name)
    return {"deleted": table_name}