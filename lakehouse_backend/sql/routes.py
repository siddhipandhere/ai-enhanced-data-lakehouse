"""SQL Explorer endpoint — filter/aggregate a chosen Gold table."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.models import User
from medallion.gold import COMBINED_TABLE_SENTINEL, find_combinable_datasets, resolve_dataset
from pipeline import registry
from sql import query_engine

router = APIRouter(prefix="/sql", tags=["sql"])


class SqlQueryRequest(BaseModel):
    table_name: str
    query: str | None = None          # pandas DataFrame.query()-style filter, e.g. "price > 100"
    group_by: list[str] | None = None
    agg_column: str | None = None
    agg_func: str | None = None       # "sum" | "mean" | "count" | "max" | "min"
    limit: int = 200


def _authorized_ready_dataset(table_name: str, current_user: User) -> None:
    """Raises if table_name isn't something this user is allowed to
    query. The combined-table sentinel is authorized by checking
    eligibility (2+ matching-schema datasets) rather than a registry
    lookup, since it isn't a real dataset entry."""
    if table_name == COMBINED_TABLE_SENTINEL:
        if not find_combinable_datasets(current_user.username):
            raise HTTPException(status_code=404, detail="No datasets are currently eligible to combine")
        return
    entry = registry.get_dataset(table_name)
    if not entry or entry.get("uploaded_by") != current_user.username:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if entry.get("gold_status") != "ready":
        raise HTTPException(status_code=409, detail=f"Dataset not ready yet (gold_status={entry.get('gold_status')})")


@router.get("/tables")
def list_queryable_tables(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    tables = [
        {"table_name": e["table_name"], "original_name": e.get("original_name"), "columns": e.get("columns", [])}
        for e in entries if e.get("gold_status") == "ready"
    ]

    combinable = find_combinable_datasets(current_user.username)
    if combinable:
        tables.insert(0, {
            "table_name": COMBINED_TABLE_SENTINEL,
            "original_name": f"All datasets (combined) — {len(combinable)} sources",
            "columns": combinable[0].get("columns", []),
        })

    return {"tables": tables}


@router.post("/query")
def run_sql(req: SqlQueryRequest, current_user: User = Depends(get_current_user)):
    _authorized_ready_dataset(req.table_name, current_user)
    df = resolve_dataset(req.table_name, current_user.username)

    if req.query:
        try:
            df = query_engine.filter_query(df, req.query)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid query expression: {e}")

    if req.group_by and req.agg_column and req.agg_func:
        try:
            df = query_engine.aggregate_query(df, group_by=req.group_by, agg_spec={req.agg_column: req.agg_func})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Aggregation failed: {e}")

    limit = max(1, min(req.limit, 1000))
    page = df.head(limit)
    return {
        "table_name": req.table_name,
        "columns": list(df.columns),
        "rows": query_engine.to_json_safe_records(page),
        "row_count": len(df),
        "truncated": len(df) > limit,
    }