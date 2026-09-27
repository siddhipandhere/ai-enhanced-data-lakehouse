"""SQL Explorer endpoint — filter/aggregate a chosen Gold table."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.models import User
from agents.query_planner import check_filter_query
from medallion.gold import COMBINED_TABLE_SENTINEL, find_combinable_datasets, resolve_dataset
from pipeline import registry
from sql import query_engine

router = APIRouter(prefix="/sql", tags=["sql"])


class SqlQueryRequest(BaseModel):
    table_name: str
    # pandas DataFrame.query()-style filter, e.g. "price > 100"
    query: str | None = None
    group_by: list[str] | None = None
    agg_column: str | None = None
    agg_func: str | None = None       # "sum" | "mean" | "count" | "max" | "min"
    limit: int = 200
    # Sort aggregated rows by the aggregated value, largest first. Without
    # it "top 15 categories" were the first 15 ALPHABETICALLY.
    sort_desc: bool = False
    # False for background calls (the Overview charts). Every Overview load
    # fired 2 chart queries that were counted as "AI queries", inflating
    # that KPI without the user asking anything.
    track: bool = True


_AGG_FUNCS = {"sum", "mean", "count", "max", "min", "median", "nunique"}


def _authorized_ready_dataset(table_name: str, current_user: User) -> None:
    """Raises if table_name isn't something this user is allowed to query."""
    if table_name == COMBINED_TABLE_SENTINEL:
        if not find_combinable_datasets(current_user.username):
            raise HTTPException(
                status_code=404, detail="No datasets are currently eligible to combine")
        return
    entry = registry.get_dataset(table_name)
    if not entry or entry.get("uploaded_by") != current_user.username:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if entry.get("gold_status") != "ready":
        raise HTTPException(
            status_code=409, detail=f"Dataset not ready yet (gold_status={entry.get('gold_status')})")


@router.get("/tables")
def list_queryable_tables(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    tables = [
        {"table_name": e["table_name"], "original_name": e.get(
            "original_name"), "columns": e.get("columns", [])}
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
        # Raw user input goes into pandas.query(); it must pass the same
        # validator the LLM planner uses (it previously went straight in,
        # and "@os.system(...)" executed on the server).
        safe_query, reason = check_filter_query(req.query, list(df.columns))
        if reason:
            raise HTTPException(
                status_code=400, detail=f"Query rejected: {reason}")
        try:
            df = query_engine.filter_query(df, safe_query)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid query expression: {e}")

    if req.group_by and req.agg_column and req.agg_func:
        missing = [c for c in [*req.group_by,
                               req.agg_column] if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Unknown column(s): {missing}")
        if req.agg_func not in _AGG_FUNCS:
            raise HTTPException(
                status_code=400, detail=f"agg_func must be one of {sorted(_AGG_FUNCS)}")
        try:
            df = query_engine.aggregate_query(df, group_by=req.group_by, agg_spec={
                                              req.agg_column: req.agg_func}, sort_desc=req.sort_desc)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Aggregation failed: {e}")

    limit = max(1, min(req.limit, 1000))
    page = df.head(limit)
    if req.track:
        registry.record_query(current_user.username)
    return {
        "table_name": req.table_name,
        "columns": list(df.columns),
        "rows": query_engine.to_json_safe_records(page),
        "row_count": len(df),
        "truncated": len(df) > limit,
    }
