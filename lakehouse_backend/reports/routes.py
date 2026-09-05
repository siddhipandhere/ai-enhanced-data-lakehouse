"""Reports endpoint — runs the agent orchestrator against a chosen dataset."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agents.orchestrator import handle_request
from auth.dependencies import get_current_user
from auth.models import User
from medallion.gold import COMBINED_TABLE_SENTINEL, find_combinable_datasets, resolve_dataset
from pipeline import registry
from sql import query_engine

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportRequest(BaseModel):
    table_name: str
    question: str


@router.get("")
def list_reports(current_user: User = Depends(get_current_user)):
    return {"reports": registry.list_reports(uploaded_by=current_user.username)}


@router.post("/generate")
def generate_report(req: ReportRequest, current_user: User = Depends(get_current_user)):
    if req.table_name == COMBINED_TABLE_SENTINEL:
        if not find_combinable_datasets(current_user.username):
            raise HTTPException(
                status_code=404, detail="No datasets are currently eligible to combine")
        original_name = f"All datasets (combined)"
    else:
        entry = registry.get_dataset(req.table_name)
        if not entry or entry.get("uploaded_by") != current_user.username:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if entry.get("gold_status") != "ready":
            raise HTTPException(
                status_code=409, detail=f"Dataset not ready yet (gold_status={entry.get('gold_status')})")
        original_name = entry.get("original_name")

    df = resolve_dataset(req.table_name, current_user.username)
    result = handle_request(req.question, df, table_name=req.table_name)

    # Keep result rows out of the persisted history (can be large/stale);
    # store the plan + summary + stats, which is what a report list needs.
    saved = registry.add_report({
        "uploaded_by": current_user.username,
        "table_name": req.table_name,
        "original_name": original_name,
        "question": req.question,
        "plan": result["plan"],
        "summary": result["summary"],
        "analysis": result["analysis"],
    })
    registry.record_query(current_user.username)

    return {
        "id": saved["id"],
        "table_name": req.table_name,
        "question": req.question,
        "plan": result["plan"],
        "summary": result["summary"],
        "analysis": result["analysis"],
        "rows": query_engine.to_json_safe_records(result["results"].head(50)),
        "columns": list(result["results"].columns),
    }
