"""Pipeline monitor endpoints: per-dataset Bronze/Silver/Gold/Vector status + retry."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from auth.dependencies import get_current_user
from auth.models import User
from pipeline import registry
from pipeline.orchestration import mark_running_stages_failed, retry_dataset
from utils.logger import get_logger

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = get_logger("pipeline_routes")


@router.get("/status")
def pipeline_status(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    return {
        "datasets": [
            {
                "table_name": e.get("table_name"),
                "original_name": e.get("original_name") or e.get("table_name"),
                "stages": {stage.replace("_status", ""): e.get(stage, "pending") for stage in registry.STAGES},
                "error": e.get("error"),
                "notes": e.get("refine_notes") or [],
                "updated_at": e.get("updated_at"),
            }
            for e in entries
        ]
    }


def _retry_safely(table_name: str) -> None:
    try:
        logger.info(f"[{table_name}] Retry: {retry_dataset(table_name)}")
    except Exception as e:
        logger.error(f"[{table_name}] Retry failed: {e}")
        mark_running_stages_failed(table_name, str(e))


@router.post("/{table_name}/retry")
def retry(table_name: str, background_tasks: BackgroundTasks,
          current_user: User = Depends(get_current_user)):
    """Re-runs the unfinished stages of a failed/interrupted dataset in the
    background; poll GET /pipeline/status for progress."""
    entry = registry.get_dataset(table_name)
    if not entry or entry.get("uploaded_by") != current_user.username:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if any(entry.get(st) == "running" for st in registry.STAGES):
        raise HTTPException(status_code=409, detail="This dataset is already being processed")
    if all(entry.get(st) in ("ready", "skipped") for st in registry.STAGES):
        return {"table_name": table_name, "status": "nothing to retry"}
    if not any(entry.get(st) == "failed" for st in registry.STAGES):
        # "pending" = queued behind another upload in the same batch; it
        # will start on its own. Retrying now would process it twice.
        raise HTTPException(status_code=409, detail="This dataset is queued and will start processing shortly")

    stale_bronze = entry.get("gold_status") != "ready" and entry.get("bronze_status") != "ready"
    if stale_bronze:
        raise HTTPException(status_code=409, detail="This upload never reached Bronze - delete it and upload the file again.")

    # Mark it running right away so the UI (and a double click) sees it.
    stage = "vector_status" if entry.get("gold_status") == "ready" else "silver_status"
    registry.upsert_dataset(table_name, **{stage: "running"}, error=None)
    background_tasks.add_task(_retry_safely, table_name)
    return {"table_name": table_name, "status": "retry started"}
