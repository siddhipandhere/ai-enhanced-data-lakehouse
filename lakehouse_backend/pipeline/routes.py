"""Pipeline monitor endpoint: per-dataset Bronze/Silver/Gold/Vector status."""

from fastapi import APIRouter, Depends

from auth.dependencies import get_current_user
from auth.models import User
from pipeline import registry

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_STAGES = ("bronze_status", "silver_status", "gold_status", "vector_status")


@router.get("/status")
def pipeline_status(current_user: User = Depends(get_current_user)):
    entries = registry.list_datasets(uploaded_by=current_user.username)
    return {
        "datasets": [
            {
                "table_name": e.get("table_name"),
                "original_name": e.get("original_name"),
                "stages": {stage.replace("_status", ""): e.get(stage, "pending") for stage in _STAGES},
                "error": e.get("error"),
                "updated_at": e.get("updated_at"),
            }
            for e in entries
        ]
    }
