"""
Bulk, multi-format dataset upload endpoint — behind login.

Accepts any mix of structured (.csv/.xlsx/.parquet), semi-structured
(.json/.xml), and unstructured (.png/.jpg/.pdf/.txt) files in a single
request, in one bulk call to ingestion.loaders.ingest_files_bulk().
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from auth.dependencies import get_current_user
from auth.models import User
from ingestion.loaders import ValidationError, ingest_files_bulk
from pipeline import registry
from pipeline.orchestration import process_dataset

router = APIRouter(prefix="/upload", tags=["upload"])


def _run_pipeline_safely(**kwargs) -> None:
    """Wrapper run in the background so a stage failure can't crash the
    worker thread silently — process_dataset already records failures
    into the registry per-stage, this is just an extra belt-and-braces
    catch since nothing awaits this call's result."""
    try:
        process_dataset(**kwargs)
    except Exception as e:
        registry.upsert_dataset(kwargs["bronze_table"], error=str(e))


@router.post("/bulk")
async def upload_bulk(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Uploads and ingests multiple datasets of any supported type at once.
    Each file lands as its own Bronze Delta table, tagged with the
    authenticated user. The whole batch commits atomically — see
    ingestion.loaders.ingest_files_bulk for the staging/rollback logic.

    The Bronze commit happens inline (fast: it's just landing the raw
    file), but Silver -> Gold -> vector-index promotion is handed to a
    background task instead of being awaited here. process_dataset() is
    a long, CPU-bound, synchronous call (a Spark job, then a pandas Gold
    build, then embedding every row with sentence-transformers) — run
    directly inside this async endpoint it blocks FastAPI's single event
    loop for its entire duration, which also freezes every other
    request on this process, INCLUDING the dashboard's own
    GET /pipeline/status polls. That's what made the Pipeline page look
    permanently stuck on "Running": the page wasn't wrong about the
    status, the server just couldn't respond to the refresh until the
    whole pipeline finished. Returning immediately and letting the
    pipeline run in the background means /pipeline/status keeps
    responding (with real "running" status) the entire time.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        for upload in files:
            dest = Path(tmp_dir) / upload.filename
            with open(dest, "wb") as f:
                f.write(await upload.read())
            saved_paths.append(dest)

        try:
            results = ingest_files_bulk(
                saved_paths, uploaded_by=current_user.username)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    for r in results:
        background_tasks.add_task(
            _run_pipeline_safely,
            bronze_table=r.bronze_table,
            bronze_path=r.bronze_path,
            category=r.category,
            original_name=r.original_name,
            uploaded_by=current_user.username,
            record_count=r.record_count,
        )

    return {
        "uploaded_by": current_user.username,
        "files_ingested": len(results),
        "results": [
            {
                "original_name": r.original_name,
                "category": r.category,
                "bronze_table": r.bronze_table,
                "record_count": r.record_count,
                # Silver/Gold/vector-index promotion now runs in the
                # background after this response is sent — poll
                # GET /pipeline/status for per-stage progress instead of
                # expecting a pipeline_error here.
                "pipeline_status": "processing_in_background",
            }
            for r in results
        ],
    }
