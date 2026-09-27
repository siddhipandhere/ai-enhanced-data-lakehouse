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
from pipeline.orchestration import mark_running_stages_failed, process_dataset

router = APIRouter(prefix="/upload", tags=["upload"])


def _run_pipeline_safely(**kwargs) -> None:
    """Background wrapper so a stage failure can't crash silently."""
    try:
        process_dataset(**kwargs)
    except Exception as e:
        # Also flip whichever stage was "running" to "failed" -- otherwise
        # the Pipeline page shows it as Running forever.
        mark_running_stages_failed(kwargs["bronze_table"], str(e))


@router.post("/bulk")
async def upload_bulk(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Uploads and ingests multiple datasets of any supported type at once.
    The Bronze commit happens inline; Silver -> Gold -> vector-index
    promotion runs as a background task so /pipeline/status keeps
    responding while it works.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        used_names: set[str] = set()
        for i, upload in enumerate(files):
            # Keep only the base name: a client-supplied filename like
            # "../../main.py" would otherwise be written OUTSIDE tmp_dir.
            # Two files with the same name in one batch would also have
            # silently overwritten each other.
            name = Path((upload.filename or "").replace(
                "\\", "/")).name or f"upload_{i}"
            if name.lower() in used_names:
                stem, suffix = Path(name).stem, Path(name).suffix
                name = f"{stem}_{i}{suffix}"
            used_names.add(name.lower())
            dest = Path(tmp_dir) / name
            with open(dest, "wb") as f:
                while chunk := await upload.read(1024 * 1024):
                    f.write(chunk)
            saved_paths.append(dest)

        try:
            results = ingest_files_bulk(
                saved_paths, uploaded_by=current_user.username)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    for r in results:
        # Register every dataset NOW, before any background work starts.
        # The background jobs run one after another; if the server stopped
        # during an early one (e.g. the first image-model download filling
        # the disk), the later files used to have no registry entry at all
        # -- their Bronze data existed but they never appeared on the
        # Pipeline page. Registered as "pending", a restart marks them
        # failed with a Retry button instead (registry.recover_interrupted).
        registry.upsert_dataset(
            r.bronze_table, create=True,
            original_name=r.original_name, category=r.category,
            uploaded_by=current_user.username, record_count=r.record_count,
            bronze_status="ready", silver_status="pending",
            gold_status="pending", vector_status="pending", error=None,
        )
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
                "pipeline_status": "processing_in_background",
            }
            for r in results
        ],
    }
