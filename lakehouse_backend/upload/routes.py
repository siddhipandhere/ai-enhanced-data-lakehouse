"""
Bulk, multi-format dataset upload endpoint — behind login.

Accepts any mix of structured (.csv/.xlsx/.parquet), semi-structured
(.json/.xml), and unstructured (.png/.jpg/.pdf/.txt) files in a single
request, in one bulk call to ingestion.loaders.ingest_files_bulk().
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from auth.dependencies import get_current_user
from auth.models import User
from ingestion.loaders import ValidationError, ingest_files_bulk
from pipeline.orchestration import process_dataset

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/bulk")
async def upload_bulk(
    files: list[UploadFile],
    current_user: User = Depends(get_current_user),
):
    """
    Uploads and ingests multiple datasets of any supported type at once.
    Each file lands as its own Bronze Delta table, tagged with the
    authenticated user. The whole batch commits atomically — see
    ingestion.loaders.ingest_files_bulk for the staging/rollback logic.
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
            results = ingest_files_bulk(saved_paths, uploaded_by=current_user.username)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    pipeline_errors: dict[str, str] = {}
    for r in results:
        try:
            process_dataset(
                bronze_table=r.bronze_table,
                bronze_path=r.bronze_path,
                category=r.category,
                original_name=r.original_name,
                uploaded_by=current_user.username,
                record_count=r.record_count,
            )
        except Exception as e:
            # Bronze already committed successfully at this point; a
            # Silver/Gold/vector failure for one file shouldn't fail the
            # whole upload response or block the other files in the batch.
            pipeline_errors[r.bronze_table] = str(e)

    return {
        "uploaded_by": current_user.username,
        "files_ingested": len(results),
        "results": [
            {
                "original_name": r.original_name,
                "category": r.category,
                "bronze_table": r.bronze_table,
                "record_count": r.record_count,
                "pipeline_error": pipeline_errors.get(r.bronze_table),
            }
            for r in results
        ],
    }
