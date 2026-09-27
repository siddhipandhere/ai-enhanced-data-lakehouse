"""
Dataset + pipeline-stage registry.

Every other piece of this backend (Spark/Delta bronze, Silver Delta
tables, Gold parquet, the FAISS index) already persists its own data to
disk — but nothing tracked, in one place, *which datasets exist* and
*how far each one got through Bronze -> Silver -> Gold -> Vector index*.
That's what the frontend's Datasets/Pipeline/Reports/Search pages need
to render anything.

This is intentionally a single small JSON file rather than a new SQL
table: it's read-modify-write, single-uvicorn-process, local-demo scale
(dozens of datasets, not millions), so a file with a lock is simpler
than standing up another table + migration for the same information.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from utils.logger import get_logger

logger = get_logger("registry")

REGISTRY_PATH = config.DATA_ROOT / "_registry.json"
BACKUP_PATH = config.DATA_ROOT / "_registry.json.bak"
# Re-entrant: readers take the lock too now, and some writers call readers.
_lock = threading.RLock()

STAGES = ("bronze_status", "silver_status", "gold_status", "vector_status")

_EMPTY = {"datasets": {}, "reports": [], "query_counts": {}}


def _load(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    data.setdefault("datasets", {})
    data.setdefault("reports", [])
    data.setdefault("query_counts", {})
    return data


def _read() -> dict:
    """Reads the registry. On a corrupt file, falls back to the last good
    backup instead of returning an EMPTY registry -- the old behaviour,
    where the very next write then permanently wiped every dataset entry."""
    if not REGISTRY_PATH.exists():
        return {"datasets": {}, "reports": [], "query_counts": {}}
    try:
        return _load(REGISTRY_PATH)
    except (json.JSONDecodeError, OSError) as e:
        if BACKUP_PATH.exists():
            try:
                logger.error(f"Registry unreadable ({e}); using backup {BACKUP_PATH.name}")
                return _load(BACKUP_PATH)
            except (json.JSONDecodeError, OSError):
                pass
        raise RuntimeError(f"Registry file {REGISTRY_PATH} is unreadable and no valid backup exists: {e}")


def _write(data: dict) -> None:
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=str, indent=2)
    if REGISTRY_PATH.exists():
        try:
            shutil.copyfile(REGISTRY_PATH, BACKUP_PATH)
        except OSError:
            pass
    # On Windows os.replace raises PermissionError if another thread or an
    # antivirus/indexer has the file open at that instant. Unhandled, that
    # exception escaped process_dataset() between stages and left the
    # dataset showing "Running" forever. Retry briefly instead.
    for attempt in range(20):
        try:
            os.replace(tmp, REGISTRY_PATH)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def upsert_dataset(table_name: str, create: bool = False, **fields: Any) -> dict | None:
    """Updates a dataset's registry entry (only the keys in ``fields``).
    Creates it only when ``create=True`` (the first write of a new upload).

    Without that rule, a background job that finished AFTER its dataset
    was deleted (e.g. a vector-index rebuild) re-created an empty ghost
    entry that then showed up as "failed" with no name."""
    with _lock:
        data = _read()
        if table_name not in data["datasets"] and not create:
            logger.info(f"Ignoring update for deleted/unknown dataset '{table_name}': {list(fields)}")
            return None
        entry = data["datasets"].setdefault(table_name, {
            "table_name": table_name,
            "created_at": datetime.now().isoformat(),
            "bronze_status": "pending",
            "silver_status": "pending",
            "gold_status": "pending",
            "vector_status": "pending",
            "error": None,
        })
        entry.update(fields)
        entry["updated_at"] = datetime.now().isoformat()
        data["datasets"][table_name] = entry
        _write(data)
        return entry


def get_dataset(table_name: str) -> dict | None:
    with _lock:
        return _read()["datasets"].get(table_name)


def list_datasets(uploaded_by: str | None = None) -> list[dict]:
    with _lock:
        entries = list(_read()["datasets"].values())
    if uploaded_by:
        entries = [e for e in entries if e.get("uploaded_by") == uploaded_by]
    return sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)


def add_report(entry: dict) -> dict:
    with _lock:
        data = _read()
        entry = {**entry, "id": entry.get("id") or f"r_{len(data['reports']) + 1}_{datetime.now().timestamp():.0f}",
                 "created_at": datetime.now().isoformat()}
        data["reports"].insert(0, entry)
        data["reports"] = data["reports"][:100]  # keep the file bounded
        _write(data)
        return entry


def list_reports(uploaded_by: str | None = None, limit: int = 50) -> list[dict]:
    with _lock:
        reports = _read()["reports"]
    if uploaded_by:
        reports = [r for r in reports if r.get("uploaded_by") == uploaded_by]
    return reports[:limit]


def record_query(uploaded_by: str) -> int:
    """
    Increments and persists this user's "AI queries" counter. Called
    once per Report / SQL Explorer / Semantic Search request that
    actually executes, so the Overview page's "AI queries" KPI reflects
    real usage instead of the hardcoded '0' it shipped with.
    """
    with _lock:
        data = _read()
        data["query_counts"][uploaded_by] = data["query_counts"].get(
            uploaded_by, 0) + 1
        _write(data)
        return data["query_counts"][uploaded_by]


def get_query_count(uploaded_by: str) -> int:
    with _lock:
        return _read()["query_counts"].get(uploaded_by, 0)


def recover_interrupted() -> list[str]:
    """
    Call once at server startup. Background pipeline jobs don't survive a
    restart (uvicorn --reload restarts on every code save), so any stage
    still marked "running"/"pending" at startup can never finish -- and
    the Pipeline page showed it as "Running" forever. Marks the first
    unfinished stage "failed" with an explanatory error and the later
    ones "skipped", so the UI tells the truth and the Retry button can
    re-run it.
    """
    fixed = []
    with _lock:
        data = _read()
        for name, entry in data["datasets"].items():
            if not any(entry.get(st) in ("running", "pending") for st in STAGES):
                continue
            first_bad = next((st for st in STAGES if entry.get(st) not in ("ready", "skipped")), None)
            if first_bad is None or entry.get(first_bad) not in ("running", "pending"):
                continue  # already failed earlier; nothing is actually stuck
            entry[first_bad] = "failed"
            if first_bad == "bronze_status":
                entry["error"] = ("Upload was interrupted before the file was saved. "
                                  "Delete this entry and upload the file again.")
            else:
                entry["error"] = (f"Interrupted: the server stopped during the "
                                  f"{first_bad.replace('_status', '')} stage. Click Retry.")
            for later in STAGES[STAGES.index(first_bad) + 1:]:
                if entry.get(later) != "skipped":
                    entry[later] = "pending"  # e.g. a stale 'ready' vector on an orphan entry
            entry["updated_at"] = datetime.now().isoformat()
            fixed.append(name)
        if fixed:
            _write(data)
            logger.warning(f"Marked interrupted pipeline runs as failed: {fixed}")
    return fixed


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def storage_used_bytes(uploaded_by: str) -> int:
    """
    Real on-disk storage for this user's datasets — Bronze + Silver
    Delta directories, the Gold parquet file, and each dataset's own
    FAISS index folder (plus their combined-view index) — instead of the
    old placeholder ``datasets.length * 3.2 MB`` estimate that had no
    relationship to actual data size.
    """
    entries = list_datasets(uploaded_by=uploaded_by)
    total = 0
    for e in entries:
        table_name = e.get("table_name")
        if not table_name:
            continue
        total += _dir_size_bytes(config.BRONZE_DIR / table_name)
        total += _dir_size_bytes(config.SILVER_DIR / table_name)
        total += _dir_size_bytes(config.GOLD_DIR / f"{table_name}.parquet")

    from embeddings.vector_store import index_size_bytes  # lazy: avoids importing faiss at registry import
    for e in entries:
        if e.get("table_name"):
            total += index_size_bytes(e["table_name"])
    total += index_size_bytes(f"__all__{uploaded_by}")

    return total


def delete_dataset(table_name: str) -> bool:
    """Removes a dataset's registry entry and every file it produced
    across Bronze/Silver/Gold, plus its cached query results. Returns
    False if the dataset wasn't in the registry to begin with (caller
    decides whether that's a 404).

    Also removes the dataset's own vector-index folder, and the owner's
    combined-view index (its contents depended on this dataset).
    """
    with _lock:
        data = _read()
        if table_name not in data["datasets"]:
            return False
        entry_owner = data["datasets"][table_name].get("uploaded_by")
        del data["datasets"][table_name]
        data["reports"] = [r for r in data["reports"]
                           if r.get("table_name") != table_name]
        _write(data)

    for base_dir, suffix in (
        (config.BRONZE_DIR, ""),
        (config.SILVER_DIR, ""),
        (config.GOLD_DIR, ".parquet"),
    ):
        path = base_dir / f"{table_name}{suffix}"
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    from embeddings.vector_store import delete_index
    delete_index(table_name)
    # The combined view's contents change when a member dataset is deleted.
    if entry_owner:
        delete_index(f"__all__{entry_owner}")

    return True
