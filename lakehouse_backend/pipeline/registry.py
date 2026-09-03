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
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import config

REGISTRY_PATH = config.DATA_ROOT / "_registry.json"
_lock = threading.Lock()

_EMPTY = {"datasets": {}, "reports": []}


def _read() -> dict:
    if not REGISTRY_PATH.exists():
        return {"datasets": {}, "reports": []}
    try:
        with open(REGISTRY_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"datasets": {}, "reports": []}
    data.setdefault("datasets", {})
    data.setdefault("reports", [])
    return data


def _write(data: dict) -> None:
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=str, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def upsert_dataset(table_name: str, **fields: Any) -> dict:
    """Creates or updates a dataset's registry entry. Only the keys
    passed in ``fields`` are touched; everything else already stored is
    left as-is."""
    with _lock:
        data = _read()
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
    return _read()["datasets"].get(table_name)


def list_datasets(uploaded_by: str | None = None) -> list[dict]:
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
    reports = _read()["reports"]
    if uploaded_by:
        reports = [r for r in reports if r.get("uploaded_by") == uploaded_by]
    return reports[:limit]


def delete_dataset(table_name: str) -> bool:
    """Removes a dataset's registry entry and every file it produced
    across Bronze/Silver/Gold, plus its cached query results. Returns
    False if the dataset wasn't in the registry to begin with (caller
    decides whether that's a 404).

    The FAISS vector index is shared, single-slot storage (see
    embeddings/vector_store.py's META_PATH/INDEX_PATH -- there's one of
    each, not one per dataset), so deleting a dataset that currently
    owns that index leaves it pointing at data that no longer exists.
    Rather than leave a dangling index, this clears it; the query/search
    agents already rebuild it on demand (build_index_if_needed) the
    next time it's actually needed, for whichever dataset is current.
    """
    import json
    import shutil

    import config

    with _lock:
        data = _read()
        if table_name not in data["datasets"]:
            return False
        del data["datasets"][table_name]
        data["reports"] = [r for r in data["reports"] if r.get("table_name") != table_name]
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

    meta_path = config.VECTOR_STORE_DIR / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("table_name") == table_name:
                for name in ("index.faiss", "id_map.csv", "meta.json"):
                    (config.VECTOR_STORE_DIR / name).unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            pass

    return True