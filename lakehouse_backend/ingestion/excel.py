"""
Excel ingestion: every useful sheet of a workbook, with its real header row.

Real-world spreadsheets (the Enron workbooks are a good example) rarely
start with a clean header in row 1. They have a title ("ENRON NORTH
AMERICA - Q2 DEALS"), a date line, blank rows, and only then the
column headers. The previous reader used pandas' default header=0,
so the title became the header, every other column got named
"Unnamed: N", and those columns were then DROPPED as stray index
columns -- silently losing most of the data. It also read only the
first sheet of the workbook.

This module reads each sheet with no header, finds the first row that
looks like a header (mostly filled, mostly text), and uses the rows
below it as data. Each useful sheet is returned separately so the
ingestion layer can land it as its own Bronze table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MIN_DATA_ROWS = 2
MIN_COLUMNS = 2
HEADER_SEARCH_ROWS = 15


class ExcelReadError(Exception):
    pass


def _clean_cell(v):
    if isinstance(v, str):
        v = v.replace("\xa0", " ").strip()
        return v or None
    return v


def _column_name(value, position: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"column_{position + 1}"
    if hasattr(value, "strftime"):  # a date used as a header, e.g. a month column
        value = value.strftime("%Y_%m_%d")
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    if not name:
        return f"column_{position + 1}"
    if name[0].isdigit():
        name = f"c_{name}"  # e.g. "2001" -> "c_2001": safe in query expressions
    return name[:60]


def detect_header_row(raw: pd.DataFrame) -> int | None:
    """Index of the first row (within the first 15) that is at least 50%
    filled and whose filled cells are at least 60% text."""
    for i in range(min(HEADER_SEARCH_ROWS, len(raw))):
        row = raw.iloc[i]
        filled = row.notna()
        if filled.mean() < 0.5 or filled.sum() < MIN_COLUMNS:
            continue
        texty = row[filled].map(lambda v: isinstance(v, str)).mean()
        if texty >= 0.6:
            return i
    return None


def clean_sheet(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Turns a header-less sheet grid into a proper table, or None if the
    sheet has no recognisable table (notes, a chart, a single cell...)."""
    raw = raw.map(_clean_cell)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if len(raw) < MIN_DATA_ROWS + 1 or raw.shape[1] < MIN_COLUMNS:
        return None

    header_idx = detect_header_row(raw)
    if header_idx is None:
        return None

    names, seen = [], {}
    for pos, value in enumerate(raw.iloc[header_idx].tolist()):
        name = _column_name(value, pos)
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    body = raw.iloc[header_idx + 1:].copy()
    body.columns = names
    body = body.dropna(how="all").dropna(axis=1, how="all")
    if len(body) < MIN_DATA_ROWS or body.shape[1] < MIN_COLUMNS:
        return None
    return body.reset_index(drop=True)


def read_workbook(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Returns [(sheet_name, table), ...] for every sheet that contains a
    usable table. .xls needs the xlrd package, .xlsx uses openpyxl."""
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    except ImportError as e:
        raise ExcelReadError(
            "Reading old .xls files needs the xlrd package: run  pip install xlrd  "
            f"in the backend's venv ({e})")
    except Exception as e:
        raise ExcelReadError(f"Could not read workbook: {e}")

    tables = []
    for name, raw in sheets.items():
        table = clean_sheet(raw)
        if table is not None:
            tables.append((str(name), table))
    return tables
