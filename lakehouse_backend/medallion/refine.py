"""
Silver refinement — structural cleaning that turns raw-looking data into
analysis-ready tables. Runs on the Silver data (after the Spark/Delta
cleaning in silver.py) right before the Gold table is built.

Each step detects its situation from the DATA itself, not from column or
file names, and is a no-op when it doesn't apply, so it's safe for any
dataset:

1. parse_raw_emails        a column of raw RFC-822 email text (headers +
                           body in one cell, as in the Kaggle Enron file)
                           is split into sender / recipients / date /
                           subject / body ... columns.
2. normalize_missing_text  "NaN", "N/A", "null", "-" ... stored as TEXT
                           are treated as missing, so a column that is
                           otherwise numeric becomes a real number column
                           (e.g. salary/bonus in the Enron financials).
3. drop_total_rows         spreadsheet summary rows ("TOTAL", "Grand
                           Total") that would otherwise be counted as a
                           real record and dominate every max/sum.
4. drop_duplicate_emails   the same email stored in several folders
                           (sent, all_documents, discussion_threads...)
                           is kept once.

refine() returns (table, notes); the notes are shown on the Pipeline
page so the cleaning that happened is visible, not silent.
"""

from __future__ import annotations

import email
import email.utils
import re

import pandas as pd

from utils.logger import get_logger

logger = get_logger("refine")

# ---------------------------------------------------------------- helpers
_MISSING_TOKENS = {"nan", "none", "null", "n/a", "na", "#n/a", "-", "--", "?", ""}
_TRUE_TOKENS = {"true", "yes", "y"}
_FALSE_TOKENS = {"false", "no", "n"}
_NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
_EMAIL_HEADER = re.compile(r"^(Message-ID|Date|From|To|Subject|Mime-Version|X-From):", re.MULTILINE)
_TOTAL_LABELS = {"total", "totals", "grand total", "sub total", "subtotal", "sum", "total:"}
_MIN_PLAUSIBLE_EMAIL_YEAR = 1990  # the Enron corpus uses 1980-01-01 as a "no date" placeholder
BODY_MAX_CHARS = 5000


def _is_text_column(s: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    first = s.dropna().head(1).tolist()
    return not (first and isinstance(first[0], (bytes, bytearray, memoryview)))


def _sample(s: pd.Series, n: int = 50) -> list:
    return [v for v in s.dropna().head(n).tolist() if isinstance(v, str)]


# ------------------------------------------------------ 1. raw email text
def find_raw_email_column(df: pd.DataFrame) -> str | None:
    """A column where most sampled values contain several RFC-822 header
    lines followed by a blank line (headers/body separator)."""
    for col in df.columns:
        if not _is_text_column(df[col]):
            continue
        sample = _sample(df[col], 40)
        if len(sample) < 3:
            continue
        hits = sum(1 for v in sample
                   if len(_EMAIL_HEADER.findall(v[:3000])) >= 3 and ("\n\n" in v or "\r\n\r\n" in v))
        if hits / len(sample) >= 0.6:
            return col
    return None


def _header(msg, name: str) -> str:
    return re.sub(r"\s+", " ", str(msg.get(name, "") or "")).strip()


def _parse_one(raw) -> dict:
    msg = email.message_from_string(raw if isinstance(raw, str) else "")
    if msg.is_multipart():
        parts = [p.get_payload() for p in msg.walk() if p.get_content_type() == "text/plain"]
        body = "\n".join(p for p in parts if isinstance(p, str))
    else:
        body = msg.get_payload()
        body = body if isinstance(body, str) else ""
    body = re.sub(r"\s+", " ", body).strip()[:BODY_MAX_CHARS]

    date, year, month = None, None, None
    try:
        dt = email.utils.parsedate_to_datetime(_header(msg, "Date"))
        if dt.year >= _MIN_PLAUSIBLE_EMAIL_YEAR:
            date, year, month = dt.strftime("%Y-%m-%d %H:%M:%S"), dt.year, dt.month
    except (TypeError, ValueError, IndexError):
        pass

    to, cc = _header(msg, "To"), _header(msg, "Cc")
    return {
        "message_id": _header(msg, "Message-ID"),
        "date": date, "year": year, "month": month,
        "sender": _header(msg, "From"),
        "sender_name": _header(msg, "X-From"),
        "recipients": to[:500],
        "recipient_count": len([a for a in to.split(",") if a.strip()]),
        "cc_count": len([a for a in cc.split(",") if a.strip()]),
        "subject": _header(msg, "Subject")[:300],
        "body": body,
        "body_length": len(body),
    }


def parse_raw_emails(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    col = find_raw_email_column(df)
    if col is None:
        return df
    parsed = pd.DataFrame([_parse_one(v) for v in df[col].tolist()], index=df.index)
    parsed["year"] = parsed["year"].astype("Int64")
    parsed["month"] = parsed["month"].astype("Int64")
    out = df.drop(columns=[col])

    # Kaggle-style "file" column: "allen-p/_sent_mail/12." -> mailbox + folder
    path_col = next((c for c in out.columns if _is_text_column(out[c]) and
                     sum(1 for v in _sample(out[c]) if re.match(r"^[^/\s]+/.+/[^/]*$", v)) >= 0.8 * max(1, len(_sample(out[c])))), None)
    if path_col is not None:
        parts = out[path_col].fillna("").str.split("/")
        out.insert(0, "mailbox", parts.str[0])
        out.insert(1, "folder", parts.str[1:-1].str.join("/"))
        out.insert(2, "is_sent", out["folder"].str.lower().str.contains("sent"))
        out = out.rename(columns={path_col: "source_path"})

    out = pd.concat([out, parsed], axis=1)
    blank_dates = int(parsed["date"].isna().sum())
    notes.append(f"Parsed raw email text in '{col}' into {parsed.shape[1]} columns "
                 f"(sender, recipients, date, subject, body, ...) for {len(out):,} emails"
                 + (f"; {blank_dates} had no valid date" if blank_dates else ""))
    out.attrs["parsed_emails"] = True
    return out


# ------------------------------------------------- 2. missing-value text
def normalize_missing_text(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    numeric_cols, bool_cols, token_hits = [], [], 0
    for col in df.columns:
        s = df[col]
        if not _is_text_column(s):
            continue
        as_text = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        lowered = as_text.map(lambda v: v.lower() if isinstance(v, str) else v)
        missing = lowered.isin(_MISSING_TOKENS) | as_text.isna()
        present = as_text[~missing]
        if present.empty:
            continue
        # Only touch a column that is clearly numeric/boolean once the
        # missing-value words are ignored -- never a real text column
        # (a "region" column containing "NA" for North America stays as is).
        cleaned_numbers = present.map(lambda v: str(v).replace(",", "") if isinstance(v, str) else v)
        is_num = cleaned_numbers.map(lambda v: isinstance(v, (int, float)) or bool(_NUMBER.match(str(v))))
        if is_num.mean() >= 0.9:
            token_hits += int((lowered.isin(_MISSING_TOKENS - {""})).sum())
            df[col] = pd.to_numeric(cleaned_numbers.reindex(df.index), errors="coerce")
            numeric_cols.append(col)
            continue
        low_present = lowered[~missing]
        if low_present.isin(_TRUE_TOKENS | _FALSE_TOKENS).all():
            df[col] = lowered.map(lambda v: True if v in _TRUE_TOKENS else False if v in _FALSE_TOKENS else None)
            bool_cols.append(col)
    if numeric_cols:
        shown = ", ".join(numeric_cols[:8]) + (" ..." if len(numeric_cols) > 8 else "")
        tokens = (f"; {token_hits:,} text values like 'NaN'/'N/A'/'null' treated as missing"
                  if token_hits else "")
        notes.append(f"Converted {len(numeric_cols)} text column(s) to numbers ({shown}){tokens}")
    if bool_cols:
        notes.append(f"Converted yes/no text to true/false in: {', '.join(bool_cols)}")
    return df


# ---------------------------------------------------- 3. total/summary rows
def drop_total_rows(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Drops rows whose label cell says TOTAL / Grand Total / Sum. Only the
    first two text columns are checked (where a row label lives), and only
    when the row also carries numbers -- a person or product literally
    named "Total" in a free-text column is not affected."""
    text_cols = [c for c in df.columns if _is_text_column(df[c])][:2]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not text_cols or not num_cols or len(df) < 3:
        return df
    label = pd.Series(False, index=df.index)
    for c in text_cols:
        label |= df[c].map(lambda v: isinstance(v, str) and v.strip().lower() in _TOTAL_LABELS)
    has_numbers = df[num_cols].notna().any(axis=1)
    mask = label & has_numbers
    if mask.any():
        notes.append(f"Removed {int(mask.sum())} spreadsheet summary row(s) labelled TOTAL/Sum")
        df = df[~mask]
    return df


# ------------------------------------------------ 4. duplicate emails
def drop_duplicate_emails(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if not df.attrs.get("parsed_emails"):
        return df
    key_cols = [c for c in ("mailbox", "sender", "date", "subject") if c in df.columns]
    body_key = df["body"].fillna("").str[:500] if "body" in df.columns else None
    # fillna before astype(str): under pandas 3 astype(str) keeps NaN as a float
    key = df[key_cols].astype(object).fillna("").astype(str).agg("|".join, axis=1)
    if body_key is not None:
        key = key + "|" + body_key
    dup = key.duplicated(keep="first")
    if dup.any():
        notes.append(f"Removed {int(dup.sum()):,} duplicate copies of the same email "
                     f"(stored in several folders); {int((~dup).sum()):,} unique emails remain")
        df = df[~dup]
    return df


# ------------------------------------------------ 5. PDF pages / images
def flag_scanned_pdf_pages(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """PDF uploads become one row per page (page_number, text, source_file).
    A scanned PDF has no text layer, so its pages come through EMPTY --
    silently, until now. Flag them so it's clear OCR would be needed."""
    if not {"page_number", "text"} <= set(df.columns):
        return df
    empty = df["text"].fillna("").astype(str).str.strip().str.len() < 30
    df["has_text"] = ~empty
    if empty.all():
        notes.append(f"All {len(df)} page(s) have no extractable text: this looks like a SCANNED PDF "
                     f"(images of pages). OCR would be needed to search or report on it")
    elif empty.any():
        notes.append(f"Extracted text from {int((~empty).sum())} of {len(df)} pages; "
                     f"{int(empty.sum())} page(s) have no text (scanned or blank)")
    else:
        notes.append(f"Extracted text from all {len(df)} pages")
    return df


def describe_images(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Image uploads arrive as raw bytes (Spark's binaryFile source). Read
    the real image properties so the table has something to query."""
    if "content" not in df.columns:
        return df
    try:
        from io import BytesIO
        from PIL import Image
    except ImportError:
        return df
    meta = []
    for blob in df["content"].tolist():
        info = {"width_px": None, "height_px": None, "image_format": None, "color_mode": None}
        if isinstance(blob, (bytes, bytearray, memoryview)):
            try:
                with Image.open(BytesIO(bytes(blob))) as im:
                    info = {"width_px": im.width, "height_px": im.height,
                            "image_format": im.format, "color_mode": im.mode}
            except Exception:
                pass
        meta.append(info)
    if not any(m["width_px"] for m in meta):
        return df
    if "original_name" in df.columns:
        # Spark reports file paths as URIs, so "my photo.jpg" can arrive as "my%20photo.jpg".
        from urllib.parse import unquote
        df["original_name"] = df["original_name"].map(lambda v: unquote(v) if isinstance(v, str) else v)
    m = pd.DataFrame(meta, index=df.index)
    for c in m.columns:
        df[c] = m[c]
    df["megapixels"] = (df["width_px"] * df["height_px"] / 1e6).round(2)
    if "length" in df.columns:
        df["size_kb"] = (pd.to_numeric(df["length"], errors="coerce") / 1024).round(1)
    readable = [m for m in meta if m["width_px"]]
    formats = pd.Series([m["image_format"] for m in readable]).value_counts()
    fmt_text = ", ".join(f"{n} {f}" for f, n in formats.items())
    bad = len(meta) - len(readable)
    notes.append(f"Read image metadata for {len(readable)} image(s) ({fmt_text}): width, height, format, "
                 f"colour mode and size are now columns you can query"
                 + (f"; {bad} file(s) could not be read as images" if bad else ""))
    return df


# ------------------------------------------------------------ entry point
def refine(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    before = len(df)
    df = df.copy()
    df = parse_raw_emails(df, notes)
    parsed = df.attrs.get("parsed_emails", False)
    df = normalize_missing_text(df, notes)
    df = drop_total_rows(df, notes)
    df.attrs["parsed_emails"] = parsed
    df = drop_duplicate_emails(df, notes)
    df = flag_scanned_pdf_pages(df, notes)
    df = describe_images(df, notes)
    df = df.reset_index(drop=True)
    if notes:
        logger.info(f"Refined {before:,} -> {len(df):,} rows: " + " | ".join(notes))
    return df, notes
