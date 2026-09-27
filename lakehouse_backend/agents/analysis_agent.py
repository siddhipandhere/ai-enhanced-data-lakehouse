"""
Analysis agent — computes summary statistics on the Query agent's output.
No LLM calls; pure computation so it stays fast and deterministic.
"""

import math

import numpy as np
import pandas as pd

import config
from utils.metrics import QueryTimer
from utils.logger import get_logger
from utils.media import is_bytes_column

logger = get_logger("analysis_agent")

_MAX_CATEGORICAL_COLUMNS = 5     # how many text columns get a "top values" breakdown
_TOP_VALUES = 5                  # values listed per column
_MAX_CARDINALITY = 2000          # above this a column is an identifier/free text, not a category
_MAX_AVG_LEN = 60                # long strings = descriptions, not categories
_SAMPLE_ROWS = 5
_SAMPLE_CELL_LEN = 80
_PREFERRED_SAMPLE_COLUMNS = ("original_name", "title", "name", "product_name", "brand", "category",
                             "sub_category", "selling_price", "price", "actual_price",
                             "discount_pct", "average_rating")


def json_safe(value):
    """Recursively converts NaN/inf to None and numpy scalars to Python.

    describe() on a 0- or 1-row result produces NaN (e.g. std of one
    value). Starlette serializes responses with allow_nan=False, so any
    NaN left in the analysis made /reports/generate return a 500 for
    every question whose filter matched 0 or 1 rows -- and a saved
    report containing NaN would then break GET /reports too.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(200)
    if sample.empty or not all(isinstance(v, str) for v in sample):
        return False
    if sample.map(len).mean() > _MAX_AVG_LEN:
        return False
    if sample.str.strip().str[:1].isin(["[", "{"]).mean() > 0.5:
        return False  # serialized JSON / lists
    nunique = series.nunique(dropna=True)
    return 1 < nunique <= _MAX_CARDINALITY


def _categorical_summary(results: pd.DataFrame, focus: list[str]) -> dict:
    """Top values for category-like text columns (brand, category, seller...)
    -- the material a useful summary needs ("mostly Clothing, led by brand
    X"), which describe() on numeric columns alone never provided."""
    ordered = [c for c in focus if c in results.columns] + \
              [c for c in results.columns if c not in focus]
    out = {}
    for c in ordered:
        if c == config.JOIN_KEY or len(out) >= _MAX_CATEGORICAL_COLUMNS:
            continue
        if not _is_categorical(results[c]):
            continue
        counts = results[c].value_counts(dropna=True).head(_TOP_VALUES)
        out[c] = {
            "distinct_values": int(results[c].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }
    return out


def _sample_rows(results: pd.DataFrame, focus: list[str]) -> list[dict]:
    cols = [c for c in focus if c in results.columns]
    cols += [c for c in _PREFERRED_SAMPLE_COLUMNS if c in results.columns and c not in cols]
    if "similarity_score" in results.columns and "similarity_score" not in cols:
        cols.insert(0, "similarity_score")
    cols = cols[:7] or [c for c in results.columns if c != config.JOIN_KEY][:5]
    # Raw image bytes mean nothing to the report writer (and can't be saved
    # as JSON); the file name next to them says which picture it is.
    cols = [c for c in cols if not is_bytes_column(results[c])]
    if not cols:
        cols = [c for c in results.columns if c != config.JOIN_KEY and not is_bytes_column(results[c])][:5]
    sample = results[cols].head(_SAMPLE_ROWS).copy()
    for c in sample.columns:
        if not pd.api.types.is_numeric_dtype(sample[c]):
            sample[c] = sample[c].map(lambda v: v[:_SAMPLE_CELL_LEN] if isinstance(v, str) else v)
    return sample.to_dict(orient="records")


_PASSAGES = 5
_PASSAGE_LEN = 700
_SOURCE_COLUMNS = ("source_file", "original_name", "subject", "sender", "mailbox", "name", "title")


def _matching_passages(results: pd.DataFrame, text_column: str | None) -> list[dict]:
    """For a semantic search over documents (PDF pages, email bodies), the
    top matching passages themselves -- long enough for the Report agent to
    answer FROM the documents. Before, it only saw 80-character snippets,
    so an answer like "found guilty of fraud and conspiracy" could only
    come from the model's general knowledge, not from the uploaded file."""
    if (not text_column or text_column not in results.columns
            or "similarity_score" not in results.columns or is_bytes_column(results[text_column])):
        return []
    out = []
    for _, row in results.head(_PASSAGES).iterrows():
        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        item = {"similarity_score": row["similarity_score"]}
        if "page_number" in results.columns:
            item["page_number"] = row["page_number"]
        for c in _SOURCE_COLUMNS:
            if c in results.columns and isinstance(row[c], str) and row[c]:
                item["source"] = row[c]
                break
        item["text"] = " ".join(text.split())[:_PASSAGE_LEN]
        out.append(item)
    return out


def run_analysis(results: pd.DataFrame, total_record_count: int | None = None,
                 filter_applied: bool = False, warnings: list[str] | None = None,
                 queried_columns: list[str] | None = None, matched_rows: int | None = None,
                 aggregated: bool = False, passage_column: str | None = None) -> dict:
    """
    total_record_count/filter_applied let the summary state a real,
    computed "N of M records matched" figure instead of the Report agent
    estimating one. categorical_summary / sample_rows give it something
    concrete to say beyond the count, which is why summaries used to be
    only a sentence or two long.
    """
    queried_columns = queried_columns or []
    if aggregated:
        return _run_aggregated_analysis(results, total_record_count, filter_applied,
                                        warnings, matched_rows)
    with QueryTimer() as timer:
        numeric_df = results.select_dtypes(include="number").drop(
            columns=[config.JOIN_KEY], errors="ignore")
        summary = {
            "record_count": len(results),
            "total_record_count": total_record_count if total_record_count is not None else len(results),
            "filter_applied": filter_applied,
            "columns": list(results.columns),
            "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty and len(results) else {},
            "warnings": warnings or [],
        }
        if filter_applied and summary["total_record_count"]:
            summary["match_rate_pct"] = round(
                100 * summary["record_count"] / summary["total_record_count"], 2)

        if queried_columns:
            summary["queried_columns"] = queried_columns
            summary["queried_column_stats"] = {
                c: results[c].describe().to_dict()
                for c in queried_columns
                if c in numeric_df.columns and len(results)
            }

        if len(results):
            summary["categorical_summary"] = _categorical_summary(results, queried_columns)
            summary["sample_rows"] = _sample_rows(results, queried_columns)
            passages = _matching_passages(results, passage_column)
            if passages:
                summary["matching_passages"] = passages
                # Score statistics say nothing about the question being asked.
                summary["numeric_summary"] = {k: v for k, v in summary["numeric_summary"].items()
                                              if k not in ("similarity_score", "page_number")}

    summary["query_execution_time_ms"] = timer.elapsed_ms
    logger.info(f"Analysis complete: {summary['record_count']} records, "
                f"{summary['query_execution_time_ms']} ms")
    return json_safe(summary)


def _run_aggregated_analysis(results: pd.DataFrame, total_record_count: int | None,
                             filter_applied: bool, warnings: list[str] | None,
                             matched_rows: int | None) -> dict:
    """
    For grouped results ("count of emails per mailbox"), each result row
    is a GROUP, not a record. The old code treated it like row-level data:
    it reported "2 of 5,232 records matched" (2 = number of groups) and
    counted each group label once ("allen-p (1), lay-k (1)"), so the
    summary said the mailboxes were tied when the real counts were 900
    vs 275. Here the grouped table itself is handed to the Report agent,
    and the match figures use the rows that passed the filter BEFORE
    grouping.
    """
    with QueryTimer() as timer:
        matched = matched_rows if matched_rows is not None else len(results)
        total = total_record_count if total_record_count is not None else matched
        summary = {
            "aggregated": True,
            "record_count": matched,
            "total_record_count": total,
            "filter_applied": filter_applied,
            "group_count": len(results),
            "columns": list(results.columns),
            "aggregated_result": results.head(25).to_dict(orient="records"),
            "warnings": warnings or [],
        }
        if filter_applied and total:
            summary["match_rate_pct"] = round(100 * matched / total, 2)
    summary["query_execution_time_ms"] = timer.elapsed_ms
    logger.info(f"Analysis complete (aggregated): {len(results)} groups from {matched} records")
    return json_safe(summary)
