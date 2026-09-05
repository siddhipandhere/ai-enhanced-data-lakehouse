"""
Query planner — the piece that makes the pipeline dataset-agnostic.

Instead of guessing which column the user meant from keyword matches
(brittle, breaks on unfamiliar column names or phrasing), this module
shows the LLM the *actual* schema of whatever dataset is loaded —
column names, dtypes, and a few sample rows — and asks it to return a
structured plan: which columns to filter/group/aggregate on, or which
column to run semantic search over. The plan is validated against the
real schema before anything executes, so a hallucinated column name
fails safely instead of crashing pandas.

This is the one LLM call the Orchestrator spends on understanding the
request; the Report agent spends the second one summarizing the result.
"""

import json
import re

import pandas as pd
from groq import Groq

import config
from utils.logger import get_logger

logger = get_logger("query_planner")

_client = None

_BANNED_SUBSTRINGS = ("import", "__", "exec", "eval",
                      "lambda", "os.", "sys.", "open(", "subprocess")

# Matches "<column> <comparison operator> <number>", e.g. "discount > 50".
_NUMERIC_COMPARISON = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)")

_PERCENT_IN_QUESTION = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# Inclusive phrases ("at least", "at most") get their own >= / <= operator
# instead of being lumped in with the strict words -- "at least 50%"
# should include exactly 50%, not exclude it the way "more than 50%" does.
_LTE_WORDS = ("at most",)
_GTE_WORDS = ("at least",)
_LT_WORDS = ("less than", "under", "below", "fewer than")
_GT_WORDS = ("more than", "greater than", "over", "above", "exceeding")

PLAN_PROMPT = """You are a query planner for a data analytics system. Given a user's
question and the schema of the currently loaded dataset, produce a JSON plan.

Dataset schema:
{schema}

Sample rows:
{sample}

User's question: {question}

Return ONLY a JSON object with this exact shape, no other text:
{{
  "mode": "sql" | "semantic" | "both",
  "filter_query": "<pandas query() expression using only the column names above, or null>",
  "group_by": ["<column name>"] | null,
  "agg_column": "<numeric column name to aggregate, or null>",
  "agg_func": "sum" | "mean" | "count" | "max" | "min" | null,
  "semantic_text_column": "<column name to run semantic search over, or null>",
  "top_k": 10
}}

Rules:
- Use "sql" when the question is answerable by filtering/grouping the structured columns.
- Use "semantic" when the question asks for things "similar to" or "about" some free-text meaning that has no numeric equivalent (e.g. "cheap and stylish", "comfortable cotton wear").
- Use "both" when it needs a structured filter narrowed further by meaning.
- CRITICAL: for any question containing a number, percentage, or comparison word ("more than", "less than", "over", "under", "at least", "greater than", "%"), always prefer "sql" mode with a numeric filter_query over "semantic" mode -- even if a text column's name or values superficially relate to the question (e.g. a "discount" text column holding "69% off" cannot be numerically compared; if a numeric column for the same concept exists, such as "discount_pct", use that in filter_query instead). Semantic search ranks by meaning-similarity to the query text, not by numeric comparison, and can never correctly answer a threshold question -- using it for one produces results that look plausible but are not actually filtered by the number requested.
- Only reference column names that appear in the schema above, exactly as spelled.
- filter_query must be a valid pandas DataFrame.query() expression, or null if no filter applies.
- If nothing in the schema clearly answers the question, set mode to "sql" and leave
  filter_query, group_by, and agg_column all null so the full table is returned.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _describe_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text"
        lines.append(f"- {col} ({dtype})")
    return "\n".join(lines)


def _sample_rows(df: pd.DataFrame, n: int = 3) -> str:
    if df.empty:
        return "(no rows available)"
    return df.head(n).to_string(index=False)


def _sanitize_filter_query(filter_query: str | None, valid_columns: list[str]) -> str | None:
    """
    Defense in depth against a malicious or hallucinated filter_query,
    since pandas.query() ultimately evaluates the expression. Rejects
    anything containing code-execution primitives, and anything that
    references a column not actually in the dataset.
    """
    if not filter_query:
        return None

    lowered = filter_query.lower()
    if any(bad in lowered for bad in _BANNED_SUBSTRINGS):
        logger.warning(
            f"Rejected filter_query containing banned pattern: {filter_query}")
        return None

    # Every bare identifier in the expression must be either a known
    # column or a Python/pandas keyword/operator — not an arbitrary name.
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", filter_query))
    allowed = set(valid_columns) | {"and", "or",
                                    "not", "in", "True", "False", "None"}
    unknown = identifiers - allowed
    if unknown:
        logger.warning(
            f"Rejected filter_query referencing unknown identifiers {unknown}: {filter_query}")
        return None

    return filter_query


def _fix_numeric_comparison_on_text_column(filter_query: str | None, df: pd.DataFrame) -> str | None:
    """
    Safety net for exactly the "discount > 50" failure mode: silver.py's
    _extract_percentage_columns() derives a numeric "<col>_pct" sibling
    whenever a text column looks like "69% off", specifically so numeric
    filters have something valid to compare against. The planner prompt
    tells the LLM to use that sibling column, but if it ever ignores the
    instruction and emits a comparison directly against the original
    text column instead, pandas.query() raises a TypeError comparing
    str > int -- which query_agent.execute_plan() catches and silently
    falls back to the FULL unfiltered dataset, with no filtering having
    actually happened at all (the exact bug behind a "which products
    have more than 50% off" report that comes back showing products
    with less than 50% off: the filter never ran, so the answer is just
    the raw table).

    Rewriting the comparison onto the numeric sibling column here — when
    one exists — fixes the filter before it ever gets a chance to fail
    silently downstream.
    """
    if not filter_query:
        return filter_query

    def _rewrite(match: re.Match) -> str:
        col, op, num = match.group(1), match.group(2), match.group(3)
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            pct_col = f"{col}_pct"
            if pct_col in df.columns and pd.api.types.is_numeric_dtype(df[pct_col]):
                logger.info(f"Rewriting filter comparison on non-numeric column '{col}' "
                            f"to its derived numeric column '{pct_col}'")
                return f"{pct_col} {op} {num}"
        return match.group(0)

    return _NUMERIC_COMPARISON.sub(_rewrite, filter_query)


def _fallback_percent_filter(question: str, df: pd.DataFrame) -> str | None:
    """
    Deterministic safety net for when the LLM returns NO filter_query at
    all for an obviously numeric-threshold question, e.g. "which
    products have more than 50% discount" planned as
    {"filter_query": null, ...}. Observed in practice: when a Gold table
    has more than one "_pct"-suffixed numeric column (a real
    "discount_pct" alongside an unrelated one derived from some other
    text column), the LLM can hedge rather than commit to either one,
    and the planner's own "if nothing clearly answers this, return the
    full table" instruction then applies -- silently turning a
    perfectly answerable threshold question into an unfiltered dump of
    every row, with nothing in the plan itself indicating a failure.

    This never overrides an LLM-produced filter_query -- it only fires
    when the model returned none. It parses the question for a percent
    value and a comparison direction, then picks whichever "<x>_pct"
    numeric column has the most keyword overlap with the question (e.g.
    "discount" in the question favors "discount_pct" over an unrelated
    "description_pct" even if both exist), refusing to guess if no
    "_pct" column shares any word with the question at all.
    """
    match = _PERCENT_IN_QUESTION.search(question)
    if not match:
        return None
    number = match.group(1)

    lowered = question.lower()
    if any(w in lowered for w in _LTE_WORDS):
        op = "<="
    elif any(w in lowered for w in _GTE_WORDS):
        op = ">="
    elif any(w in lowered for w in _LT_WORDS):
        op = "<"
    elif any(w in lowered for w in _GT_WORDS):
        op = ">"
    else:
        return None

    pct_columns = [c for c in df.columns if c.endswith(
        "_pct") and pd.api.types.is_numeric_dtype(df[c])]
    if not pct_columns:
        return None

    question_words = set(re.findall(r"[a-z]+", lowered))

    def _overlap(col: str) -> int:
        base_words = set(col[: -len("_pct")].split("_"))
        return len(question_words & base_words)

    best = max(pct_columns, key=_overlap)
    if _overlap(best) == 0:
        return None  # no column name relates to anything in the question -- don't guess blindly

    return f"{best} {op} {number}"


def _validate_plan(plan: dict, df: pd.DataFrame) -> dict:
    valid_columns = list(df.columns)

    mode = plan.get("mode")
    if mode not in {"sql", "semantic", "both"}:
        mode = "sql"

    group_by = plan.get("group_by")
    if group_by:
        group_by = [c for c in group_by if c in valid_columns] or None

    agg_column = plan.get("agg_column")
    if agg_column not in valid_columns:
        agg_column = None

    agg_func = plan.get("agg_func")
    if agg_func not in {"sum", "mean", "count", "max", "min"}:
        agg_func = "sum" if agg_column else None

    semantic_text_column = plan.get("semantic_text_column")
    if semantic_text_column not in valid_columns:
        semantic_text_column = None
        if mode in {"semantic", "both"}:
            mode = "sql"  # can't do semantic search without a valid text column

    filter_query = _sanitize_filter_query(
        plan.get("filter_query"), valid_columns)
    filter_query = _fix_numeric_comparison_on_text_column(filter_query, df)

    return {
        "mode": mode,
        "filter_query": filter_query,
        "group_by": group_by,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "semantic_text_column": semantic_text_column,
        "top_k": plan.get("top_k") if isinstance(plan.get("top_k"), int) else config.DEFAULT_TOP_K,
    }


def _fallback_plan(df: pd.DataFrame) -> dict:
    """Used if the LLM call fails outright: return the full table, no filtering."""
    return {
        "mode": "sql", "filter_query": None, "group_by": None,
        "agg_column": None, "agg_func": None,
        "semantic_text_column": None, "top_k": config.DEFAULT_TOP_K,
    }


def build_plan(user_request: str, df: pd.DataFrame) -> dict:
    """
    Main entry point. Returns a validated plan dict ready for
    query_agent.execute_plan(). Never raises — falls back to a safe
    no-op plan (full table, no filter) if the LLM call or JSON parsing fails.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": PLAN_PROMPT.format(
                    schema=_describe_schema(df),
                    sample=_sample_rows(df),
                    question=user_request,
                ),
            }],
            max_tokens=800,
            temperature=0,
            reasoning_effort="low",
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        if not raw:
            raise ValueError(
                "Model returned empty content (likely exhausted max_tokens on reasoning before answering)")
        plan = json.loads(raw)
        validated = _validate_plan(plan, df)

        if not validated["filter_query"] and validated["mode"] == "sql":
            fallback_filter = _fallback_percent_filter(user_request, df)
            if fallback_filter:
                logger.info(f"LLM returned no filter for an apparent percent-threshold question; "
                            f"deterministic fallback applied: {fallback_filter}")
                validated["filter_query"] = fallback_filter

        logger.info(f"Query plan: {validated}")
        return validated
    except Exception as e:
        logger.error(
            f"Query planning failed, falling back to full-table result: {e}")
        return _fallback_plan(df)
