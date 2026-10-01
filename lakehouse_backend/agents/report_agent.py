"""
Report agent — the second (and last) LLM call per query, per the
2-calls-per-query budget the Orchestrator is designed around.
"""

import json
import re

from groq import Groq

import config
from utils.logger import get_logger

logger = get_logger("report_agent")

_client = None

REPORT_PROMPT = """You are a data analyst writing the summary shown at the top of a dashboard report.

User's question: {question}

Computed result (JSON). Fields: record_count, total_record_count, filter_applied,
match_rate_pct, warnings, queried_columns + queried_column_stats (the column(s)
the question is actually about), numeric_summary (every numeric column),
categorical_summary (top values of category-like columns in the RESULT rows),
sample_rows (first few result rows):
{analysis}

If "aggregated" is true, the result is a GROUPED table: "aggregated_result" has one row
per group with its computed value (already sorted largest first), "group_count" is the
number of groups, and record_count/total_record_count are the rows that matched the
filter BEFORE grouping. Answer from aggregated_result (name the top group and its value,
and compare with the others); for part 3 list every group with its value.

If "matching_passages" is present, this was a search over documents (PDF pages, emails):
the passages are the most relevant text found, best match first. Answer the question FROM
that text -- quote a short phrase and name its source / page_number. If the passages do not
contain the answer, say so plainly instead of answering from general knowledge. For part 2,
say how many passages matched and where they come from; never report similarity-score
statistics. For part 4, cite the passages you used.

Write the report in this structure, as plain text (no markdown headers, no bold, no tables):

1. Answer (1-2 sentences): answer the question directly. If filter_applied is true,
   state "X of Y records (Z%) matched" using record_count / total_record_count /
   match_rate_pct exactly as given. If warnings is non-empty, or the question asked
   for a subset but filter_applied is false, say so FIRST and plainly.
2. Key figures (2-3 sentences): the min / median (50%) / mean / max of the
   queried column(s) from queried_column_stats; if there are none, use the most
   relevant columns of numeric_summary (prices, ratings, percentages).
3. Breakdown (2-3 short lines, each starting with "- "): what dominates the result,
   from categorical_summary, with counts, e.g. "- Category: Clothing and Accessories
   (812), Footwear (95)". Skip this part if categorical_summary is empty.
4. Examples (1 sentence): name 2-3 concrete items from sample_rows.
5. Takeaway (1 sentence): one practical insight or a useful follow-up question.

Put a blank line between parts. Rules:
- Use ONLY numbers that literally appear in the JSON above. Never estimate,
  round into vague words ("roughly half"), or invent counts.
- If queried_columns is present, do not quote numbers for unrelated columns
  that happen to be in numeric_summary (e.g. a "description_pct" when the
  question was about "discount_pct").
- If record_count is 0, say nothing matched, suggest how to relax the question,
  and skip parts 2-4.
- If a part has nothing to say (e.g. no numeric stats), leave it out entirely --
  never output an empty numbered line.
"""


def _get_client() -> Groq:
    global _client
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (add it to lakehouse_backend/.env)")
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _fallback_summary(analysis: dict) -> str:
    """Deterministic multi-part summary used when the LLM is unavailable.
    Built only from computed numbers, so it's always accurate."""
    parts = []
    warnings = analysis.get("warnings") or []
    if warnings:
        parts.append("Note: " + " ".join(warnings))

    n, total = analysis.get("record_count", 0), analysis.get("total_record_count", 0)
    if analysis.get("filter_applied"):
        parts.append(f"{_fmt(n)} of {_fmt(total)} records ({analysis.get('match_rate_pct', 0)}%) matched.")
    else:
        parts.append(f"The result contains {_fmt(n)} records.")
    if not n:
        return "\n\n".join(parts + ["Nothing matched - try relaxing the condition."])

    if analysis.get("aggregated"):
        rows = analysis.get("aggregated_result") or []
        lines = ["- " + ", ".join(f"{k}: {_fmt(v)}" for k, v in r.items()) for r in rows[:10]]
        if lines:
            parts.append(f"{analysis.get('group_count', len(rows))} groups (largest first):\n" + "\n".join(lines))
        return "\n\n".join(parts)

    stats = analysis.get("queried_column_stats") or {}
    lines = [f"{col}: min {_fmt(s.get('min'))}, median {_fmt(s.get('50%'))}, "
             f"mean {_fmt(s.get('mean'))}, max {_fmt(s.get('max'))}."
             for col, s in stats.items() if s.get("mean") is not None]
    if lines:
        parts.append(" ".join(lines))

    cats = analysis.get("categorical_summary") or {}
    breakdown = []
    for col, info in list(cats.items())[:3]:
        top = ", ".join(f"{k} ({_fmt(v)})" for k, v in list(info["top_values"].items())[:3])
        breakdown.append(f"- {col}: {top}")
    if breakdown:
        parts.append("\n".join(breakdown))
    return "\n\n".join(parts)


def generate_report(user_request: str, analysis: dict) -> str:
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": REPORT_PROMPT.format(
                    question=user_request,
                    analysis=json.dumps(analysis, default=str, indent=1)),
            }],
            # gpt-oss is a reasoning model: its hidden reasoning tokens count
            # against max_tokens. At 600 the visible answer was routinely cut
            # to one or two sentences (or empty).
            max_tokens=2000,
            temperature=0.3,
            reasoning_effort="low",
        )
        text = (response.choices[0].message.content or "").strip()
        # The dashboard shows plain text: drop markdown bold/italics markers
        # and numbered parts the model left empty (e.g. a lone "3.").
        text = text.replace("**", "").replace("__", "")
        text = "\n".join(l for l in text.splitlines() if not re.fullmatch(r"\s*\d+\.\s*", l))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            raise ValueError(
                "Model returned empty content (likely exhausted max_tokens on reasoning before answering)")
        return text
    except Exception as e:
        logger.error(
            f"Report generation failed, falling back to a computed summary: {e}")
        return _fallback_summary(analysis)
