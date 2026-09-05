"""
Report agent — the second (and last) LLM call per query, per the
2-calls-per-query budget the Orchestrator is designed around.
"""

from groq import Groq

import config
from utils.logger import get_logger

logger = get_logger("report_agent")

_client = None

REPORT_PROMPT = """You are summarizing a data query result for a dashboard.

User's original question: {question}
Result summary (record count, total record count, whether a filter was
successfully applied, match rate, columns, numeric stats, any warnings,
and — when present — "queried_columns"/"queried_column_stats" naming the
specific column(s) this query actually filtered/grouped/aggregated on):
{analysis}

Rules:
- Use ONLY the numbers present in the result summary above. Never state a
  count, percentage, or statistic that isn't literally there (e.g. don't
  estimate "roughly half" or invent a record count) -- if the summary
  doesn't contain the figure needed to answer precisely, say what IS
  known instead of guessing.
- If "queried_column_stats" is present, that is THE column(s) this
  question is actually about -- lead with those numbers specifically.
  "numeric_summary" covers every numeric column in the result and will
  usually contain other, unrelated columns (e.g. a dataset can have both
  a real "discount_pct" and an unrelated "description_pct" derived from
  something else entirely, like fabric composition text) -- do not pull
  a number from "numeric_summary" for a column that isn't in
  "queried_columns" when "queried_columns" is present.
- If "warnings" is non-empty, or "filter_applied" is false while the
  question clearly asked for a filtered subset, say so plainly up front
  (e.g. "the requested filter couldn't be applied, so this reflects the
  full dataset") rather than presenting the data as if the filter worked.
- If "filter_applied" is true, lead with the real match: "X of Y records
  (Z%) matched" using record_count/total_record_count/match_rate_pct
  exactly as given.
- Write 3-5 sentences in plain language, no markdown headers or bullet
  lists. Be substantive: mention the concrete numbers that ARE available
  (counts, relevant min/max/mean from numeric_summary) rather than only
  restating the question.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate_report(user_request: str, analysis: dict) -> str:
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": REPORT_PROMPT.format(question=user_request, analysis=analysis),
            }],
            max_tokens=600,
            temperature=0.3,
            reasoning_effort="low",
        )
        text = response.choices[0].message.content.strip()
        if not text:
            raise ValueError(
                "Model returned empty content (likely exhausted max_tokens on reasoning before answering)")
        return text
    except Exception as e:
        logger.error(
            f"Report generation failed, falling back to a plain summary: {e}")
        warnings = analysis.get("warnings") or []
        warning_note = f" Note: {' '.join(warnings)}" if warnings else ""
        if analysis.get("filter_applied"):
            return (f"{analysis.get('record_count', 0)} of {analysis.get('total_record_count', 0)} records "
                    f"({analysis.get('match_rate_pct', 0)}%) matched, across columns "
                    f"{analysis.get('columns', [])}.{warning_note}")
        return (f"Found {analysis.get('record_count', 0)} records "
                f"across columns {analysis.get('columns', [])}.{warning_note}")
