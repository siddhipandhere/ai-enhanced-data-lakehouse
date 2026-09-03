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
Be concise: 2-3 sentences, plain language, no markdown headers.

User's original question: {question}
Result summary (record count, columns, stats): {analysis}
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
            max_tokens=400,
            temperature=0.3,
            reasoning_effort="low",
        )
        text = response.choices[0].message.content.strip()
        if not text:
            raise ValueError("Model returned empty content (likely exhausted max_tokens on reasoning before answering)")
        return text
    except Exception as e:
        logger.error(f"Report generation failed, falling back to a plain summary: {e}")
        return (f"Found {analysis.get('record_count', 0)} matching records "
                f"across columns {analysis.get('columns', [])}.")