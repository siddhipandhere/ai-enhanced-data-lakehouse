# Run from lakehouse_backend:  venv\Scripts\python check_reports.py
# Check for the grouped-report fix: the analysis must say which aggregate a value
# is, and carry real stats of the matched rows (not the per-group values).
import sys, types, json
import os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["LAKEHOUSE_DATA_ROOT"] = tempfile.mkdtemp()  # keep test files out of data/
vs = types.ModuleType("embeddings.vector_store")
vs.semantic_search = vs.index_exists = vs.build_index_if_needed = lambda *a, **k: None
sys.modules["embeddings"] = types.ModuleType("embeddings"); sys.modules["embeddings.vector_store"] = vs
sys.modules["groq"] = types.SimpleNamespace(Groq=None)
import pandas as pd
from agents.query_agent import run_query
from agents.analysis_agent import run_analysis
from agents.report_agent import _fallback_summary

df = pd.DataFrame({"id": range(8), "category": ["Bags", "Bags", "Shoes", "Shoes", "Shoes", "Bags", "Shoes", "Bags"],
                   "discount_pct": [55, 80, 60, 80, 70, 10, 20, 30]})
plan = {"mode": "sql", "filter_query": "discount_pct > 50", "group_by": ["category"], "agg_column": "discount_pct",
        "agg_func": "max", "semantic_text_column": None, "top_k": 10}
res, warn = run_query("q", df, plan)
a = run_analysis(res, total_record_count=len(df), filter_applied=True, warnings=warn,
                 matched_rows=res.attrs["matched_rows"], aggregated=res.attrs["aggregated"])

assert a["record_count"] == 5 and a["group_count"] == 2
assert a["aggregated_result"] == [{"category": "Bags", "max_discount_pct": 80}, {"category": "Shoes", "max_discount_pct": 80}]
assert a["aggregation"].startswith("max of discount_pct per category")
s = a["queried_column_stats"]["discount_pct"]
assert (s["min"], s["max"], s["count"], s["mean"]) == (55, 80, 5, 69), s   # the 5 matched rows, not the 2 maxima
assert "max_discount_pct: 80" in _fallback_summary(a)

# count of the group column itself keeps gold.aggregate's own "<col>_count" name
res, _ = run_query("q", df, {**plan, "filter_query": None, "agg_column": "category", "agg_func": "count"})
a = run_analysis(res, total_record_count=8, aggregated=True, matched_rows=8)
assert a["aggregated_result"][0] == {"category": "Bags", "category_count": 4}, a["aggregated_result"]
assert "queried_column_stats" not in a
print("report analysis checks OK")

from agents.report_agent import _renumber_parts
assert _renumber_parts("1. Answer\n\n2. Figures\n\n3. Breakdown:\n- a\n- b\n\n5. Takeaway") == \
    "1. Answer\n\n2. Figures\n\n3. Breakdown:\n- a\n- b\n\n4. Takeaway"
assert _renumber_parts("Note: x\n\n2. Only part") == "Note: x\n\n1. Only part"
print("report numbering checks OK")
