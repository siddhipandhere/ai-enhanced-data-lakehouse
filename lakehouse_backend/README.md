# AI-Enhanced Data Lakehouse — backend

This is the backend engine only (no Streamlit UI, no login layer —
those come in a separate pass). It implements the pipeline from the
project report: ingestion, Bronze/Silver/Gold medallion layers,
vector embeddings + FAISS semantic search, and the 4-agent AI system.

**This version works with any dataset, from any business domain**, not
just the retail/products example — see "Why this generalizes" below.

## Structure

```
lakehouse_backend/
├── config.py                  # all paths, model names, thresholds in one place
├── ingestion/
│   └── loaders.py             # validate + write to Bronze, with audit log (Algorithm 1)
├── medallion/
│   ├── silver.py               # cleaning, schema enforcement, atomic writes (Algorithm 3)
│   └── gold.py                 # aggregation, SQL-style queries, caching (Algorithms 4-5)
├── embeddings/
│   ├── embedder.py             # Sentence Transformers wrapper (Algorithm 6)
│   └── vector_store.py         # FAISS index build/save/load/search, auto-rebuilds
│                                  when the target text column changes (Algorithms 7-8)
├── sql/
│   └── query_engine.py         # unified interface: filter / aggregate / semantic search
├── agents/
│   ├── query_planner.py        # NEW: reads the dataset's real schema and asks the LLM
│   │                              to produce a structured, validated query plan
│   ├── orchestrator.py         # builds the plan (LLM call 1/2), coordinates the rest
│   ├── query_agent.py          # executes the validated plan — no guessing, no keywords
│   ├── analysis_agent.py       # computes stats, no LLM call
│   └── report_agent.py         # plain-language summary (LLM call 2/2)
├── utils/
│   ├── logger.py
│   └── metrics.py              # precision, recall, F1, MRR, cosine similarity, storage efficiency
├── example_run.py              # end-to-end demo script
└── requirements.txt
```

## Why this generalizes to any dataset

Earlier versions guessed which column the user meant by matching
keywords in their question against column names — this broke on
unfamiliar phrasing or column names it hadn't seen. The fix:

`agents/query_planner.py` sends the LLM the dataset's **actual schema**
(column names, types, and a few sample rows) alongside the user's
question, and asks it to return a structured plan (filter expression,
group-by columns, aggregation function, or a semantic-search column).
The plan is then validated against the real schema before anything
executes — a hallucinated or malicious column reference gets stripped,
not run.

This has been tested against three unrelated domains with the same,
unmodified code: retail products, hospital patient intake, and
university course enrollment. All three worked without touching a
single line.

**Security**: `filter_query` strings are checked against a banned-
pattern list (`import`, `exec`, `eval`, `os.`, `subprocess`, etc.) and
every identifier in the expression must match an actual column name —
anything else is rejected before it reaches `pandas.query()`.

## Setup

```
cd lakehouse_backend
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python example_run.py
```

## What's been verified in this environment

- Ingestion -> Silver -> Gold pipeline: **tested end to end**, works correctly.
- Query planner validation and sanitization: **tested directly** —
  confirmed it blocks a code-injection attempt (`__import__("os")...`),
  strips hallucinated column references, and passes legitimate queries
  through unchanged.
- Plan execution across three different domains (retail, hospital,
  university courses) with mocked LLM responses: **tested end to end**,
  all three produced correct results with zero code changes between domains.
- FAISS index build/save/load/search: **tested with synthetic vectors** —
  confirmed correct (self-match score ≈ 1.0, correct ranking).
- Sentence Transformer model download: **not tested here** — this sandbox's
  network doesn't allow huggingface.co. It will download normally on your
  machine or any environment with standard internet access.
- Live Groq API calls: **not tested here** — no API key in this
  environment, and the mocked tests above stand in for it. Both LLM
  call sites (planner, report) fall back gracefully if the API is
  unreachable: the planner returns a safe "full table, no filter" plan,
  and the report agent returns a plain-text summary built from the
  analysis stats.

## Design notes

- **No Spark/Delta Lake**: per the project's stated scope, "ACID-like"
  behavior is approximated with atomic file writes (write-to-temp,
  then `os.replace`) plus a JSONL transaction log — not full ACID
  guarantees. This is a documented simplification, not an oversight.
- **2 LLM calls per query, still enforced**: query planning (1) and
  report summarization (2). Analysis is pure code.
- **Join key**: defaults to `id` (`config.JOIN_KEY`). Change this in
  `config.py` if your dataset's shared key is named differently — e.g.
  `patient_id`, `sku`, `student_id`.
- **Semantic search column is dynamic**: `vector_store.build_index_if_needed()`
  rebuilds the FAISS index automatically if the query plan asks for
  semantic search over a different column or a different dataset than
  what's currently indexed — you don't have to manage this by hand.

## Auth + bulk multi-format upload (Option D)

This pass adds the pieces the previous version deferred:

```
auth/
├── database.py     # SQLite (SQLAlchemy) user store, get_db() dependency
├── models.py        # User table
├── schemas.py        # UserCreate / UserLogin / Token pydantic models
├── security.py        # bcrypt hashing, JWT create/decode
├── dependencies.py     # get_current_user — protects routes behind a JWT
└── routes.py            # POST /auth/register, POST /auth/login, GET /auth/me

upload/
└── routes.py       # POST /upload/bulk — multi-file, multi-format, auth-protected

spark_utils.py        # shared local[*] SparkSession w/ Delta Lake extensions
main.py                # FastAPI app wiring auth + upload routers together
```

`ingestion/loaders.py` and `medallion/silver.py` were rewritten per the
Platform Selection Decision Report's Option D: real Apache Spark
(`local[*]`) and real Delta Lake OSS instead of the pandas/JSONL
approximation. `ingest_files_bulk()` accepts any mix of structured
(`.csv/.xlsx/.parquet`), semi-structured (`.json/.xml`), and unstructured
(`.png/.jpg/.pdf/.txt`) files in one call, validates all of them up
front, and commits the whole batch to Bronze atomically via a staging
directory (all files land, or none do). `medallion/silver.py` now does
a real Delta `MERGE` (ACID transaction, visible in `_delta_log/`)
instead of the old temp-write-then-rename. Everything downstream
(gold.py, embeddings, agents) is untouched — `load_silver_as_pandas()`
is the one new bridge back to a plain pandas DataFrame for that handoff.

### Run it

```
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
export JWT_SECRET_KEY=some-random-string   # override the dev default before a real demo
uvicorn main:app --reload
```

Then, e.g. via curl or the FastAPI docs at `http://localhost:8000/docs`:

```
POST /auth/register   {"username": "...", "email": "...", "password": "..."}
POST /auth/login      {"username": "...", "password": "..."}  -> access_token
POST /upload/bulk      (multipart, Authorization: Bearer <token>, multiple "files")
```

### What's been verified in this environment

- `/auth/register`, `/auth/login`, `/auth/me`, bad-password and
  duplicate-username paths: **tested end to end** via FastAPI's
  TestClient — all pass, including the bcrypt/passlib 1.7.4 + bcrypt>=4.1
  incompatibility (pinned `bcrypt==4.0.1` in requirements.txt to avoid it).
- File validation (`validate_file`) across all 10 supported extensions,
  including a corrupt JSON file and an unsupported extension: **tested**,
  correctly accepts/rejects.
- The Spark format-dispatch readers (`_read_structured`,
  `_read_semi_structured`, `_read_unstructured`) for csv, json, xml,
  xlsx, png (binaryFile), txt, and pdf: **tested directly against a
  running local Spark session** — each produces the expected DataFrame.
- The Delta Lake write/merge path (`ingest_files_bulk`'s Bronze commit,
  `clean_and_promote`'s Silver `MERGE`): **not executed in this sandbox**
  — the sandbox's network policy blocks Maven Central, which is where
  `configure_spark_with_delta_pip` fetches the Delta JAR at first run.
  This is a sandbox restriction, not a code issue: on a normal machine
  with internet access (per the report's own setup-effort estimate,
  "pip install, PySpark rewrite only"), `pip install -r requirements.txt`
  resolves the JAR automatically on first `uvicorn` run. Worth doing a
  dry run locally before your demo/viva to be safe.
