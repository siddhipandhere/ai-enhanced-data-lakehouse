# AI-Enhanced Data Lakehouse

A local data lakehouse that takes structured, semi-structured and unstructured files, refines them through
Bronze → Silver → Gold layers, and lets you query them three ways: SQL-style filters, semantic (meaning-based)
search over text and images, and plain-English reports written by an LLM.

## Architecture

```
Upload (dashboard.html)
   │  POST /upload/bulk  (JWT login required)
   ▼
Bronze   ingestion/loaders.py     Spark reads each file as-is → Delta table (atomic batch commit, audit log)
   ▼
Silver   medallion/silver.py      Spark cleaning + schema enforcement → Delta MERGE (ACID, _delta_log)
         medallion/refine.py      Structural refinement: parse raw emails, "NaN"/"N/A" text → missing,
                                  drop TOTAL rows, remove duplicate emails, flag scanned PDFs, read image metadata
   ▼
Gold     medallion/gold.py        Query-ready Parquet table per dataset (+ combined view of matching datasets)
   ▼
Vector   embeddings/              FAISS index per dataset:
                                    text   → all-MiniLM-L6-v2 (384-d)
                                    images → CLIP clip-ViT-B-32 (512-d), searchable by description
   ▼
Query    sql/        SQL Explorer: filter / group-by / aggregate (safe pandas expressions)
         search/     Semantic Search: FAISS nearest neighbours, image results as thumbnails
         agents/     Reports: planner → query → analysis → report (2 Groq LLM calls per question)
```

`pipeline/orchestration.py` runs Silver → Gold → Vector automatically in the background after every upload;
`pipeline/registry.py` tracks each stage, and the Pipeline page shows progress, refinement notes and a Retry button.

## Supported files

| Type | Extensions | Becomes |
|---|---|---|
| Structured | .csv, .xlsx, .xls, .parquet | One table per file (one per useful sheet for Excel; the real header row is detected) |
| Semi-structured | .json, .xml | Flattened table |
| Unstructured | .pdf | One row per page with its text (scanned PDFs are flagged: OCR would be needed) |
| Unstructured | .png, .jpg, .jpeg | Images uploaded together become ONE collection, one row per picture |
| Unstructured | .txt | One row per line |

## Setup (Windows, PowerShell)

```
cd lakehouse_backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then put your Groq key in .env
uvicorn main:app --reload
```

Open `login.html` (project root) in a browser, register / log in, and you land on `dashboard.html`.

`.env` settings:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM for Reports. Without it, reports still work with a computed summary. |
| `JWT_SECRET_KEY` | Signs login tokens. Set any long random string before a demo. |
| `HF_HOME` (optional) | Where the embedding models are stored, e.g. `D:\hf_cache` if C: is low on space. |

First run notes: Spark downloads the Delta Lake JAR once; the text model (~90 MB) and the image model
(~600 MB, only when images are uploaded) download once from Hugging Face.

## Using it

- **Pipeline**: per-dataset Bronze / Silver / Gold / Vector status, green notes describing what refinement did, Retry.
- **SQL Explorer**: filter expressions such as `bonus > 5000000`, `is_sent == True`,
  `subject.str.contains('california', case=False, na=False)`, `image_format == 'PNG'`; optional group-by + aggregate.
- **Semantic Search**: search by meaning, e.g. "gas pipeline outage" on emails, "broadband strategy" on the annual
  report, "an office building" or "a stock price chart" on an image collection.
- **Reports**: ask a question in plain English ("Which mailbox sent the most emails?"); the planner turns it into a
  validated query, the result is analysed in code, and the LLM writes the summary.
- **All datasets (combined)**: appears when 2+ datasets have exactly the same columns (e.g. the three Enron mailboxes).

## Security

- Every route except register/login needs a JWT.
- LLM-generated and user-typed filters are checked by a tokenizer (`agents/query_planner.check_filter_query`):
  only real column names, literals and a whitelist of methods are allowed; `@`, backticks, dunder names and unknown
  identifiers are rejected, and `DataFrame.query` runs with empty local/global scopes.
- Upload file names are reduced to a base name (no path traversal); each batch is validated before anything is written.
- Secrets live only in `.env` (git-ignored).

## Tests

The test harness (stubbed Spark / models / Groq) covers the filter validator and injection attempts, aggregation,
combined datasets, raw-email parsing and de-duplication, Excel header detection, PDF text / scanned detection,
image collections, CLIP index build and search ranking, thumbnails, crash recovery and Retry.

## Project structure

```
lakehouse_backend/
├── main.py                 FastAPI app, startup recovery checks
├── config.py               paths, models, limits (reads .env)
├── spark_utils.py          local[*] SparkSession with Delta Lake
├── auth/                   users (SQLite), bcrypt, JWT
├── upload/                 POST /upload/bulk
├── ingestion/              loaders.py (Bronze), excel.py (sheet + header detection)
├── medallion/              silver.py, refine.py, gold.py
├── embeddings/             embedder.py (text + CLIP), vector_store.py (FAISS per dataset)
├── pipeline/               orchestration.py, registry.py, routes.py (status + retry)
├── sql/  search/  reports/  datasets/   API routes
├── agents/                 query_planner, orchestrator, query_agent, analysis_agent, report_agent
└── utils/                  logger, metrics, media (image helpers, thumbnails)
```
