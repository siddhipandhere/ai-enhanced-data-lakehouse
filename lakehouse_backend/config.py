"""
Central configuration for the AI-Enhanced Data Lakehouse backend.

Keep every path, model name, and tunable threshold here so the rest of
the codebase never hardcodes a string. Values can be overridden with
environment variables where noted.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Loads GROQ_API_KEY (and any other secrets) from a .env file next to this
# config.py, so you don't have to re-export it in every new terminal
# session. If the file doesn't exist, this is a harmless no-op and
# os.getenv() below still falls through to whatever's in the real
# environment.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Storage paths (the "Bronze / Silver / Gold" medallion layers) ---
# Anchored to this file's folder, not the current working directory.
# Previously a plain relative "data" meant that starting uvicorn from the
# repo root instead of lakehouse_backend/ silently created a second,
# empty data folder -- every dataset appeared to have "disappeared".
_data_root_env = Path(os.getenv("LAKEHOUSE_DATA_ROOT", "data"))
DATA_ROOT = _data_root_env if _data_root_env.is_absolute() else BASE_DIR / _data_root_env
BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"

for _dir in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, VECTOR_STORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Ingestion ---
# Grouped by data category so the ingestion dispatcher (Option D:
# Spark/Delta backend) can route each upload to the right reader.
STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}  # .xls needs: pip install xlrd
SEMI_STRUCTURED_EXTENSIONS = {".json", ".xml"}
UNSTRUCTURED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".txt"}
ALLOWED_EXTENSIONS = STRUCTURED_EXTENSIONS | SEMI_STRUCTURED_EXTENSIONS | UNSTRUCTURED_EXTENSIONS
MAX_FILE_SIZE_MB = 200
MAX_FILES_PER_BULK_UPLOAD = 25

# --- Spark / Delta Lake (Option D: local[*] mode, no HDFS/cluster) ---
SPARK_APP_NAME = "ai-lakehouse-local"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTITIONS = int(
    os.getenv("SPARK_SHUFFLE_PARTITIONS", "4"))  # small local datasets

# --- Auth ---
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "dev-secret-change-me-before-viva")
JWT_ALGORITHM = "HS256"
# Normal sign-in: token lasts this many minutes.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES") or "60")
# "Keep me signed in" ticked on the login page: token lasts this many days.
REMEMBER_ME_EXPIRE_DAYS = int(os.getenv("REMEMBER_ME_EXPIRE_DAYS") or "7")
USERS_DB_PATH = DATA_ROOT / "users.db"

# --- Embeddings / vector search ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2
# Image collections are searched with CLIP: it embeds pictures and text into
# one space, so typing "an office building" finds matching photos.
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
FAISS_INDEX_TYPE = "IndexFlatIP"  # cosine similarity via normalized inner product
DEFAULT_TOP_K = 10

# --- LLM (Groq) ---
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
# NEVER hardcode the key here -- this file is committed to Git. Put it in
# lakehouse_backend/.env (already in .gitignore) as:
#     GROQ_API_KEY=gsk_...
# If it's missing, both LLM call sites fall back gracefully (full-table
# plan + a computed plain-text summary) instead of crashing.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_LLM_CALLS_PER_QUERY = 2  # Orchestrator classification + Report summary

# --- Record keying ---
JOIN_KEY = "id"  # the shared field CSV / JSON / image records are joined on
