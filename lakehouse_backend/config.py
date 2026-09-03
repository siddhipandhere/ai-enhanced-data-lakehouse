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
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Storage paths (the "Bronze / Silver / Gold" medallion layers) ---
DATA_ROOT = Path(os.getenv("LAKEHOUSE_DATA_ROOT", "data"))
BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"

for _dir in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, VECTOR_STORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Ingestion ---
# Grouped by data category so the ingestion dispatcher (Option D:
# Spark/Delta backend) can route each upload to the right reader.
STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".parquet"}
SEMI_STRUCTURED_EXTENSIONS = {".json", ".xml"}
UNSTRUCTURED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".txt"}
ALLOWED_EXTENSIONS = STRUCTURED_EXTENSIONS | SEMI_STRUCTURED_EXTENSIONS | UNSTRUCTURED_EXTENSIONS
MAX_FILE_SIZE_MB = 200
MAX_FILES_PER_BULK_UPLOAD = 25

# --- Spark / Delta Lake (Option D: local[*] mode, no HDFS/cluster) ---
SPARK_APP_NAME = "ai-lakehouse-local"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "4"))  # small local datasets

# --- Auth ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me-before-viva")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
USERS_DB_PATH = DATA_ROOT / "users.db"

# --- Embeddings / vector search ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2
FAISS_INDEX_TYPE = "IndexFlatIP"  # cosine similarity via normalized inner product
DEFAULT_TOP_K = 10

# --- LLM (Groq) ---
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_LLM_CALLS_PER_QUERY = 2  # Orchestrator classification + Report summary

# --- Record keying ---
JOIN_KEY = "id"  # the shared field CSV / JSON / image records are joined on