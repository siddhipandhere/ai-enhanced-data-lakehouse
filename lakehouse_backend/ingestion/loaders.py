"""
Data ingestion layer — Option D (local-mode PySpark + Delta Lake OSS).

Report Algorithm 1 (Data Ingestion), upgraded from the pandas prototype
to real Spark DataFrames written as real Delta tables, per the Platform
Selection Decision Report (Section 8, Final Platform Stack).

Responsibilities:
- Accept structured, semi-structured, and unstructured files in a single
  bulk request (multiple datasets uploaded simultaneously)
- Validate every file (type, size, structural integrity) before anything
  is written
- Route each file to the right Spark reader based on its data category
- Write validated data into the Bronze layer as Delta tables (real ACID
  transaction log, ``data/bronze/<table>/_delta_log/``)
- Commit the whole bulk batch atomically at the filesystem level: either
  every file in the batch lands in Bronze, or none do
- Log ingestion metadata for an audit trail
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import input_file_name, lit

import config
from spark_utils import get_spark
from utils.logger import get_logger

logger = get_logger("ingestion")

AUDIT_LOG_PATH = config.BRONZE_DIR / "_ingestion_audit_log.jsonl"


class ValidationError(Exception):
    pass


def _category_for_extension(ext: str) -> str:
    if ext in config.STRUCTURED_EXTENSIONS:
        return "structured"
    if ext in config.SEMI_STRUCTURED_EXTENSIONS:
        return "semi_structured"
    if ext in config.UNSTRUCTURED_EXTENSIONS:
        return "unstructured"
    raise ValidationError(f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")


def validate_file(file_path: Path) -> str:
    """
    Raises ValidationError if the file fails type, size, or structural
    checks. Returns the file's data category on success.
    """
    ext = file_path.suffix.lower()
    category = _category_for_extension(ext)

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise ValidationError(f"File is {size_mb:.1f} MB, exceeds {config.MAX_FILE_SIZE_MB} MB limit")

    try:
        if ext == ".csv":
            pd.read_csv(file_path, nrows=5)
        elif ext == ".xlsx":
            pd.read_excel(file_path, nrows=5)
        elif ext == ".parquet":
            pd.read_parquet(file_path).head(5)
        elif ext == ".json":
            with open(file_path) as f:
                json.load(f)
        elif ext == ".xml":
            import xml.etree.ElementTree as ET
            ET.parse(file_path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            with Image.open(file_path) as img:
                img.verify()
        elif ext == ".pdf":
            from pypdf import PdfReader
            PdfReader(str(file_path))
        elif ext == ".txt":
            with open(file_path, encoding="utf-8", errors="strict") as f:
                f.read(1024)
    except Exception as e:
        raise ValidationError(f"Structural validation failed: {e}")

    return category


# --- Format-specific Spark readers ---------------------------------------

def _dedupe_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    """Renames duplicate column labels (e.g. two JSON keys that normalize to
    the same name) to `name`, `name__2`, `name__3`, ... Both Spark and
    Parquet reject duplicate column names outright, so this has to happen
    before the frame ever reaches Spark.

    Dedupe is done case-insensitively (tracking lowercased names in `seen`)
    because Spark resolves column names case-insensitively by default
    (spark.sql.caseSensitive=false). Two columns that differ only by case
    -- e.g. "Care Instructions" vs "care instructions" from two different
    nested JSON paths -- are exact-distinct to pandas/Parquet but collide
    the moment Spark reads the file back, raising COLUMN_ALREADY_EXISTS.
    """
    seen: dict[str, int] = {}
    new_cols = []
    for c in pdf.columns:
        name = str(c)
        key = name.lower()
        if key not in seen:
            seen[key] = 1
            new_cols.append(name)
        else:
            seen[key] += 1
            new_cols.append(f"{name}__{seen[key]}")
    pdf = pdf.copy()
    pdf.columns = new_cols
    return pdf


def _flatten_nested_objects(pdf: pd.DataFrame) -> pd.DataFrame:
    """Serializes any dict/list-valued cells to JSON strings.

    _dedupe_columns only protects against *top-level* column-name
    collisions. It can't see inside a column like `product_details` that
    holds a Python dict/list per row (e.g. scraped per-product attribute
    key/value pairs). When such a column reaches PyArrow's Parquet writer,
    PyArrow infers a nested struct schema by unioning the keys across
    every row -- and if two rows used differently-cased keys for the same
    attribute (a common real-world scraping inconsistency, e.g.
    "Care Instructions" vs "care instructions"), that produces two struct
    fields that only collide once Spark reads them back with its default
    case-insensitive resolution (COLUMN_ALREADY_EXISTS), even though no
    top-level column was ever duplicated.

    Serializing dict/list cells to JSON strings up front sidesteps struct
    inference -- and its case-collision risk -- entirely. The values are
    still fully present and can be json.loads()'d downstream.
    """
    pdf = pdf.copy()
    for col in pdf.columns:
        if pdf[col].apply(lambda v: isinstance(v, (dict, list))).any():
            pdf[col] = pdf[col].apply(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    return pdf


def _pdf_to_spark(spark: SparkSession, pdf: pd.DataFrame, tmp_name: str) -> DataFrame:
    """Hands a pandas DataFrame to Spark via a Parquet round-trip instead of
    spark.createDataFrame(pdf).

    With Arrow disabled (see spark_utils.py), createDataFrame(pandas_df)
    backs the resulting DataFrame with an RDD of pickled Python Row objects
    (built via sc.parallelize). Every downstream action -- count(),
    collect(), even a plain column-wise aggregate with no UDFs -- then has
    to deserialize that RDD through a spawned Python worker subprocess, and
    on this Windows setup that worker process has been crashing
    (EOFException / "Python worker exited unexpectedly") on ordinary
    uploads. Writing the pandas frame to Parquet (pure pyarrow, no Spark
    involved) and reading it back with Spark's native JVM Parquet reader
    sidesteps that worker path entirely for ingestion.
    """
    # Files that were originally saved via pandas without index=False leave
    # a stray "Unnamed: 0" (or "Unnamed: 1", etc.) column holding the old
    # row index. It's not real data, but its name contains a space and a
    # colon -- both are in Delta's banned character set -- so it fails the
    # write with DELTA_INVALID_CHARACTERS_IN_COLUMN_NAMES if left in.
    unnamed_cols = [c for c in pdf.columns if str(c).startswith("Unnamed: ")]
    if unnamed_cols:
        logger.info(f"{tmp_name}: dropping stray index column(s) {unnamed_cols}")
        pdf = pdf.drop(columns=unnamed_cols)
    pdf = _dedupe_columns(pdf)
    pdf = _flatten_nested_objects(pdf)
    logger.info(f"{tmp_name}: {len(pdf.columns)} columns after dedupe -> {list(pdf.columns)}")
    tmp_dir = config.DATA_ROOT / "_tmp_pdf_bridge"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{tmp_name}_{uuid.uuid4().hex[:8]}.parquet"
    try:
        # object columns with mixed/unhashable types (e.g. None mixed with
        # dict from a ragged XML/JSON record) can't always be written by
        # pyarrow as-is; stringify anything it chokes on rather than fail
        # the whole upload.
        try:
            pdf.to_parquet(tmp_path, index=False)
        except Exception:
            pdf = pdf.astype(str)
            pdf.to_parquet(tmp_path, index=False)
        return spark.read.parquet(str(tmp_path))
    finally:
        # Spark reads the file lazily/at collect time in some paths, but by
        # the time _pdf_to_spark returns here the read has already been
        # planned against the file; deleting immediately after read.parquet()
        # can race with that plan on Windows, so leave cleanup to the
        # periodic sweep in ingest_files_bulk's staging cleanup instead.
        pass


def _read_structured(spark: SparkSession, path: Path, ext: str) -> DataFrame:
    if ext == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(str(path))
    if ext == ".parquet":
        return spark.read.parquet(str(path))
    if ext == ".xlsx":
        # Spark has no native xlsx reader; pandas (openpyxl) does the parsing,
        # then it's handed to Spark via the Parquet bridge (see _pdf_to_spark).
        pdf = pd.read_excel(path)
        return _pdf_to_spark(spark, pdf, path.stem)
    raise ValidationError(f"No structured reader for '{ext}'")


def _read_semi_structured(spark: SparkSession, path: Path, ext: str) -> DataFrame:
    if ext == ".json":
        try:
            return spark.read.option("multiLine", True).json(str(path))
        except Exception as e:
            # This was previously a silent `except Exception: pass`-equivalent
            # fallback, which hid the real reason Spark's native JSON reader
            # rejected the file (often the *same* duplicate-column problem
            # the pandas path below then hits again). Log it so the actual
            # cause is visible instead of guessed at.
            logger.warning(f"{path.name}: Spark native JSON reader failed ({e}); falling back to pandas.json_normalize")
            with open(path) as f:
                data = json.load(f)
            pdf = pd.json_normalize(data if isinstance(data, list) else [data])
            logger.info(f"{path.name}: {len(pdf.columns)} columns after json_normalize, before dedupe -> {list(pdf.columns)}")
            return _pdf_to_spark(spark, pdf, path.stem)
    if ext == ".xml":
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        records = [{child.tag: child.text for child in elem} for elem in list(root)]
        if not records:  # flat/single-record XML
            records = [{child.tag: child.text for child in root}]
        pdf = pd.DataFrame(records)
        return _pdf_to_spark(spark, pdf, path.stem)
    raise ValidationError(f"No semi-structured reader for '{ext}'")


def _read_unstructured(spark: SparkSession, path: Path, ext: str) -> DataFrame:
    if ext in {".png", ".jpg", ".jpeg"}:
        # binaryFile source: real Spark DataFrame of path/length/modificationTime/content,
        # keeps large blobs out of the driver's Python heap.
        df = spark.read.format("binaryFile").load(str(path))
        return df.withColumn("original_name", lit(path.name))
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [{"page_number": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(reader.pages)]
        pdf = pd.DataFrame(pages)
        pdf["source_file"] = path.name
        return _pdf_to_spark(spark, pdf, path.stem)
    if ext == ".txt":
        df = spark.read.text(str(path))  # one row per line, column "value"
        return df.withColumn("source_file", lit(path.name))
    raise ValidationError(f"No unstructured reader for '{ext}'")


def _read_any(spark: SparkSession, path: Path, category: str, ext: str) -> DataFrame:
    if category == "structured":
        return _read_structured(spark, path, ext)
    if category == "semi_structured":
        return _read_semi_structured(spark, path, ext)
    return _read_unstructured(spark, path, ext)


def _clean_tabular(df: DataFrame, source_name: str) -> DataFrame:
    """Drops exact duplicate rows and logs null counts per column, in Spark."""
    from pyspark.sql.functions import col, count, when

    null_counts_row = df.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    nulls_found = {c: n for c, n in null_counts_row.items() if n > 0}
    if nulls_found:
        logger.info(f"{source_name}: null values found -> {nulls_found}")

    before = df.count()
    df = df.dropDuplicates()
    removed = before - df.count()
    if removed:
        logger.info(f"{source_name}: removed {removed} duplicate rows")

    return df


@dataclass
class IngestResult:
    original_name: str
    category: str
    bronze_table: str
    bronze_path: Path
    record_count: int


def _log_audit_entry(entry: dict) -> None:
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def ingest_files_bulk(file_paths: list[str | Path], uploaded_by: str = "system") -> list[IngestResult]:
    """
    Validates and ingests any mix of structured / semi-structured /
    unstructured files in one bulk call. Each file becomes its own Bronze
    Delta table (data/bronze/<table_name>/), written via a staging
    directory that is only swapped into place once every file in the
    batch has validated *and* written successfully — so a batch either
    fully commits to Bronze or leaves Bronze untouched (no partial bulk
    upload visible to readers).
    """
    if len(file_paths) > config.MAX_FILES_PER_BULK_UPLOAD:
        raise ValidationError(
            f"{len(file_paths)} files exceeds the {config.MAX_FILES_PER_BULK_UPLOAD}-file bulk upload limit"
        )

    file_paths = [Path(p) for p in file_paths]

    # Phase 1: validate everything up front. One bad file fails the whole
    # batch before any Spark write happens.
    categories = {}
    for fp in file_paths:
        categories[fp] = validate_file(fp)

    spark = get_spark()
    batch_id = f"{uploaded_by}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    staging_root = config.BRONZE_DIR / f"_staging_{batch_id}"
    staging_root.mkdir(parents=True, exist_ok=True)

    staged: list[IngestResult] = []
    try:
        for fp in file_paths:
            ext = fp.suffix.lower()
            category = categories[fp]
            table_name = f"{uploaded_by}_{fp.stem}_{uuid.uuid4().hex[:6]}"
            staged_path = staging_root / table_name

            df = _read_any(spark, fp, category, ext)
            if category in ("structured", "semi_structured"):
                df = _clean_tabular(df, fp.name)
            record_count = df.count()

            df.write.format("delta").mode("overwrite").save(str(staged_path))

            staged.append(IngestResult(
                original_name=fp.name,
                category=category,
                bronze_table=table_name,
                bronze_path=config.BRONZE_DIR / table_name,
                record_count=record_count,
            ))

        # Phase 2: all writes succeeded -> atomically promote staging into Bronze.
        for result in staged:
            staged_path = staging_root / result.bronze_table
            shutil.move(str(staged_path), str(result.bronze_path))

    except Exception:
        logger.error(f"Bulk ingest batch {batch_id} failed; rolling back, Bronze left untouched")
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        # Clean up the pandas->Spark parquet bridge files (see _pdf_to_spark).
        # Safe to remove now: every df.write(...) above has already fully
        # materialized these into Bronze/staging, so nothing still needs them.
        shutil.rmtree(config.DATA_ROOT / "_tmp_pdf_bridge", ignore_errors=True)

    for result in staged:
        _log_audit_entry({
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "uploaded_by": uploaded_by,
            "original_name": result.original_name,
            "category": result.category,
            "bronze_table": result.bronze_table,
            "bronze_path": str(result.bronze_path),
            "record_count": result.record_count,
        })
        logger.info(
            f"Ingested {result.original_name} ({result.category}) -> "
            f"{result.bronze_path} ({result.record_count} records)"
        )

    return staged


def ingest_file(file_path: str | Path, uploaded_by: str = "system") -> Path:
    """Single-file convenience wrapper around ingest_files_bulk(), kept for
    backward compatibility with example_run.py and any single-upload caller."""
    results = ingest_files_bulk([file_path], uploaded_by=uploaded_by)
    return results[0].bronze_path


def read_audit_log() -> pd.DataFrame:
    if not AUDIT_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_json(AUDIT_LOG_PATH, lines=True)