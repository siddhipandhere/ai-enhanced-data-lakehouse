"""
Silver layer — Option D (local-mode PySpark + Delta Lake OSS).

Report Algorithm 3 (Data Management, ACID-style transactions), upgraded
from the pandas write-temp-then-rename approximation to real Delta Lake
ACID transactions: every promote_to_silver() call is a single Delta
commit with its own entry in Delta's own ``_delta_log/`` transaction
log — no hand-rolled JSONL log needed, no partial writes ever visible
to readers, and MERGE-based idempotent re-runs instead of blind overwrite.

Downstream code (medallion/gold.py, agents/*) is untouched: it still
receives a plain pandas DataFrame. load_silver_as_pandas() is the one
new seam that bridges Spark back to pandas for that handoff.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

import config
from spark_utils import get_spark
from utils.logger import get_logger

logger = get_logger("silver")


def enforce_schema(df: DataFrame, required_columns: list[str]) -> DataFrame:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Schema enforcement failed, missing columns: {missing}")
    return df.select(*required_columns)


def _coerce_numeric_like(df: DataFrame) -> DataFrame:
    """Mirrors the old pandas heuristic: string columns that are >90% numeric
    get cast to double, in Spark rather than pandas."""
    from pyspark.sql.functions import count, when

    total = df.count()
    if total == 0:
        return df

    for field in df.schema.fields:
        if str(field.dataType) != "StringType()":
            continue
        numeric_count = df.select(
            count(when(col(field.name).rlike(r"^-?\d+(\.\d+)?$"), field.name))
        ).collect()[0][0]
        if total and (numeric_count / total) > 0.9:
            df = df.withColumn(field.name, col(field.name).cast("double"))
    return df


def _extract_percentage_columns(df: DataFrame) -> DataFrame:
    """Some source data stores a percentage as display text rather than a
    number -- e.g. a "discount" column holding "69% off" instead of 69.
    _coerce_numeric_like() intentionally leaves these alone (most of the
    string isn't numeric, so the >90%-numeric heuristic correctly skips
    it), but that means a question like "products over 50% off" has no
    numeric column to filter on at all downstream.

    For any string column where most non-null values start with a number
    followed by '%', this adds a sibling numeric column named
    '<col>_pct' (e.g. discount -> discount_pct) holding just that number,
    while leaving the original text column untouched for display. This
    is dataset-agnostic: columns that don't match this "NN% ..." shape
    are left completely alone, so it's a no-op for datasets without it.
    """
    from pyspark.sql.functions import count, regexp_extract, when

    total = df.count()
    if total == 0:
        return df

    pct_pattern = r"^\s*(-?\d+(?:\.\d+)?)\s*%"
    for field in df.schema.fields:
        if str(field.dataType) != "StringType()":
            continue
        match_count = df.select(
            count(when(col(field.name).rlike(pct_pattern), field.name))
        ).collect()[0][0]
        if total and (match_count / total) > 0.5:
            derived_col = f"{field.name}_pct"
            df = df.withColumn(
                derived_col,
                regexp_extract(col(field.name), pct_pattern, 1).cast("double"),
            )
            logger.info(f"Derived numeric column '{derived_col}' from '{field.name}' ({match_count}/{total} rows matched 'NN%' pattern)")
    return df


def clean_and_promote(bronze_path: str | Path, table_name: str | None = None,
                       required_columns: list[str] | None = None) -> Path:
    """
    Reads a Bronze-layer Delta table, applies cleaning + optional schema
    enforcement, and MERGEs it into the Silver Delta table — a real
    atomic, ACID-logged transaction (Delta's ``_delta_log/``), replacing
    the previous temp-file-then-rename approximation.
    """
    bronze_path = Path(bronze_path)
    spark = get_spark()

    df = spark.read.format("delta").load(str(bronze_path))
    df = _coerce_numeric_like(df)
    df = _extract_percentage_columns(df)
    df = df.dropna(how="all")

    if required_columns:
        df = enforce_schema(df, required_columns)

    table_name = table_name or bronze_path.name
    dest_path = config.SILVER_DIR / table_name

    if DeltaTable.isDeltaTable(spark, str(dest_path)):
        # Real ACID upsert: re-promoting the same source is idempotent
        # instead of silently duplicating rows, using the join key where
        # present and falling back to a full overwrite otherwise.
        if config.JOIN_KEY in df.columns:
            target = DeltaTable.forPath(spark, str(dest_path))
            (target.alias("t")
             .merge(df.alias("s"), f"t.{config.JOIN_KEY} = s.{config.JOIN_KEY}")
             .whenMatchedUpdateAll()
             .whenNotMatchedInsertAll()
             .execute())
        else:
            df.write.format("delta").mode("overwrite").save(str(dest_path))
    else:
        df.write.format("delta").mode("overwrite").save(str(dest_path))

    record_count = spark.read.format("delta").load(str(dest_path)).count()
    logger.info(f"Promoted to Silver: {dest_path} ({record_count} records, "
                f"Delta version history in {dest_path}/_delta_log)")
    return dest_path


def join_by_id(silver_table_path: str | Path, other_bronze_path: str | Path | None = None,
                join_key: str = config.JOIN_KEY) -> DataFrame:
    """Joins the Silver Delta table with another Bronze Delta table on the
    shared id field, as described in the report's data model."""
    spark = get_spark()
    df = spark.read.format("delta").load(str(silver_table_path))

    if other_bronze_path:
        other_df = spark.read.format("delta").load(str(other_bronze_path))
        if join_key not in df.columns or join_key not in other_df.columns:
            raise ValueError(f"Join key '{join_key}' not present in both sources")
        df = df.join(other_df, on=join_key, how="left")

    return df


def load_silver_as_pandas(table_name_or_path: str | Path) -> pd.DataFrame:
    """Bridges Spark back to pandas for the unchanged Gold-layer code
    (medallion/gold.py takes a pandas DataFrame, not a Spark one)."""
    spark = get_spark()
    path = Path(table_name_or_path)
    if not path.is_absolute() and path.parent == Path("."):
        path = config.SILVER_DIR / path
    return spark.read.format("delta").load(str(path)).toPandas()


def get_history(table_name_or_path: str | Path) -> pd.DataFrame:
    """Returns the Delta transaction log as a DataFrame — real version
    history (Section 8: 'real transaction log, real time-travel'),
    useful to show directly in a viva."""
    spark = get_spark()
    path = Path(table_name_or_path)
    if not path.is_absolute() and path.parent == Path("."):
        path = config.SILVER_DIR / path
    return DeltaTable.forPath(spark, str(path)).history().toPandas()