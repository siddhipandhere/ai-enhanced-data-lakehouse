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
        raise ValueError(
            f"Schema enforcement failed, missing columns: {missing}")
    return df.select(*required_columns)


def _coerce_numeric_like(df: DataFrame) -> DataFrame:
    """Mirrors the old pandas heuristic: string columns that are >90% numeric
    get cast to double, in Spark rather than pandas.

    Two robustness fixes over the original version, both found against
    the same real Flipkart export:

    1. The ratio is computed against the column's own NON-NULL/NON-BLANK
       count, not the total row count. A real e-commerce column is often
       legitimately blank for a share of rows -- against the total row
       count that blank share counts against the column and can keep a
       genuinely-numeric field from ever crossing the 90% bar, even
       though every value that IS present is a clean number.
    2. Thousands-separator commas ("2,999") are stripped before the
       numeric check and the cast. Without this, a price column
       formatted with commas never matches "is this numeric" at all and
       is left as a string all the way to Gold -- which is exactly why
       aggregations like "average selling_price" (SQL Explorer, the
       Overview charts) were failing with a 400: pandas can't average a
       column that's still text.
    """
    from pyspark.sql.functions import count, regexp_replace, trim, when

    total = df.count()
    if total == 0:
        return df

    for field in df.schema.fields:
        if str(field.dataType) != "StringType()":
            continue
        c = col(field.name)
        present = c.isNotNull() & (trim(c) != "")
        non_null = df.select(count(when(present, field.name))).collect()[0][0]
        if not non_null:
            continue
        no_commas = regexp_replace(c, ",", "")
        is_numeric = no_commas.rlike(r"^-?\d+(\.\d+)?$")
        numeric_count = df.select(
            count(when(present & is_numeric, field.name))).collect()[0][0]
        if (numeric_count / non_null) > 0.9:
            df = df.withColumn(field.name, no_commas.cast("double"))
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

    Requires EXACTLY ONE '%' in the value (not just a leading "NN%") AND
    that the WHOLE value be short (<= MAX_LEN characters). Two different
    real-world false positives showed up in testing against actual
    Flipkart data, and one condition alone doesn't catch both:

    1. A composition/ingredient list -- "80% Cotton, 20% Elastane,
       breathable fit..." -- starts with "80%" and would otherwise
       derive a spurious percentage from whichever material happened to
       be listed first. This has 2+ '%' signs (the parts of a whole),
       so the single-'%' check alone rules it out.
    2. A long product description that opens with a one-off promo
       banner -- e.g. "10% Instant Discount on Kotak Bank Cards. This
       premium cotton shirt features..." -- has only ONE '%' sign, so it
       survives the single-'%' check, but the number has nothing to do
       with the product's actual discount. What distinguishes it from a
       genuine discount field ("69% off", "Flat 20% off") is length: a
       real discount/rate label is always short, while a description is
       a full sentence or paragraph. Capping the WHOLE value's length
       catches this case regardless of how many '%' signs appear.

    A genuine discount/rate field satisfies both conditions at once;
    neither a composition list nor a promo-prefixed description does.
    This stays fully dataset-agnostic -- no column-name keywords.

    Like _coerce_numeric_like(), the match rate is measured against the
    column's own NON-NULL/NON-BLANK count, not the total row count. A
    real "discount" field is routinely blank for every product that
    isn't on sale -- against the total row count that can keep the rate
    under the 50% bar even when literally every non-blank value is a
    clean "NN%" discount label. This was confirmed against a real
    Flipkart export: with the total-row-count denominator, 'discount'
    never got derived at all (its non-blank values were all valid, but
    diluted below 50% by blanks), while an unrelated 'description'
    column -- which has no blanks -- cleared the bar and got derived
    instead. Fixing the denominator fixes this directly.
    """
    from pyspark.sql.functions import count, length, regexp_extract, regexp_replace, trim, when

    total = df.count()
    if total == 0:
        return df

    # generous for "Flat 70% off on MRP today!" (27 chars), too short for a sentence
    MAX_LEN = 30
    pct_pattern = r"^\s*(-?\d+(?:\.\d+)?)\s*%"
    near_misses = []
    for field in df.schema.fields:
        if str(field.dataType) != "StringType()":
            continue
        c = col(field.name)
        present = c.isNotNull() & (trim(c) != "")
        non_null = df.select(count(when(present, field.name))).collect()[0][0]
        if not non_null:
            continue
        single_pct = (length(c) - length(regexp_replace(c, "%", ""))) == 1
        is_short = length(c) <= MAX_LEN
        is_candidate = present & c.rlike(pct_pattern) & single_pct & is_short
        match_count = df.select(
            count(when(is_candidate, field.name))).collect()[0][0]
        rate = match_count / non_null
        if rate > 0.5:
            derived_col = f"{field.name}_pct"
            df = df.withColumn(
                derived_col,
                when(is_candidate, regexp_extract(
                    c, pct_pattern, 1).cast("double")).otherwise(None),
            )
            logger.info(f"Derived numeric column '{derived_col}' from '{field.name}' "
                        f"({match_count}/{non_null} non-blank values matched a short, single 'NN%' value)")
        elif match_count > 0:
            near_misses.append(
                f"{field.name} ({match_count}/{non_null} = {rate:.0%})")
    if near_misses:
        logger.info(f"Percentage-pattern columns that did NOT clear the 50% threshold "
                    f"(no '_pct' column derived for these): {near_misses}")
    return df


_ORIGINAL_PRICE_NAMES = {"actual_price",
                         "original_price", "mrp", "list_price", "listed_price"}
_SOLD_PRICE_NAMES = {"selling_price", "sale_price",
                     "discounted_price", "final_price", "current_price"}


def _derive_price_based_discount(df: DataFrame) -> DataFrame:
    """
    A second, independent source for a discount percentage: when the
    dataset carries a real "before" and "after" price pair (e.g.
    Flipkart's actual_price vs. selling_price), compute
    discount_pct = (before - after) / before * 100 directly from the
    numbers instead of relying on parsing a free-text "discount" field
    at all.

    This is strictly more reliable than text-parsing wherever it
    applies: a price pair, once cleaned to numbers, is unambiguous,
    whereas a "discount" field's text format varies dataset to dataset
    and (see _extract_percentage_columns above) can be too sparse for
    a match-rate heuristic to trust. Only runs when a 'discount_pct'
    column doesn't already exist, and only when a recognized price pair
    is both present and numeric-parseable -- a no-op otherwise, so this
    stays safe for datasets without this shape.
    """
    from pyspark.sql.functions import regexp_replace, when

    if "discount_pct" in df.columns:
        return df

    cols_lower = {c.lower(): c for c in df.columns}
    before_col = next(
        (cols_lower[n] for n in _ORIGINAL_PRICE_NAMES if n in cols_lower), None)
    after_col = next(
        (cols_lower[n] for n in _SOLD_PRICE_NAMES if n in cols_lower), None)
    if not before_col or not after_col:
        return df

    def _numeric(colname: str):
        cleaned = regexp_replace(col(colname).cast("string"), r"[^0-9.\-]", "")
        return when(cleaned == "", None).otherwise(cleaned.cast("double"))

    before = _numeric(before_col)
    after = _numeric(after_col)
    df = df.withColumn(
        "discount_pct",
        when(before.isNotNull() & after.isNotNull() & (before > 0),
             ((before - after) / before) * 100.0).otherwise(None),
    )
    logger.info(f"Derived 'discount_pct' from price columns '{before_col}' and '{after_col}' "
                f"as (before-after)/before * 100 -- computed from prices rather than text-parsed")
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
    df = _derive_price_based_discount(df)
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
            raise ValueError(
                f"Join key '{join_key}' not present in both sources")
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
