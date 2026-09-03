"""
Spark session management — Option D (local-mode PySpark + Delta Lake OSS).

Single shared SparkSession, configured with the Delta Lake extensions and
catalog, running entirely in local[*] mode: no HDFS, no cluster manager,
no networked nodes. This is the piece that makes the report's "Apache
Spark" and "Delta Lake" claims literally true in code (Section 8 of the
Platform Selection Decision Report) while keeping setup to `pip install`.

Every other module (ingestion, silver) imports get_spark() rather than
constructing its own session, so the whole pipeline shares one JVM.
"""

import os
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

import config

# Windows fixes:
# 1. Spark's RPC URL parser chokes on hostnames with underscores
#    (Windows machine names often have them) -> pin to loopback.
# 2. Without an explicit PYSPARK_PYTHON, Spark can spawn its worker
#    subprocess using a different Python install than this venv,
#    which crashes with "Python worker exited unexpectedly".
#    sys.executable is always this exact interpreter, so this is
#    correct no matter which machine/venv path runs it.
# 3. If the project path itself contains spaces (e.g. "FINAL YEAR
#    PROJ"), Spark launches the worker without quoting the path,
#    so Windows splits it on the spaces and tries to run a
#    nonexistent command (observed as "'YEAR' is not recognized as
#    an internal or external command"), silently killing the worker
#    before it can respond -> resolve to the space-free Windows 8.3
#    short path name instead, which sidesteps the quoting bug
#    entirely regardless of where the repo lives on disk.
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def _spark_safe_python_path(path: str) -> str:
    """Returns a space-free path for use in PYSPARK_PYTHON on Windows.

    Falls back to the original path on any failure (non-Windows,
    short names disabled on the volume, etc.) so this never breaks
    startup outright -- it just reintroduces the original bug.
    """
    if sys.platform != "win32":
        return path
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 260):
            return buf.value
    except Exception:
        pass
    return path


_python_exe = _spark_safe_python_path(sys.executable)
os.environ.setdefault("PYSPARK_PYTHON", _python_exe)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", _python_exe)

_spark: SparkSession | None = None


def get_spark() -> SparkSession:
    """Returns the shared local-mode Spark session, creating it on first use."""
    global _spark
    if _spark is not None:
        return _spark

    builder = (
        SparkSession.builder.appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", str(config.DATA_ROOT / "_spark_warehouse"))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # Disabled: spark.createDataFrame(pandas_df) (used for .xlsx/.json/
        # .xml/.pdf ingestion) uses Arrow-based serialization by default,
        # which is sensitive to the installed pyarrow version matching what
        # this PySpark build expects. A too-new/unpinned pyarrow here has
        # been observed to silently kill the Python worker mid-conversion
        # on Windows (seen as "Python worker exited unexpectedly" /
        # EOFException with no readable error). Kept disabled even now that
        # ingestion.loaders routes pandas->Spark through a Parquet file
        # (see _pdf_to_spark) instead of createDataFrame -- other code
        # paths may still call createDataFrame directly, and this stays
        # the safe default until pyarrow is verified compatible.
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    )
    # configure_spark_with_delta_pip wires in the correct delta-spark JAR
    # for the pyspark version installed, so no manual --packages flag or
    # cluster-side JAR management is needed (Option D's whole point).
    _spark = configure_spark_with_delta_pip(builder).getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")
    return _spark


def stop_spark() -> None:
    """Stops the shared session. Mainly useful for tests / clean shutdown."""
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
