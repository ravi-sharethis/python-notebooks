# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — connected components (Spark/GraphFrames)
# MAGIC
# MAGIC Runs after the build script. Reads `node_types`/`edges`, runs
# MAGIC `connectedComponents()`, and writes `result` (node/type/component/
# MAGIC component_size). **This notebook does ONLY that** -- for all downstream
# MAGIC analysis (per-type summary, type matrix, histograms, component-size
# MAGIC diagnostics, point-query helpers), see `IDGraph_CC_Analysis_Databricks.py`,
# MAGIC which reads this notebook's `result` output and does NOT need GraphFrames
# MAGIC installed at all -- splitting it out means the expensive, failure-prone
# MAGIC compute step only needs to run once (or whenever MAX_FANOUT/STAR_THRESHOLD/
# MAGIC the underlying edges change), while analysis can be iterated on freely
# MAGIC without ever touching this step again (`stage_dataframe` skips it anyway
# MAGIC if `result` already exists, but keeping them in separate notebooks means
# MAGIC you don't even need GraphFrames attached to the cluster you use for
# MAGIC day-to-day analysis).
# MAGIC
# MAGIC **Every join/groupBy/union goes through `@stage_dataframe`** (from
# MAGIC `staging_df.py`, pasted inline below): each stage reads from and writes to its
# MAGIC own path under `IDGraph3`, so a cluster restart mid-run re-reads already-
# MAGIC completed stages instead of recomputing them -- the same problem that made the
# MAGIC build-phase self-joins so painful to re-run after every restart.
# MAGIC
# MAGIC **Setup**: `%pip install --force-reinstall graphframes-py==0.12.1` --
# MAGIC confirmed working on Databricks Runtime 17.3 LTS (Spark 4.0.0 / Scala 2.13).
# MAGIC This resolves the GraphFrames/Spark-4.0/Scala-2.13 compatibility that was
# MAGIC an open, unverified risk earlier in the project -- the pip-installable
# MAGIC `graphframes-py` package (vs. the older Maven-coordinate-JAR approach)
# MAGIC is what actually works on this runtime.

# COMMAND ----------

# MAGIC %md ## `staging_df.py`, pasted inline

# COMMAND ----------


def _path_exists(path):
    try:
        dbutils.fs.ls(path)
        print(f"{path} exists")
        return True
    except Exception:
        return False


import logging
from functools import wraps
from typing import List, Literal, Optional

from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

logger = logging.getLogger(__name__)


def spark_read_parquet_or_delta(path: str, spark: Optional[SparkSession] = None) -> DataFrame:
    if spark is None:
        spark = SparkSession.builder.appName("SparkReader").getOrCreate()

    if not DeltaTable.isDeltaTable(spark, path):
        return spark.read.parquet(path)
    return spark.read.format("delta").load(path)


def check_if_dataframe_exists(
    path: str, spark: SparkSession, write_format: Literal["parquet, delta"] = "delta"
) -> bool:
    if write_format not in ["parquet", "delta"]:
        raise ValueError("Expected 'delta' or 'parquet'")

    try:
        _ = spark_read_parquet_or_delta(path, spark)
        return True
    except Exception:
        return False


def stage_dataframe(
    partitionby: Optional[List[str]] = None,
    mode: Optional[Literal["overwrite", "append"]] = "overwrite",
    compression: Optional[Literal["snappy", "gzip"]] = "snappy",
    write_format: Optional[Literal["parquet", "delta"]] = "parquet",
    repartition: Optional[int] = None,
    unpersist_result_after_write: bool = False,
):
    """
    Stage a Spark DataFrame and return the saved data.

    - overwrite: reuse existing readable data without executing the function.
    - append: execute and append, then return the full saved dataset.
    - No write_path: execute without writing (spark is still required).
    - unpersist_result_after_write: release the produced DataFrame after
      a write attempt, whether the write succeeds or fails.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            write_path = kwargs.get("write_path")
            spark = kwargs.get("spark")

            # Find Spark session from args or self._spark.
            if not spark:
                for arg in args:
                    if isinstance(arg, SparkSession):
                        spark = arg
                        break
                    if hasattr(arg, "_spark") and isinstance(arg._spark, SparkSession):
                        spark = arg._spark
                        break
            if not spark:
                raise RuntimeError(
                    "[stage_dataframe] SparkSession not found. "
                    "Pass it explicitly or ensure the class has `self._spark`."
                )

            def write_produced_dataframe(produced_df, save_mode):
                primary_error = None
                try:
                    write_df = produced_df
                    if repartition:
                        write_df = produced_df.repartition(repartition)
                        logger.debug(
                            f"=-=-=- Repartitioned DataFrame to {repartition} partition(s)."
                        )

                    writer = (
                        write_df.write.format(write_format)
                        .mode(save_mode)
                        .option("compression", compression)
                    )
                    if partitionby:
                        writer = writer.partitionBy(*partitionby)
                    writer.save(write_path)
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    if unpersist_result_after_write:
                        try:
                            produced_df.unpersist()
                        except Exception:
                            if primary_error is None:
                                raise
                            logger.exception(
                                "Failed to unpersist a staged DataFrame after "
                                "the primary write operation failed."
                            )

            if write_path:
                path_exists = check_if_dataframe_exists(write_path, spark, write_format=write_format)

                if path_exists:
                    if mode == "overwrite":
                        logger.info(
                            f"=-=-=- File exists at {write_path}. Skipping execution and reading existing DataFrame."
                        )
                        return spark_read_parquet_or_delta(write_path, spark)

                    elif mode == "append":
                        logger.info(
                            f"=-=-=- File exists at {write_path}. Running {func.__name__} and appending results."
                        )
                        produced_df = func(*args, **kwargs)
                        write_produced_dataframe(produced_df, "append")
                        return spark_read_parquet_or_delta(write_path, spark)

                logger.debug(
                    f"=-=-=- Path does not exist. Running {func.__name__} and writing to {write_path}."
                )
                produced_df = func(*args, **kwargs)
                write_produced_dataframe(produced_df, mode)
                logger.info(f"Data successfully written to {write_path}.")
                return spark_read_parquet_or_delta(write_path, spark)

            logger.debug(f"=-=-=- No write_path provided. Running {func.__name__} without writing.")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

BUILD_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph3/build_output/"
CC_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph3/cc_output/"
CHECKPOINT_PATH = "s3://data-science-research/ravi/IDGraph3/gf_checkpoints/"
RESULT_PATH = f"{CC_OUTPUT_PATH}/result"

spark.sparkContext.setCheckpointDir(CHECKPOINT_PATH)

# COMMAND ----------

# MAGIC %md ## Load build output, build the GraphFrame
# MAGIC Plain reads + renames -- not staged, since these aren't new computations,
# MAGIC just reading/relabeling what the build script already produced.

# COMMAND ----------

node_types = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/node_types")
edges = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/edges")

vertices = node_types.withColumnRenamed("node_id", "id")
edges_gf = edges.withColumnRenamed("node_a", "src").withColumnRenamed("node_b", "dst")

g = GraphFrame(vertices, edges_gf)
print(f"{vertices.count():,} vertices, {edges_gf.count():,} edges")

# COMMAND ----------

# MAGIC %md ## Connected components
# MAGIC The most expensive single step here (an iterative distributed algorithm) --
# MAGIC definitely worth staging so a restart doesn't force a full re-run.
# MAGIC
# MAGIC **`MetadataFetchFailedException` / executor-loss fix**: the original
# MAGIC MAX_FANOUT=64/STAR_THRESHOLD=20 config (16.95B edges) failed with an
# MAGIC executor becoming completely unreachable mid-shuffle -- almost certainly an
# MAGIC OOM kill, likely from GraphFrames' default `broadcastThreshold=1000000`
# MAGIC attempting a broadcast join that blew up under skew. `broadcastThreshold=-1`
# MAGIC disables broadcasting entirely, forcing plain shuffle joins throughout.
# MAGIC Combined with the config being dialed back to MAX_FANOUT=32/STAR_THRESHOLD=10
# MAGIC (6.65B edges), the rerun succeeded -- exactly which of the two fixed it
# MAGIC (broadcastThreshold, or just less data) isn't disambiguated yet.

# COMMAND ----------

# Tried clearing cached state before the CC call, in case stale cached
# DataFrames/Databricks IO cache were contributing to the OOM -- didn't turn
# out to be needed (the broadcastThreshold + config changes above were
# sufficient), left here commented as a "tried this, didn't need it" marker
# rather than deleted, in case it's worth revisiting on a future larger run.
# spark.catalog.clearCache()
# spark.conf.set("spark.databricks.io.cache.enabled", "false")


@stage_dataframe(write_format="parquet")
def connected_components_fn(graph, spark=None, write_path=None):
    return graph.connectedComponents(
        algorithm="graphframes", checkpointInterval=2, broadcastThreshold=-1
    )


components = connected_components_fn(
    g, spark=spark, write_path=f"{CC_OUTPUT_PATH}/components"
)


# Delta, not partitioned by `component` -- that column is extremely high-cardinality
# (tens of millions of distinct values), and partitioning Delta by a high-cardinality
# column creates millions of tiny files, worse for read/write than not partitioning at
# all. Z-ORDER (below, a separate OPTIMIZE step) is the right tool for fast point-
# lookups on a high-cardinality key instead.
@stage_dataframe(write_format="delta")
def result_with_size_fn(components_df, spark=None, write_path=None):
    component_sizes = (
        components_df.groupBy("component").count().withColumnRenamed("count", "component_size")
    )
    return components_df.join(component_sizes, on="component")


result = result_with_size_fn(components, spark=spark, write_path=RESULT_PATH)

# One-time optimization for the point-query helpers / analysis notebook -- clusters
# rows by `component` within files so a lookup skips most of the table instead of
# scanning all ~4B rows.
spark.sql(f"OPTIMIZE delta.`{RESULT_PATH}` ZORDER BY (component)")

n_components = result.select("component").distinct().count()
print(f"{n_components:,} components over {result.count():,} nodes")

# COMMAND ----------

# MAGIC %md ## Done -- next steps
# MAGIC `result` is written to `RESULT_PATH` (Delta, Z-ORDERed by `component`).
# MAGIC All downstream analysis -- per-type summary, type matrix, histograms,
# MAGIC component-size diagnostics, point-query helpers -- lives in
# MAGIC `IDGraph_CC_Analysis_Databricks.py`, which reads this output directly and
# MAGIC does not need GraphFrames or this notebook re-run.
