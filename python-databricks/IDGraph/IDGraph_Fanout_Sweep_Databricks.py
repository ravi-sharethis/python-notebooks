# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — MAX_FANOUT singleton sweep (Spark, no GraphFrames, no CC)
# MAGIC
# MAGIC Predicts the **singleton count for every possible MAX_FANOUT at once**,
# MAGIC without rebuilding edges or re-running `connectedComponents()`.
# MAGIC
# MAGIC A node gets at least one edge iff at least one of its grouping keys (across
# MAGIC id5/tapad/liid) has a group size `n_distinct` in `[2, MAX_FANOUT]`.
# MAGIC STAR_THRESHOLD never matters here -- star and clique both connect every
# MAGIC member of a surviving group. So per node, `min_group_size` = the smallest
# MAGIC `n_distinct >= 2` among its keys, and:
# MAGIC
# MAGIC     singletons(F) = total_nodes - #{nodes with min_group_size <= F}
# MAGIC
# MAGIC Nodes with no `min_group_size` at all (every key is a size-1 group, or no
# MAGIC key) are singletons at ANY fanout -- the irreducible baseline.
# MAGIC
# MAGIC **Inputs** (all from the build notebook, none depend on MAX_FANOUT/
# MAGIC STAR_THRESHOLD, so one run of this notebook covers every config):
# MAGIC `{via}_filtered_base`, `{via}_fanout`, `node_types`.
# MAGIC
# MAGIC **Only predicts singletons**, not giant-component size -- that still needs
# MAGIC a real CC run. Use this to choose which config is worth that run.

# COMMAND ----------

# Spark settings used for every IDGraph Databricks notebook.
spark.conf.set("spark.sql.shuffle.partitions", "20000")
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")

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

import pandas as pd

# Any build dir works -- filtered_base/fanout/node_types don't depend on MAX_FANOUT/
# STAR_THRESHOLD, so IDGraph4 (32/10) and IDGraph5 (16/6) should give identical
# results. The KNOWN_SINGLETONS cross-check below validates both configs from either.
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
BUILD_OUTPUT_PATH = f"{GRAPH_ROOT}/build_output"
# node_types lives only in IDGraph3 -- IDGraph4/5 builds reused it rather than rebuilding.
NODE_TYPES_PATH = "s3://data-science-research/ravi/IDGraph3/build_output/node_types"
SWEEP_OUTPUT_PATH = f"{GRAPH_ROOT}/fanout_sweep"

VIA_KEY_COLS = [
    ("id5", "individual_id5"),
    ("tapad", "individual_id_tapad"),
    ("liid", "liid"),
]

# Must match DENYLIST in IDGraph_Build_Databricks.py -- the build applies it AFTER
# filtered_base is staged, so the staged filtered_base still contains these rows.
# If the two drift apart, the F=16/F=32 cross-check below will stop matching.
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}

# Fanout values to report in the sweep table (None = no cap at all).
SWEEP_FANOUTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64, 96, 128, 256, None]

# Known CC-measured singleton counts -- the sweep must reproduce these exactly.
KNOWN_SINGLETONS = {32: 1_228_969_090, 16: 2_418_307_483}
KNOWN_TOTAL_NODES = 4_914_094_515

# COMMAND ----------

# MAGIC %md ## Input check
# MAGIC Fail fast before the expensive join if any input is missing.

# COMMAND ----------

_inputs = [NODE_TYPES_PATH] + [
    f"{BUILD_OUTPUT_PATH}/{via_label}_{suffix}"
    for via_label, _ in VIA_KEY_COLS
    for suffix in ("filtered_base", "fanout")
]
_missing = [p for p in _inputs if not _path_exists(p)]
if _missing:
    raise RuntimeError("missing sweep inputs:\n  " + "\n  ".join(_missing))

# COMMAND ----------

# MAGIC %md ## Per-node minimum surviving group size
# MAGIC One row per node that has at least one key with `n_distinct >= 2`.
# MAGIC Nodes absent from this table are baseline singletons (never linkable).

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def node_min_group_size_one_via_fn(via_label, key_col, spark=None, write_path=None):
    """Per-via_type min group size, staged on its own so the final cross-via
    groupBy reads materialized (deterministic) input, and a failure only redoes
    whichever via_type hadn't finished."""
    b = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_filtered_base")
    deny = DENYLIST.get(via_label, set())
    if deny:
        b = b.filter(~F.col(key_col).isin(list(deny)))
    fanout = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_fanout").filter(
        F.col("n_distinct") >= 2
    )
    return (
        b.join(fanout, on=key_col)
        .groupBy("node_id", "type")
        .agg(F.min("n_distinct").alias("group_size"))
    )


@stage_dataframe(write_format="parquet")
def node_min_group_size_fn(parts, spark=None, write_path=None):
    unioned = parts[0]
    for p in parts[1:]:
        unioned = unioned.unionByName(p)
    return unioned.groupBy("node_id", "type").agg(F.min("group_size").alias("min_group_size"))


node_min_group_size_parts = [
    node_min_group_size_one_via_fn(
        via_label, key_col, spark=spark, write_path=f"{SWEEP_OUTPUT_PATH}/node_min_group_size_{via_label}"
    )
    for via_label, key_col in VIA_KEY_COLS
]
node_min_group_size = node_min_group_size_fn(
    node_min_group_size_parts, spark=spark, write_path=f"{SWEEP_OUTPUT_PATH}/node_min_group_size"
)

# COMMAND ----------

# MAGIC %md ## Histogram of min_group_size, by type
# MAGIC Small (types x distinct sizes) -- safe to pull into pandas. Totals come from
# MAGIC `node_types` directly rather than a 4.9B x 4.9B left join; baseline
# MAGIC singletons = total - linkable.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def min_group_size_histogram_fn(min_df, spark=None, write_path=None):
    return min_df.groupBy("type", "min_group_size").agg(F.count("*").alias("n_nodes"))


@stage_dataframe(write_format="parquet")
def node_totals_by_type_fn(spark=None, write_path=None):
    return (
        spark.read.parquet(NODE_TYPES_PATH)
        .groupBy("type")
        .agg(F.count("*").alias("n_nodes"))
    )


min_group_size_histogram = min_group_size_histogram_fn(
    node_min_group_size, spark=spark, write_path=f"{SWEEP_OUTPUT_PATH}/min_group_size_histogram"
)
node_totals_by_type = node_totals_by_type_fn(
    spark=spark, write_path=f"{SWEEP_OUTPUT_PATH}/node_totals_by_type"
)

hist_pd = min_group_size_histogram.toPandas()
totals_pd = node_totals_by_type.toPandas().set_index("type")["n_nodes"]
total_nodes = int(totals_pd.sum())
print(f"total nodes: {total_nodes:,}")
if total_nodes != KNOWN_TOTAL_NODES:
    print(f"WARNING: expected {KNOWN_TOTAL_NODES:,} nodes -- wrong node_types path, or a SAMPLE=True build?")

# COMMAND ----------

# MAGIC %md ## Singleton sweep over MAX_FANOUT

# COMMAND ----------


def singleton_sweep(hist, totals, fanouts):
    rows = []
    for f in fanouts:
        h = hist if f is None else hist[hist["min_group_size"] <= f]
        linked = h.groupby("type")["n_nodes"].sum().reindex(totals.index, fill_value=0)
        singletons = totals - linked
        row = {"max_fanout": "none" if f is None else f, "singletons": int(singletons.sum())}
        row["pct_singleton"] = round(100 * row["singletons"] / totals.sum(), 2)
        for t in totals.index:
            row[f"pct_{t}"] = round(100 * singletons[t] / totals[t], 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("max_fanout")


sweep = singleton_sweep(hist_pd, totals_pd, SWEEP_FANOUTS)
baseline = int(sweep.loc["none", "singletons"])
sweep["fanout_stranded"] = sweep["singletons"] - baseline

print("--- predicted singletons by MAX_FANOUT (pct_<type> = % of that type's nodes) ---")
print(sweep.to_string())
print()
print(f"baseline (singleton at any fanout): {baseline:,} ({100 * baseline / total_nodes:.2f}%)")

# Cross-check against the real CC runs.
for f, expected in KNOWN_SINGLETONS.items():
    got = int(sweep.loc[f, "singletons"])
    status = "OK" if got == expected else f"MISMATCH (diff {got - expected:+,})"
    print(f"F={f}: predicted {got:,} vs CC-measured {expected:,} -- {status}")

# COMMAND ----------

# MAGIC %md ## Full per-size table (for charting)
# MAGIC Every integer fanout from 2 to the largest observed group size is too many
# MAGIC rows past a few thousand, so this goes to 1024 -- beyond that the sweep
# MAGIC table's 256/none rows cover it.

# COMMAND ----------

full_sweep = singleton_sweep(hist_pd, totals_pd, list(range(2, 1025)) + [None])
print(full_sweep[["singletons", "pct_singleton"]].to_csv())
