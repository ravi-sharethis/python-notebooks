# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — connected components + summary stats (Spark/GraphFrames)
# MAGIC
# MAGIC Runs after the build script. Reads `node_types`/`edges` and does everything
# MAGIC the local notebook's `run_cc()` + per-type-summary + type-matrix cells do, but
# MAGIC on Spark -- the full-scale graph (~4B nodes, implied by node_types' ~186GB)
# MAGIC does not fit in one machine's RAM the way scipy's `connected_components` needs.
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
# MAGIC
# MAGIC **Output** (small, safe to download in full): `type_summary`, `type_matrix`,
# MAGIC `weight_histogram`, `via_type_count_histogram`.
# MAGIC **Stays on Databricks** (too big to bring down): `result` -- the full
# MAGIC node_id/type/component/component_size table, ~4B rows. Point-query lookups
# MAGIC need to filter this in place, not assume it fits in local pandas.

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

spark.sparkContext.setCheckpointDir(CHECKPOINT_PATH)

FILTER_TYPES = ["HARDWARE_IDFA", "HARDWARE_ANDROID_AD_ID", "HEM_MD5", "ID5_UID", "TTD", "ST"]

# COMMAND ----------

# MAGIC %md ## Load build output, build the GraphFrame
# MAGIC Plain reads + renames -- not staged, since these aren't new computations,
# MAGIC just reading/relabeling what the build script already produced.

# COMMAND ----------

node_types = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/node_types")
edges = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/edges")

vertices = node_types.withColumnRenamed("node_id", "id")
edges_gf = edges.withColumnRenamed("node_a", "src").withColumnRenamed("node_b", "dst")
type_map = vertices.select("id", "type")

g = GraphFrame(vertices, edges_gf)
print(f"{vertices.count():,} vertices, {edges_gf.count():,} edges")

# COMMAND ----------

# MAGIC %md ## Node counts by type, edge counts by via_type
# MAGIC `via_type` isn't a column on `edges` (edges_fn collapses it into
# MAGIC via_type_count during aggregation) -- getting edge counts by via_type means
# MAGIC reading the per-via_type `pairs` tables the build script staged, not something
# MAGIC derivable from `edges` alone. Two numbers per via_type, not one: `pair_rows` is
# MAGIC the raw row count (can double-count a pair if multiple keys of that via_type
# MAGIC link it), `distinct_edges` is how many distinct (node_a,node_b) pairs that
# MAGIC via_type contributes to at least once -- the gap between them tells you how
# MAGIC much repeat-linking is happening within a single via_type.

# COMMAND ----------

node_counts_by_type = node_types.groupBy("type").count().withColumnRenamed("count", "n_nodes")
print("--- node counts by type ---")
node_counts_by_type.orderBy(F.desc("n_nodes")).show(truncate=False)

all_pairs = spark.read.parquet(
    f"{BUILD_OUTPUT_PATH}/id5_pairs", f"{BUILD_OUTPUT_PATH}/tapad_pairs", f"{BUILD_OUTPUT_PATH}/liid_pairs"
)


@stage_dataframe(write_format="parquet")
def edge_counts_by_via_type_fn(all_pairs_df, spark=None, write_path=None):
    pair_rows = all_pairs_df.groupBy("via_type").count().withColumnRenamed("count", "pair_rows")
    distinct_edges = (
        all_pairs_df.select("node_a", "node_b", "via_type")
        .distinct()
        .groupBy("via_type")
        .count()
        .withColumnRenamed("count", "distinct_edges")
    )
    return pair_rows.join(distinct_edges, on="via_type")


edge_counts_by_via_type = edge_counts_by_via_type_fn(
    all_pairs, spark=spark, write_path=f"{CC_OUTPUT_PATH}/edge_counts_by_via_type"
)
print("--- edge counts by via_type ---")
edge_counts_by_via_type.orderBy(F.desc("pair_rows")).show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Node distribution by type and via_type: singletons vs. plurals
# MAGIC `via_keys` only ever contains a row for a node if it survived into a real
# MAGIC (>=2-member, non-excluded) group -- a node with ZERO via_keys rows across
# MAGIC all 3 via_types can have no edges at all, and a node with >=1 via_keys row
# MAGIC is guaranteed at least one edge. So "has a via_keys row" and "is a CC
# MAGIC singleton" are two different measurements of the EXACT SAME population --
# MAGIC this section derives singleton counts independently from via_keys (no
# MAGIC dependency on connectedComponents() having run) and cross-checks them
# MAGIC against the CC-derived singleton counts later, in the per-type summary
# MAGIC section below.
# MAGIC
# MAGIC "Plural" nodes (>=1 via_keys row) are broken down two ways: by (type,
# MAGIC via_type) -- a node can appear in more than one via_type's bucket here,
# MAGIC this isn't a partition -- and by (type, n_via_type), which IS a partition
# MAGIC (every plural node has exactly one n_via_type value: 1, 2, or 3).

# COMMAND ----------

via_keys = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/via_keys")


@stage_dataframe(write_format="parquet")
def plural_by_type_via_type_fn(via_keys_df, type_map_df, spark=None, write_path=None):
    return (
        via_keys_df.select("node_id", "via_type")
        .distinct()
        .join(type_map_df.withColumnRenamed("id", "node_id"), on="node_id")
        .groupBy("type", "via_type")
        .agg(F.countDistinct("node_id").alias("n_plural_nodes"))
    )


plural_by_type_via_type = plural_by_type_via_type_fn(
    via_keys, type_map, spark=spark, write_path=f"{CC_OUTPUT_PATH}/plural_by_type_via_type"
)
print("--- plural nodes by type and via_type (a node can count under >1 via_type) ---")
plural_by_type_via_type.orderBy("type", "via_type").show(30, truncate=False)


@stage_dataframe(write_format="parquet")
def plural_by_type_n_via_type_fn(via_keys_df, type_map_df, spark=None, write_path=None):
    per_node_n_via_type = via_keys_df.groupBy("node_id").agg(
        F.countDistinct("via_type").alias("n_via_type")
    )
    return (
        per_node_n_via_type.join(type_map_df.withColumnRenamed("id", "node_id"), on="node_id")
        .groupBy("type", "n_via_type")
        .count()
        .withColumnRenamed("count", "n_nodes")
    )


plural_by_type_n_via_type = plural_by_type_n_via_type_fn(
    via_keys, type_map, spark=spark, write_path=f"{CC_OUTPUT_PATH}/plural_by_type_n_via_type"
)
print("--- plural nodes by type and n_via_type (partition -- sums to total plural nodes per type) ---")
plural_by_type_n_via_type.orderBy("type", "n_via_type").show(30, truncate=False)


@stage_dataframe(write_format="parquet")
def singleton_by_type_from_via_keys_fn(node_types_df, via_keys_df, spark=None, write_path=None):
    plural_node_ids = via_keys_df.select("node_id").distinct()
    return (
        node_types_df.join(plural_node_ids, on="node_id", how="left_anti")
        .groupBy("type")
        .count()
        .withColumnRenamed("count", "n_singleton_from_via_keys")
    )


singleton_by_type_from_via_keys = singleton_by_type_from_via_keys_fn(
    node_types, via_keys, spark=spark, write_path=f"{CC_OUTPUT_PATH}/singleton_by_type_from_via_keys"
)
print("--- singleton nodes by type, derived independently from via_keys (no CC dependency) ---")
singleton_by_type_from_via_keys.orderBy("type").show(truncate=False)

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


RESULT_PATH = f"{CC_OUTPUT_PATH}/result"
result = result_with_size_fn(components, spark=spark, write_path=RESULT_PATH)
result = result.withColumn("is_singleton", F.col("component_size") == 1)

# One-time optimization for the point-query helpers below (view_component/
# component_of_node) -- clusters rows by `component` within files so a lookup skips
# most of the table instead of scanning all ~4B rows.
spark.sql(f"OPTIMIZE delta.`{RESULT_PATH}` ZORDER BY (component)")

n_components = result.select("component").distinct().count()
print(f"{n_components:,} components over {result.count():,} nodes")

# COMMAND ----------

# MAGIC %md ## Component-size diagnostics
# MAGIC The IDGraph2 sample run (see chat) surfaced a giant component swallowing ~32%
# MAGIC of all sampled nodes -- this is the top validation priority for whether
# MAGIC MAX_FANOUT=64 / STAR_THRESHOLD=20 actually curb that, or whether it persists via
# MAGIC chains of individually-small groups. Run these against IDGraph3's result and
# MAGIC compare against the IDGraph2 numbers.
# MAGIC
# MAGIC **Caution on `component_size_histogram_fn` vs. `per_type_component_size_histogram_fn`**:
# MAGIC these are NOT interchangeable, and summing the per-type one over type does NOT
# MAGIC give the global one -- a single component with, say, 1 TTD node and 1 HEM_MD5
# MAGIC node contributes ONE row (size=2) to the global histogram, but TWO rows
# MAGIC (TTD count=1, HEM_MD5 count=1) to the per-type histogram. Conflating them
# MAGIC produced a misleading "overall" table earlier in chat -- ALWAYS derive the
# MAGIC global histogram from `result_df` deduped on `(component, component_size)`
# MAGIC directly, never by summing the per-type table over type.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def component_size_histogram_fn(result_df, spark=None, write_path=None):
    """TRUE global component-size distribution: one row per component, deduped
    before counting, so each component is counted exactly once regardless of how
    many types/nodes it contains."""
    return result_df.select("component", "component_size").distinct().groupBy("component_size").count()


component_size_histogram = component_size_histogram_fn(
    result, spark=spark, write_path=f"{CC_OUTPUT_PATH}/component_size_histogram"
)
component_size_histogram.orderBy("component_size").show(300)


@stage_dataframe(write_format="parquet")
def per_type_component_size_histogram_fn(result_df, spark=None, write_path=None):
    """For each type, distribution of 'how many nodes of this type appear together
    in one component' -- NOT the same as overall component size (see markdown
    above). Useful for decomposing which types dominate the largest component(s)."""
    per_type_counts = (
        result_df.groupBy("component", "type").count().withColumnRenamed("count", "type_count_in_component")
    )
    return per_type_counts.groupBy("type", "type_count_in_component").count().withColumnRenamed(
        "count", "num_components"
    )


per_type_component_size_histogram = per_type_component_size_histogram_fn(
    result, spark=spark, write_path=f"{CC_OUTPUT_PATH}/per_type_component_size_histogram"
)


@stage_dataframe(write_format="parquet")
def top_component_type_breakdown_fn(result_df, top_n, spark=None, write_path=None):
    """Per-type node counts for the TOP_N largest components -- e.g. to see exactly
    which types dominate the largest component, and confirm the type-count columns
    sum to that component's true size."""
    top_components = (
        result_df.select("component", "component_size")
        .distinct()
        .orderBy(F.desc("component_size"))
        .limit(top_n)
    )
    return (
        result_df.join(top_components.select("component"), on="component")
        .groupBy("component", "type")
        .count()
        .withColumnRenamed("count", "type_count")
    )


TOP_N_COMPONENTS = 5
top_component_breakdown = top_component_type_breakdown_fn(
    result,
    TOP_N_COMPONENTS,
    spark=spark,
    write_path=f"{CC_OUTPUT_PATH}/top_{TOP_N_COMPONENTS}_component_breakdown",
)
print(f"--- per-type composition of the top {TOP_N_COMPONENTS} largest components ---")
top_component_breakdown.orderBy(F.desc("type_count")).show(100, truncate=False)

# COMMAND ----------

# MAGIC %md ## Per-type summary
# MAGIC Broken into staged sub-tables (totals, conn_identities, degree_counts) before
# MAGIC the final join -- same granularity as the build script's `id5_pre`/`tapad_pre`/
# MAGIC `liid_pre` pattern, so each piece is independently resumable.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def type_totals_fn(result_df, spark=None, write_path=None):
    return result_df.groupBy("type").agg(
        F.count("id").alias("total"),
        F.sum(F.col("is_singleton").cast("int")).alias("singleton"),
    )


type_totals = type_totals_fn(result, spark=spark, write_path=f"{CC_OUTPUT_PATH}/type_totals")

# Cross-check against the via_keys-derived singleton counts from earlier (see
# the "Node distribution by type and via_type" section) -- these are two
# independent measurements of the same population (CC's is_singleton vs.
# "zero via_keys rows") and should match exactly. A mismatch would mean
# either a real bug in the pipeline or a stale/mismatched via_keys vs. edges
# build (e.g. one was rebuilt and the other wasn't).
singleton_cross_check = type_totals.select("type", "singleton").join(
    singleton_by_type_from_via_keys, on="type"
).withColumn("matches", F.col("singleton") == F.col("n_singleton_from_via_keys"))
print("--- singleton cross-check: CC-derived vs. via_keys-derived (should all be True) ---")
singleton_cross_check.orderBy("type").show(truncate=False)


@stage_dataframe(write_format="parquet")
def conn_identities_fn(result_df, spark=None, write_path=None):
    return (
        result_df.filter(F.col("component_size") > 1)
        .groupBy("type")
        .agg(F.countDistinct("component").alias("conn_identities"))
    )


conn_identities = conn_identities_fn(
    result, spark=spark, write_path=f"{CC_OUTPUT_PATH}/conn_identities"
)


@stage_dataframe(write_format="parquet")
def degree_counts_fn(edges_df, type_map_df, spark=None, write_path=None):
    deg_a = edges_df.join(type_map_df, edges_df.src == type_map_df.id).select(type_map_df.type)
    deg_b = edges_df.join(type_map_df, edges_df.dst == type_map_df.id).select(type_map_df.type)
    return deg_a.unionByName(deg_b).groupBy("type").count().withColumnRenamed("count", "deg_sum")


degree_counts = degree_counts_fn(
    edges_gf, type_map, spark=spark, write_path=f"{CC_OUTPUT_PATH}/degree_counts"
)


@stage_dataframe(write_format="parquet")
def type_summary_fn(totals_df, identities_df, degree_df, spark=None, write_path=None):
    summary = totals_df.join(identities_df, on="type", how="left")
    summary = summary.withColumn("connected", F.col("total") - F.col("singleton"))
    summary = summary.withColumn("pct_conn", 100 * F.col("connected") / F.col("total"))
    summary = summary.withColumn("nodes_per_id", F.col("connected") / F.col("conn_identities"))

    merged_away = F.col("connected") - F.col("conn_identities")
    summary = summary.withColumn("reduce_conn", 100 * merged_away / F.col("connected"))
    summary = summary.withColumn("reduce_all", 100 * merged_away / F.col("total"))

    summary = summary.join(degree_df, on="type", how="left")
    summary = summary.withColumn("avg_deg_conn", F.col("deg_sum") / F.col("connected"))
    return summary.drop("deg_sum")


summary = type_summary_fn(
    type_totals,
    conn_identities,
    degree_counts,
    spark=spark,
    write_path=f"{CC_OUTPUT_PATH}/type_summary",
)
summary.orderBy("type").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Type x type matrix
# MAGIC Result is <=36 rows -- `.toPandas()` + pivot at the very end is plain pandas
# MAGIC on an already-tiny, already-collected table, not staged (the aggregation
# MAGIC itself, which is the expensive part, is staged below).

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def directed_pairs_fn(edges_df, type_map_df, spark=None, write_path=None):
    e2 = edges_df.join(
        type_map_df.withColumnRenamed("id", "src").withColumnRenamed("type", "type_a"), on="src"
    ).join(
        type_map_df.withColumnRenamed("id", "dst").withColumnRenamed("type", "type_b"), on="dst"
    )
    fwd = e2.select(F.col("type_a").alias("row"), F.col("type_b").alias("col"))
    bwd = e2.select(F.col("type_b").alias("row"), F.col("type_a").alias("col"))
    return fwd.unionByName(bwd)


directed = directed_pairs_fn(
    edges_gf, type_map, spark=spark, write_path=f"{CC_OUTPUT_PATH}/directed_pairs"
)


@stage_dataframe(write_format="parquet")
def type_matrix_fn(directed_df, spark=None, write_path=None):
    count_matrix = directed_df.groupBy("row", "col").count()
    row_totals = directed_df.groupBy("row").count().withColumnRenamed("count", "row_total")
    return count_matrix.join(row_totals, on="row").withColumn(
        "pct", 100 * F.col("count") / F.col("row_total")
    )


type_matrix = type_matrix_fn(
    directed, spark=spark, write_path=f"{CC_OUTPUT_PATH}/type_matrix"
)

type_matrix_pd = type_matrix.toPandas()
print(type_matrix_pd.pivot(index="row", columns="col", values="pct").round(2).to_string())

# `directed` counts each edge twice (fwd + bwd -- even on the diagonal, where
# type_a == type_b), so type_matrix's raw `count` column is 2x the true undirected
# edge count between a type pair. Filtering to row <= col keeps exactly one of each
# (A,B)/(B,A) pair, giving the real count directly without a manual /2.
undirected_type_counts = (
    type_matrix.filter(F.col("row") <= F.col("col"))
    .select("row", "col", "count")
    .orderBy(F.desc("count"))
)
print("--- edge counts by endpoint type pair (undirected, true counts) ---")
undirected_type_counts.show(50, truncate=False)

# COMMAND ----------

# MAGIC %md ## Weight / via_type_count histograms
# MAGIC Same bucketing convention as the local notebook -- exact counts for the low
# MAGIC end, bucketed for the tail.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def weight_histogram_fn(edges_df, spark=None, write_path=None):
    bucketed = edges_df.withColumn(
        "weight_bucket",
        F.when(F.col("weight") <= 10, F.col("weight").cast("string")).otherwise(
            F.concat(F.lit(">"), (F.floor(F.col("weight") / 100) * 100).cast("string"))
        ),
    )
    return bucketed.groupBy("weight_bucket").count()


weight_histogram = weight_histogram_fn(
    edges, spark=spark, write_path=f"{CC_OUTPUT_PATH}/weight_histogram"
)
weight_histogram.orderBy(F.desc("count")).show(30, truncate=False)


@stage_dataframe(write_format="parquet")
def via_type_count_histogram_fn(edges_df, spark=None, write_path=None):
    return edges_df.groupBy("via_type_count").count()


via_type_count_histogram = via_type_count_histogram_fn(
    edges, spark=spark, write_path=f"{CC_OUTPUT_PATH}/via_type_count_histogram"
)
via_type_count_histogram.orderBy("via_type_count").show()

# COMMAND ----------

# MAGIC %md ## Point-query helpers
# MAGIC Ad-hoc lookups against the full-scale `result`/`edges` tables, run here in
# MAGIC Spark -- `result` is ~4B rows, nowhere close to fitting locally. Not wrapped
# MAGIC in `@stage_dataframe`: these are interactive, different-args-every-call
# MAGIC lookups, not reusable pipeline stages worth persisting to their own path.
# MAGIC
# MAGIC Note: `result.filter(F.col("component") == X)` is a full scan over ~4B rows on
# MAGIC plain Parquet -- fine occasionally, but if these become a regular workflow,
# MAGIC worth writing `result` as Delta partitioned/Z-ordered by `component` (
# MAGIC `stage_dataframe` already supports `write_format="delta"` and `partitionby=`)
# MAGIC so lookups skip most of the data instead of scanning all of it.

# COMMAND ----------


def view_component(component_id, spark=spark, result_path=f"{CC_OUTPUT_PATH}/result",
                    edges_path=f"{BUILD_OUTPUT_PATH}/edges"):
    """Members of one component, plus the edges among them (with weight/via_type_count).
    Spark equivalent of the local notebook's view_component() -- same signature/output
    shape, but queries the full-scale table in place instead of filtering local pandas."""
    result_df = spark.read.format("delta").load(result_path)
    members = result_df.filter(F.col("component") == component_id).select("id", "type")
    member_ids = [row["id"] for row in members.collect()]  # fine as long as the component itself is small

    edges_df = spark.read.parquet(edges_path)
    edges_in = edges_df.filter(
        F.col("node_a").isin(member_ids) & F.col("node_b").isin(member_ids)
    )
    print(f"Component {component_id}: {members.count()} nodes, {edges_in.count()} internal edges")
    return members, edges_in


def component_of_node(node_id, spark=spark, result_path=f"{CC_OUTPUT_PATH}/result"):
    """Find the component containing a node, then view_component() it."""
    result_df = spark.read.format("delta").load(result_path)
    match = result_df.filter(F.col("id") == node_id).select("component").first()
    if match is None:
        raise ValueError(f"node_id not found: {node_id}")
    return view_component(match["component"], spark=spark)


def component_of_node_pair(node_a, node_b, spark=spark, result_path=f"{CC_OUTPUT_PATH}/result",
                            edges_path=f"{BUILD_OUTPUT_PATH}/edges"):
    """Find the component containing both endpoints of a node pair -- they must be
    the same component, i.e. connected directly or transitively."""
    result_df = spark.read.format("delta").load(result_path)
    comp_a = result_df.filter(F.col("id") == node_a).select("component").first()
    comp_b = result_df.filter(F.col("id") == node_b).select("component").first()
    if comp_a is None or comp_b is None:
        raise ValueError("one or both node_ids not found")
    if comp_a["component"] != comp_b["component"]:
        raise ValueError("nodes are in different components -- no path between them")

    edges_df = spark.read.parquet(edges_path)
    direct = edges_df.filter(
        ((F.col("node_a") == node_a) & (F.col("node_b") == node_b))
        | ((F.col("node_a") == node_b) & (F.col("node_b") == node_a))
    )
    direct_row = direct.first()
    if direct_row is not None:
        print(f"Direct edge: weight={direct_row['weight']}, via_type_count={direct_row['via_type_count']}")
    else:
        print("No direct edge between these two nodes -- connected only transitively.")

    return view_component(comp_a["component"], spark=spark)


# example usage:
# members, edges_in = view_component(653346)
# members, edges_in = component_of_node("HARDWARE_ANDROID_AD_ID:d6063de0-b218-43c4-a768-ba23f939652c")

# COMMAND ----------

# MAGIC %md ## Done -- next steps
# MAGIC 1. Download `type_summary`, `type_matrix`, `weight_histogram`,
# MAGIC    `via_type_count_histogram` locally -- all small.
# MAGIC 2. `result` (the full node/component table) stays on Databricks. Any point
# MAGIC    lookup (a specific component, a specific node's neighbors) should query it
# MAGIC    in place here rather than assume it fits in a local pandas DataFrame.
# MAGIC 3. Re-running this notebook after a restart re-reads every already-completed
# MAGIC    staged path instead of recomputing -- only genuinely new/unfinished stages
# MAGIC    actually run.
