# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — edge labels from Affinity (Steps 0 + 1)
# MAGIC
# MAGIC Uses the Affinity crosswalk as an **out-of-sample judge** of our grouping
# MAGIC keys. Affinity NEVER feeds graph construction -- it only scores the links
# MAGIC our own sources (id5/tapad/liid) make. Full plan and rationale:
# MAGIC `IDGraph_Journal.md` §12.
# MAGIC
# MAGIC **The test**: every grouping key claims its members are one person. For
# MAGIC every pair of Affinity-matched nodes (HEM_MD5/IDFA/AAID) that share a key,
# MAGIC does Affinity agree they're the same person?
# MAGIC
# MAGIC | Affinity says | label |
# MAGIC |---|---|
# MAGIC | same AFS individual (afs id sets overlap) | `agree` |
# MAGIC | different, both nodes' AFS individuals have >= 2 crosswalk IDs | `strong_disagree` |
# MAGIC | different, at least one side has a single-ID AFS individual | `weak_disagree` |
# MAGIC
# MAGIC A single-ID AFS individual may just be one Affinity never linked to
# MAGIC anything, so its "different person" is weaker evidence. Reporting rates
# MAGIC with and without weak labels tests that assumption.
# MAGIC
# MAGIC **Step 0** -- node labels + tune/holdout split (by hash of
# MAGIC `AFS_INDIVIDUAL_ID`, so one person never lands in both) + coverage check.
# MAGIC **Step 1** -- labeled pairs per via_type from `{via}_filtered_base`, which
# MAGIC has full key membership (not limited by the star layout or MAX_FANOUT);
# MAGIC summaries by via_type x type pair x group size, and by which via_types
# MAGIC corroborate a pair. **Summaries use the tune split only** -- holdout is
# MAGIC reserved for scoring graphs (Step 3) and must not be looked at while tuning.
# MAGIC
# MAGIC Independent of MAX_FANOUT/STAR_THRESHOLD: one run serves every config.
# MAGIC Not yet included: chained (2-hop) pairs (Step 2), graph scoring (Step 3),
# MAGIC household (`AFS_HHID`), vendor-split liid self-agreement check.

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

# filtered_base/fanout are threshold-independent -- IDGraph4 (32/10) and
# IDGraph5 (16/6) hold identical copies (verified by the fanout sweep).
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
BUILD_OUTPUT_PATH = f"{GRAPH_ROOT}/build_output"
# node_types lives only in IDGraph3 -- IDGraph4/5 builds reused it.
NODE_TYPES_PATH = "s3://data-science-research/ravi/IDGraph3/build_output/node_types"
EDGE_LABELS_PATH = f"{GRAPH_ROOT}/edge_labels"

AFFINITY_RDATE = "20261008"
# Working copy of the 20261008 crosswalk -- use this for now.
AFFINITY_CROSSWALK_PATH = f"s3://data-science-research/ravi/afs_crosswalk_{AFFINITY_RDATE}"
# Prod / original location -- switch back to this when moving to prod:
# AFFINITY_CROSSWALK_PATH = (
#     f"s3://data-science-research/affinity_matching/DCR/refresh/{AFFINITY_RDATE}/crosswalk/all_id_to_afs_id"
# )

VIA_KEY_COLS = [
    ("id5", "individual_id5"),
    ("tapad", "individual_id_tapad"),
    ("liid", "liid"),
]

# Must match DENYLIST in IDGraph_Build_Databricks.py (applied after filtered_base
# is staged, so the staged table still contains these rows).
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}

# our node type -> (crosswalk Type, crosswalk Subtype or None)
CW_TYPE_MAP = [
    ("HEM_MD5", "hem", "md5"),
    ("HARDWARE_ANDROID_AD_ID", "android", None),
    ("HARDWARE_IDFA", "ios", None),
]
TYPE_SHORT = {"HEM_MD5": "HEM", "HARDWARE_IDFA": "IDFA", "HARDWARE_ANDROID_AD_ID": "AAID"}

# Split by AFS individual. Changing the salt or fraction reshuffles the
# holdout -- fix both before looking at any results and never change them after.
SPLIT_SALT = "idgraph-edge-labels-v1"
TUNE_PCT = 70

# Pairs are enumerated within keys up to this group size (well past any cap
# considered; the fanout sweep shows only ~0.06% of nodes depend on larger groups).
# Bounds the per-key self-join at C(1024, 2).
MAX_PAIR_GROUP_SIZE = 1024

# COMMAND ----------

# MAGIC %md ## Step 0a -- node labels
# MAGIC One row per graph node of an eligible type that Affinity matches:
# MAGIC `afs_ids` (sorted array -- a node can map to several AFS individuals),
# MAGIC `afs_class` (`multi` if ANY of its AFS individuals has >= 2 distinct
# MAGIC crosswalk IDs across hem/android/ios, else `single`), and `split`.
# MAGIC `n_cw_ids` counts distinct IDs, never rows (the crosswalk has ~8 rows per
# MAGIC ID). IP-type crosswalk rows are ignored (we have no IP nodes).
# MAGIC A node's split comes from its smallest AFS id, so multi-mapped nodes
# MAGIC still get exactly one split.

# COMMAND ----------


def _split_expr(afs_col):
    bucket = F.pmod(F.xxhash64(F.lit(SPLIT_SALT), afs_col), F.lit(100))
    return F.when(bucket < TUNE_PCT, F.lit("tune")).otherwise(F.lit("holdout"))


@stage_dataframe(write_format="parquet")
def cw_nodes_fn(cw_df, spark=None, write_path=None):
    """Crosswalk rows mapped onto our node_id format, deduplicated."""
    parts = []
    for node_type, cw_type, cw_subtype in CW_TYPE_MAP:
        c = cw_df.filter(F.col("Type") == cw_type)
        if cw_subtype is not None:
            c = c.filter(F.col("Subtype") == cw_subtype)
        parts.append(
            c.select(
                F.concat(F.lit(f"{node_type}:"), F.col("ID")).alias("node_id"),
                F.lit(node_type).alias("type"),
                "AFS_INDIVIDUAL_ID",
            )
        )
    unioned = parts[0]
    for p in parts[1:]:
        unioned = unioned.unionByName(p)
    return unioned.distinct()


@stage_dataframe(write_format="parquet")
def node_labels_fn(cw_nodes_df, node_types_df, spark=None, write_path=None):
    afs_counts = cw_nodes_df.groupBy("AFS_INDIVIDUAL_ID").agg(F.count("*").alias("n_cw_ids"))
    graph_nodes = node_types_df.filter(F.col("type").isin(list(TYPE_SHORT))).select("node_id")
    return (
        cw_nodes_df.join(graph_nodes, on="node_id", how="leftsemi")
        .join(afs_counts, on="AFS_INDIVIDUAL_ID")
        .groupBy("node_id", "type")
        .agg(
            F.array_sort(F.collect_set("AFS_INDIVIDUAL_ID")).alias("afs_ids"),
            F.max("n_cw_ids").alias("max_n_cw_ids"),
            F.min("AFS_INDIVIDUAL_ID").alias("split_afs"),
        )
        .withColumn("afs_class", F.when(F.col("max_n_cw_ids") >= 2, F.lit("multi")).otherwise(F.lit("single")))
        .withColumn("split", _split_expr(F.col("split_afs")))
        .drop("split_afs")
    )


affinity_cw = spark.read.parquet(AFFINITY_CROSSWALK_PATH)
node_types = spark.read.parquet(NODE_TYPES_PATH)

cw_nodes = cw_nodes_fn(affinity_cw, spark=spark, write_path=f"{EDGE_LABELS_PATH}/cw_nodes")
node_labels = node_labels_fn(cw_nodes, node_types, spark=spark, write_path=f"{EDGE_LABELS_PATH}/node_labels")

# COMMAND ----------

# MAGIC %md ## Step 0b -- coverage check
# MAGIC Positive (`agree`) pairs only exist for AFS individuals with >= 2 of OUR
# MAGIC nodes. If that population is small, the whole approach is starved of
# MAGIC positives -- check this before reading Step 1.
# MAGIC Expected sanity numbers: matched HEM_MD5 89,884,356 / IDFA 48,713,684 /
# MAGIC AAID 46,698,298 (from the earlier affinity match, same refresh).

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def node_label_summary_fn(labels_df, spark=None, write_path=None):
    return labels_df.groupBy("type", "afs_class", "split").agg(
        F.count("*").alias("n_nodes"),
        F.sum((F.size("afs_ids") > 1).cast("int")).alias("n_multi_mapped"),
    )


@stage_dataframe(write_format="parquet")
def afs_graph_coverage_fn(labels_df, spark=None, write_path=None):
    per_afs = (
        labels_df.select(F.explode("afs_ids").alias("AFS_INDIVIDUAL_ID"), "type")
        .groupBy("AFS_INDIVIDUAL_ID")
        .agg(
            F.count("*").alias("n_graph_nodes"),
            F.sum((F.col("type") == "HEM_MD5").cast("int")).alias("n_hem"),
        )
        .withColumn("split", _split_expr(F.col("AFS_INDIVIDUAL_ID")))
    )
    return (
        per_afs.withColumn(
            "n_graph_nodes_bucket",
            F.when(F.col("n_graph_nodes") <= 4, F.col("n_graph_nodes").cast("string"))
            .when(F.col("n_graph_nodes") <= 9, F.lit("5-9"))
            .otherwise(F.lit("10+")),
        )
        .groupBy("split", "n_graph_nodes_bucket")
        .agg(
            F.count("*").alias("n_afs"),
            F.sum((F.col("n_graph_nodes") * (F.col("n_graph_nodes") - 1) / 2).cast("long")).alias(
                "max_possible_agree_pairs"
            ),
            F.sum((F.col("n_hem") >= 2).cast("int")).alias("n_afs_with_2plus_hem"),
        )
    )


node_label_summary = node_label_summary_fn(
    node_labels, spark=spark, write_path=f"{EDGE_LABELS_PATH}/node_label_summary"
)
afs_graph_coverage = afs_graph_coverage_fn(
    node_labels, spark=spark, write_path=f"{EDGE_LABELS_PATH}/afs_graph_coverage"
)

print("--- matched graph nodes by type / afs_class / split ---")
print(node_label_summary.orderBy("type", "afs_class", "split").toPandas().to_string(index=False))
print()
print("--- AFS individuals by how many of OUR nodes they map to ---")
print(afs_graph_coverage.orderBy("split", "n_graph_nodes_bucket").toPandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md ## Step 1a -- labeled pairs per via_type
# MAGIC Per via_type, staged separately (see the fanout-sweep notebook for why:
# MAGIC one big union+shuffle hit Spark's indeterminate-stage abort). Members =
# MAGIC `filtered_base` rows of matched nodes, in keys of size 2..MAX_PAIR_GROUP_SIZE.
# MAGIC Pairs = every two members of the same key, same split only. One row per
# MAGIC (key, pair): a pair linked by several keys of the same via_type appears
# MAGIC once per key here and is collapsed in Step 1b.

# COMMAND ----------

_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


@stage_dataframe(write_format="parquet")
def labeled_members_fn(labels_df, via_label, key_col, spark=None, write_path=None):
    b = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_filtered_base")
    deny = DENYLIST.get(via_label, set())
    if deny:
        b = b.filter(~F.col(key_col).isin(list(deny)))
    fanout = (
        spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_fanout")
        .filter((F.col("n_distinct") >= 2) & (F.col("n_distinct") <= MAX_PAIR_GROUP_SIZE))
        .select(key_col, F.col("n_distinct").alias("group_size"))
    )
    return (
        b.select("node_id", key_col)
        .join(labels_df.select("node_id", "type", "afs_ids", "afs_class", "split"), on="node_id")
        .join(fanout, on=key_col)
        .select(F.col(key_col).cast("string").alias("key"), "group_size", "node_id", "type", "afs_ids", "afs_class", "split")
    )


@stage_dataframe(write_format="parquet")
def labeled_pairs_fn(members_df, via_label, spark=None, write_path=None):
    a = members_df.select(
        "key", "group_size",
        F.col("node_id").alias("node_a"), F.col("type").alias("type_a"),
        F.col("afs_ids").alias("afs_a"), F.col("afs_class").alias("class_a"), "split",
    )
    b = members_df.select(
        "key",
        F.col("node_id").alias("node_b"), F.col("type").alias("type_b"),
        F.col("afs_ids").alias("afs_b"), F.col("afs_class").alias("class_b"), F.col("split").alias("split_b"),
    )
    label = (
        F.when(F.arrays_overlap("afs_a", "afs_b"), F.lit("agree"))
        .when((F.col("class_a") == "multi") & (F.col("class_b") == "multi"), F.lit("strong_disagree"))
        .otherwise(F.lit("weak_disagree"))
    )
    return (
        a.join(b, on="key")
        .filter((F.col("node_a") < F.col("node_b")) & (F.col("split") == F.col("split_b")))
        .select(
            F.lit(via_label).alias("via_type"),
            "key",
            "group_size",
            "node_a",
            "node_b",
            F.concat_ws("-", F.array_sort(F.array(_short_map[F.col("type_a")], _short_map[F.col("type_b")]))).alias(
                "type_pair"
            ),
            label.alias("label"),
            "split",
        )
    )


labeled_pairs_parts = []
for via_label, key_col in VIA_KEY_COLS:
    _members = labeled_members_fn(
        node_labels, via_label, key_col, spark=spark, write_path=f"{EDGE_LABELS_PATH}/labeled_members_{via_label}"
    )
    labeled_pairs_parts.append(
        labeled_pairs_fn(_members, via_label, spark=spark, write_path=f"{EDGE_LABELS_PATH}/labeled_pairs_{via_label}")
    )

# COMMAND ----------

# MAGIC %md ## Step 1b -- one row per (via_type, pair), then per pair
# MAGIC `pairs_by_via`: a pair linked by several keys of the same via_type
# MAGIC collapses to one row at its SMALLEST group size (the strongest evidence
# MAGIC that via_type has for it), with `n_keys`.
# MAGIC `pair_corroboration`: one row per pair with `via_set` (e.g. `liid+tapad`)
# MAGIC -- which of our sources link it at all (any key up to MAX_PAIR_GROUP_SIZE,
# MAGIC not just keys that survive a particular cap).

# COMMAND ----------

PAIR_COLS = ["node_a", "node_b", "type_pair", "label", "split"]


@stage_dataframe(write_format="parquet")
def pairs_by_via_fn(parts, spark=None, write_path=None):
    unioned = parts[0]
    for p in parts[1:]:
        unioned = unioned.unionByName(p)
    return unioned.groupBy("via_type", *PAIR_COLS).agg(
        F.min("group_size").alias("group_size"), F.count("*").alias("n_keys")
    )


@stage_dataframe(write_format="parquet")
def pair_corroboration_fn(pairs_by_via_df, spark=None, write_path=None):
    return pairs_by_via_df.groupBy(*PAIR_COLS).agg(
        F.concat_ws("+", F.array_sort(F.collect_set("via_type"))).alias("via_set"),
        F.min("group_size").alias("min_group_size"),
    )


pairs_by_via = pairs_by_via_fn(labeled_pairs_parts, spark=spark, write_path=f"{EDGE_LABELS_PATH}/pairs_by_via")
pair_corroboration = pair_corroboration_fn(
    pairs_by_via, spark=spark, write_path=f"{EDGE_LABELS_PATH}/pair_corroboration"
)

# COMMAND ----------

# MAGIC %md ## Step 1c -- agreement summaries (TUNE SPLIT ONLY)
# MAGIC Rates, always shown with counts:
# MAGIC - `agree_rate_strong` = agree / (agree + strong_disagree) -- ignores weak labels
# MAGIC - `agree_rate_all` = agree / (agree + strong_disagree + weak_disagree)
# MAGIC The gap between them shows how much single-ID AFS individuals move the answer.

# COMMAND ----------


def _group_size_bucket(col):
    edges = [2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 128, 256, MAX_PAIR_GROUP_SIZE]
    expr = None
    lo = 2
    for hi in edges:
        label = str(hi) if lo == hi else f"{lo}-{hi}"
        cond = F.col(col) <= hi
        expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
        lo = hi + 1
    return expr


def _label_counts(df, group_cols):
    return df.groupBy(*group_cols).agg(
        F.count("*").alias("n_pairs"),
        F.sum((F.col("label") == "agree").cast("long")).alias("n_agree"),
        F.sum((F.col("label") == "strong_disagree").cast("long")).alias("n_strong_disagree"),
        F.sum((F.col("label") == "weak_disagree").cast("long")).alias("n_weak_disagree"),
    )


@stage_dataframe(write_format="parquet")
def agreement_by_via_type_pair_size_fn(pairs_by_via_df, spark=None, write_path=None):
    tune = pairs_by_via_df.filter(F.col("split") == "tune").withColumn(
        "size_bucket", _group_size_bucket("group_size")
    ).withColumn("size_order", F.col("group_size"))
    counts = _label_counts(tune, ["via_type", "type_pair", "size_bucket"])
    order = tune.groupBy("size_bucket").agg(F.min("size_order").alias("size_order"))
    return counts.join(order, on="size_bucket")


@stage_dataframe(write_format="parquet")
def agreement_by_corroboration_fn(corroboration_df, spark=None, write_path=None):
    return _label_counts(corroboration_df.filter(F.col("split") == "tune"), ["via_set", "type_pair"])


agreement_by_via_type_pair_size = agreement_by_via_type_pair_size_fn(
    pairs_by_via, spark=spark, write_path=f"{EDGE_LABELS_PATH}/agreement_by_via_type_pair_size"
)
agreement_by_corroboration = agreement_by_corroboration_fn(
    pair_corroboration, spark=spark, write_path=f"{EDGE_LABELS_PATH}/agreement_by_corroboration"
)


def _add_rates(pdf):
    pdf = pdf.copy()
    strong_den = pdf["n_agree"] + pdf["n_strong_disagree"]
    pdf["agree_rate_strong"] = (100 * pdf["n_agree"] / strong_den.where(strong_den > 0)).round(2)
    pdf["agree_rate_all"] = (100 * pdf["n_agree"] / pdf["n_pairs"].where(pdf["n_pairs"] > 0)).round(2)
    return pdf


_COUNT_COLS = ["n_pairs", "n_agree", "n_strong_disagree", "n_weak_disagree"]
_size_pd = agreement_by_via_type_pair_size.toPandas()

print("=== [tune] agreement by via_type (all type pairs, all sizes) ===")
print(_add_rates(_size_pd.groupby("via_type", as_index=False)[_COUNT_COLS].sum()).to_string(index=False))
print()
print("=== [tune] agreement by via_type x type_pair (all sizes) ===")
print(
    _add_rates(_size_pd.groupby(["via_type", "type_pair"], as_index=False)[_COUNT_COLS].sum())
    .sort_values(["via_type", "type_pair"])
    .to_string(index=False)
)
print()
print("=== [tune] agreement by via_type x type_pair x group size ===")
print(
    _add_rates(_size_pd)
    .sort_values(["via_type", "type_pair", "size_order"])
    .drop(columns="size_order")
    .to_string(index=False)
)
print()
print("=== [tune] agreement by which via_types link the pair (corroboration) ===")
print(
    _add_rates(agreement_by_corroboration.toPandas())
    .sort_values(["type_pair", "via_set"])
    .to_string(index=False)
)

# COMMAND ----------

# MAGIC %md ## Full tables as CSV (paste back to save locally)

# COMMAND ----------

print(_add_rates(_size_pd).sort_values(["via_type", "type_pair", "size_order"]).to_csv(index=False))
print(_add_rates(agreement_by_corroboration.toPandas()).sort_values(["type_pair", "via_set"]).to_csv(index=False))
