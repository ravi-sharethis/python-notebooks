# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph build (Spark/Databricks) -- staged version
# MAGIC
# MAGIC Same hybrid star/clique logic as before, rewritten so every join/groupBy/
# MAGIC union goes through `@stage_dataframe` (from `staging_df.py`, pasted inline
# MAGIC below): each stage reads from and writes to its own path, so a cluster
# MAGIC restart mid-run re-reads already-completed stages instead of recomputing them.
# MAGIC
# MAGIC **`node_types` and `via_keys` don't depend on the star/clique choice at all**
# MAGIC (see chat) -- if they already exist from a prior run, point `NODE_TYPES_PATH`/
# MAGIC `VIA_KEYS_PATH` at that existing location and the decorator will just read them
# MAGIC back rather than recomputing. Only `edges` (built from `all_pairs`, which does
# MAGIC change shape under star vs. clique) needs a fresh run whenever
# MAGIC `STAR_THRESHOLD`/`MAX_FANOUT`/`DENYLIST` change.

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

import random

RAW_PATH = "s3://data-science-research/ravi/IDGraph2/id5_tapad_li_joined_outer"
BUILD_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph3/build_output/"
IDGRAPH2_BUILD_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph2/build_output/"  # TODO: confirm this is the right prior-run path

# base/node_types are pure functions of the raw filtered rows -- no dependency on
# MAX_FANOUT/STAR_THRESHOLD/DENYLIST/hub-priority at all -- so reusing them from
# IDGraph2 is safe AS LONG AS FILTER_TYPES/RAW_PATH/sampling used to build them there
# still match what you want here.
#
# via_keys is NOT safe to reuse by default, unlike the markdown below used to claim:
# it's derived from all_pairs, which only contains rows for keys that survived the
# MAX_FANOUT/DENYLIST filter -- so via_keys silently encodes whatever MAX_FANOUT/
# DENYLIST/hub-priority logic was active when it was built. This run uses MAX_FANOUT=64
# and hub type-priority selection, almost certainly different from whatever IDGraph2
# used -- reusing its via_keys as-is would produce a node-membership table inconsistent
# with the edges this run computes. Only flip REUSE_VIA_KEYS if you've confirmed
# IDGraph2's via_keys was built under the identical MAX_FANOUT/STAR_THRESHOLD/DENYLIST/
# hub-priority config as this run.
REUSE_BASE = True
REUSE_NODE_TYPES = True
REUSE_VIA_KEYS = False

BASE_PATH = f"{IDGRAPH2_BUILD_OUTPUT_PATH}/base" if REUSE_BASE else f"{BUILD_OUTPUT_PATH}/base"
NODE_TYPES_PATH = f"{IDGRAPH2_BUILD_OUTPUT_PATH}/node_types" if REUSE_NODE_TYPES else f"{BUILD_OUTPUT_PATH}/node_types"
VIA_KEYS_PATH = f"{IDGRAPH2_BUILD_OUTPUT_PATH}/via_keys" if REUSE_VIA_KEYS else f"{BUILD_OUTPUT_PATH}/via_keys"
# always fresh under IDGraph3 -- depends on MAX_FANOUT/STAR_THRESHOLD/DENYLIST/hub logic.
# Also depends on the per-via_type node_id/type/key_col dedup in pairs_via (see chat)
# -- fixes a real weight-inflation bug where a node with one fixed key value but
# several different OTHER-key values (e.g. one tapad value, many liid values) was
# being self-joined once per duplicate row instead of once per node. If EDGES_PATH
# already has data from a build run BEFORE that fix, delete it and rebuild -- its
# weight values are inflated and stage_dataframe would otherwise silently reuse them.
EDGES_PATH = f"{BUILD_OUTPUT_PATH}/edges"

SAMPLE = True
SAMPLE_FILE_COUNT = 500
SAMPLE_SEED = 42

FILTER_TYPES = ["HARDWARE_IDFA", "HARDWARE_ANDROID_AD_ID", "HEM_MD5", "ID5_UID", "TTD", "ST"]

# Safety valve: a single grouping key fanning out to an unreasonable number of distinct
# nodes gets excluded entirely -- this is the ACTUAL individual-vs-family control knob
# (not STAR_THRESHOLD below, which only affects edge representation within groups that
# survive this cap -- see chat for the full reasoning). Set from the real group-size
# histograms (fan-out audit section below), not a guess -- id5's histogram tops out
# ~32 with a smooth organic-looking decay; tapad's smooth-but-longer tail made 100
# look too permissive (100 distinct nodes sharing one key reads as bot/shared-default
# behavior, not a household) once actually reasoned through. 64 is the settled
# compromise: comfortably above id5's observed natural max, well below the range that
# started looking synthetic for tapad.
MAX_FANOUT = 64

# Hybrid star/clique threshold -- PERFORMANCE lever only, does not change which nodes
# merge into the same identity (a star still fully connects every group member; see
# chat). Groups size <= STAR_THRESHOLD get full pairwise clique (C(k,2) edges, cheap
# and preserves per-pair weight/via_type_count fidelity); larger groups (up to
# MAX_FANOUT) get a star (hub = MIN(node_id), k-1 edges instead of k(k-1)/2). Kept at
# 20 rather than dropping to 10: the id5 histogram showed the 11-20 size band alone
# holds ~11.5M groups (more aggregate clique cost than the whole 21-32 band), and
# that's also the size range most likely to be genuine household/churn evidence worth
# preserving pairwise weight/via_type_count fidelity for -- worth the extra edges.
STAR_THRESHOLD = 20

# Value-level denylist: specific grouping-key VALUES excluded by identity, regardless
# of fanout -- confirmed recurring bad actor across dataset generations (see chat).
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}

# COMMAND ----------

# MAGIC %md ## Base: filtered, projected rows
# MAGIC `base_fn` always builds/persists the FULL, unsampled base -- staged like
# MAGIC everything else, so if `BASE_PATH` already has data (e.g. `REUSE_BASE` pointing
# MAGIC at IDGraph2), it reads that back and skips execution entirely. That's exactly
# MAGIC why SAMPLE can't live inside base_fn: `stage_dataframe` short-circuits before
# MAGIC ever calling the function once BASE_PATH has data, so any sampling logic in
# MAGIC there would silently never run whenever REUSE_BASE is on. Sampling instead
# MAGIC happens as its own step below, against base's own (possibly-reused) storage
# MAGIC path -- works the same way regardless of whether base was just freshly built or
# MAGIC read back from IDGraph2.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def base_fn(raw, spark=None, write_path=None):
    return raw.filter(F.col("type").isin(FILTER_TYPES) & F.col("value").isNotNull()).select(
        F.concat(F.col("type"), F.lit(":"), F.col("value")).alias("node_id"),
        "type",
        "individual_id5",
        "individual_id_tapad",
        "liid",
    )


def load_base(spark):
    """Two distinct paths, matching the two distinct costs involved:

    REUSE_BASE=True: base is assumed to already exist in full at BASE_PATH (e.g.
    IDGraph2). Sampling here just means reading fewer of its already-materialized
    files -- no fresh compute either way, so full-vs-sample can safely share
    BASE_PATH.

    REUSE_BASE=False (building fresh from RAW_PATH): sample RAW_PATH's files FIRST,
    before any filter/project/write happens -- a SAMPLE=True run should only ever
    touch a small slice of the ~9B-row raw dataset, not pay for a full build and
    subsample afterward. Since building fresh writes new data, sample and full runs
    use DIFFERENT write paths (`_sample` suffix) so toggling SAMPLE across runs can
    never cause stage_dataframe to skip a real build by finding stale sample-scale
    data sitting at the same path.
    """
    if REUSE_BASE:
        if not _path_exists(BASE_PATH):
            raise RuntimeError(
                f"REUSE_BASE=True but no existing base found at {BASE_PATH} -- "
                "set REUSE_BASE=False to build fresh from RAW_PATH instead of "
                "silently doing a full, expensive build into what's meant to be a "
                "reused prior-run path."
            )
        if not SAMPLE:
            return spark.read.parquet(BASE_PATH)

        all_base_files = [f.path for f in dbutils.fs.ls(BASE_PATH) if f.path.endswith(".parquet")]
        rng = random.Random(SAMPLE_SEED)
        sample_files = rng.sample(all_base_files, min(SAMPLE_FILE_COUNT, len(all_base_files)))
        print(
            f"SAMPLE=True (reused base): reading {len(sample_files)}/{len(all_base_files)} files "
            f"({100 * len(sample_files) / len(all_base_files):.2f}%)"
        )
        return spark.read.parquet(*sample_files)

    if not SAMPLE:
        raw = spark.read.parquet(RAW_PATH)
        return base_fn(raw, spark=spark, write_path=BASE_PATH)

    all_raw_files = [f.path for f in dbutils.fs.ls(RAW_PATH) if f.path.endswith(".parquet")]
    rng = random.Random(SAMPLE_SEED)
    sample_files = rng.sample(all_raw_files, min(SAMPLE_FILE_COUNT, len(all_raw_files)))
    print(
        f"SAMPLE=True: reading {len(sample_files)}/{len(all_raw_files)} randomly-selected raw files "
        f"({100 * len(sample_files) / len(all_raw_files):.2f}%)"
    )
    raw = spark.read.parquet(*sample_files)
    return base_fn(raw, spark=spark, write_path=f"{BASE_PATH}_sample")


base = load_base(spark)
base.cache()
print(f"base rows: {base.count():,}")

# COMMAND ----------

# MAGIC %md ## node_types
# MAGIC Staged so a prior run's output is reused automatically -- unaffected by
# MAGIC STAR_THRESHOLD, never needs to change alongside it.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def node_types_fn(base_df, spark=None, write_path=None):
    return base_df.select("node_id", "type").distinct()


node_types = node_types_fn(base, spark=spark, write_path=NODE_TYPES_PATH)
print(f"node_types rows: {node_types.count():,}")

# COMMAND ----------

# MAGIC %md ## Fan-out audit + group-size histogram
# MAGIC Run BEFORE the pairs build. The histogram (not just top-20) is what should
# MAGIC actually inform MAX_FANOUT/STAR_THRESHOLD -- see chat for why guessing round
# MAGIC numbers already went badly wrong once (the 8TB edges surprise).

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def fanout_fn(base_df, key_col, spark=None, write_path=None):
    return (
        base_df.filter(F.col(key_col).isNotNull())
        .groupBy(key_col)
        .agg(F.countDistinct("node_id").alias("n_distinct"))
    )


fanout_tables = {}
for via_label, key_col in [
    ("id5", "individual_id5"),
    ("tapad", "individual_id_tapad"),
    ("liid", "liid"),
]:
    fanout = fanout_fn(
        base, key_col, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_fanout"
    )
    fanout_tables[via_label] = fanout

    top = fanout.filter(F.col("n_distinct") >= 10).orderBy(F.desc("n_distinct")).limit(20)
    print(f"--- {via_label} top fan-out ---")
    top.show(truncate=60)

    hist = (
        fanout.filter(F.col("n_distinct") > 1)
        .groupBy("n_distinct")
        .count()
        .orderBy("n_distinct")
    )
    print(f"--- {via_label} group-size histogram ---")
    hist.show(100)

# COMMAND ----------

# MAGIC %md ## Pairs per grouping-key type -- hybrid star/clique
# MAGIC Same worked example / reasoning as before (see the local notebook's Connected
# MAGIC Components section for the full "why pair at all" writeup). Each sub-stage
# MAGIC (fanout already above, small-group clique, large-group star, final union) is
# MAGIC independently staged per via_type, so a restart only redoes whatever specific
# MAGIC piece hadn't finished.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def small_pairs_fn(filtered_base, small_keys, key_col, spark=None, write_path=None):
    small_filtered = filtered_base.join(small_keys, on=key_col, how="leftsemi")
    left = small_filtered.select(F.col("node_id").alias("node_a"), F.col(key_col).alias("key"))
    right = small_filtered.select(F.col("node_id").alias("node_b"), F.col(key_col).alias("key"))
    return (
        left.join(right, on="key")
        .filter(F.col("node_a") < F.col("node_b"))
        .select("node_a", "node_b", "key")
    )


@stage_dataframe(write_format="parquet")
def hub_selection_fn(filtered_base, large_keys, key_col, via_label, spark=None, write_path=None):
    """One row per large group: which node got picked as hub, and under which
    via_type. Staged on its own (not inlined into large_pairs_fn) so the hub
    identity survives past pairs_fn's node_a/node_b canonicalization -- that step
    destroys which side of a pair was originally the hub, so anything wanting hub
    identity (e.g. hub_diversity_fn below) has to read it from here instead."""
    large_filtered = filtered_base.join(large_keys, on=key_col, how="leftsemi")
    # Hub type priority: HEM_MD5 > any other HEM_* variant > ST > everything else
    # (hardware/device/cookie IDs, which churn most). Falls back to MIN(node_id)
    # within whichever priority tier actually has members in the group -- plain
    # MIN(node_id) alone (no priority) biases toward "HARDWARE_..." since it sorts
    # alphabetically before "HEM_MD5", picking the least stable available type.
    ranked = large_filtered.withColumn(
        "hub_priority",
        F.when(F.col("type") == "HEM_MD5", 0)
        .when(F.col("type").startswith("HEM_"), 1)
        .when(F.col("type") == "ST", 2)
        .otherwise(3),
    )
    return (
        ranked.groupBy(key_col)
        .agg(F.min(F.struct("hub_priority", "node_id")).alias("hub_struct"))
        .select(key_col, F.col("hub_struct.node_id").alias("hub"), F.lit(via_label).alias("via_type"))
    )


@stage_dataframe(write_format="parquet")
def large_pairs_fn(filtered_base, large_keys, hubs, key_col, spark=None, write_path=None):
    large_filtered = filtered_base.join(large_keys, on=key_col, how="leftsemi")
    return (
        large_filtered.join(hubs.select(key_col, "hub"), on=key_col)
        .filter(F.col("node_id") != F.col("hub"))
        .select(F.col("hub").alias("node_a"), F.col("node_id").alias("node_b"), F.col(key_col).alias("key"))
    )


@stage_dataframe(write_format="parquet")
def pairs_fn(small_df, large_df, via_label, spark=None, write_path=None):
    combined = small_df.unionByName(large_df)
    # Canonicalize node_a/node_b ordering: star hubs are now chosen by type priority
    # (HEM_MD5 preferred) rather than MIN(node_id), so node_a is no longer guaranteed
    # to sort before node_b the way small_pairs_fn's clique output already does.
    # Without this, the same physical pair could land as (A,B) from one group and
    # (B,A) from another, fragmenting weight/via_type_count across two edges_fn rows
    # instead of merging into one.
    canonical = combined.select(
        F.least("node_a", "node_b").alias("node_a"),
        F.greatest("node_a", "node_b").alias("node_b"),
        "key",
    )
    return canonical.select(
        "node_a",
        "node_b",
        F.lit(via_label).alias("via_type"),
        F.col("key").cast("string").alias("via_key"),
    )


@stage_dataframe(write_format="parquet")
def filtered_base_fn(base_df, key_col, spark=None, write_path=None):
    # base is an outer join across id5/tapad/liid -- a node with one fixed tapad
    # value but several different liid values (or vice versa) legitimately produces
    # multiple base rows that are identical from THIS via_type's perspective (see
    # chat: e.g. one Android ID with 10 different liid rows, same tapad value on
    # every one). Without this dedup, small_pairs_fn's self-join and large_pairs_fn's
    # join both multiply every pair that node participates in by however many such
    # duplicate rows it has -- inflating weight on evidence that isn't real
    # repetition, just an artifact of the OTHER key varying. fanout_fn is unaffected
    # (countDistinct already collapses this), which is why the group-size histograms
    # used to tune MAX_FANOUT/STAR_THRESHOLD are still trustworthy.
    return base_df.filter(F.col(key_col).isNotNull()).select("node_id", "type", key_col).distinct()


def pairs_via(key_col, via_label, fanout):
    # The diagnostic .count() calls below are real Spark actions -- unlike the
    # @stage_dataframe-wrapped calls further down, they are NOT skipped just because
    # their downstream output already exists. Without this guard, a rerun where
    # everything is already staged (e.g. after a cluster restart) would still pay
    # for a full scan+count of `b` (denylist check, derived from `base` -- can be
    # billions of rows) plus three counts on `fanout`, every single time. Skip them
    # entirely when the final `pairs` output already exists -- that information was
    # already printed the first time this stage actually ran.
    already_built = _path_exists(f"{BUILD_OUTPUT_PATH}/{via_label}_pairs")

    # Staged like everything else, so a restart (or the liid failure/retry from
    # chat) doesn't force re-scanning base + re-deduping from scratch -- only
    # whichever via_type's filtered_base hadn't finished actually reruns. Denylist
    # filtering deliberately stays OUTSIDE this stage (applied to the returned `b`
    # below) since DENYLIST can change between runs without wanting to force a
    # rebuild of this (base-derived, expensive) filter+dedup step.
    b = filtered_base_fn(
        base, key_col, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_filtered_base"
    )

    deny = DENYLIST.get(via_label, set())
    if deny:
        b_excl_deny = b.filter(~F.col(key_col).isin(list(deny)))
        if not already_built:
            n_denied = b.filter(F.col(key_col).isin(list(deny))).count()
            if n_denied:
                print(f"[{via_label}] excluding {n_denied:,} rows matching {len(deny)} denylisted value(s)")
        b = b_excl_deny

    if not already_built:
        excluded_count = fanout.filter(F.col("n_distinct") > MAX_FANOUT).count()
        if excluded_count:
            print(f"[{via_label}] excluding {excluded_count:,} keys with fan-out > {MAX_FANOUT}")

    small_keys = fanout.filter(
        (F.col("n_distinct") > 1) & (F.col("n_distinct") <= STAR_THRESHOLD)
    ).select(key_col)
    large_keys = fanout.filter(
        (F.col("n_distinct") > STAR_THRESHOLD) & (F.col("n_distinct") <= MAX_FANOUT)
    ).select(key_col)

    if not already_built:
        print(f"[{via_label}] {small_keys.count():,} small groups (<= {STAR_THRESHOLD}, full pairwise) / "
              f"{large_keys.count():,} large groups ({STAR_THRESHOLD}-{MAX_FANOUT}, star)")
    else:
        print(f"[{via_label}] pairs already built at {BUILD_OUTPUT_PATH}/{via_label}_pairs -- reusing, skipping diagnostics")

    small_pairs = small_pairs_fn(
        b, small_keys, key_col, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_small_pairs"
    )
    hubs = hub_selection_fn(
        b, large_keys, key_col, via_label, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_hubs"
    )
    large_pairs = large_pairs_fn(
        b, large_keys, hubs, key_col, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_large_pairs"
    )
    pairs = pairs_fn(
        small_pairs, large_pairs, via_label, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_pairs"
    )
    return pairs, hubs


id5_pairs, id5_hubs = pairs_via("individual_id5", "id5", fanout_tables["id5"])
tapad_pairs, tapad_hubs = pairs_via("individual_id_tapad", "tapad", fanout_tables["tapad"])
liid_pairs, liid_hubs = pairs_via("liid", "liid", fanout_tables["liid"])

# COMMAND ----------

# MAGIC %md ## Hub diversity (cheap cross-via_type approximation)
# MAGIC For nodes that were selected as hub in a large/starred group under more than
# MAGIC one via_type, this is a cheap proxy for cross-via_type corroboration -- it
# MAGIC specifically helps for the population edges_fn's real via_type_count is blind
# MAGIC to: two non-hub members of a starred group never get a direct edge, so they
# MAGIC never contribute to each other's via_type_count even if genuinely linked (see
# MAGIC chat). Not a substitute for real via_type_count -- noisy, since whether a node
# MAGIC becomes hub in any one group depends on which other nodes happen to be in that
# MAGIC group -- just a cheap, byproduct-of-hub-selection signal.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def hub_diversity_fn(id5_hubs_df, tapad_hubs_df, liid_hubs_df, spark=None, write_path=None):
    all_hubs = (
        id5_hubs_df.select("hub", "via_type")
        .unionByName(tapad_hubs_df.select("hub", "via_type"))
        .unionByName(liid_hubs_df.select("hub", "via_type"))
        .distinct()
    )
    return all_hubs.groupBy("hub").agg(
        F.countDistinct("via_type").alias("hub_via_type_diversity"),
        F.collect_set("via_type").alias("hub_via_types"),
    )


hub_diversity = hub_diversity_fn(
    id5_hubs, tapad_hubs, liid_hubs, spark=spark, write_path=f"{BUILD_OUTPUT_PATH}/hub_diversity"
)
print(f"hub_diversity rows: {hub_diversity.count():,}")
hub_diversity.groupBy("hub_via_type_diversity").count().orderBy("hub_via_type_diversity").show()

# COMMAND ----------

# MAGIC %md ## Denylist candidate review
# MAGIC Raw fanout alone (the existing top-20 audit above) is a weak signal on its own --
# MAGIC a value with high fanout could just be an organically popular cookie/device.
# MAGIC The combined signal used here is stronger: for each large-group key, look up
# MAGIC whether its own hub node ALSO got selected as hub in a large group under a
# MAGIC DIFFERENT via_type (via `hub_diversity`). A key whose hub is independently
# MAGIC anchoring large groups across multiple via_types looks much more like a
# MAGIC shared/default/bot-network value than one that's merely locally popular --
# MAGIC this is the same shape of evidence that originally surfaced the
# MAGIC already-denylisted tapad value (high fanout AND reappeared across dataset
# MAGIC generations). This is a REVIEW tool, not an automatic denylist writer -- confirm
# MAGIC a candidate actually looks like a bad actor (e.g. via view_component-style
# MAGIC inspection once CC has run) before adding its value to DENYLIST.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def denylist_candidates_fn(fanout_df, hubs_df, hub_diversity_df, key_col, via_label, spark=None, write_path=None):
    return (
        fanout_df.join(hubs_df.select(key_col, "hub"), on=key_col)
        .join(hub_diversity_df, on="hub")
        .filter(F.col("hub_via_type_diversity") > 1)
        .select(
            F.lit(via_label).alias("via_type"),
            F.col(key_col).alias("key_value"),
            "n_distinct",
            "hub",
            "hub_via_type_diversity",
            "hub_via_types",
        )
    )


for via_label, key_col, hubs_df in [
    ("id5", "individual_id5", id5_hubs),
    ("tapad", "individual_id_tapad", tapad_hubs),
    ("liid", "liid", liid_hubs),
]:
    denylist_candidates_fn(
        fanout_tables[via_label],
        hubs_df,
        hub_diversity,
        key_col,
        via_label,
        spark=spark,
        write_path=f"{BUILD_OUTPUT_PATH}/{via_label}_denylist_candidates",
    )

# Each via_type's candidates are already separately staged above -- unioning and
# rewriting them to a new path would just duplicate data already on disk (same
# reasoning as all_pairs earlier). A multi-path read is a metadata-only union, no
# new write, same as that fix.
denylist_candidates = spark.read.parquet(
    *[f"{BUILD_OUTPUT_PATH}/{via_label}_denylist_candidates" for via_label in ["id5", "tapad", "liid"]]
)

print("--- denylist candidates: large-group keys whose hub also anchors >1 via_type ---")
denylist_candidates.orderBy(F.desc("hub_via_type_diversity"), F.desc("n_distinct")).show(50, truncate=60)

# COMMAND ----------

# MAGIC %md ## all_pairs, via_keys, edges

# COMMAND ----------


# id5_pairs/tapad_pairs/liid_pairs are already separately staged, same-schema
# parquet directories -- unioning and rewriting them to a new `all_pairs` path would
# just be a duplicate copy of data that's already on disk. Reading all three paths
# in one spark.read.parquet(...) call is a metadata-only union (Spark appends each
# path's files to the input file list) -- no shuffle, no new write. Downstream
# stages that need real computation (via_keys_fn's distinct, edges_fn's groupBy)
# still do their own actual work; they just take this lazy union as input instead
# of a materialized all_pairs file.
all_pairs = spark.read.parquet(
    f"{BUILD_OUTPUT_PATH}/id5_pairs", f"{BUILD_OUTPUT_PATH}/tapad_pairs", f"{BUILD_OUTPUT_PATH}/liid_pairs"
)
print(f"all_pairs rows: {all_pairs.count():,}")


# repartition targets below: no stage anywhere previously called repartition(), so
# every output inherited whatever partition count fell out of Spark's shuffle --
# observed as 15,003 files for via_keys (522.5GB, ~35MB/file) and a much worse
# 63,047 files for edges (1.7TB, ~27MB/file). Both are smaller-than-ideal for
# Parquet (a ~128-256MB/file sweet spot is typical), and edges' fragmentation is
# disproportionate even relative to its size -- more files than via_keys despite
# being ~3x the data. Targets below aim for ~200MB/file based on the CURRENT
# observed sizes; rescale proportionally if a future run is at meaningfully
# different scale (e.g. a smaller SAMPLE run, or the dataset growing further).


@stage_dataframe(write_format="parquet", repartition=2600)
def via_keys_fn(all_pairs_df, spark=None, write_path=None):
    return (
        all_pairs_df.select(F.col("node_a").alias("node_id"), "via_type", "via_key")
        .unionByName(all_pairs_df.select(F.col("node_b").alias("node_id"), "via_type", "via_key"))
        .distinct()
    )


via_keys = via_keys_fn(all_pairs, spark=spark, write_path=VIA_KEYS_PATH)
print(f"via_keys rows: {via_keys.count():,}")


@stage_dataframe(write_format="parquet", repartition=8500)
def edges_fn(all_pairs_df, spark=None, write_path=None):
    return all_pairs_df.groupBy("node_a", "node_b").agg(
        F.count(F.lit(1)).alias("weight"),
        F.countDistinct("via_type").alias("via_type_count"),
    )


edges = edges_fn(all_pairs, spark=spark, write_path=EDGES_PATH)
print(f"edges rows: {edges.count():,}")

# COMMAND ----------

# MAGIC %md ## Done -- next steps
# MAGIC 1. If `node_types`/`via_keys` already existed at `NODE_TYPES_PATH`/
# MAGIC    `VIA_KEYS_PATH` when this ran, they were read back, not recomputed --
# MAGIC    check the printed log lines ("File exists ... Skipping execution") to
# MAGIC    confirm.
# MAGIC 2. Feed `node_types`/`edges` into `IDGraph_CC_Databricks.py` for connected
# MAGIC    components + summary stats.
