# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — component quality vs Affinity (notebook A: foundation)
# MAGIC
# MAGIC Tags every component (singletons included) against the Affinity crosswalk
# MAGIC and stores the inputs any quality model needs. No modelling here -- the
# MAGIC models (notebook B, `IDGraph_Component_Quality_Models_Databricks.py`) all
# MAGIC read these outputs, so they're compared on one foundation. Design:
# MAGIC `IDGraph_Journal.md` §14.
# MAGIC
# MAGIC Affinity is used ONLY to label and evaluate -- never to split or merge a
# MAGIC specific component.
# MAGIC
# MAGIC | class | meaning |
# MAGIC |---|---|
# MAGIC | 0 | no node in the component maps to exactly one `AFS_INDIVIDUAL_ID` |
# MAGIC | 1 | its single-person nodes all map to the same `AFS_INDIVIDUAL_ID` |
# MAGIC | 2 | its single-person nodes map to more than one `AFS_INDIVIDUAL_ID` |
# MAGIC
# MAGIC Only single-person nodes (matching exactly one AFS individual) are labels.
# MAGIC A node Affinity maps to 2+ people (a shared device, a household email) says
# MAGIC nothing about what the graph merged -- on IDGraph5 they were 44% of matched
# MAGIC singletons, all class 2 under the old rule. They're counted
# MAGIC (`n_multi_afs`) but don't set the class, `incomplete` or `person_spread`.
# MAGIC
# MAGIC Observed labels per component (from Affinity, not predictions):
# MAGIC `over_merged` (class 2), `incomplete` (an AFS person in it also has nodes in
# MAGIC another component) -- both null for class 0. `judgeable` = class 2 was
# MAGIC possible (>= 2 single-person nodes).
# MAGIC Predicted probabilities for every component come from notebook B.
# MAGIC
# MAGIC Outputs live in three places, by what they depend on:
# MAGIC - `COMMON_PATH` -- source data + denylist only. Reused by every graph
# MAGIC   built from the same filtered_base, whatever its MAX_FANOUT.
# MAGIC - `GRAPH_PATH` -- this graph's caps and components; shared by tune and holdout.
# MAGIC - `SPLIT_PATH` -- this graph x SCORE_SPLIT. Only AFS people in that split
# MAGIC   count as matches (others are treated as unmatched).
# MAGIC - `OUT_GRAPH_PATH` / `OUT_SPLIT_PATH` -- the per-component outputs. Same as
# MAGIC   GRAPH_PATH / SPLIT_PATH unless SAMPLE_PCT < 100, then under
# MAGIC   `component_quality_sample<pct>/`.

# COMMAND ----------

# Spark settings used for every IDGraph Databricks notebook.
spark.conf.set("spark.sql.shuffle.partitions", "20000")
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
# One-pass job: the disk cache only competes with shuffle files for local disk
# (a "No space left on device" in dbio_cache_root killed an earlier run).
spark.conf.set("spark.databricks.io.cache.enabled", "false")

# COMMAND ----------

# MAGIC %md ## Imports -- `stage_dataframe` from `python-databricks/staging_df.py`

# COMMAND ----------

import os
import sys

# In a Databricks Git folder the working directory is this notebook's folder;
# staging_df.py lives one level up, in python-databricks/.
sys.path.append(os.path.abspath(".."))

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from staging_df import _path_exists, in_slice, stage_dataframe, stage_sliced

# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

# The graph being scored. Any build works; MAX_FANOUT must match the caps that
# build used, per source (IDGraph4 = 32 everywhere, IDGraph5 = 16 everywhere).
# It decides which keys count as linking a node vs "dropped by cap".
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
MAX_FANOUT = {"id5": 16, "liid": 16, "tapad": 16}
RESULT_PATH = f"{GRAPH_ROOT}/cc_output/result"  # delta: id, component, component_size

# filtered_base/fanout don't depend on MAX_FANOUT -- IDGraph4 and IDGraph5 hold
# identical copies (verified by the fanout sweep), so read one for every graph.
SOURCE_BUILD_PATH = "s3://data-science-research/ravi/IDGraph5/build_output"

# node_labels is graph-independent (edge-labels notebook, Step 0a).
NODE_LABELS_PATH = "s3://data-science-research/ravi/IDGraph5/edge_labels/node_labels"

# "tune" for analysis / model training, "holdout" for final scoring only.
# Widget so a job can run this notebook twice: a task with no parameter gets
# "tune"; the second task passes SCORE_SPLIT=holdout and must depend on the
# first -- both write the shared graph/ folder, so never run them at once.
dbutils.widgets.dropdown("SCORE_SPLIT", "tune", ["tune", "holdout"])
SCORE_SPLIT = dbutils.widgets.get("SCORE_SPLIT")

# Reused across graphs built from the same filtered_base + DENYLIST. If either
# changes, point this at a new folder.
COMMON_PATH = "s3://data-science-research/ravi/IDGraph_common/component_quality"
GRAPH_PATH = f"{GRAPH_ROOT}/component_quality/graph"
SPLIT_PATH = f"{GRAPH_ROOT}/component_quality/{SCORE_SPLIT}"

# Which Affinity-matched nodes make the labels: "all" = HEM + IDFA + AAID (every
# type Affinity matches), "hem" = HEM only. Everything from component_afs on is
# written to its own folder ("_hem" suffix); matched_nodes and the graph tables
# are shared. A widget, so one job can run both label sets.
dbutils.widgets.dropdown("LABEL_SET", "all", ["all", "hem"])
LABEL_SET = dbutils.widgets.get("LABEL_SET")
_LABEL_SUFFIX = "" if LABEL_SET == "all" else f"_{LABEL_SET}"

# Percent of components analysed (by hash of the component id), e.g. 5 for a
# quick end-to-end test. A widget, so one job-level parameter sets it for every
# task (A tune, A holdout, B). The common tables, node_sources and the Affinity
# matching (matched_nodes .. component_matches) always cover all nodes: they're
# shared with the full run, and `incomplete` needs all of a person's components.
# Everything per component is sampled and written under OUT_*_PATH.
dbutils.widgets.text("SAMPLE_PCT", "100")
SAMPLE_PCT = int(dbutils.widgets.get("SAMPLE_PCT"))
_SAMPLE_ROOT = f"{GRAPH_ROOT}/component_quality" + ("" if SAMPLE_PCT >= 100 else f"_sample{SAMPLE_PCT}")
OUT_GRAPH_PATH = f"{_SAMPLE_ROOT}/graph"
OUT_SPLIT_PATH = f"{_SAMPLE_ROOT}/{SCORE_SPLIT}{_LABEL_SUFFIX}"

VIA_KEY_COLS = [
    ("id5", "individual_id5"),
    ("liid", "liid"),
    ("tapad", "individual_id_tapad"),
]
VIA_LABELS = [v for v, _ in VIA_KEY_COLS]

# Must match DENYLIST in IDGraph_Build_Databricks.py (applied after filtered_base
# is staged, so the staged table still contains these rows).
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}

# Off by default: run with elastic disk (autoscaling local storage) on the job
# cluster. If a run still fills local disk, set True -- the heaviest steps (common
# node_via_groups, nodes, component tokens/structure, matched nodes,
# component_summary) then run in N_SLICES passes, each shuffling 1/N of the data,
# at the cost of re-reading their inputs N times. Delete a step's folder before
# changing SLICE_HEAVY_STEPS or N_SLICES for it.
SLICE_HEAVY_STEPS = False
N_SLICES = 8
_N = N_SLICES if SLICE_HEAVY_STEPS else 1

TYPE_SHORT = {"HEM_MD5": "HEM", "HARDWARE_IDFA": "IDFA", "HARDWARE_ANDROID_AD_ID": "AAID"}
MAID_TYPES = ["HARDWARE_IDFA", "HARDWARE_ANDROID_AD_ID"]
# Short type names that get their own count column (n_type_<T>) and has/hasn't
# breakdown. Any other type still shows up in type_set and n_other.
ID_TYPES = ["HEM", "IDFA", "AAID", "TTD", "ID5_UID", "ST"]

_inputs = [RESULT_PATH, NODE_LABELS_PATH] + [
    f"{SOURCE_BUILD_PATH}/{v}_{t}" for v in VIA_LABELS for t in ("fanout", "filtered_base")
]
_missing = [p for p in _inputs if not _path_exists(p)]
assert not _missing, f"missing inputs: {_missing}"
print(
    f"scoring {GRAPH_ROOT} (MAX_FANOUT={MAX_FANOUT}) on the {SCORE_SPLIT} split, "
    f"labels from {LABEL_SET} matched nodes, {SAMPLE_PCT}% of components"
)

# COMMAND ----------

# MAGIC %md ## Helpers

# COMMAND ----------


def _node_type(col="node_id"):
    # node_id is "<type>:<value>" (see base_fn in the build notebook).
    return F.substring_index(F.col(col), ":", 1)


_type_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


def _type_short(type_col):
    return F.coalesce(_type_short_map[F.col(type_col)], F.col(type_col))


def _type_group(type_col):
    return (
        F.when(F.col(type_col).startswith("HEM_"), F.lit("HEM"))
        .when(F.col(type_col).isin(MAID_TYPES), F.lit("MAID"))
        .otherwise(F.lit("OTHER"))
    )


def _in_sample(col_name):
    # Salted so the sample doesn't line up with in_slice's hash of the same column.
    if SAMPLE_PCT >= 100:
        return F.lit(True)
    return F.pmod(F.xxhash64(F.lit("sample"), F.col(col_name)), F.lit(100)) < F.lit(SAMPLE_PCT)


def _sorted_join(arr):
    # "" from an empty array becomes null, so "no sources" stays distinguishable.
    return F.when(F.size(arr) > 0, F.concat_ws("+", F.array_sort(arr)))


def _size_bucket(col, bounds):
    expr = None
    lo = bounds[0]
    for hi in bounds:
        label = str(hi) if lo == hi else f"{lo}-{hi}"
        cond = F.col(col) <= hi
        expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
        lo = hi + 1
    return expr.otherwise(F.lit(f"{bounds[-1] + 1}+"))


COMPONENT_SIZE_BOUNDS = [1, 2, 4, 8, 16, 32, 64, 128, 1_000, 10_000, 100_000]
GROUP_SIZE_BOUNDS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]
DROPPED_SIZE_BOUNDS = [min(MAX_FANOUT.values()) + 1, 24, 32, 48, 64, 128, 256, 1024]

# COMMAND ----------

# MAGIC %md ## Common -- every node's group sizes per source (cap-independent)
# MAGIC 1. `node_key_sizes_{via}`: `{via}_filtered_base` x `{via}_fanout`, one source
# MAGIC    at a time -- narrow distinct (node, group size, denylisted) rows. The only
# MAGIC    shuffle of `fanout`, and AQE's skew join splits huge keys.
# MAGIC 2. `node_via_groups`: those three unioned and grouped by node once -- one row
# MAGIC    per node with, per source, the sorted distinct sizes of its non-denylisted
# MAGIC    keys (`group_sizes_{via}`) and whether it has a denylisted key
# MAGIC    (`any_denylisted_{via}`, null = node not in that source). If sliced, each
# MAGIC    slice only re-reads the narrow tables.
# MAGIC
# MAGIC Any graph's caps are applied to this later, row by row, so it's computed
# MAGIC once and reused, and no graph has to shuffle by node again.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def node_key_sizes_fn(via_label, key_col, spark=None, write_path=None):
    base = spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_filtered_base").select("node_id", key_col)
    fanout = spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_fanout").select(key_col, "n_distinct")
    deny = list(DENYLIST.get(via_label, set()))
    is_deny = F.col(key_col).isin(deny) if deny else F.lit(False)
    return (
        base.join(fanout, on=key_col)
        .select("node_id", F.col("n_distinct").alias("size"), is_deny.alias("is_deny"))
        .distinct()
    )


@stage_dataframe(write_format="parquet")
def node_via_groups_fn(parts, slice_id=0, n_slices=1, spark=None, write_path=None):
    # parts are in VIA_KEY_COLS order (built by the list comprehension below).
    unioned = None
    for p, v in zip(parts, VIA_LABELS):
        p = p.filter(in_slice("node_id", slice_id, n_slices)).withColumn("via_type", F.lit(v))
        unioned = p if unioned is None else unioned.unionByName(p)
    aggs = []
    for v in VIA_LABELS:
        is_v = F.col("via_type") == v
        # collect_set drops nulls, so other sources' rows and denylisted keys are left out.
        aggs.append(F.array_sort(F.collect_set(F.when(is_v & ~F.col("is_deny"), F.col("size")))).alias(f"group_sizes_{v}"))
        aggs.append(F.max(F.when(is_v, F.col("is_deny").cast("int"))).alias(f"any_denylisted_{v}"))
    return unioned.groupBy("node_id").agg(*aggs)


node_key_sizes = [
    node_key_sizes_fn(via_label, key_col, spark=spark, write_path=f"{COMMON_PATH}/node_key_sizes_{via_label}")
    for via_label, key_col in VIA_KEY_COLS
]
node_via_groups = stage_sliced(
    node_via_groups_fn, node_key_sizes, spark=spark, write_path=f"{COMMON_PATH}/node_via_groups", n_slices=_N
)

# COMMAND ----------

# MAGIC %md ## Graph 1 -- apply this graph's caps: sources and singleton reasons per node
# MAGIC Per node and source, the build links the node iff it has a key of size
# MAGIC 2..MAX_FANOUT[source] that isn't denylisted. Otherwise its reason is:
# MAGIC - `dropped` -- it has a key of size > cap (`dropped_size_{via}` = smallest)
# MAGIC - `denylisted` -- its only shared key is on the denylist
# MAGIC - `alone` -- every key it has contains only itself
# MAGIC
# MAGIC `source_set` = sources that link the node (null = none: should be a
# MAGIC singleton); `raw_source_set` = every source it appears in;
# MAGIC `singleton_reasons` = reasons from the sources that don't link it.
# MAGIC Row by row on `node_via_groups` -- no shuffle.

# COMMAND ----------


def _reason(via_label):
    cap = MAX_FANOUT[via_label]
    sizes = F.col(f"group_sizes_{via_label}")
    deny = F.col(f"any_denylisted_{via_label}")
    kept = F.filter(sizes, lambda x: (x >= 2) & (x <= cap))
    dropped = F.filter(sizes, lambda x: x > cap)
    reason = (
        F.when(deny.isNull(), F.lit(None).cast("string"))  # not in this source
        .when(F.size(kept) > 0, F.lit("linked"))
        .when(F.size(dropped) > 0, F.lit("dropped"))
        .when(deny == 1, F.lit("denylisted"))
        .otherwise(F.lit("alone"))
    )
    return reason, F.array_max(kept), F.array_min(dropped)


@stage_dataframe(write_format="parquet")
def node_sources_fn(groups_df, spark=None, write_path=None):
    cols = {}
    for v in VIA_LABELS:
        reason, max_kept, min_dropped = _reason(v)
        cols[f"reason_{v}"] = reason
        cols[f"max_group_size_{v}"] = F.when(reason == "linked", max_kept)
        cols[f"dropped_size_{v}"] = F.when(reason == "dropped", min_dropped)
    df = groups_df.select("node_id", *[c.alias(name) for name, c in cols.items()])

    def labels(cond_fn):
        # array_compact drops the nulls from sources where cond_fn is false.
        return F.array_compact(F.array(*[F.when(cond_fn(v), F.lit(v)) for v in VIA_LABELS]))

    reasons = F.array_compact(F.array(*[
        F.when(F.col(f"reason_{v}") != "linked", F.concat(F.lit(f"{v}:"), F.col(f"reason_{v}")))
        for v in VIA_LABELS
    ]))
    return df.select(
        "node_id",
        _sorted_join(labels(lambda v: F.col(f"reason_{v}") == "linked")).alias("source_set"),
        _sorted_join(labels(lambda v: F.col(f"reason_{v}").isNotNull())).alias("raw_source_set"),
        _sorted_join(reasons).alias("singleton_reasons"),
        F.least(*[F.col(f"dropped_size_{v}") for v in VIA_LABELS]).alias("min_dropped_size"),
        *[F.col(f"max_group_size_{v}") for v in VIA_LABELS],
        *[F.col(f"dropped_size_{v}") for v in VIA_LABELS],
    )


node_sources = node_sources_fn(node_via_groups, spark=spark, write_path=f"{GRAPH_PATH}/node_sources")

# COMMAND ----------

# MAGIC %md ## Graph 2 -- nodes with components; structure, tokens, singletons
# MAGIC `nodes`: every node with its component and sources (one join of the
# MAGIC component table with `node_sources`).
# MAGIC Consistency checks (both expected 0 -- otherwise MAX_FANOUT/DENYLIST don't
# MAGIC match the build, or filtered_base differs from it):
# MAGIC - nodes in components of size >= 2 that no source links (`n_unlinked`)
# MAGIC - singletons that some source links, or that appear in no source
# MAGIC
# MAGIC `component_tokens` is the one group-by over all nodes; `component_structure`
# MAGIC rolls it up by component.
# MAGIC
# MAGIC Token = source set x ID type, one per node (e.g. `liid+tapad:HEM`,
# MAGIC `tapad:TTD`). HEM_MD5/IDFA/AAID are shortened; other types keep their name.

# COMMAND ----------

result = spark.read.format("delta").load(RESULT_PATH)


@stage_dataframe(write_format="parquet")
def nodes_fn(result_df, node_sources_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    return (
        result_df.select(F.col("id").alias("node_id"), "component", "component_size")
        .filter(in_slice("node_id", slice_id, n_slices) & _in_sample("component"))
        .join(node_sources_df.filter(in_slice("node_id", slice_id, n_slices)), on="node_id", how="left")
        .withColumn("type", _node_type())
    )


@stage_dataframe(write_format="parquet")
def component_tokens_fn(nodes_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    # The one pass over all nodes grouped by component: token counts, plus what
    # component_structure needs (it rolls this much smaller table up by component).
    token = F.concat(F.coalesce(F.col("source_set"), F.lit("unlinked")), F.lit(":"), _type_short("type"))
    per_via = [F.max(f"max_group_size_{v}").alias(f"max_group_size_{v}") for v in VIA_LABELS]
    return (
        nodes_df.filter((F.col("component_size") > 1) & in_slice("component", slice_id, n_slices))
        .groupBy("component", token.alias("token"))
        .agg(
            F.count("*").alias("n_nodes"),
            F.first("component_size").alias("component_size"),
            # The token fixes the node type and whether it's linked, so these are constant per row.
            F.first(_type_group("type")).alias("type_group"),
            F.first(F.col("source_set").isNull()).alias("unlinked"),
            *per_via,
        )
    )


@stage_dataframe(write_format="parquet")
def component_structure_fn(tokens_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    tshort = F.substring_index("token", ":", -1)  # token = "<sources>:<short type>"
    per_via = [F.max(f"max_group_size_{v}").alias(f"max_group_size_{v}") for v in VIA_LABELS]
    s = (
        tokens_df.filter(in_slice("component", slice_id, n_slices))
        .groupBy("component")
        .agg(
            F.first("component_size").alias("component_size"),
            F.sum(F.when(F.col("type_group") == "HEM", F.col("n_nodes")).otherwise(0)).alias("n_hem"),
            F.sum(F.when(F.col("type_group") == "MAID", F.col("n_nodes")).otherwise(0)).alias("n_maid"),
            F.sum(F.when(F.col("type_group") == "OTHER", F.col("n_nodes")).otherwise(0)).alias("n_other"),
            F.sum(F.when(F.col("unlinked"), F.col("n_nodes")).otherwise(0)).alias("n_unlinked"),
            *[F.sum(F.when(tshort == t, F.col("n_nodes")).otherwise(0)).alias(f"n_type_{t}") for t in ID_TYPES],
            _sorted_join(F.collect_set(tshort)).alias("type_set"),
            *per_via,
        )
    )
    # concat_ws skips nulls -> only the sources that link something in the component.
    sources = F.concat_ws(
        "+", *[F.when(F.col(f"max_group_size_{v}").isNotNull(), F.lit(v)) for v in VIA_LABELS]
    )
    return s.withColumn("sources", sources)


@stage_dataframe(write_format="parquet")
def singletons_fn(nodes_df, spark=None, write_path=None):
    return (
        nodes_df.filter(F.col("component_size") == 1)
        .select(
            "component",
            "node_id",
            "type",
            "source_set",
            "raw_source_set",
            "singleton_reasons",
            "min_dropped_size",
            *[f"dropped_size_{v}" for v in VIA_LABELS],
        )
        .fillna({"raw_source_set": "none", "singleton_reasons": "none"})
    )


nodes = stage_sliced(nodes_fn, result, node_sources, spark=spark, write_path=f"{OUT_GRAPH_PATH}/nodes", n_slices=_N)
component_tokens = stage_sliced(
    component_tokens_fn, nodes, spark=spark, write_path=f"{OUT_GRAPH_PATH}/component_tokens", n_slices=_N
)
component_structure = stage_sliced(
    component_structure_fn, component_tokens, spark=spark, write_path=f"{OUT_GRAPH_PATH}/component_structure", n_slices=_N
)
singletons = singletons_fn(nodes, spark=spark, write_path=f"{OUT_GRAPH_PATH}/singletons")

_n_unlinked = component_structure.filter(F.col("n_unlinked") > 0).count()
_n_bad_singletons = singletons.filter(F.col("source_set").isNotNull() | (F.col("raw_source_set") == "none")).count()
print(f"components with nodes no source links (expected 0): {_n_unlinked:,}")
print(f"singletons linked by a source, or in no source (expected 0): {_n_bad_singletons:,}")

# COMMAND ----------

# MAGIC %md ## Split 1 -- Affinity matches, people, per-component match stats
# MAGIC `matched_nodes`: graph nodes whose AFS individual(s) fall in SCORE_SPLIT.
# MAGIC `person_spread`: per AFS person, how many of our nodes and components
# MAGIC they're in -- `n_components >= 2` means split (under-merged).
# MAGIC `component_matches`: per component, n matched nodes, n distinct AFS
# MAGIC people, and `incomplete`.

# COMMAND ----------

node_labels = spark.read.parquet(NODE_LABELS_PATH)


@stage_dataframe(write_format="parquet")
def matched_nodes_fn(result_df, labels_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    labels = labels_df.filter(
        (F.col("split") == SCORE_SPLIT) & in_slice("node_id", slice_id, n_slices)
    ).select("node_id", "afs_ids")
    return (
        result_df.select(F.col("id").alias("node_id"), "component")
        .filter(in_slice("node_id", slice_id, n_slices))
        .join(labels, on="node_id")
    )


@stage_dataframe(write_format="parquet")
def component_afs_fn(matched_df, spark=None, write_path=None):
    # Single-person nodes only -- see the class table at the top.
    return (
        matched_df.filter(F.size("afs_ids") == 1)
        .select("component", F.col("afs_ids")[0].alias("afs_id"))
        .groupBy("component", "afs_id")
        .agg(F.count("*").alias("n_nodes"))
    )


@stage_dataframe(write_format="parquet")
def person_spread_fn(component_afs_df, spark=None, write_path=None):
    return component_afs_df.groupBy("afs_id").agg(
        F.sum("n_nodes").alias("n_nodes"), F.count("*").alias("n_components")
    )


@stage_dataframe(write_format="parquet")
def component_matches_fn(matched_df, component_afs_df, person_df, spark=None, write_path=None):
    multi = (F.size("afs_ids") >= 2).cast("long")
    n_matched = matched_df.groupBy("component").agg(
        F.count("*").alias("n_matched"),
        F.sum(multi).alias("n_multi_afs"),
        F.sum(1 - multi).alias("n_single_afs"),
    )
    per_comp = (
        component_afs_df.join(person_df.select("afs_id", "n_components"), on="afs_id")
        .groupBy("component")
        .agg(
            F.count("*").alias("n_afs"),
            F.max((F.col("n_components") >= 2).cast("int")).alias("incomplete"),
        )
    )
    # Left: components whose matches are all multi-person nodes keep their counts.
    return n_matched.join(per_comp, on="component", how="left")


matched_nodes = stage_sliced(
    matched_nodes_fn, result, node_labels, spark=spark, write_path=f"{SPLIT_PATH}/matched_nodes", n_slices=_N
)
# "_single_afs" = built from single-person nodes only (2026-11-10). The older
# component_afs / person_spread / component_matches folders counted every node
# and can be deleted.
# The nodes whose matches make the labels (LABEL_SET).
label_nodes = matched_nodes if LABEL_SET == "all" else matched_nodes.filter(F.col("node_id").startswith("HEM_"))
component_afs = component_afs_fn(
    label_nodes, spark=spark, write_path=f"{SPLIT_PATH}/component_afs_single_afs{_LABEL_SUFFIX}"
)
person_spread = person_spread_fn(
    component_afs, spark=spark, write_path=f"{SPLIT_PATH}/person_spread_single_afs{_LABEL_SUFFIX}"
)
component_matches = component_matches_fn(
    label_nodes, component_afs, person_spread,
    spark=spark, write_path=f"{SPLIT_PATH}/component_matches_single_afs{_LABEL_SUFFIX}",
)

# COMMAND ----------

# MAGIC %md ## Split 2 -- `component_summary`: one row per component (the model input)
# MAGIC Non-singletons from `component_structure`, singletons from `singletons`
# MAGIC (`sources = "singleton"`, plus their reasons), joined to match stats.
# MAGIC Notebook B reads this table and `graph/component_tokens`.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def component_summary_fn(structure_df, singletons_df, matches_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    group = _type_group("type")
    this_slice = in_slice("component", slice_id, n_slices)
    single = singletons_df.filter(this_slice).select(
        "component",
        F.lit(1).cast("long").alias("component_size"),
        (group == "HEM").cast("long").alias("n_hem"),
        (group == "MAID").cast("long").alias("n_maid"),
        (group == "OTHER").cast("long").alias("n_other"),
        F.lit("singleton").alias("sources"),
        _type_short("type").alias("singleton_type"),
        _type_short("type").alias("type_set"),
        *[(_type_short("type") == t).cast("long").alias(f"n_type_{t}") for t in ID_TYPES],
        "raw_source_set",
        "singleton_reasons",
        "min_dropped_size",
        *[f"dropped_size_{v}" for v in VIA_LABELS],
    )
    multi = structure_df.filter(this_slice).drop("n_unlinked")
    s = (
        multi.unionByName(single, allowMissingColumns=True)
        .join(matches_df.filter(this_slice), on="component", how="left")
        .fillna({"n_matched": 0, "n_multi_afs": 0, "n_single_afs": 0, "n_afs": 0, "incomplete": 0})
    )
    cls = F.when(F.col("n_afs") == 0, 0).when(F.col("n_afs") == 1, 1).otherwise(2)
    id_types = (
        F.when((F.col("n_hem") > 0) & (F.col("n_maid") > 0), F.lit("HEM+MAID"))
        .when(F.col("n_hem") > 0, F.lit("HEM only"))
        .when(F.col("n_maid") > 0, F.lit("MAID only"))
        .otherwise(F.lit("neither"))
    )
    # over_merged / incomplete are observed labels -- unknown (null) for class 0,
    # where Affinity says nothing. Predictions for every component come from notebook B.
    matched = F.col("class") > 0
    return (
        s.withColumn("class", cls)
        .withColumn("over_merged", F.when(matched, F.col("class") == 2))
        .withColumn("incomplete", F.when(matched, F.col("incomplete") == 1))
        .withColumn("judgeable", F.col("n_single_afs") >= 2)
        .withColumn("id_types", id_types)
        .withColumn("size_bucket", _size_bucket("component_size", COMPONENT_SIZE_BOUNDS))
        .withColumn("size_order", F.col("component_size"))
    )


component_summary = stage_sliced(
    component_summary_fn,
    component_structure,
    singletons,
    component_matches,
    spark=spark,
    write_path=f"{OUT_SPLIT_PATH}/component_summary",
    n_slices=_N,
)

# COMMAND ----------

# MAGIC %md ## Split 3 -- breakdown tables
# MAGIC Per cell: class counts, plus (computed in pandas below)
# MAGIC - `pct_over_merged` = class 2 / (class 1 + class 2)
# MAGIC - `pct_over_merged_judgeable` = class 2 / judgeable (only where class 2 was possible)
# MAGIC - `pct_incomplete` = incomplete / (class 1 + class 2)
# MAGIC - `pct_ids_over_merged` = nodes in class-2 components / nodes in class 1 + 2 components
# MAGIC - `pct_multi_afs_nodes` = matched nodes Affinity maps to 2+ people / all matched nodes
# MAGIC   (ignored for labels); `n_multi_afs_only` = components matched only through such nodes
# MAGIC   (class 0)
# MAGIC
# MAGIC `type_set` = the ID types present in a component (e.g. `AAID+HEM+TTD`), like
# MAGIC `sources` for ID types. `breakdown_has_type_size` compares components with vs
# MAGIC without each type in `ID_TYPES`, size bucket by size bucket -- e.g. whether
# MAGIC components containing TTD over-merge more than those of the same size without.
# MAGIC `breakdown_source_group_size`: non-singletons by each linking source's largest
# MAGIC group (`max_group_size_{via}` bucket) x component size -- the cap question:
# MAGIC at which group size does each source start over-merging?

# COMMAND ----------


def _breakdown(df, cols, *extra_aggs):
    matched = F.col("class") > 0
    return df.groupBy(*cols).agg(
        *extra_aggs,
        F.count("*").alias("n_components"),
        F.sum((F.col("class") == 0).cast("long")).alias("n_class0"),
        F.sum((F.col("class") == 1).cast("long")).alias("n_class1"),
        F.sum((F.col("class") == 2).cast("long")).alias("n_class2"),
        F.sum(F.col("judgeable").cast("long")).alias("n_judgeable"),
        F.sum(F.col("incomplete").cast("long")).alias("n_incomplete"),
        F.sum(F.when(matched, F.col("component_size")).otherwise(0)).alias("ids_matched_components"),
        F.sum(F.when(F.col("over_merged"), F.col("component_size")).otherwise(0)).alias("ids_over_merged"),
        F.sum("n_matched").alias("nodes_matched"),
        F.sum("n_multi_afs").alias("nodes_multi_afs"),
        F.sum(((F.col("n_matched") > 0) & (F.col("class") == 0)).cast("long")).alias("n_multi_afs_only"),
        F.min("size_order").alias("size_order"),
    )


BREAKDOWNS = {
    "by_size": ["size_bucket"],
    "by_sources": ["sources"],
    "by_id_types": ["id_types"],
    "by_sources_size": ["sources", "size_bucket"],
    "by_sources_id_types": ["sources", "id_types"],
    "by_type_set": ["type_set"],
    "by_type_set_size": ["type_set", "size_bucket"],
}


@stage_dataframe(write_format="parquet")
def breakdown_fn(summary_df, cols, spark=None, write_path=None):
    return _breakdown(summary_df, cols)


@stage_dataframe(write_format="parquet")
def singleton_breakdown_fn(summary_df, spark=None, write_path=None):
    s = summary_df.filter(F.col("sources") == "singleton").withColumn(
        "dropped_size_bucket",
        F.when(F.col("min_dropped_size").isNull(), F.lit("-")).otherwise(
            _size_bucket("min_dropped_size", DROPPED_SIZE_BOUNDS)
        ),
    )
    return _breakdown(s, ["singleton_type", "raw_source_set", "singleton_reasons", "dropped_size_bucket"]).withColumn(
        "size_order", F.lit(1)
    )


@stage_dataframe(write_format="parquet")
def has_type_breakdown_fn(summary_df, spark=None, write_path=None):
    # Components with vs without each ID type, by size -- one row per component
    # and type after the explode, so a single pass covers every type.
    per_type = F.explode(F.array(*[
        F.struct(F.lit(t).alias("id_type"), (F.col(f"n_type_{t}") > 0).alias("has_type")) for t in ID_TYPES
    ]))
    s = summary_df.withColumn("_t", per_type).select("*", "_t.id_type", "_t.has_type").drop("_t")
    return _breakdown(s, ["id_type", "has_type", "size_bucket"])


@stage_dataframe(write_format="parquet")
def source_group_breakdown_fn(summary_df, spark=None, write_path=None):
    # Non-singletons by each source's largest linking group: one row per
    # component and source that linked it. Kept by component size too, so a
    # big-group effect can be told apart from a big-component one.
    per_src = F.explode(F.array(*[
        F.struct(F.lit(v).alias("source"), F.col(f"max_group_size_{v}").alias("max_group_size")) for v in VIA_LABELS
    ]))
    s = (
        summary_df.filter(F.col("sources") != "singleton")
        .withColumn("_s", per_src)
        .select("*", "_s.source", "_s.max_group_size")
        .drop("_s")
        .filter(F.col("max_group_size").isNotNull())
        .withColumn("max_group_bucket", _size_bucket("max_group_size", GROUP_SIZE_BOUNDS))
    )
    return _breakdown(s, ["source", "max_group_bucket", "size_bucket"], F.min("max_group_size").alias("group_order"))


@stage_dataframe(write_format="parquet")
def person_split_summary_fn(person_df, spark=None, write_path=None):
    # Only people with 2+ of their nodes in our graph can be split.
    return (
        person_df.filter(F.col("n_nodes") >= 2)
        .agg(
            F.count("*").alias("n_people_splittable"),
            F.sum((F.col("n_components") >= 2).cast("long")).alias("n_people_split"),
        )
    )


breakdowns = {
    name: breakdown_fn(component_summary, cols, spark=spark, write_path=f"{OUT_SPLIT_PATH}/breakdown_{name}")
    for name, cols in BREAKDOWNS.items()
}
source_group_breakdown = source_group_breakdown_fn(
    component_summary, spark=spark, write_path=f"{OUT_SPLIT_PATH}/breakdown_source_group_size"
)
has_type_breakdown = has_type_breakdown_fn(
    component_summary, spark=spark, write_path=f"{OUT_SPLIT_PATH}/breakdown_has_type_size"
)
singleton_breakdown = singleton_breakdown_fn(
    component_summary, spark=spark, write_path=f"{OUT_SPLIT_PATH}/breakdown_singletons"
)
person_split_summary = person_split_summary_fn(
    person_spread, spark=spark, write_path=f"{OUT_SPLIT_PATH}/person_split_summary"
)

# COMMAND ----------

# MAGIC %md ## Results

# COMMAND ----------


def _add_rates(pdf):
    pdf = pdf.copy()
    matched = pdf["n_class1"] + pdf["n_class2"]
    for c in (0, 1, 2):
        pdf[f"pct_class{c}"] = (100 * pdf[f"n_class{c}"] / pdf["n_components"]).round(2)
    pdf["pct_over_merged"] = (100 * pdf["n_class2"] / matched.where(matched > 0)).round(2)
    pdf["pct_over_merged_judgeable"] = (100 * pdf["n_class2"] / pdf["n_judgeable"].where(pdf["n_judgeable"] > 0)).round(2)
    pdf["pct_incomplete"] = (100 * pdf["n_incomplete"] / matched.where(matched > 0)).round(2)
    ids = pdf["ids_matched_components"]
    pdf["pct_ids_over_merged"] = (100 * pdf["ids_over_merged"] / ids.where(ids > 0)).round(2)
    nodes = pdf["nodes_matched"]
    pdf["pct_multi_afs_nodes"] = (100 * pdf["nodes_multi_afs"] / nodes.where(nodes > 0)).round(2)
    return pdf


_SHOW = [
    "n_components", "pct_class0", "pct_class1", "pct_class2",
    "pct_over_merged", "pct_over_merged_judgeable", "pct_incomplete", "pct_ids_over_merged",
    "pct_multi_afs_nodes",
]
# Components at least this big count as "giant" in the headline -- IDGraph5 has
# one (697M nodes) and it dominates any node-weighted rate.
GIANT_MIN_SIZE = 100_001

breakdowns_pd = {name: _add_rates(df.toPandas()) for name, df in breakdowns.items()}
has_type_pd = _add_rates(has_type_breakdown.toPandas())
source_group_size_pd = _add_rates(source_group_breakdown.toPandas())
# Collapsed over component size (counts add up, so rates are recomputed).
_g = source_group_breakdown.toPandas().drop(columns=["size_bucket", "size_order"])
source_group_pd = _add_rates(
    _g.groupby(["source", "max_group_bucket"], as_index=False).agg(
        {c: ("min" if c == "group_order" else "sum") for c in _g.columns if c not in ("source", "max_group_bucket")}
    )
)

_size_pd = breakdowns_pd["by_size"]
_count_cols = [c for c in _size_pd.columns if c.startswith(("n_", "ids_", "nodes_"))]
_tot = _add_rates(_size_pd[_count_cols].sum().to_frame().T)
_is_giant = _size_pd["size_order"] >= GIANT_MIN_SIZE
_giant = _size_pd[_is_giant][_count_cols].sum()
_tot_ng = _add_rates(_size_pd[~_is_giant][_count_cols].sum().to_frame().T)
_people = person_split_summary.toPandas().iloc[0]
print(f"=== [{SCORE_SPLIT}, labels: {LABEL_SET}] headline: {GRAPH_ROOT}, {SAMPLE_PCT}% of components ===")
print(f"components:                {int(_tot['n_components'].iloc[0]):,}")
print(f"% over-merged (components): {_tot['pct_over_merged'].iloc[0]}  (judgeable only: {_tot['pct_over_merged_judgeable'].iloc[0]})")
print(f"% over-merged (IDs):        {_tot['pct_ids_over_merged'].iloc[0]}")
print(
    f"  excl. {int(_giant['n_components']):,} giant (>= {GIANT_MIN_SIZE:,} IDs, "
    f"{int(_giant['ids_matched_components']):,} IDs): {_tot_ng['pct_ids_over_merged'].iloc[0]}"
)
print(f"% incomplete (components):  {_tot['pct_incomplete'].iloc[0]}")
print(
    f"% matched nodes -> 2+ people: {_tot['pct_multi_afs_nodes'].iloc[0]}  (not used as labels; "
    f"{int(_tot['n_multi_afs_only'].iloc[0]):,} components matched only through them -> class 0)"
)
_split_pct = 100 * _people["n_people_split"] / max(_people["n_people_splittable"], 1)
print(
    f"% people split:             {_split_pct:.2f}  "
    f"({int(_people['n_people_split']):,} of {int(_people['n_people_splittable']):,} with 2+ nodes in the graph)"
)
print()

_SORT = {
    "by_size": ["size_order"],
    "by_sources": ["sources"],
    "by_id_types": ["id_types"],
    "by_sources_size": ["sources", "size_order"],
    "by_sources_id_types": ["sources", "id_types"],
    "by_type_set": ["type_set"],
    "by_type_set_size": ["type_set", "size_order"],
}
# Long tables (one row per type set / singleton reason) stay on S3 only --
# printed, they push the key tables past Databricks' output limit.
_LONG = ["by_type_set", "by_type_set_size"]
for name, cols in BREAKDOWNS.items():
    if name in _LONG:
        continue
    print(f"=== [{SCORE_SPLIT}] {name} ===")
    print(breakdowns_pd[name].sort_values(_SORT[name])[cols + _SHOW].to_string(index=False))
    print()

_g_cols = ["source", "max_group_bucket"]
print(f"=== [{SCORE_SPLIT}] components by each source's largest linking group (MAX_FANOUT={MAX_FANOUT}) ===")
print(source_group_pd.sort_values(["source", "group_order"])[_g_cols + _SHOW].to_string(index=False))
print()

_h_cols = ["id_type", "size_bucket", "has_type"]
print(f"=== [{SCORE_SPLIT}] with vs without each ID type, by size ===")
print(has_type_pd.sort_values(["id_type", "size_order", "has_type"])[_h_cols + _SHOW].to_string(index=False))
print()

print(f"not printed (read from S3): " + ", ".join(
    [f"{OUT_SPLIT_PATH}/breakdown_{n}" for n in _LONG] + [f"{OUT_SPLIT_PATH}/breakdown_singletons"]
))

# COMMAND ----------

# MAGIC %md ## Full tables as CSV (paste back to save locally)

# COMMAND ----------

for name, cols in BREAKDOWNS.items():
    if name in _LONG:
        continue
    print(f"# {name}")
    print(breakdowns_pd[name].sort_values(_SORT[name]).drop(columns="size_order").to_csv(index=False))
print("# source_group_size")
print(
    source_group_size_pd.sort_values(["source", "group_order", "size_order"])
    .drop(columns=["group_order", "size_order"]).to_csv(index=False)
)
print("# has_type_size")
print(has_type_pd.sort_values(["id_type", "size_order", "has_type"]).drop(columns="size_order").to_csv(index=False))

# COMMAND ----------

# MAGIC %md ## Which source + node type make healthy components
# MAGIC Judgeable multi-node components only (a singleton can't show 2 people). A token
# MAGIC is `<sources linking the node>:<node type>`, as in `component_tokens`; a
# MAGIC component counts once for each token it contains. On the judgeable base:
# MAGIC - `pct_over_merged` = 2+ people in the component
# MAGIC - `pct_incomplete` = its person also has nodes in another component
# MAGIC - `pct_healthy` = neither
# MAGIC
# MAGIC Size drives both errors, so each rate is also compared with components of the
# MAGIC same size: `*_lift` = observed / expected from the overall rate in each size
# MAGIC bucket, weighted by the token's size mix. `healthy_lift` > 1 = healthier than
# MAGIC size alone predicts. Absolute rates depend on the split's size (journal);
# MAGIC lifts compare within one split.

# COMMAND ----------

_OUTCOMES = ["over_merged", "incomplete", "healthy"]


def _health_counts(df, cols):
    return df.groupBy(*cols).agg(
        F.count("*").alias("n_judgeable"),
        F.sum(F.col("over_merged").cast("long")).alias("n_over_merged"),
        F.sum(F.col("incomplete").cast("long")).alias("n_incomplete"),
        F.sum((~F.col("over_merged") & ~F.col("incomplete")).cast("long")).alias("n_healthy"),
    )


@stage_dataframe(write_format="parquet")
def token_health_fn(tokens_df, summary_df, spark=None, write_path=None):
    judged = summary_df.filter(F.col("judgeable")).select("component", "size_bucket", "over_merged", "incomplete")
    return _health_counts(tokens_df.select("component", "token").join(judged, on="component"), ["token", "size_bucket"])


token_health = token_health_fn(
    component_tokens, component_summary, spark=spark, write_path=f"{OUT_SPLIT_PATH}/token_health"
)
size_health = _health_counts(component_summary.filter(F.col("judgeable")), ["size_bucket"]).toPandas()

_th = token_health.toPandas()
_th["sources"] = _th["token"].str.rsplit(":", n=1).str[0]
_th["node_type"] = _th["token"].str.rsplit(":", n=1).str[1]


def _health_view(df, cols, n_col="n_judgeable", outcomes=_OUTCOMES, base=None):
    """Counts summed over `cols`, with % of n_col and lift vs the same size buckets
    (expected = each row's n_col x the rate in its size bucket, from `base` --
    default: df itself)."""
    base = df if base is None else base
    rate = base.groupby("size_bucket")[[n_col] + [f"n_{o}" for o in outcomes]].sum()
    df = df.copy()
    for o in outcomes:
        df[f"exp_{o}"] = df[n_col] * df["size_bucket"].map(rate[f"n_{o}"] / rate[n_col])
    g = df.groupby(cols, as_index=False)[[n_col] + [f"n_{o}" for o in outcomes] + [f"exp_{o}" for o in outcomes]].sum()
    for o in outcomes:
        g[f"pct_{o}"] = (100 * g[f"n_{o}"] / g[n_col]).round(1)
        g[f"{o}_lift"] = (g[f"n_{o}"] / g[f"exp_{o}"]).round(2)
    show = [n_col] + [f"pct_{o}" for o in outcomes] + [f"{o}_lift" for o in outcomes]
    return g[cols + show].sort_values("healthy_lift", ascending=False)


for _cols in [["node_type"], ["sources"], ["sources", "node_type"]]:
    print(f"=== [{SCORE_SPLIT}] health by {' x '.join(_cols)} (judgeable multi-node components) ===")
    print(_health_view(_th, _cols, base=size_health).to_string(index=False))
    print()

print("# token_health by size (CSV)")
print(_th.sort_values(["token", "size_bucket"]).to_csv(index=False))

# COMMAND ----------

# MAGIC %md ## Pairs: which sources + type pairs make healthy components
# MAGIC A pair = two nodes of a judgeable sampled component that share a key of size
# MAGIC 2..MAX_FANOUT in at least one source -- a direct link in this graph, taken
# MAGIC from the graph alone (all nodes, matched or not). `sources` = every source
# MAGIC with such a key for the pair (e.g. `liid+tapad` = both sources linked these
# MAGIC two nodes). Affinity enters only through the component's label. Counted per
# MAGIC component, like the token table: `n_judgeable` = components holding at least
# MAGIC one such pair; rates and lifts are over those components, against all
# MAGIC judgeable components of the same size. The giant component is left out (one
# MAGIC component, billions of pairs). Pair-level extras: `n_pairs`; `n_labeled` =
# MAGIC pairs whose two nodes each match one AFS person, and `pct_same_person` = the
# MAGIC share of those that are the same person (is the link itself right).

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def pair_members_fn(nodes_df, matched_df, summary_df, via_label, key_col, spark=None, write_path=None):
    base = spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_filtered_base").select("node_id", key_col)
    deny = list(DENYLIST.get(via_label, set()))
    if deny:
        base = base.filter(~F.col(key_col).isin(deny))
    keys = (
        spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_fanout")
        .filter((F.col("n_distinct") >= 2) & (F.col("n_distinct") <= MAX_FANOUT[via_label]))
        .select(key_col)
    )
    judged = summary_df.filter(F.col("judgeable") & (F.col("component_size") < GIANT_MIN_SIZE)).select("component")
    members = nodes_df.select("node_id", "component").join(judged, on="component")
    single = matched_df.filter(F.size("afs_ids") == 1).select("node_id", F.col("afs_ids")[0].alias("afs_id"))
    return (
        base.join(members, on="node_id")
        .join(keys, on=key_col, how="leftsemi")
        .join(single, on="node_id", how="left")
        .select(F.col(key_col).cast("string").alias("key"), "node_id", "component", "afs_id")
    )


@stage_dataframe(write_format="parquet")
def source_pairs_fn(members_df, spark=None, write_path=None):
    # same_person is null unless both nodes match one AFS person.
    a = members_df.select("key", F.col("node_id").alias("node_a"), "component", F.col("afs_id").alias("afs_a"))
    b = members_df.select("key", F.col("node_id").alias("node_b"), F.col("afs_id").alias("afs_b"))
    return (
        a.join(b, on="key")
        .filter(F.col("node_a") < F.col("node_b"))
        .select("node_a", "node_b", "component", (F.col("afs_a") == F.col("afs_b")).alias("same_person"))
        .distinct()
    )


@stage_dataframe(write_format="parquet")
def pair_health_fn(parts, summary_df, spark=None, write_path=None):
    unioned = None
    for p, v in zip(parts, VIA_LABELS):
        p = p.withColumn("via_type", F.lit(v))
        unioned = p if unioned is None else unioned.unionByName(p)
    type_pair = F.concat_ws("-", F.array_sort(F.array(_type_short("type_a"), _type_short("type_b"))))
    pairs = (
        unioned.groupBy("node_a", "node_b", "component", "same_person")
        .agg(_sorted_join(F.collect_set("via_type")).alias("sources"))
        .withColumn("type_a", _node_type("node_a"))
        .withColumn("type_b", _node_type("node_b"))
        .withColumn("type_pair", type_pair)
    )
    per_comp = pairs.groupBy("component", "sources", "type_pair").agg(
        F.count("*").alias("n_pairs"),
        F.count("same_person").alias("n_labeled"),
        F.sum(F.col("same_person").cast("long")).alias("n_same_person"),
    )
    comp = summary_df.select("component", "size_bucket", "over_merged", "incomplete")
    return (
        per_comp.join(comp, on="component")
        .groupBy("sources", "type_pair", "size_bucket")
        .agg(
            F.count("*").alias("n_judgeable"),
            F.sum("n_pairs").alias("n_pairs"),
            F.sum("n_labeled").alias("n_labeled"),
            F.sum("n_same_person").alias("n_same_person"),
            F.sum(F.col("over_merged").cast("long")).alias("n_over_merged"),
            F.sum(F.col("incomplete").cast("long")).alias("n_incomplete"),
            F.sum((~F.col("over_merged") & ~F.col("incomplete")).cast("long")).alias("n_healthy"),
        )
    )


_pair_parts = []
for _via, _key_col in VIA_KEY_COLS:
    _members = pair_members_fn(
        nodes, label_nodes, component_summary, _via, _key_col,
        spark=spark, write_path=f"{OUT_SPLIT_PATH}/graph_pair_members_{_via}",
    )
    _pair_parts.append(source_pairs_fn(_members, spark=spark, write_path=f"{OUT_SPLIT_PATH}/graph_pairs_{_via}"))
pair_health = pair_health_fn(_pair_parts, component_summary, spark=spark, write_path=f"{OUT_SPLIT_PATH}/graph_pair_health")

_ph = pair_health.toPandas()
for _cols in [["sources"], ["type_pair"], ["sources", "type_pair"]]:
    _same = _ph.groupby(_cols, as_index=False)[["n_pairs", "n_labeled", "n_same_person"]].sum()
    _same["pct_same_person"] = (100 * _same["n_same_person"] / _same["n_labeled"]).round(1)
    _view = _health_view(_ph, _cols, base=size_health).merge(
        _same[_cols + ["n_pairs", "n_labeled", "pct_same_person"]], on=_cols
    )
    print(f"=== [{SCORE_SPLIT}] components by linking pair: {' x '.join(_cols)} ===")
    print(_view.to_string(index=False))
    print()

print("# pair_health by size (CSV)")
print(_ph.sort_values(["sources", "type_pair", "size_bucket"]).to_csv(index=False))
