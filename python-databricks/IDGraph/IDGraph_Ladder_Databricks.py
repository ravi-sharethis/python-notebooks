# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph -- link-class ladder (V1..V5) on the IDGraph5 sample
# MAGIC
# MAGIC Rebuilds components from individual links, adding one class of links per
# MAGIC version, best first, and scores each version with notebook A's labels:
# MAGIC
# MAGIC | version | definition |
# MAGIC |---|---|
# MAGIC | V1 | links seen in 2+ sources (any node types) |
# MAGIC | V2 | V1 + the 3 rated single-source classes at >= 60% same Affinity person: tapad HEM-HEM, liid HEM-IDFA, tapad IDFA-IDFA |
# MAGIC | V3 | V2 + the next 2, at 50-60%: liid IDFA-IDFA, liid AAID-HEM |
# MAGIC | V4 | V3 + the remaining 6 rated classes, < 50%: liid AAID-AAID; tapad AAID-HEM, HEM-IDFA, AAID-AAID, AAID-IDFA; liid AAID-IDFA |
# MAGIC | V5 | V4 + every other single-source link (any TTD / ID5_UID / ST node, or unrated) = today's graph |
# MAGIC
# MAGIC A link = two nodes sharing a key of size 2..MAX_FANOUT in a source (the
# MAGIC full pair list per key, not the build's star layout). Class rates come from
# MAGIC notebook A's pair table (tune, 15% sample, 2026-11-12).
# MAGIC
# MAGIC **Only inside the sample.** Every version is a subset of today's links, so
# MAGIC its components never cross today's component boundaries: components are
# MAGIC recomputed inside each sampled multi-node component (2 .. GIANT_MIN_SIZE-1
# MAGIC nodes). Singletons stay singletons; the giant is left out.
# MAGIC
# MAGIC **Labels, joined once.** Each evaluated node is tagged once with its
# MAGIC Affinity match per split (`eval_nodes`); every version's metrics are then
# MAGIC group-bys on its new component ids. A person's nodes outside the evaluated
# MAGIC components sit in separate components in every version, so a component is
# MAGIC incomplete when one of its Affinity persons has nodes in another version
# MAGIC component OR outside the evaluated set (V5 reproduces notebook A).
# MAGIC
# MAGIC Run notebook A first (both splits, same SAMPLE_PCT and LABEL_SET).
# MAGIC Writes only under `IDGraph5_ladder/`; reads IDGraph5's build and A's tables.

# COMMAND ----------

# Spark settings used for every IDGraph Databricks notebook.
spark.conf.set("spark.sql.shuffle.partitions", "20000")
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")

# COMMAND ----------

# MAGIC %md ## Imports -- `stage_dataframe` from `python-databricks/staging_df.py`

# COMMAND ----------

import os
import sys

# In a Databricks Git folder the working directory is this notebook's folder;
# staging_df.py lives one level up, in python-databricks/.
sys.path.append(os.path.abspath(".."))

from pyspark.sql import functions as F
from pyspark.sql import types as T
from staging_df import _path_exists, in_slice, stage_dataframe, stage_sliced

# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

import pandas as pd

GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
SOURCE_BUILD_PATH = f"{GRAPH_ROOT}/build_output"
LADDER_ROOT = "s3://data-science-research/ravi/IDGraph5_ladder"

# Must match the notebook A run (job-level parameters).
dbutils.widgets.text("SAMPLE_PCT", "100")
SAMPLE_PCT = int(dbutils.widgets.get("SAMPLE_PCT"))
dbutils.widgets.dropdown("LABEL_SET", "all", ["all", "hem"])
LABEL_SET = dbutils.widgets.get("LABEL_SET")
_LABEL_SUFFIX = "" if LABEL_SET == "all" else f"_{LABEL_SET}"

SPLITS = ["tune", "holdout"]
_QUALITY = f"{GRAPH_ROOT}/component_quality"
A_SAMPLE_ROOT = _QUALITY + ("" if SAMPLE_PCT >= 100 else f"_sample{SAMPLE_PCT}")
OUT_PATH = f"{LADDER_ROOT}/sample{SAMPLE_PCT}"
OUT_SPLIT_PATH = {s: f"{OUT_PATH}/{s}{_LABEL_SUFFIX}" for s in SPLITS}

VIA_KEY_COLS = [
    ("id5", "individual_id5"),
    ("liid", "liid"),
    ("tapad", "individual_id_tapad"),
]
VIA_LABELS = [v for v, _ in VIA_KEY_COLS]

# Must match notebook A / the build.
MAX_FANOUT = {"id5": 16, "liid": 16, "tapad": 16}
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}
TYPE_SHORT = {"HEM_MD5": "HEM", "HARDWARE_IDFA": "IDFA", "HARDWARE_ANDROID_AD_ID": "AAID"}
GIANT_MIN_SIZE = 100_001
COMPONENT_SIZE_BOUNDS = [1, 2, 4, 8, 16, 32, 64, 128, 1_000, 10_000, 100_000]

# Single-source class ("<source> <sorted type pair>") -> the version that adds it.
# Links seen in 2+ sources are V1; every other single-source link is V5.
CLASS_VERSION = {
    "tapad HEM-HEM": 2, "liid HEM-IDFA": 2, "tapad IDFA-IDFA": 2,
    "liid IDFA-IDFA": 3, "liid AAID-HEM": 3,
    "liid AAID-AAID": 4, "tapad AAID-HEM": 4, "tapad HEM-IDFA": 4,
    "tapad AAID-AAID": 4, "tapad AAID-IDFA": 4, "liid AAID-IDFA": 4,
}
VERSIONS = [1, 2, 3, 4, 5]

_inputs = [f"{A_SAMPLE_ROOT}/graph/nodes"] + [
    f"{_QUALITY}/{s}/{t}"
    for s in SPLITS
    for t in ("matched_nodes", f"component_afs_single_afs{_LABEL_SUFFIX}", f"person_spread_single_afs{_LABEL_SUFFIX}")
] + [f"{SOURCE_BUILD_PATH}/{v}_{t}" for v in VIA_LABELS for t in ("fanout", "filtered_base")]
_missing = [p for p in _inputs if not _path_exists(p)]
assert not _missing, f"missing inputs (run notebook A on both splits first): {_missing}"
print(f"ladder on {GRAPH_ROOT}, {SAMPLE_PCT}% of components, labels from {LABEL_SET} matched nodes -> {OUT_PATH}")

# COMMAND ----------

# MAGIC %md ## Helpers

# COMMAND ----------

_type_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


def _short_type(col):
    # node_id is "<type>:<value>".
    t = F.substring_index(F.col(col), ":", 1)
    return F.coalesce(_type_short_map[t], t)


def _size_bucket(col, bounds):
    expr = None
    lo = bounds[0]
    for hi in bounds:
        label = str(hi) if lo == hi else f"{lo}-{hi}"
        cond = F.col(col) <= hi
        expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
        lo = hi + 1
    return expr.otherwise(F.lit(f"{bounds[-1] + 1}+"))


def _read(path):
    # Notebook A may have written these in slices -- drop the partition column.
    return spark.read.parquet(path).drop("slice")


# COMMAND ----------

# MAGIC %md ## Evaluated nodes, tagged once with their Affinity matches
# MAGIC Sampled nodes in components of 2 .. GIANT_MIN_SIZE-1 nodes. Per split:
# MAGIC `m_<split>` = Affinity matched the node (any number of persons),
# MAGIC `afs_<split>` = its Affinity person when it matches exactly one (the
# MAGIC nodes notebook A's labels count). LABEL_SET = "hem" keeps HEM matches only.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def eval_nodes_fn(nodes_df, matched, spark=None, write_path=None):
    out = nodes_df.filter(
        (F.col("component_size") >= 2) & (F.col("component_size") < GIANT_MIN_SIZE)
    ).select("node_id", "component")
    for s in SPLITS:
        m = matched[s]
        if LABEL_SET == "hem":
            m = m.filter(F.col("node_id").startswith("HEM_"))
        m = m.select(
            "node_id",
            F.lit(True).alias(f"m_{s}"),
            F.when(F.size("afs_ids") == 1, F.col("afs_ids")[0].cast("string")).alias(f"afs_{s}"),
        )
        out = out.join(m, on="node_id", how="left")
    return out.fillna({f"m_{s}": False for s in SPLITS})


nodes = _read(f"{A_SAMPLE_ROOT}/graph/nodes")
matched = {s: _read(f"{_QUALITY}/{s}/matched_nodes") for s in SPLITS}
eval_nodes = eval_nodes_fn(nodes, matched, spark=spark, write_path=f"{OUT_PATH}/eval_nodes{_LABEL_SUFFIX}")

# COMMAND ----------

# MAGIC %md ## Links, each with the version that adds it
# MAGIC Per source: the keys of size 2..MAX_FANOUT among evaluated nodes, expanded
# MAGIC to every pair. `ladder_pairs`: one row per pair with `level` = 1 if 2+
# MAGIC sources link it, else CLASS_VERSION of its single-source class, else 5.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def pair_members_fn(eval_df, via_label, key_col, spark=None, write_path=None):
    base = spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_filtered_base").select("node_id", key_col)
    deny = list(DENYLIST.get(via_label, set()))
    if deny:
        base = base.filter(~F.col(key_col).isin(deny))
    keys = (
        spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_fanout")
        .filter((F.col("n_distinct") >= 2) & (F.col("n_distinct") <= MAX_FANOUT[via_label]))
        .select(key_col)
    )
    return (
        base.join(eval_df.select("node_id", "component"), on="node_id")
        .join(keys, on=key_col, how="leftsemi")
        .select(F.col(key_col).cast("string").alias("key"), "node_id", "component")
    )


@stage_dataframe(write_format="parquet")
def source_pairs_fn(members_df, spark=None, write_path=None):
    a = members_df.select("key", F.col("node_id").alias("node_a"), "component")
    b = members_df.select("key", F.col("node_id").alias("node_b"))
    return a.join(b, on="key").filter(F.col("node_a") < F.col("node_b")).select("node_a", "node_b", "component").distinct()


_class_map = F.create_map(*[F.lit(x) for kv in CLASS_VERSION.items() for x in kv])


@stage_dataframe(write_format="parquet")
def ladder_pairs_fn(parts, spark=None, write_path=None):
    unioned = None
    for p, v in zip(parts, VIA_LABELS):
        p = p.withColumn("via_type", F.lit(v))
        unioned = p if unioned is None else unioned.unionByName(p)
    pairs = unioned.groupBy("component", "node_a", "node_b").agg(F.collect_set("via_type").alias("sources"))
    type_pair = F.concat_ws("-", F.array_sort(F.array(_short_type("node_a"), _short_type("node_b"))))
    cls = F.concat(F.col("sources")[0], F.lit(" "), type_pair)
    level = F.when(F.size("sources") >= 2, F.lit(1)).otherwise(F.coalesce(_class_map[cls], F.lit(5)))
    return pairs.select("component", "node_a", "node_b", level.cast("int").alias("level"))


_pair_parts = []
for _via, _key_col in VIA_KEY_COLS:
    _members = pair_members_fn(
        eval_nodes, _via, _key_col, spark=spark, write_path=f"{OUT_PATH}/pair_members_{_via}{_LABEL_SUFFIX}"
    )
    _pair_parts.append(source_pairs_fn(_members, spark=spark, write_path=f"{OUT_PATH}/pairs_{_via}{_LABEL_SUFFIX}"))
ladder_pairs = ladder_pairs_fn(_pair_parts, spark=spark, write_path=f"{OUT_PATH}/ladder_pairs{_LABEL_SUFFIX}")

# COMMAND ----------

# MAGIC %md ## Components per version, inside each sampled component
# MAGIC One pass per sampled component (applyInPandas): union-find adds the links
# MAGIC level by level and, after each level, emits that version's components.
# MAGIC A version component with no Affinity match in either split is only
# MAGIC counted (`n_comps` per size); matched ones get a row with their counts and
# MAGIC distinct single-person Affinity ids per split.

# COMMAND ----------

_VC_SCHEMA = T.StructType(
    [
        T.StructField("version", T.IntegerType()),
        T.StructField("component", T.StringType()),
        T.StructField("vcomp", T.StringType()),
        T.StructField("size", T.LongType()),
        T.StructField("n_comps", T.LongType()),
    ]
    + [
        f
        for s in SPLITS
        for f in (
            T.StructField(f"n_matched_{s}", T.LongType()),
            T.StructField(f"n_single_{s}", T.LongType()),
            T.StructField(f"afs_list_{s}", T.ArrayType(T.StringType())),
        )
    ]
)


def _version_components(pdf):
    comp = str(pdf["component"].iloc[0])
    is_node = pdf["level"] == 0
    node_rows = pdf[is_node]
    edges = pdf[~is_node].sort_values("level")
    nodes_ = list(node_rows["node_a"])
    flags = {
        s: (dict(zip(node_rows["node_a"], node_rows[f"m_{s}"])), dict(zip(node_rows["node_a"], node_rows[f"afs_{s}"])))
        for s in SPLITS
    }
    parent = {n: n for n in nodes_}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    elist = list(zip(edges["node_a"], edges["node_b"], edges["level"]))
    i = 0
    for v in VERSIONS:
        while i < len(elist) and elist[i][2] <= v:
            a, b, _ = elist[i]
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            i += 1
        groups = {}
        for n in nodes_:
            groups.setdefault(find(n), []).append(n)
        unlabelled = {}
        for root, members in groups.items():
            row = {"version": v, "component": comp, "size": len(members), "n_comps": 1}
            labelled = False
            for s in SPLITS:
                m, afs = flags[s]
                n_matched = sum(1 for n in members if bool(m[n]))
                ids = sorted({afs[n] for n in members if isinstance(afs[n], str)})
                n_single = sum(1 for n in members if isinstance(afs[n], str))
                row.update({f"n_matched_{s}": n_matched, f"n_single_{s}": n_single, f"afs_list_{s}": ids})
                labelled = labelled or n_matched > 0
            if labelled:
                row["vcomp"] = f"{comp}|{root}"
                out.append(row)
            else:
                unlabelled[len(members)] = unlabelled.get(len(members), 0) + 1
        for size, n in unlabelled.items():
            row = {"version": v, "component": comp, "vcomp": None, "size": size, "n_comps": n}
            for s in SPLITS:
                row.update({f"n_matched_{s}": 0, f"n_single_{s}": 0, f"afs_list_{s}": []})
            out.append(row)
    return pd.DataFrame(out, columns=[f.name for f in _VC_SCHEMA.fields])


@stage_dataframe(write_format="parquet")
def version_components_fn(eval_df, pairs_df, spark=None, write_path=None):
    node_rows = eval_df.select(
        "component",
        F.col("node_id").alias("node_a"),
        F.col("node_id").alias("node_b"),
        F.lit(0).alias("level"),
        *[c for s in SPLITS for c in (f"m_{s}", f"afs_{s}")],
    )
    edge_rows = pairs_df.select(
        "component", "node_a", "node_b", "level",
        *[c for s in SPLITS for c in (F.lit(None).cast("boolean").alias(f"m_{s}"), F.lit(None).cast("string").alias(f"afs_{s}"))],
    )
    return node_rows.unionByName(edge_rows).groupBy("component").applyInPandas(_version_components, schema=_VC_SCHEMA)


version_components = version_components_fn(
    eval_nodes, ladder_pairs, spark=spark, write_path=f"{OUT_PATH}/version_components{_LABEL_SUFFIX}"
)

# COMMAND ----------

# MAGIC %md ## Per split: incomplete, then metrics by version x size
# MAGIC Person spread in a version = its version components in the evaluated set
# MAGIC + its components outside it (`n_components` from notebook A minus those
# MAGIC inside). Over-merged / judgeable / healthy follow notebook A: judgeable =
# MAGIC 2+ single-person nodes, over-merged = 2+ distinct Affinity persons,
# MAGIC incomplete = an Affinity person in it has nodes in another component.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def person_versions_fn(vc_df, eval_df, comp_afs_df, spread_df, split, spark=None, write_path=None):
    afs_col = f"afs_list_{split}"
    in_v = (
        vc_df.filter(F.size(afs_col) > 0)
        .select("version", "vcomp", F.explode(afs_col).alias("afs_id"))
        .groupBy("version", "afs_id")
        .agg(F.countDistinct("vcomp").alias("n_v_in"))
    )
    eval_comps = eval_df.select("component").distinct()
    base_in = (
        comp_afs_df.select("component", F.col("afs_id").cast("string").alias("afs_id"))
        .join(eval_comps, on="component", how="leftsemi")
        .groupBy("afs_id")
        .agg(F.countDistinct("component").alias("n_base_in"))
    )
    spread = spread_df.select(F.col("afs_id").cast("string").alias("afs_id"), "n_nodes", F.col("n_components").alias("n_base_total"))
    return (
        in_v.join(base_in, on="afs_id")
        .join(spread, on="afs_id")
        .withColumn("n_spread", F.col("n_v_in") + F.col("n_base_total") - F.col("n_base_in"))
    )


@stage_dataframe(write_format="parquet")
def version_metrics_fn(vc_df, person_df, split, spark=None, write_path=None):
    afs_col = f"afs_list_{split}"
    labelled = vc_df.filter(F.size(afs_col) > 0)
    incomplete = (
        labelled.select("version", "vcomp", F.explode(afs_col).alias("afs_id"))
        .join(person_df.select("version", "afs_id", "n_spread"), on=["version", "afs_id"])
        .groupBy("version", "vcomp")
        .agg(F.max((F.col("n_spread") >= 2).cast("int")).alias("incomplete"))
    )
    v = (
        vc_df.join(incomplete, on=["version", "vcomp"], how="left")
        .fillna({"incomplete": 0})
        .withColumn("n_afs", F.size(afs_col))
        .withColumn("size_bucket", _size_bucket("size", COMPONENT_SIZE_BOUNDS))
    )
    judgeable = F.col(f"n_single_{split}") >= 2
    over = F.col("n_afs") >= 2
    inc = (F.col("n_afs") >= 1) & (F.col("incomplete") == 1)
    n = F.col("n_comps")
    return v.groupBy("version", "size_bucket").agg(
        F.min("size").alias("size_order"),
        F.sum(n).alias("n_components"),
        F.sum(n * F.col("size")).alias("n_nodes"),
        F.sum(F.when(F.col(f"n_matched_{split}") > 0, n).otherwise(0)).alias("n_targetable"),
        F.sum(F.when(F.col("n_afs") >= 1, n).otherwise(0)).alias("n_labelled"),
        F.sum(F.when(judgeable, n).otherwise(0)).alias("n_judgeable"),
        F.sum(F.when(over, n).otherwise(0)).alias("n_over_merged"),
        F.sum(F.when(inc, n).otherwise(0)).alias("n_incomplete_labelled"),
        F.sum(F.when(judgeable & inc, n).otherwise(0)).alias("n_incomplete"),
        F.sum(F.when(judgeable & ~over & ~inc, n).otherwise(0)).alias("n_healthy"),
    )


person_versions = {}
version_metrics = {}
for _s in SPLITS:
    person_versions[_s] = person_versions_fn(
        version_components, eval_nodes,
        _read(f"{_QUALITY}/{_s}/component_afs_single_afs{_LABEL_SUFFIX}"),
        _read(f"{_QUALITY}/{_s}/person_spread_single_afs{_LABEL_SUFFIX}"),
        _s, spark=spark, write_path=f"{OUT_SPLIT_PATH[_s]}/person_versions",
    )
    version_metrics[_s] = version_metrics_fn(
        version_components, person_versions[_s], _s, spark=spark, write_path=f"{OUT_SPLIT_PATH[_s]}/version_metrics"
    )

# COMMAND ----------

# MAGIC %md ## Results
# MAGIC - links added per version (cumulative share)
# MAGIC - scale per version (no labels): components, singletons, multi-node
# MAGIC   components, mean / max size, size distribution, and a whole-graph estimate
# MAGIC - per split, per version: components, singletons, % targetable, and the
# MAGIC   judgeable rates (% over-merged, % incomplete, % healthy), plus % of
# MAGIC   Affinity persons split
# MAGIC - per split: % healthy of judgeable by version x size (compare at a fixed size)
# MAGIC
# MAGIC Tune vs holdout absolute rates differ (70% vs 30% of persons): compare
# MAGIC versions within a split.

# COMMAND ----------

pd.set_option("display.width", 250)

_lp = ladder_pairs.groupBy("level").agg(F.count("*").alias("n_pairs")).toPandas().sort_values("level")
_lp["cum_pct"] = (100 * _lp["n_pairs"].cumsum() / _lp["n_pairs"].sum()).round(1)
print("=== links added per version (level = the version that adds them) ===")
print(_lp.to_string(index=False))
print()

# Scale (no labels): same nodes in every version, split into more, smaller components
# as links are removed. V5 = today's graph on these components. est_full_* scales the
# sample to the whole graph (excluding the giant and today's singletons).
_sc = (
    version_components.groupBy("version")
    .agg(
        F.sum("n_comps").alias("n_components"),
        F.sum(F.when(F.col("size") == 1, F.col("n_comps")).otherwise(0)).alias("n_singletons"),
        F.sum(F.when(F.col("size") >= 2, F.col("n_comps")).otherwise(0)).alias("n_multi_node"),
        F.sum(F.col("size") * F.col("n_comps")).alias("n_nodes"),
        F.sum(F.when(F.col("size") >= 2, F.col("size") * F.col("n_comps")).otherwise(0)).alias("nodes_in_multi"),
        F.max("size").alias("max_size"),
    )
    .toPandas()
    .sort_values("version")
)
_sc["n_links"] = _lp.set_index("level")["n_pairs"].cumsum().reindex(_sc["version"]).values
_sc["pct_nodes_singleton"] = (100 * _sc["n_singletons"] / _sc["n_nodes"]).round(2)
_sc["mean_multi_size"] = (_sc["nodes_in_multi"] / _sc["n_multi_node"].where(_sc["n_multi_node"] > 0)).round(2)
_scale = 100 / SAMPLE_PCT
for _c in ("n_components", "n_singletons", "n_multi_node"):
    _sc[f"est_full_{_c}"] = (_sc[_c] * _scale).round().astype("int64")
print(f"=== scale per version: {SAMPLE_PCT}% sample of today's multi-node components (giant excluded) ===")
print(_sc[[
    "version", "n_links", "n_nodes", "n_components", "n_singletons", "pct_nodes_singleton",
    "n_multi_node", "mean_multi_size", "max_size", "est_full_n_components", "est_full_n_singletons", "est_full_n_multi_node",
]].to_string(index=False))
print()
_sd = (
    version_components.withColumn("size_bucket", _size_bucket("size", COMPONENT_SIZE_BOUNDS))
    .groupBy("version", "size_bucket")
    .agg(F.min("size").alias("size_order"), F.sum("n_comps").alias("n_components"))
    .toPandas()
)
print("=== components by size, per version ===")
print(_sd.pivot(index=["size_order", "size_bucket"], columns="version", values="n_components").fillna(0).astype("int64").reset_index(level=0, drop=True).to_string())
print()


def _rates(df):
    df = df.copy()
    df["pct_targetable"] = (100 * df["n_targetable"] / df["n_components"]).round(2)
    df["pct_incomplete_labelled"] = (100 * df["n_incomplete_labelled"] / df["n_labelled"].where(df["n_labelled"] > 0)).round(2)
    for k in ("over_merged", "incomplete", "healthy"):
        df[f"pct_{k}"] = (100 * df[f"n_{k}"] / df["n_judgeable"].where(df["n_judgeable"] > 0)).round(2)
    return df


# pct_over_merged / pct_incomplete / pct_healthy are of judgeable components (as in notebook A's
# health tables); pct_incomplete_labelled is of all labelled components (A's headline pct_incomplete).
_SHOW = [
    "n_components", "n_singletons", "n_targetable", "pct_targetable", "n_judgeable",
    "pct_over_merged", "pct_incomplete", "pct_healthy", "pct_incomplete_labelled",
]
for _s in SPLITS:
    _m = version_metrics[_s].toPandas()
    _tot = _m.groupby("version", as_index=False)[[c for c in _m.columns if c.startswith("n_")]].sum()
    _tot["n_singletons"] = _tot["version"].map(_m[_m["size_bucket"] == "1"].set_index("version")["n_components"]).fillna(0).astype(int)
    _tot = _rates(_tot)
    _pv = person_versions[_s].filter(F.col("n_nodes") >= 2).groupBy("version").agg(
        F.count("*").alias("n_persons"), F.sum((F.col("n_spread") >= 2).cast("int")).alias("n_split")
    ).toPandas()
    _pv["pct_persons_split"] = (100 * _pv["n_split"] / _pv["n_persons"]).round(2)
    _tot = _tot.merge(_pv[["version", "pct_persons_split"]], on="version", how="left")
    print(f"=== [{_s}] per version (rates of judgeable components; % Affinity persons split = of those with 2+ nodes) ===")
    print(_tot[["version"] + _SHOW + ["pct_persons_split"]].to_string(index=False))
    print()
    _bs = _rates(_m[_m["n_judgeable"] > 0].sort_values(["size_order", "version"]))
    print(f"=== [{_s}] % healthy of judgeable components, version x size (n_judgeable in brackets) ===")
    _bs["cell"] = _bs["pct_healthy"].astype(str) + " (" + _bs["n_judgeable"].astype(int).astype(str) + ")"
    _piv = _bs.pivot(index=["size_order", "size_bucket"], columns="version", values="cell").fillna("")
    print(_piv.reset_index(level=0, drop=True).to_string())
    print()
    for _k in ("over_merged", "incomplete"):
        _piv = _bs.pivot(index=["size_order", "size_bucket"], columns="version", values=f"pct_{_k}").fillna("")
        print(f"=== [{_s}] % {_k} of judgeable components, version x size ===")
        print(_piv.reset_index(level=0, drop=True).to_string())
        print()

# COMMAND ----------

# MAGIC %md ## Full tables as CSV (paste back to save locally)

# COMMAND ----------

print("# links per version")
print(_lp.to_csv(index=False))
for _s in SPLITS:
    print(f"# [{_s}] version_metrics")
    print(version_metrics[_s].toPandas().sort_values(["version", "size_order"]).to_csv(index=False))
