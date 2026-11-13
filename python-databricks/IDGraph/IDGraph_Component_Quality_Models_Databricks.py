# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — component quality models (notebook B)
# MAGIC
# MAGIC Predicts, for EVERY component (including the class-0 ones Affinity can't
# MAGIC judge), the probability that it is over-merged, incomplete, or healthy
# MAGIC (judgeable, neither) -- from the component's own constituents, never from
# MAGIC Affinity. Affinity only provides the training labels (tune split) and the
# MAGIC test labels (holdout split), both from notebook A
# MAGIC (`IDGraph_Component_Quality_Databricks.py`). Design: `IDGraph_Journal.md` §14.
# MAGIC
# MAGIC **Run notebook A twice first** (`SCORE_SPLIT = "tune"` and `"holdout"`) on
# MAGIC the same GRAPH_ROOT.
# MAGIC
# MAGIC | model | idea |
# MAGIC |---|---|
# MAGIC | `lookup` | baseline: rate of the component's cell (sources x type set x size x singleton reason), falling back to coarser cells |
# MAGIC | `nb_sum` | Bayesian naive Bayes: sum of token log-odds (classic NB -- big components saturate), calibrated with size |
# MAGIC | `nb_mean` | same token rates, but node-weighted MEAN token log-odds + size -- doesn't saturate |
# MAGIC | `lr` | logistic regression on size, node counts (type, source x type) and multi-source pair counts (type pair, source) -- see the glossary |
# MAGIC
# MAGIC Token rates: `(own positives + PRIOR_STRENGTH x type rate) / (own total +
# MAGIC PRIOR_STRENGTH)`, type rate = pooled rate of every token with that ID type
# MAGIC (so a token never seen in training, e.g. `id5+liid+tapad:AAID`, sits at
# MAGIC its type's rate; an unknown type at the global rate).
# MAGIC
# MAGIC Evaluation is on holdout people only: AUC (overall and within size strata
# MAGIC -- size alone predicts a lot, the strata show what a model adds beyond it)
# MAGIC and calibration by 10% band. Caveat: a component can hold both tune and
# MAGIC holdout people, so the same component may be a training row (tune label)
# MAGIC and a test row (holdout label). The features are coarse (no component id),
# MAGIC so no model can memorise a component, but the two labels are correlated.
# MAGIC
# MAGIC **Over-merge rates depend on the split's size.** Class 2 needs two
# MAGIC different people from the SAME split in a component; tune holds 70% of
# MAGIC people and holdout 30%, so holdout sees fewer over-merges (on synthetic
# MAGIC data: 21% vs 9% of judgeable components). AUC (ranking) is comparable
# MAGIC across splits; over-merge probabilities and calibration are not -- read
# MAGIC them as "over-merge visible with 70% of people". Incomplete doesn't have
# MAGIC this problem (a person's IDs are all in one split). Comparing graphs is
# MAGIC fine as long as it's on the same split.

# COMMAND ----------

# Spark settings used for every IDGraph Databricks notebook.
spark.conf.set("spark.sql.shuffle.partitions", "20000")
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
# One-pass job: the disk cache only competes with shuffle files for local disk.
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

import itertools

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

# Must be a graph notebook A has already scored on both splits.
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
# Must match the SAMPLE_PCT notebook A ran with (a job-level parameter sets both).
dbutils.widgets.text("SAMPLE_PCT", "100")
SAMPLE_PCT = int(dbutils.widgets.get("SAMPLE_PCT"))
QUALITY_ROOT = f"{GRAPH_ROOT}/component_quality" + ("" if SAMPLE_PCT >= 100 else f"_sample{SAMPLE_PCT}")
GRAPH_PATH = f"{QUALITY_ROOT}/graph"
TRAIN_SPLIT_PATH = f"{QUALITY_ROOT}/tune"
EVAL_SPLIT_PATH = f"{QUALITY_ROOT}/holdout"
# v2: node + multi-source pair count features, healthy target.
MODELS_PATH = f"{QUALITY_ROOT}/models_v2"
# filtered_base/fanout, for the pair features.
SOURCE_BUILD_PATH = f"{GRAPH_ROOT}/build_output"

VIA_KEY_COLS = [
    ("id5", "individual_id5"),
    ("liid", "liid"),
    ("tapad", "individual_id_tapad"),
]
VIA_LABELS = [v for v, _ in VIA_KEY_COLS]
TARGETS = ["over_merged", "incomplete", "healthy"]

# Must match notebook A.
MAX_FANOUT = {"id5": 16, "liid": 16, "tapad": 16}
DENYLIST = {
    "tapad": {
        "No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf",
    },
}
TYPE_SHORT = {"HEM_MD5": "HEM", "HARDWARE_IDFA": "IDFA", "HARDWARE_ANDROID_AD_ID": "AAID"}
ID_TYPES = ["HEM", "IDFA", "AAID", "TTD", "ID5_UID", "ST"]
# Pair features skip components this big (the giant); its pair features are 0.
GIANT_MIN_SIZE = 100_001

# Every (sorted) type pair, one pair feature each.
TYPE_PAIRS = [f"{a}-{b}" for a, b in itertools.combinations_with_replacement(sorted(ID_TYPES), 2)]

# A component with one matched ID can never be class 2; training on those
# would teach "less Affinity data = good". Applies to train AND test rows, for
# over_merged and healthy.
OVERMERGE_TRAIN_JUDGEABLE_ONLY = True

# Lookup cells with fewer labelled components fall back to the next coarser level.
LOOKUP_MIN_CELL = 100
LOOKUP_LEVELS = [
    ["sources", "type_set", "size_bucket", "reason_key"],
    ["sources", "type_set", "size_bucket"],
    ["sources", "size_bucket"],
    ["size_bucket"],
]

# Pseudo-observations borrowed from the token's ID-type rate. "fit" estimates
# one strength per type from the spread of its token rates (beta moments),
# falling back to 100 where that's not identifiable.
PRIOR_STRENGTH = 100

# Logistic regressions (lr + the NB calibrators) train on a sample of the
# labelled tune rows; lookup and NB token rates always use all of them.
LR_TRAIN_FRACTION = 0.1
LR_REG_PARAM = 0.001
SEED = 7

# Predictions are binned this finely for AUC / calibration (cheap, exact to 0.001).
N_SCORE_BINS = 1000

# Off by default (run with elastic disk on the job cluster). If True, steps that
# shuffle every component (features, NB token counts / scores, model inputs,
# training rows, eval histogram) run in N_SLICES passes by component hash.
# Delete a step's folder before changing SLICE_HEAVY_STEPS or N_SLICES for it.
SLICE_HEAVY_STEPS = False
N_SLICES = 8
_N = N_SLICES if SLICE_HEAVY_STEPS else 1

DROPPED_SIZE_BOUNDS = [16, 24, 32, 48, 64, 128, 256, 1024]

_inputs = [
    f"{GRAPH_PATH}/component_tokens",
    f"{GRAPH_PATH}/nodes",
    f"{TRAIN_SPLIT_PATH}/component_summary",
    f"{EVAL_SPLIT_PATH}/component_summary",
]
_inputs += [f"{SOURCE_BUILD_PATH}/{v}_{t}" for v in VIA_LABELS for t in ("fanout", "filtered_base")]
_missing = [p for p in _inputs if not _path_exists(p)]
assert not _missing, f"missing inputs (run notebook A on both splits first): {_missing}"

# COMMAND ----------

# MAGIC %md ## Helpers

# COMMAND ----------


def _size_bucket(col, bounds):
    expr = None
    lo = bounds[0]
    for hi in bounds:
        label = str(hi) if lo == hi else f"{lo}-{hi}"
        cond = F.col(col) <= hi
        expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
        lo = hi + 1
    return expr.otherwise(F.lit(f"{bounds[-1] + 1}+"))


STRATA_BOUNDS = [1, 4, 16, 128]  # -> "1", "2-4", "5-16", "17-128", "129+"
_LOG2 = lambda c: F.log2(F.lit(1.0) + F.coalesce(c, F.lit(0)).cast("double"))


_type_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


def _short_type(col):
    # node_id is "<type>:<value>".
    t = F.substring_index(F.col(col), ":", 1)
    return F.coalesce(_type_short_map[t], t)


def _col_name(prefix, value):
    return prefix + value.replace("+", "_").replace("-", "_")


def _logit(p):
    return F.log(p / (F.lit(1.0) - p))


def _clip(p, eps=1e-4):
    return F.least(F.greatest(p, F.lit(eps)), F.lit(1 - eps))


def _fit_or_load(path, fit):
    """Fitted models are saved, so a rerun after a crash doesn't refit."""
    if _path_exists(path):
        return PipelineModel.load(path)
    model = fit()
    model.write().overwrite().save(path)
    return model


def _p1(prob_col="probability"):
    return vector_to_array(F.col(prob_col))[1]


# COMMAND ----------

# MAGIC %md ## Multi-source pairs per component (graph only)
# MAGIC A pair = two nodes of one component that share a key of size 2..MAX_FANOUT in
# MAGIC a source. Only pairs that 2+ sources link are kept (e.g. liid and tapad both
# MAGIC put the same two nodes in a group) -- single-source pairs are the graph's
# MAGIC ordinary links. Only nodes linked by 2+ sources can be in such a pair, so
# MAGIC each source's pairs are built from those nodes only. No Affinity.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def pair_members_fn(nodes_df, via_label, key_col, spark=None, write_path=None):
    base = spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_filtered_base").select("node_id", key_col)
    deny = list(DENYLIST.get(via_label, set()))
    if deny:
        base = base.filter(~F.col(key_col).isin(deny))
    keys = (
        spark.read.parquet(f"{SOURCE_BUILD_PATH}/{via_label}_fanout")
        .filter((F.col("n_distinct") >= 2) & (F.col("n_distinct") <= MAX_FANOUT[via_label]))
        .select(key_col)
    )
    multi_source = nodes_df.filter(
        F.col("source_set").contains("+")
        & F.col("source_set").contains(via_label)
        & (F.col("component_size") < GIANT_MIN_SIZE)
    ).select("node_id", "component")
    return (
        base.join(multi_source, on="node_id")
        .join(keys, on=key_col, how="leftsemi")
        .select(F.col(key_col).cast("string").alias("key"), "node_id", "component")
    )


@stage_dataframe(write_format="parquet")
def source_pairs_fn(members_df, spark=None, write_path=None):
    a = members_df.select("key", F.col("node_id").alias("node_a"), "component")
    b = members_df.select("key", F.col("node_id").alias("node_b"))
    return a.join(b, on="key").filter(F.col("node_a") < F.col("node_b")).select("node_a", "node_b", "component").distinct()


@stage_dataframe(write_format="parquet")
def pair_features_fn(parts, spark=None, write_path=None):
    unioned = None
    for p, v in zip(parts, VIA_LABELS):
        p = p.withColumn("via_type", F.lit(v))
        unioned = p if unioned is None else unioned.unionByName(p)
    pairs = (
        unioned.groupBy("node_a", "node_b", "component")
        .agg(F.array_sort(F.collect_set("via_type")).alias("sources"))
        .filter(F.size("sources") >= 2)
        .withColumn("type_pair", F.concat_ws("-", F.array_sort(F.array(_short_type("node_a"), _short_type("node_b")))))
    )
    return pairs.groupBy("component").agg(
        *[F.sum((F.col("type_pair") == tp).cast("long")).alias(_col_name("pair_", tp)) for tp in TYPE_PAIRS],
        *[F.sum(F.array_contains("sources", v).cast("long")).alias(f"pair_{v}") for v in VIA_LABELS],
        # Not a feature: >1 means pair_<type> and pair_<source> no longer say
        # which pair had which sources (the caveat in the glossary).
        F.countDistinct(F.concat_ws("+", "sources")).alias("n_pair_source_sets"),
    )


# Notebook A may have written nodes in slices -- drop the partition column.
nodes = spark.read.parquet(f"{GRAPH_PATH}/nodes").drop("slice")
_pair_parts = []
for _via, _key_col in VIA_KEY_COLS:
    _members = pair_members_fn(nodes, _via, _key_col, spark=spark, write_path=f"{MODELS_PATH}/pair_members_{_via}")
    _pair_parts.append(source_pairs_fn(_members, spark=spark, write_path=f"{MODELS_PATH}/pairs_{_via}"))
pair_features = pair_features_fn(_pair_parts, spark=spark, write_path=f"{MODELS_PATH}/pair_features")

_pf = pair_features.agg(
    F.count("*").alias("n"), F.sum((F.col("n_pair_source_sets") > 1).cast("long")).alias("n_mixed")
).first()
print(
    f"components with a multi-source pair: {_pf['n']:,}; with pairs from 2+ different source sets "
    f"(the glossary caveat): {_pf['n_mixed']:,} ({100 * _pf['n_mixed'] / max(_pf['n'], 1):.2f}%)"
)

# COMMAND ----------

# MAGIC %md ## Features (one row per component) and labels
# MAGIC Features come from the graph only; labels come from the tune (training) and
# MAGIC holdout (test) summaries. ID type = the node id's type prefix, for every
# MAGIC node, matched or not. Types: HEM, IDFA, AAID, TTD, ID5_UID, ST. Sources:
# MAGIC id5, liid, tapad.
# MAGIC
# MAGIC **Glossary** (all counts per component; the model sees log2(1 + count))
# MAGIC
# MAGIC | feature | meaning |
# MAGIC |---|---|
# MAGIC | `log2_size` | log2 of the number of nodes |
# MAGIC | `node_<T>` | nodes of type T, e.g. `node_HEM` |
# MAGIC | `node_<source>_<T>` | nodes of type T that the source links (the node shares a kept key with another node in that source), e.g. `node_tapad_IDFA`. A node linked by two sources counts under both |
# MAGIC | `pair_<A>_<B>` | multi-source pairs (two nodes that 2+ sources link) joining types A and B, e.g. `pair_HEM_IDFA` |
# MAGIC | `pair_<source>` | multi-source pairs that the source links, e.g. `pair_liid` |
# MAGIC
# MAGIC Example: one HEM-IDFA pair linked by liid and tapad gives `pair_HEM_IDFA` = 1,
# MAGIC `pair_liid` = 1, `pair_tapad` = 1.
# MAGIC
# MAGIC **Caveat:** with two multi-source pairs from different source sets (say
# MAGIC HEM-IDFA by liid+tapad, AAID-IDFA by id5+tapad), the counts don't say which
# MAGIC pair had which sources. The pair cell above prints how often that happens.
# MAGIC
# MAGIC Other columns kept for the lookup and NB models only (not LR features):
# MAGIC `sources`, `type_set`, `size_bucket`, singleton reason and dropped size.

# COMMAND ----------

NODE_FEATURES = [f"node_{t}" for t in ID_TYPES] + [f"node_{v}_{t}" for v in VIA_LABELS for t in ID_TYPES]
PAIR_FEATURES = [_col_name("pair_", tp) for tp in TYPE_PAIRS] + [f"pair_{v}" for v in VIA_LABELS]
COUNT_FEATURES = NODE_FEATURES + PAIR_FEATURES
LR_FEATURES = ["log2_size"] + [f"log2_{c}" for c in COUNT_FEATURES]


@stage_dataframe(write_format="parquet")
def features_fn(summary_df, tokens_df, pairs_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    # token = "<source_set>:<type>", one row per (component, token) with its node count.
    src = F.split(F.substring_index("token", ":", 1), "\\+")
    typ = F.substring_index("token", ":", -1)
    node_src = tokens_df.filter(this_slice).groupBy("component").agg(
        *[
            F.sum(F.when(F.array_contains(src, v) & (typ == t), F.col("n_nodes")).otherwise(0)).alias(f"node_{v}_{t}")
            for v in VIA_LABELS
            for t in ID_TYPES
        ]
    )
    keep = (
        ["component", "component_size", "sources", "type_set", "size_bucket"]
        + ["singleton_type", "singleton_reasons", "min_dropped_size"]
        + [F.col(f"n_type_{t}").alias(f"node_{t}") for t in ID_TYPES]
    )
    f = (
        summary_df.filter(this_slice)
        .select(*keep)
        .join(node_src, on="component", how="left")
        .join(pairs_df.filter(this_slice).drop("n_pair_source_sets"), on="component", how="left")
        .fillna(0, subset=COUNT_FEATURES)
    )
    dropped_bucket = F.when(F.col("min_dropped_size").isNull(), F.lit("-")).otherwise(
        _size_bucket("min_dropped_size", DROPPED_SIZE_BOUNDS)
    )
    f = (
        f.withColumn("is_singleton", (F.col("sources") == "singleton").cast("double"))
        .withColumn("log2_size", F.log2(F.col("component_size").cast("double")))
        .withColumn("dropped_bucket", dropped_bucket)
        .withColumn("reason_key", F.concat_ws("|", F.coalesce(F.col("singleton_reasons"), F.lit("-")), dropped_bucket))
        .withColumn("stratum", _size_bucket("component_size", STRATA_BOUNDS))
    )
    for c in COUNT_FEATURES:
        f = f.withColumn(f"log2_{c}", _LOG2(F.col(c)))
    return f


@stage_dataframe(write_format="parquet")
def labels_fn(summary_df, target, spark=None, write_path=None):
    # healthy = judgeable and neither over-merged nor incomplete.
    rows = summary_df.filter(F.col("class") > 0)
    if target in ("over_merged", "healthy") and OVERMERGE_TRAIN_JUDGEABLE_ONLY:
        rows = rows.filter(F.col("judgeable"))
    label = ~F.col("over_merged") & ~F.col("incomplete") if target == "healthy" else F.col(target)
    return rows.select("component", label.cast("double").alias("label"))


# Notebook A may have written these in slices -- drop the partition column.
component_tokens = spark.read.parquet(f"{GRAPH_PATH}/component_tokens").drop("slice")
train_summary = spark.read.parquet(f"{TRAIN_SPLIT_PATH}/component_summary").drop("slice")
eval_summary = spark.read.parquet(f"{EVAL_SPLIT_PATH}/component_summary").drop("slice")

features = stage_sliced(
    features_fn, train_summary, component_tokens, pair_features, spark=spark, write_path=f"{MODELS_PATH}/features", n_slices=_N
)
train_labels = {
    t: labels_fn(train_summary, t, spark=spark, write_path=f"{MODELS_PATH}/labels_tune_{t}") for t in TARGETS
}
eval_labels = {
    t: labels_fn(eval_summary, t, spark=spark, write_path=f"{MODELS_PATH}/labels_holdout_{t}") for t in TARGETS
}

# COMMAND ----------

# MAGIC %md ## Tokens for every component
# MAGIC Non-singletons: `component_tokens` (source set x ID type, one per node).
# MAGIC Singletons have no links, so they get `singleton[<reasons>]:<type>` (e.g.
# MAGIC `singleton[tapad:dropped]:HEM`) and, if dropped, `dropped_size=<bucket>:DROPPED_SIZE`.
# MAGIC A token's ID type is the part after its last `:`.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def all_tokens_fn(tokens_df, features_df, spark=None, write_path=None):
    single = features_df.filter(F.col("is_singleton") == 1)
    reason_token = single.select(
        "component",
        F.concat(F.lit("singleton["), F.col("singleton_reasons"), F.lit("]:"), F.col("singleton_type")).alias("token"),
        F.lit(1).cast("long").alias("n_nodes"),
    )
    dropped_token = single.filter(F.col("min_dropped_size").isNotNull()).select(
        "component",
        F.concat(F.lit("dropped_size="), F.col("dropped_bucket"), F.lit(":DROPPED_SIZE")).alias("token"),
        F.lit(1).cast("long").alias("n_nodes"),
    )
    return tokens_df.select("component", "token", "n_nodes").unionByName(reason_token).unionByName(dropped_token)


all_tokens = all_tokens_fn(component_tokens, features, spark=spark, write_path=f"{MODELS_PATH}/all_tokens")

# COMMAND ----------

# MAGIC %md ## Model 1 -- lookup baseline (rates per cell, all levels)

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def lookup_rates_fn(train_rows_df, spark=None, write_path=None):
    # train_rows = features x labels, so no join here.
    rows = train_rows_df
    parts = [
        rows.groupBy(F.concat_ws("|", *cols).alias("key"))
        .agg(F.count("*").alias("n"), F.sum("label").alias("n_pos"))
        .withColumn("level", F.lit(i))
        for i, cols in enumerate(LOOKUP_LEVELS)
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out


def _with_lookup(df, rates_df):
    """p_lookup = finest level with >= LOOKUP_MIN_CELL labelled rows, else the global rate."""
    tot = rates_df.filter(F.col("level") == len(LOOKUP_LEVELS) - 1).agg(F.sum("n_pos") / F.sum("n")).first()[0]
    out = df
    probs = []
    for i, cols in enumerate(LOOKUP_LEVELS):
        lvl = rates_df.filter((F.col("level") == i) & (F.col("n") >= LOOKUP_MIN_CELL)).select(
            F.col("key").alias(f"_k{i}"), (F.col("n_pos") / F.col("n")).alias(f"_p{i}")
        )
        out = out.withColumn(f"_k{i}", F.concat_ws("|", *cols)).join(F.broadcast(lvl), on=f"_k{i}", how="left")
        probs.append(F.col(f"_p{i}"))
    out = out.withColumn("p_lookup", F.coalesce(*probs, F.lit(tot)))
    return out.drop(*[f"_k{i}" for i in range(len(LOOKUP_LEVELS))], *[f"_p{i}" for i in range(len(LOOKUP_LEVELS))])


# Computed after train_rows (below) -- it's a groupBy of those rows.

# COMMAND ----------

# MAGIC %md ## Model 2 -- Bayesian naive Bayes token rates
# MAGIC Per token: `n` = labelled training components carrying it, `n_pos` =
# MAGIC those that are positive. Counted per component, not per node: node counts
# MAGIC let the one giant component (697M nodes on IDGraph5) set every token's
# MAGIC rate to ~0.99. Rates are shrunk toward the
# MAGIC token's ID-type rate with PRIOR_STRENGTH pseudo-observations. Small table
# MAGIC (one row per token) -- the shrinkage runs in pandas.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def nb_token_counts_fn(all_tokens_df, labels_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    return (
        all_tokens_df.filter(this_slice)
        .join(labels_df.filter(this_slice), on="component")
        .groupBy("token")
        .agg(F.count("*").alias("n"), F.sum("label").alias("n_pos"))
    )


@stage_dataframe(write_format="parquet")
def nb_token_counts_total_fn(sliced_counts_df, spark=None, write_path=None):
    # Per-slice counts -> one row per token (a no-op sum when unsliced).
    return sliced_counts_df.groupBy("token").agg(F.sum("n").alias("n"), F.sum("n_pos").alias("n_pos"))


def _beta_strength(g):
    # Method of moments: token rates of one type ~ Beta(m * mu, m * (1 - mu)).
    if len(g) < 3:
        return np.nan
    r = g["n_pos"] / g["n"]
    w = g["n"]
    mu = (w * r).sum() / w.sum()
    var = (w * (r - mu) ** 2).sum() / w.sum()
    if var <= 0:
        return np.nan
    m = mu * (1 - mu) / var - 1
    return m if m > 0 else np.nan


@stage_dataframe(write_format="parquet")
def nb_token_rates_fn(counts_df, spark=None, write_path=None):
    pdf = counts_df.toPandas()
    pdf["n_pos"] = pdf["n_pos"].astype(float)
    pdf["type"] = pdf["token"].str.rsplit(":", n=1).str[1]
    by_type = pdf.groupby("type")[["n", "n_pos"]].sum()
    pdf["type_rate"] = pdf["type"].map(by_type["n_pos"] / by_type["n"])
    if PRIOR_STRENGTH == "fit":
        strength = pdf.groupby("type").apply(_beta_strength).fillna(100.0)
        pdf["prior_strength"] = pdf["type"].map(strength)
    else:
        pdf["prior_strength"] = float(PRIOR_STRENGTH)
    pdf["rate"] = (pdf["n_pos"] + pdf["prior_strength"] * pdf["type_rate"]) / (pdf["n"] + pdf["prior_strength"])
    return spark.createDataFrame(pdf[["token", "type", "n", "n_pos", "type_rate", "prior_strength", "rate"]])


@stage_dataframe(write_format="parquet")
def nb_scores_fn(all_tokens_df, rates_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    """Per component: nb_sum = sum over nodes of (token log-odds - base log-odds)
    (classic NB evidence); nb_mean = node-weighted mean token log-odds."""
    tot = rates_df.agg(F.sum("n_pos") / F.sum("n")).first()[0]
    type_rates = rates_df.select("type", "type_rate").distinct()
    p = _clip(F.coalesce(F.col("rate"), F.col("type_rate"), F.lit(tot)))
    base = float(np.log(tot / (1 - tot))) if 0 < tot < 1 else 0.0
    scored = (
        all_tokens_df.filter(in_slice("component", slice_id, n_slices))
        .withColumn("type", F.substring_index("token", ":", -1))
        .join(F.broadcast(rates_df.select("token", "rate")), on="token", how="left")
        .join(F.broadcast(type_rates), on="type", how="left")
        .withColumn("token_logit", _logit(p))
    )
    return scored.groupBy("component").agg(
        F.sum(F.col("n_nodes") * (F.col("token_logit") - F.lit(base))).alias("nb_sum"),
        (F.sum(F.col("n_nodes") * F.col("token_logit")) / F.sum("n_nodes")).alias("nb_mean"),
    )


nb_token_rates = {}
nb_scores = {}
for t in TARGETS:
    _sliced = stage_sliced(
        nb_token_counts_fn, all_tokens, train_labels[t],
        spark=spark, write_path=f"{MODELS_PATH}/nb_token_counts_sliced_{t}", n_slices=_N,
    )
    _counts = nb_token_counts_total_fn(_sliced, spark=spark, write_path=f"{MODELS_PATH}/nb_token_counts_{t}")
    nb_token_rates[t] = nb_token_rates_fn(_counts, spark=spark, write_path=f"{MODELS_PATH}/nb_token_rates_{t}")
    nb_scores[t] = stage_sliced(
        nb_scores_fn, all_tokens, nb_token_rates[t],
        spark=spark, write_path=f"{MODELS_PATH}/nb_scores_{t}", n_slices=_N,
    )

# COMMAND ----------

# MAGIC %md ## Model inputs and training rows
# MAGIC `model_inputs_{target}`: features + that target's NB scores, every
# MAGIC component. `train_rows_{target}`: the labelled tune components.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def model_inputs_fn(features_df, nb_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    return (
        features_df.filter(this_slice)
        .join(nb_df.filter(this_slice), on="component", how="left")
        .fillna({"nb_sum": 0.0, "nb_mean": 0.0})
    )


@stage_dataframe(write_format="parquet")
def train_rows_fn(inputs_df, labels_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    return inputs_df.filter(this_slice).join(labels_df.filter(this_slice), on="component")


model_inputs = {
    t: stage_sliced(
        model_inputs_fn, features, nb_scores[t],
        spark=spark, write_path=f"{MODELS_PATH}/model_inputs_{t}", n_slices=_N,
    )
    for t in TARGETS
}
train_rows = {
    t: stage_sliced(
        train_rows_fn, model_inputs[t], train_labels[t],
        spark=spark, write_path=f"{MODELS_PATH}/train_rows_{t}", n_slices=_N,
    )
    for t in TARGETS
}
lookup_rates = {
    t: lookup_rates_fn(train_rows[t], spark=spark, write_path=f"{MODELS_PATH}/lookup_rates_{t}") for t in TARGETS
}

# COMMAND ----------

# MAGIC %md ## Fit -- logistic regression, and the NB calibrators
# MAGIC The NB scores aren't probabilities; each is mapped to one with a 2-input
# MAGIC logistic regression (score, log2 size). Models are saved under
# MAGIC `MODELS_PATH` and reloaded on a rerun.

# COMMAND ----------


def _lr_pipeline(cols, out_prefix):
    assembler = VectorAssembler(inputCols=cols, outputCol=f"{out_prefix}_x")
    lr = LogisticRegression(
        featuresCol=f"{out_prefix}_x",
        labelCol="label",
        regParam=LR_REG_PARAM,
        probabilityCol=f"{out_prefix}_prob",
        rawPredictionCol=f"{out_prefix}_raw",
        predictionCol=f"{out_prefix}_pred",
    )
    return Pipeline(stages=[assembler, lr])


MODEL_SPECS = {
    "lr": LR_FEATURES,
    "nb_sum": ["nb_sum", "log2_size"],
    "nb_mean": ["nb_mean", "log2_size"],
}

fitted = {}
for t in TARGETS:
    _sample = train_rows[t].sample(fraction=LR_TRAIN_FRACTION, seed=SEED)
    for name, cols in MODEL_SPECS.items():
        fitted[(t, name)] = _fit_or_load(
            f"{MODELS_PATH}/fitted_{name}_{t}", lambda cols=cols, name=name: _lr_pipeline(cols, name).fit(_sample)
        )

# COMMAND ----------

# MAGIC %md ## Score every component
# MAGIC `scores_{target}`: one row per component (all classes, including 0) with
# MAGIC each model's probability. This is the table to use downstream.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def scores_fn(inputs_df, rates_df, target, spark=None, write_path=None):
    out = _with_lookup(inputs_df, rates_df)
    for name in MODEL_SPECS:
        out = fitted[(target, name)].transform(out).withColumn(f"p_{name}", _p1(f"{name}_prob"))
    return out.select("component", "component_size", "stratum", "p_lookup", *[f"p_{n}" for n in MODEL_SPECS])


MODEL_NAMES = ["lookup"] + list(MODEL_SPECS)
scores = {
    t: scores_fn(model_inputs[t], lookup_rates[t], t, spark=spark, write_path=f"{MODELS_PATH}/scores_{t}")
    for t in TARGETS
}

# COMMAND ----------

# MAGIC %md ## Evaluate on holdout
# MAGIC Scores are binned to N_SCORE_BINS per (stratum, model); AUC and
# MAGIC calibration are computed from the bins in pandas.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def eval_hist_fn(scores_df, labels_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    rows = scores_df.filter(this_slice).join(labels_df.filter(this_slice), on="component")
    long = rows.select(
        "stratum",
        "label",
        F.explode(
            F.array(*[F.struct(F.lit(m).alias("model"), F.col(f"p_{m}").alias("p")) for m in MODEL_NAMES])
        ).alias("mp"),
    ).select("stratum", "label", "mp.model", "mp.p")
    return long.groupBy(
        "stratum", "model", F.least(F.floor(F.col("p") * N_SCORE_BINS), F.lit(N_SCORE_BINS - 1)).alias("bin")
    ).agg(F.count("*").alias("n"), F.sum("label").alias("n_pos"), F.sum("p").alias("sum_p"))


@stage_dataframe(write_format="parquet")
def eval_hist_total_fn(sliced_hist_df, spark=None, write_path=None):
    # Per-slice histograms -> one (a no-op sum when unsliced).
    return sliced_hist_df.groupBy("stratum", "model", "bin").agg(
        F.sum("n").alias("n"), F.sum("n_pos").alias("n_pos"), F.sum("sum_p").alias("sum_p")
    )


eval_hists = {}
for t in TARGETS:
    _sliced = stage_sliced(
        eval_hist_fn, scores[t], eval_labels[t],
        spark=spark, write_path=f"{MODELS_PATH}/eval_hist_sliced_{t}", n_slices=_N,
    )
    eval_hists[t] = eval_hist_total_fn(_sliced, spark=spark, write_path=f"{MODELS_PATH}/eval_hist_{t}")


def _auc(h):
    h = h.sort_values("bin", ascending=False)
    pos = h["n_pos"].astype(float)
    neg = h["n"] - pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Negatives in a bin are beaten by every positive in higher bins, tied with half of those in it.
    return float((neg * (pos.cumsum() - pos + pos / 2)).sum() / (n_pos * n_neg))


def _metrics(hist_pd):
    all_strata = hist_pd.groupby(["model", "bin"], as_index=False)[["n", "n_pos", "sum_p"]].sum()
    all_strata["stratum"] = "all"
    # "2+" = everything but singletons, whose labels are easy and can carry "all".
    multi = hist_pd[hist_pd["stratum"] != "1"].groupby(["model", "bin"], as_index=False)[["n", "n_pos", "sum_p"]].sum()
    multi["stratum"] = "2+"
    h = pd.concat([hist_pd, all_strata, multi], ignore_index=True)
    auc = h.groupby(["stratum", "model"]).apply(_auc, include_groups=False).unstack("model")[MODEL_NAMES]
    counts = h[h["model"] == MODEL_NAMES[0]].groupby("stratum")[["n", "n_pos"]].sum()
    auc.insert(0, "base_rate", (counts["n_pos"] / counts["n"]).round(4))
    auc.insert(0, "n_eval", counts["n"])
    order = [s for s in ["all", "2+", "1", "2-4", "5-16", "17-128", "129+"] if s in auc.index]
    auc = auc.reindex(order)
    calib = all_strata.assign(band=(all_strata["bin"] * 10 // N_SCORE_BINS))
    calib = calib.groupby(["model", "band"])[["n", "n_pos", "sum_p"]].sum()
    calib["predicted"] = (calib["sum_p"] / calib["n"]).round(4)
    calib["actual"] = (calib["n_pos"] / calib["n"]).round(4)
    return auc, calib[["n", "predicted", "actual"]].reset_index()


# COMMAND ----------

# MAGIC %md ## Results

# COMMAND ----------

for t in TARGETS:
    auc, calib = _metrics(eval_hists[t].toPandas())
    print(f"=== {t}: holdout AUC by size stratum (base_rate = share positive) ===")
    print(auc.round(4).to_string())
    print()
    print(f"=== {t}: holdout calibration (all strata) -- predicted vs actual by 10% band ===")
    print(calib.pivot(index="band", columns="model", values=["n", "predicted", "actual"]).to_string())
    print()

# COMMAND ----------

# MAGIC %md ## What the models learned

# COMMAND ----------

for t in TARGETS:
    lr_model = fitted[(t, "lr")].stages[-1]
    coef = pd.Series(lr_model.coefficients.toArray(), index=LR_FEATURES).sort_values()
    print(f"=== {t}: logistic regression coefficients (intercept {lr_model.intercept:.3f}) ===")
    print(coef.round(4).to_string())
    print()
    rates_pd = nb_token_rates[t].toPandas().sort_values("n", ascending=False)
    print(f"=== {t}: NB token rates (top 40 by nodes; rate = shrunk, type_rate = its prior) ===")
    print(rates_pd.head(40).round(4).to_string(index=False))
    print()

# COMMAND ----------

# MAGIC %md ## Full tables as CSV (paste back to save locally)

# COMMAND ----------

for t in TARGETS:
    auc, calib = _metrics(eval_hists[t].toPandas())
    print(f"# {t} auc")
    print(auc.round(4).to_csv())
    print(f"# {t} calibration")
    print(calib.to_csv(index=False))
    print(f"# {t} nb token rates")
    print(nb_token_rates[t].toPandas().sort_values("n", ascending=False).round(6).to_csv(index=False))
