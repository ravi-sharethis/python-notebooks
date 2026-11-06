# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — component quality models (notebook B)
# MAGIC
# MAGIC Predicts, for EVERY component (including the class-0 ones Affinity can't
# MAGIC judge), the probability that it is over-merged and the probability that it
# MAGIC is incomplete -- from the component's own constituents, never from
# MAGIC Affinity. Affinity only provides the training labels (tune split) and the
# MAGIC test labels (holdout split), both from notebook A
# MAGIC (`IDGraph_Component_Quality_Databricks.py`). Design: `IDGraph_Journal.md` §14.
# MAGIC
# MAGIC **Run notebook A twice first** (`SCORE_SPLIT = "tune"` and `"holdout"`) on
# MAGIC the same GRAPH_ROOT.
# MAGIC
# MAGIC | model | idea |
# MAGIC |---|---|
# MAGIC | `lookup` | baseline: rate of the component's cell (sources x ID types x size x singleton reason), falling back to coarser cells |
# MAGIC | `nb_sum` | Bayesian naive Bayes: sum of token log-odds (classic NB -- big components saturate), calibrated with size |
# MAGIC | `nb_mean` | same token rates, but node-weighted MEAN token log-odds + size -- doesn't saturate |
# MAGIC | `lr` | logistic regression on size, source shares, ID-type shares, group sizes, singleton reasons |
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

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

# Must be a graph notebook A has already scored on both splits.
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
QUALITY_ROOT = f"{GRAPH_ROOT}/component_quality"
GRAPH_PATH = f"{QUALITY_ROOT}/graph"
TRAIN_SPLIT_PATH = f"{QUALITY_ROOT}/tune"
EVAL_SPLIT_PATH = f"{QUALITY_ROOT}/holdout"
MODELS_PATH = f"{QUALITY_ROOT}/models"

VIA_LABELS = ["id5", "liid", "tapad"]
TARGETS = ["over_merged", "incomplete"]

# A component with one matched ID can never be class 2; training on those
# would teach "less Affinity data = good". Applies to train AND test rows.
OVERMERGE_TRAIN_JUDGEABLE_ONLY = True

# Lookup cells with fewer labelled components fall back to the next coarser level.
LOOKUP_MIN_CELL = 100
LOOKUP_LEVELS = [
    ["sources", "id_types", "size_bucket", "reason_key"],
    ["sources", "id_types", "size_bucket"],
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
    f"{TRAIN_SPLIT_PATH}/component_summary",
    f"{EVAL_SPLIT_PATH}/component_summary",
]
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

# MAGIC %md ## Features (one row per component) and labels
# MAGIC Features come from the graph only (tune `component_summary` minus its label
# MAGIC columns, plus source shares from `component_tokens`); labels come from the
# MAGIC tune (training) and holdout (test) summaries.

# COMMAND ----------

LR_FEATURES = (
    ["log2_size", "is_singleton", "share_hem", "share_maid", "share_other", "has_hem_and_maid", "denylisted"]
    + [f"share_{v}" for v in VIA_LABELS]
    + [f"has_group_{v}" for v in VIA_LABELS]
    + [f"log2_max_group_{v}" for v in VIA_LABELS]
    + [f"log2_dropped_{v}" for v in VIA_LABELS]
    + [f"alone_{v}" for v in VIA_LABELS]
)


@stage_dataframe(write_format="parquet")
def features_fn(summary_df, tokens_df, slice_id=0, n_slices=1, spark=None, write_path=None):
    this_slice = in_slice("component", slice_id, n_slices)
    src = F.split(F.substring_index("token", ":", 1), "\\+")
    src_counts = tokens_df.filter(this_slice).groupBy("component").agg(
        *[F.sum(F.when(F.array_contains(src, v), F.col("n_nodes")).otherwise(0)).alias(f"n_src_{v}") for v in VIA_LABELS]
    )
    keep = (
        ["component", "component_size", "n_hem", "n_maid", "n_other", "sources", "id_types", "size_bucket"]
        + ["singleton_type", "singleton_reasons", "min_dropped_size"]
        + [f"max_group_size_{v}" for v in VIA_LABELS]
        + [f"dropped_size_{v}" for v in VIA_LABELS]
    )
    f = summary_df.filter(this_slice).select(*keep).join(src_counts, on="component", how="left")
    size = F.col("component_size").cast("double")
    reasons = F.coalesce(F.col("singleton_reasons"), F.lit(""))
    dropped_bucket = F.when(F.col("min_dropped_size").isNull(), F.lit("-")).otherwise(
        _size_bucket("min_dropped_size", DROPPED_SIZE_BOUNDS)
    )
    f = (
        f.withColumn("is_singleton", (F.col("sources") == "singleton").cast("double"))
        .withColumn("log2_size", F.log2(size))
        .withColumn("share_hem", F.col("n_hem") / size)
        .withColumn("share_maid", F.col("n_maid") / size)
        .withColumn("share_other", F.col("n_other") / size)
        .withColumn("has_hem_and_maid", ((F.col("n_hem") > 0) & (F.col("n_maid") > 0)).cast("double"))
        .withColumn("denylisted", reasons.contains("denylisted").cast("double"))
        .withColumn("dropped_bucket", dropped_bucket)
        .withColumn("reason_key", F.concat_ws("|", F.coalesce(F.col("singleton_reasons"), F.lit("-")), dropped_bucket))
        .withColumn("stratum", _size_bucket("component_size", STRATA_BOUNDS))
    )
    for v in VIA_LABELS:
        f = (
            f.withColumn(f"share_{v}", F.coalesce(F.col(f"n_src_{v}"), F.lit(0)) / size)
            .withColumn(f"has_group_{v}", F.col(f"max_group_size_{v}").isNotNull().cast("double"))
            .withColumn(f"log2_max_group_{v}", _LOG2(F.col(f"max_group_size_{v}")))
            .withColumn(f"log2_dropped_{v}", _LOG2(F.col(f"dropped_size_{v}")))
            .withColumn(f"alone_{v}", reasons.contains(f"{v}:alone").cast("double"))
            .drop(f"n_src_{v}")
        )
    return f


@stage_dataframe(write_format="parquet")
def labels_fn(summary_df, target, spark=None, write_path=None):
    rows = summary_df.filter(F.col("class") > 0)
    if target == "over_merged" and OVERMERGE_TRAIN_JUDGEABLE_ONLY:
        rows = rows.filter(F.col("judgeable"))
    return rows.select("component", F.col(target).cast("double").alias("label"))


# Notebook A may have written these in slices -- drop the partition column.
component_tokens = spark.read.parquet(f"{GRAPH_PATH}/component_tokens").drop("slice")
train_summary = spark.read.parquet(f"{TRAIN_SPLIT_PATH}/component_summary").drop("slice")
eval_summary = spark.read.parquet(f"{EVAL_SPLIT_PATH}/component_summary").drop("slice")

features = stage_sliced(
    features_fn, train_summary, component_tokens, spark=spark, write_path=f"{MODELS_PATH}/features", n_slices=_N
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
# MAGIC Per token: `n` = nodes carrying it in labelled training components,
# MAGIC `n_pos` = those in positive components. Rates are shrunk toward the
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
        .agg(F.sum("n_nodes").alias("n"), F.sum(F.col("n_nodes") * F.col("label")).alias("n_pos"))
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
    h = pd.concat([hist_pd, all_strata], ignore_index=True)
    auc = h.groupby(["stratum", "model"]).apply(_auc).unstack("model")[MODEL_NAMES]
    counts = h[h["model"] == MODEL_NAMES[0]].groupby("stratum")[["n", "n_pos"]].sum()
    auc.insert(0, "base_rate", (counts["n_pos"] / counts["n"]).round(4))
    auc.insert(0, "n_eval", counts["n"])
    order = ["all"] + [s for s in ["1", "2-4", "5-16", "17-128", "129+"] if s in auc.index]
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
