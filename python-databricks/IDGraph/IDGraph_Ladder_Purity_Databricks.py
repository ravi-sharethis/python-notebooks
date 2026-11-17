# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph -- ladder purity: naive Bayes P(pure) for every component of V1..V5
# MAGIC
# MAGIC Scores how pure each multi-node component is (pure = one Affinity person),
# MAGIC whatever version built it. Uses ONLY rates notebook A measured on tune --
# MAGIC no rates are learned here. Full write-up with worked examples:
# MAGIC `IDGraph-local/reports/ladder_purity_model.md`.
# MAGIC
# MAGIC **Rates (A, tune, judgeable components, per size bucket):**
# MAGIC - link class (`<sources> <type pair>`, A's pair table) and node token
# MAGIC   (`<source set>:<type>`, A's token table): share of judgeable components of
# MAGIC   that size containing it that are pure; a thin cell falls back to the
# MAGIC   token's over-merge lift x the size's over-merge rate
# MAGIC - size prior: share of judgeable components of that size that are pure
# MAGIC
# MAGIC **Links count as the sources provided them.** Candidate links = pairs that
# MAGIC share a key, labelled with all the sources that put them together (one link
# MAGIC per pair). For a component of n nodes, n - 1 links are picked best-rated
# MAGIC first (a maximum spanning tree), so a pair seen in 2 sources counts once,
# MAGIC and a group of k nodes counts k - 1 links.
# MAGIC
# MAGIC **Scores:** start = logit(size prior); evidence of a token = logit(rate at
# MAGIC the size) - logit(size prior); tokens without a rate add 0.
# MAGIC - `p_raw` = 1 / (1 + e^-(start + all picked links + all nodes))
# MAGIC - `p_best` = from start + the strongest single token (link or node)
# MAGIC - `p_worst` = from start + the weakest picked link (the link the component
# MAGIC   can least do without)
# MAGIC - `p_pure` = logistic calibration of (nb_sum, log2 size) on tune components with
# MAGIC   REFERENCE_LABELLED labelled nodes
# MAGIC Checked on holdout, per version.
# MAGIC
# MAGIC Assumptions: A's rates come from today's graph (V5) and are applied to V1..V4 too.
# MAGIC The ladder's cut-offs were set with pair-level rates, a different measure.
# MAGIC
# MAGIC Run after notebook A (both splits) and the ladder notebook, same SAMPLE_PCT and
# MAGIC LABEL_SET. Writes under `IDGraph5_ladder/sample<pct>/purity_v2/`.

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

import math

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
LADDER_ROOT = "s3://data-science-research/ravi/IDGraph5_ladder"

# Must match notebook A and the ladder (job-level parameters).
dbutils.widgets.text("SAMPLE_PCT", "100")
SAMPLE_PCT = int(dbutils.widgets.get("SAMPLE_PCT"))
dbutils.widgets.dropdown("LABEL_SET", "all", ["all", "hem"])
LABEL_SET = dbutils.widgets.get("LABEL_SET")
_LABEL_SUFFIX = "" if LABEL_SET == "all" else f"_{LABEL_SET}"

SPLITS = ["tune", "holdout"]
LADDER_PATH = f"{LADDER_ROOT}/sample{SAMPLE_PCT}"
PURITY_PATH = f"{LADDER_PATH}/purity_v2{_LABEL_SUFFIX}"
# Notebook A's tune tables (the rates).
A_TUNE_PATH = f"{GRAPH_ROOT}/component_quality" + ("" if SAMPLE_PCT >= 100 else f"_sample{SAMPLE_PCT}") + f"/tune{_LABEL_SUFFIX}"

VIA_LABELS = ["id5", "liid", "tapad"]
TYPE_SHORT = {"HEM_MD5": "HEM", "HARDWARE_IDFA": "IDFA", "HARDWARE_ANDROID_AD_ID": "AAID"}

# Same ladder as IDGraph_Ladder_Databricks.py: the version that adds each class.
CLASS_VERSION = {
    "tapad HEM-HEM": 2, "liid HEM-IDFA": 2, "tapad IDFA-IDFA": 2,
    "liid IDFA-IDFA": 3, "liid AAID-HEM": 3,
    "liid AAID-AAID": 4, "tapad AAID-HEM": 4, "tapad HEM-IDFA": 4,
    "tapad AAID-AAID": 4, "tapad AAID-IDFA": 4, "liid AAID-IDFA": 4,
}
VERSIONS = [1, 2, 3, 4, 5]

# Must match notebook A's size buckets.
COMPONENT_SIZE_BOUNDS = [1, 2, 4, 8, 16, 32, 64, 128, 1_000, 10_000, 100_000]
# A link class / node token needs this many judgeable components at a size for its own
# rate there; otherwise lift x the size's over-merge rate.
MIN_TOKEN_COMPONENTS = 50
RATE_CLIP = 0.01  # keeps a rate of 0 or 1 from giving unbounded evidence
# Calibration uses tune components with exactly this many labelled nodes; the holdout
# check uses holdout components with this many.
REFERENCE_LABELLED = 2
# Predicted-purity distribution: bands of this width from 0 to 1.
PURITY_BAND_WIDTH = 0.05
N_SCORE_BINS = 1000
LR_REG_PARAM = 0.001

_inputs = [f"{LADDER_PATH}/eval_nodes{_LABEL_SUFFIX}"] + [f"{LADDER_PATH}/pairs_{v}{_LABEL_SUFFIX}" for v in VIA_LABELS] + [
    f"{A_TUNE_PATH}/{t}" for t in ("graph_pair_health", "token_health", "component_summary")
]
_missing = [p for p in _inputs if not _path_exists(p)]
assert not _missing, f"missing inputs (run notebook A on both splits, then the ladder): {_missing}"
print(f"ladder purity on {LADDER_PATH}, labels from {LABEL_SET} matched nodes -> {PURITY_PATH}")

# COMMAND ----------

# MAGIC %md ## Helpers

# COMMAND ----------

_type_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


def _short_type(col):
    # node_id is "<type>:<value>".
    t = F.substring_index(F.col(col), ":", 1)
    return F.coalesce(_type_short_map[t], t)


def _py_short_type(node_id):
    t = node_id.split(":", 1)[0]
    return TYPE_SHORT.get(t, t)


def _read(path):
    return spark.read.parquet(path).drop("slice")


def _logit(p):
    p = min(max(p, RATE_CLIP), 1 - RATE_CLIP)
    return math.log(p / (1 - p))


def _py_size_bucket(n):
    lo = COMPONENT_SIZE_BOUNDS[0]
    for hi in COMPONENT_SIZE_BOUNDS:
        if n <= hi:
            return str(hi) if lo == hi else f"{lo}-{hi}"
        lo = hi + 1
    return f"{COMPONENT_SIZE_BOUNDS[-1] + 1}+"


def _auc(h):
    """AUC from a histogram of (bin, n, n_pos)."""
    h = h.sort_values("bin", ascending=False)
    pos = h["n_pos"].astype(float)
    neg = h["n"] - pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((neg * (pos.cumsum() - pos + pos / 2)).sum() / (n_pos * n_neg))


# COMMAND ----------

# MAGIC %md ## 1. Rates from notebook A (tune)
# MAGIC Small tables, built in pandas:
# MAGIC - `size_priors_pd`: per size bucket, pure share of judgeable components
# MAGIC - `rates_pd`: per token (`link <sources> <type pair>` / `node <source set>:<type>`)
# MAGIC   x size bucket: rate, where it came from, and evidence = logit(rate) - logit(size prior)

# COMMAND ----------


def _token_size_rates(df, prefix, over_rate):
    """df: token, size_bucket, n_judgeable, n_over_merged -> one row per token x size bucket."""
    df = df.merge(over_rate.rename("over_rate").reset_index(), on="size_bucket", how="left")
    df["exp_over"] = df["n_judgeable"] * df["over_rate"]
    lift = df.groupby("token")[["n_over_merged", "exp_over"]].sum()
    lift = (lift["n_over_merged"] / lift["exp_over"]).rename("lift")
    grid = pd.MultiIndex.from_product([lift.index, over_rate.index], names=["token", "size_bucket"]).to_frame(index=False)
    grid = grid.merge(df[["token", "size_bucket", "n_judgeable", "n_over_merged"]], on=["token", "size_bucket"], how="left")
    grid = grid.merge(over_rate.rename("over_rate").reset_index(), on="size_bucket").merge(lift.reset_index(), on="token")
    own = grid["n_judgeable"].fillna(0) >= MIN_TOKEN_COMPONENTS
    grid["rate"] = np.where(own, 1 - grid["n_over_merged"] / grid["n_judgeable"].where(own, 1), 1 - grid["lift"] * grid["over_rate"])
    grid["rate_source"] = np.where(own, "token x size", "lift fallback")
    grid["token"] = prefix + grid["token"]
    return grid


def _a_rates():
    cs = _read(f"{A_TUNE_PATH}/component_summary").filter(F.col("judgeable")).groupBy("size_bucket").agg(
        F.count("*").alias("n"), F.sum(F.col("over_merged").cast("long")).alias("n_over")
    ).toPandas().set_index("size_bucket")
    cs["over_rate"] = cs["n_over"] / cs["n"]
    cs["prior"] = 1 - cs["over_rate"]
    global_prior = 1 - cs["n_over"].sum() / cs["n"].sum()

    ph = _read(f"{A_TUNE_PATH}/graph_pair_health").groupBy("sources", "type_pair", "size_bucket").agg(
        F.sum("n_judgeable").alias("n_judgeable"), F.sum("n_over_merged").alias("n_over_merged")
    ).toPandas()
    ph["token"] = ph["sources"] + " " + ph["type_pair"]
    th = _read(f"{A_TUNE_PATH}/token_health").select("token", "size_bucket", "n_judgeable", "n_over_merged").toPandas()
    rates = pd.concat([
        _token_size_rates(ph[["token", "size_bucket", "n_judgeable", "n_over_merged"]], "link ", cs["over_rate"]),
        _token_size_rates(th, "node ", cs["over_rate"]),
    ], ignore_index=True)
    rates["prior"] = rates["size_bucket"].map(cs["prior"])
    rates["evidence"] = [_logit(r) - _logit(p) for r, p in zip(rates["rate"], rates["prior"])]
    return cs.reset_index(), global_prior, rates


size_priors_pd, GLOBAL_PRIOR, rates_pd = _a_rates()
# Plain dicts for the per-component pass (small: tokens x size buckets).
EVIDENCE = {(t, b): e for t, b, e in zip(rates_pd["token"], rates_pd["size_bucket"], rates_pd["evidence"])}
START = {b: _logit(p) for b, p in zip(size_priors_pd["size_bucket"], size_priors_pd["prior"])}
print(f"component prior over all sizes (A, tune): {GLOBAL_PRIOR:.4f}; {len(rates_pd)} token x size rates")

# COMMAND ----------

# MAGIC %md ## 2. Links with their class and version
# MAGIC Every candidate link (pairs per source from the ladder, unioned): `cls` =
# MAGIC `<all sources that link the pair> <type pair>`, `level` = the version that adds it.

# COMMAND ----------

_class_map = F.create_map(*[F.lit(x) for kv in CLASS_VERSION.items() for x in kv])


@stage_dataframe(write_format="parquet")
def ladder_links_fn(parts, spark=None, write_path=None):
    unioned = None
    for p, v in zip(parts, VIA_LABELS):
        p = p.withColumn("via_type", F.lit(v))
        unioned = p if unioned is None else unioned.unionByName(p)
    pairs = unioned.groupBy("component", "node_a", "node_b").agg(F.array_sort(F.collect_set("via_type")).alias("sources"))
    type_pair = F.concat_ws("-", F.array_sort(F.array(_short_type("node_a"), _short_type("node_b"))))
    single_cls = F.concat(F.col("sources")[0], F.lit(" "), type_pair)
    level = F.when(F.size("sources") >= 2, F.lit(1)).otherwise(F.coalesce(_class_map[single_cls], F.lit(5)))
    return pairs.select(
        "component", "node_a", "node_b",
        F.concat(F.concat_ws("+", "sources"), F.lit(" "), type_pair).alias("cls"),
        level.cast("int").alias("level"),
    )


eval_nodes = _read(f"{LADDER_PATH}/eval_nodes{_LABEL_SUFFIX}")
ladder_links = ladder_links_fn(
    [_read(f"{LADDER_PATH}/pairs_{v}{_LABEL_SUFFIX}") for v in VIA_LABELS], spark=spark, write_path=f"{PURITY_PATH}/ladder_links"
)

# COMMAND ----------

# MAGIC %md ## 3. Score every multi-node component of every version
# MAGIC One pass per sampled component (applyInPandas). For each version: the
# MAGIC version's links form its components; for each multi-node component, its
# MAGIC size bucket selects the rates; n - 1 links are picked best evidence first
# MAGIC (unrated last) as a maximum spanning tree; tokens = picked links + one node
# MAGIC token per node (`<source set>:<type>`, source set = sources of the node's
# MAGIC links in that version). Output per component: `tokens` (picked links and
# MAGIC nodes, with counts), `start`, `nb_sum`, `ev_best`, `ev_worst`, and the labels
# MAGIC per split (`n_single_*` labelled nodes, `n_afs_*` distinct Affinity persons).

# COMMAND ----------

_SCORE_SCHEMA = T.StructType(
    [
        T.StructField("version", T.IntegerType()),
        T.StructField("vcomp", T.StringType()),
        T.StructField("size", T.LongType()),
        T.StructField("size_bucket", T.StringType()),
        T.StructField("tokens", T.MapType(T.StringType(), T.LongType())),
        T.StructField("start", T.DoubleType()),
        T.StructField("nb_sum", T.DoubleType()),
        T.StructField("ev_best", T.DoubleType()),
        T.StructField("ev_worst", T.DoubleType()),
    ]
    + [f for s in SPLITS for f in (T.StructField(f"n_single_{s}", T.LongType()), T.StructField(f"n_afs_{s}", T.LongType()))]
)


def _make_scorer(evidence, start, global_start):
    def _score_components(pdf):
        comp = str(pdf["component"].iloc[0])
        is_node = pdf["level"] == 0
        nodes_ = list(pdf.loc[is_node, "node_a"])
        afs = {s: dict(zip(pdf.loc[is_node, "node_a"], pdf.loc[is_node, f"afs_{s}"])) for s in SPLITS}
        e = pdf[~is_node]
        edges = list(zip(e["node_a"], e["node_b"], e["level"], e["cls"]))
        out = []
        for v in VERSIONS:
            ev_links = [x for x in edges if x[2] <= v]
            # components of this version
            parent = {n: n for n in nodes_}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            node_src = {}
            for a, b, _, cls in ev_links:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                srcs = cls.split(" ", 1)[0].split("+")
                node_src.setdefault(a, set()).update(srcs)
                node_src.setdefault(b, set()).update(srcs)
            groups, comp_links = {}, {}
            for n in nodes_:
                groups.setdefault(find(n), []).append(n)
            for x in ev_links:
                comp_links.setdefault(find(x[0]), []).append(x)
            for root, members in groups.items():
                if len(members) < 2:
                    continue
                bucket = _py_size_bucket(len(members))
                st = start.get(bucket, global_start)
                # maximum spanning tree: best evidence first, unrated (None) last
                cand = [(evidence.get((f"link {cls}", bucket)), a, b, cls) for a, b, _, cls in comp_links.get(root, [])]
                cand.sort(key=lambda c: 1e9 if c[0] is None else -c[0])
                p2 = {n: n for n in members}

                def find2(x):
                    while p2[x] != x:
                        p2[x] = p2[p2[x]]
                        x = p2[x]
                    return x

                tokens, link_evs = {}, []
                for ev, a, b, cls in cand:
                    ra, rb = find2(a), find2(b)
                    if ra == rb:
                        continue
                    p2[max(ra, rb)] = min(ra, rb)
                    tokens[f"link {cls}"] = tokens.get(f"link {cls}", 0) + 1
                    link_evs.append(0.0 if ev is None else ev)
                node_evs = []
                for n in members:
                    k = f"node {'+'.join(sorted(node_src.get(n, ())))}:{_py_short_type(n)}"
                    tokens[k] = tokens.get(k, 0) + 1
                    node_evs.append(evidence.get((k, bucket), 0.0))
                row = {
                    "version": v, "vcomp": f"{comp}|{root}", "size": len(members), "size_bucket": bucket,
                    "tokens": tokens, "start": st, "nb_sum": st + sum(link_evs) + sum(node_evs),
                    "ev_best": max(link_evs + node_evs), "ev_worst": min(link_evs),
                }
                for s in SPLITS:
                    ids = [afs[s][n] for n in members if isinstance(afs[s][n], str)]
                    row[f"n_single_{s}"] = len(ids)
                    row[f"n_afs_{s}"] = len(set(ids))
                out.append(row)
        return pd.DataFrame(out, columns=[f.name for f in _SCORE_SCHEMA.fields])

    return _score_components


@stage_dataframe(write_format="parquet")
def component_scores_fn(eval_df, links_df, spark=None, write_path=None):
    node_rows = eval_df.select(
        "component", F.col("node_id").alias("node_a"), F.col("node_id").alias("node_b"),
        F.lit(0).alias("level"), F.lit(None).cast("string").alias("cls"), *[f"afs_{s}" for s in SPLITS],
    )
    edge_rows = links_df.select(
        "component", "node_a", "node_b", "level", "cls", *[F.lit(None).cast("string").alias(f"afs_{s}") for s in SPLITS]
    )
    scorer = _make_scorer(EVIDENCE, START, _logit(GLOBAL_PRIOR))
    sig = lambda c: 1 / (1 + F.exp(-c))
    return (
        node_rows.unionByName(edge_rows).groupBy("component").applyInPandas(scorer, schema=_SCORE_SCHEMA)
        .withColumn("p_raw", sig(F.col("nb_sum")))
        .withColumn("p_best", sig(F.col("start") + F.col("ev_best")))
        .withColumn("p_worst", sig(F.col("start") + F.col("ev_worst")))
        .withColumn("log2_size", F.log2(F.col("size").cast("double")))
    )


component_scores = component_scores_fn(eval_nodes, ladder_links, spark=spark, write_path=f"{PURITY_PATH}/component_scores")

# COMMAND ----------

# MAGIC %md ## 4. Calibrate (tune), score every component
# MAGIC `p_pure` = logistic regression on (nb_sum, log2 size), fitted on tune
# MAGIC components with REFERENCE_LABELLED labelled nodes.

# COMMAND ----------


def _reference(df, split):
    return df.filter(F.col(f"n_single_{split}") == REFERENCE_LABELLED).withColumn(
        "label", (F.col(f"n_afs_{split}") == 1).cast("double")
    )


_calib_path = f"{PURITY_PATH}/fitted_calibration"
if _path_exists(_calib_path):
    calibration = PipelineModel.load(_calib_path)
else:
    calibration = Pipeline(stages=[
        VectorAssembler(inputCols=["nb_sum", "log2_size"], outputCol="cal_x"),
        LogisticRegression(featuresCol="cal_x", labelCol="label", regParam=LR_REG_PARAM, probabilityCol="cal_prob",
                           rawPredictionCol="cal_raw", predictionCol="cal_pred"),
    ]).fit(_reference(component_scores, "tune"))
    calibration.write().overwrite().save(_calib_path)


@stage_dataframe(write_format="parquet")
def purity_scores_fn(scores_df, spark=None, write_path=None):
    return calibration.transform(scores_df).withColumn("p_pure", vector_to_array("cal_prob")[1]).select(
        "version", "vcomp", "size", "size_bucket", "nb_sum", "p_raw", "p_best", "p_worst", "p_pure",
        *[c for s in SPLITS for c in (f"n_single_{s}", f"n_afs_{s}")],
    )


purity_scores = purity_scores_fn(component_scores, spark=spark, write_path=f"{PURITY_PATH}/purity_scores")

# COMMAND ----------

# MAGIC %md ## Results
# MAGIC
# MAGIC **Glossary**
# MAGIC
# MAGIC | column | meaning |
# MAGIC |---|---|
# MAGIC | `token` | `link <sources> <type pair>` or `node <source set>:<type>` |
# MAGIC | `prior` (size bucket) | A: share of judgeable components of that size that are pure |
# MAGIC | token `rate` | A: share of judgeable components of that size containing the token that are pure (`rate_source`: own cell, or lift fallback) |
# MAGIC | `evidence` | logit(rate) - logit(size prior); tokens without a rate add 0 |
# MAGIC | `p_raw` | 1 / (1 + e^-nb_sum), nb_sum = logit(size prior) + evidence of the n - 1 picked links and every node |
# MAGIC | `p_best`, `p_worst` | from logit(size prior) + the strongest single token / the weakest picked link |
# MAGIC | `p_pure` | calibrated P(pure) from (nb_sum, log2 size) |
# MAGIC | `auc` | holdout AUC on components with REFERENCE_LABELLED labelled nodes (0.5 = chance) |
# MAGIC | `predicted`, `actual` | mean p_pure vs actual share pure, holdout, per 10% band |
# MAGIC | band rows | share of ALL multi-node components of the version whose p_pure falls in the band (PURITY_BAND_WIDTH wide, 0 to 1) |

# COMMAND ----------

pd.set_option("display.width", 250)

print("=== component prior per size bucket (A, tune, judgeable components) ===")
print(size_priors_pd[["size_bucket", "n", "prior"]].round(3).to_string(index=False))
print()
for _kind in ("link", "node"):
    _r = rates_pd[rates_pd["token"].str.startswith(_kind + " ")]
    print(f"=== {_kind} rates from notebook A: P(pure | token) per size bucket (tune) ===")
    print(_r.pivot(index="token", columns="size_bucket", values="rate").round(3).to_string())
    print(f"(lift fallback used in {int((_r['rate_source'] == 'lift fallback').sum())} of {len(_r)} cells)")
    print()


def _hist(df, p_col, split, by):
    rows = _reference(df, split)
    return (
        rows.groupBy(*by, F.least(F.floor(F.col(p_col) * N_SCORE_BINS), F.lit(N_SCORE_BINS - 1)).alias("bin"))
        .agg(F.count("*").alias("n"), F.sum("label").alias("n_pos"), F.sum(p_col).alias("sum_p"))
        .toPandas()
    )


_rows = []
for _p in ("p_pure", "p_raw", "p_best", "p_worst"):
    _h = _hist(purity_scores, _p, "holdout", ["version"])
    _rows.append({"score": _p, "version": "all", "auc": _auc(_h.groupby("bin", as_index=False)[["n", "n_pos"]].sum())})
    for _v, _g in _h.groupby("version"):
        _rows.append({"score": _p, "version": _v, "auc": _auc(_g)})
_n_ref = _reference(purity_scores, "holdout").groupBy("version").count().toPandas().set_index("version")["count"]
_aucs = pd.DataFrame(_rows).pivot(index="version", columns="score", values="auc")
_aucs["n"] = [int(_n_ref.sum()) if v == "all" else int(_n_ref.get(v, 0)) for v in _aucs.index]
print(f"=== holdout AUC (holdout components with {REFERENCE_LABELLED} labelled nodes) ===")
print(_aucs[["n", "p_pure", "p_raw", "p_best", "p_worst"]].round(4).to_string())
print()
_h = _hist(purity_scores, "p_pure", "holdout", [])
_cal = _h.assign(band=_h["bin"] * 10 // N_SCORE_BINS).groupby("band")[["n", "n_pos", "sum_p"]].sum()
_cal["predicted"] = (_cal["sum_p"] / _cal["n"]).round(3)
_cal["actual"] = (_cal["n_pos"] / _cal["n"]).round(3)
print("=== holdout calibration of p_pure, by 10% band ===")
print(_cal[["n", "predicted", "actual"]].to_string())
print()

_n_bands = int(round(1 / PURITY_BAND_WIDTH))
_band_i = F.least(F.floor(F.col("p_pure") / PURITY_BAND_WIDTH), F.lit(_n_bands - 1)).cast("int")
_pd = purity_scores.groupBy("version", _band_i.alias("band_i")).agg(F.count("*").alias("n")).toPandas()
_pd["pct"] = 100 * _pd["n"] / _pd.groupby("version")["n"].transform("sum")
_labels = [f"{i * PURITY_BAND_WIDTH:.2f}-{(i + 1) * PURITY_BAND_WIDTH:.2f}" for i in range(_n_bands)]
_dist = _pd.pivot(index="band_i", columns="version", values="pct").reindex(range(_n_bands)).fillna(0).round(1)
_dist.index = pd.Index(_labels, name="p_pure band")
_mean = (
    purity_scores.groupBy("version").agg(
        F.count("*").alias("n_multi_node"), F.avg("p_pure").alias("mean_p_pure"),
        F.avg("p_best").alias("mean_p_best"), F.avg("p_worst").alias("mean_p_worst"),
    ).toPandas().set_index("version").sort_index()
)
print("=== predicted purity of ALL multi-node components: % per p_pure band, by version ===")
print(_dist.to_string())
print(_mean.round(3).T.to_string())

# COMMAND ----------

# MAGIC %md ## Full tables as CSV (paste back to save locally)

# COMMAND ----------

print("# rates (token x size bucket)")
print(rates_pd[["token", "size_bucket", "n_judgeable", "rate", "rate_source", "evidence"]].to_csv(index=False))
print("# predicted purity distribution (% per band, columns = version)")
print(_dist.to_csv())
print("# means per version")
print(_mean.to_csv())
