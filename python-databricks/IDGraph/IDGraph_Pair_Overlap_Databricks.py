# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph -- pairs by via_type_count x node type pair
# MAGIC
# MAGIC How many pairs are linked by 1, 2 or 3 of our sources (id5/tapad/liid),
# MAGIC and within each `via_type_count`, which node types the pairs join.
# MAGIC
# MAGIC **Pairs** = every two nodes that share a grouping key of size
# MAGIC 2..`MAX_GROUP_SIZE` (16 = the IDGraph5 cap), enumerated from
# MAGIC `{via}_filtered_base` as full cliques. Unlike the built `edges` table this
# MAGIC is not thinned by the star layout (IDGraph5 stars every group above 6, and
# MAGIC two non-hub members of a starred group never get a direct edge, which
# MAGIC hides cross-source overlap). A pair linked by several keys of one source
# MAGIC counts once for that source.
# MAGIC
# MAGIC Graph-only: no Affinity labels, so it covers every pair, not just matched ones.

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
from staging_df import stage_dataframe

# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

import pandas as pd

# filtered_base/fanout are threshold-independent -- IDGraph4 and IDGraph5 hold
# identical copies. MAX_GROUP_SIZE sets which keys count.
GRAPH_ROOT = "s3://data-science-research/ravi/IDGraph5"
BUILD_OUTPUT_PATH = f"{GRAPH_ROOT}/build_output"
MAX_GROUP_SIZE = 16
OUTPUT_PATH = f"{GRAPH_ROOT}/pair_overlap/max{MAX_GROUP_SIZE}"

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

TYPE_SHORT = {
    "HEM_MD5": "HEM",
    "HARDWARE_IDFA": "IDFA",
    "HARDWARE_ANDROID_AD_ID": "AAID",
    "ID5_UID": "ID5",
    "TTD": "TTD",
    "ST": "ST",
}

# COMMAND ----------

# MAGIC %md ## Pairs per source
# MAGIC Staged per via_type (one big union+shuffle has hit Spark's
# MAGIC indeterminate-stage abort before). One row per distinct pair per source.

# COMMAND ----------


@stage_dataframe(write_format="parquet")
def source_pairs_fn(via_label, key_col, spark=None, write_path=None):
    b = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_filtered_base")
    deny = DENYLIST.get(via_label, set())
    if deny:
        b = b.filter(~F.col(key_col).isin(list(deny)))
    keys = (
        spark.read.parquet(f"{BUILD_OUTPUT_PATH}/{via_label}_fanout")
        .filter((F.col("n_distinct") >= 2) & (F.col("n_distinct") <= MAX_GROUP_SIZE))
        .select(key_col)
    )
    members = b.select("node_id", key_col).join(keys, on=key_col, how="leftsemi")
    left = members.select(F.col(key_col).alias("key"), F.col("node_id").alias("node_a"))
    right = members.select(F.col(key_col).alias("key"), F.col("node_id").alias("node_b"))
    return (
        left.join(right, on="key")
        .filter(F.col("node_a") < F.col("node_b"))
        .select("node_a", "node_b")
        .distinct()
    )


source_pairs = {
    via_label: source_pairs_fn(via_label, key_col, spark=spark, write_path=f"{OUTPUT_PATH}/pairs_{via_label}")
    for via_label, key_col in VIA_KEY_COLS
}

# COMMAND ----------

# MAGIC %md ## Pairs by source set x type pair
# MAGIC One row per (`via_set`, `type_pair`) with its pair count. `via_type_count`
# MAGIC is the number of sources in `via_set`. Node type comes from the
# MAGIC `TYPE:value` node id, so no join is needed.

# COMMAND ----------

_short_map = F.create_map(*[F.lit(x) for kv in TYPE_SHORT.items() for x in kv])


def _short_type(col):
    return _short_map[F.substring_index(F.col(col), ":", 1)]


@stage_dataframe(write_format="parquet")
def pair_overlap_fn(source_pairs, spark=None, write_path=None):
    parts = [df.withColumn("via_type", F.lit(via)) for via, df in source_pairs.items()]
    unioned = parts[0]
    for p in parts[1:]:
        unioned = unioned.unionByName(p)
    per_pair = unioned.groupBy("node_a", "node_b").agg(
        F.concat_ws("+", F.array_sort(F.collect_set("via_type"))).alias("via_set"),
        F.countDistinct("via_type").alias("via_type_count"),
    )
    type_pair = F.concat_ws("-", F.array_sort(F.array(_short_type("node_a"), _short_type("node_b"))))
    return (
        per_pair.withColumn("type_pair", type_pair)
        .groupBy("via_type_count", "via_set", "type_pair")
        .agg(F.count("*").alias("n_pairs"))
    )


pair_overlap = pair_overlap_fn(source_pairs, spark=spark, write_path=f"{OUTPUT_PATH}/pair_overlap")

# COMMAND ----------

# MAGIC %md ## Results
# MAGIC - by `via_type_count`, and by `via_set`
# MAGIC - within each `via_type_count`: the type-pair mix (`pct` = share of that
# MAGIC   via_type_count's pairs)
# MAGIC - per type pair: the share linked by 2+ sources

# COMMAND ----------

pd.set_option("display.width", 250)
_po = pair_overlap.toPandas()
_total = _po["n_pairs"].sum()


def _with_pct(df, by=None):
    df = df.copy()
    den = df.groupby(by)["n_pairs"].transform("sum") if by else df["n_pairs"].sum()
    df["pct"] = (100 * df["n_pairs"] / den).round(3)
    return df


print(f"distinct pairs (groups 2..{MAX_GROUP_SIZE}): {_total:,}")
print()
print("=== pairs by via_type_count ===")
_by_vtc = _with_pct(_po.groupby("via_type_count", as_index=False)["n_pairs"].sum())
print(_by_vtc.to_string(index=False))
print()
print("=== pairs by via_set ===")
_by_set = _with_pct(_po.groupby(["via_type_count", "via_set"], as_index=False)["n_pairs"].sum())
print(_by_set.sort_values(["via_type_count", "n_pairs"], ascending=[True, False]).to_string(index=False))
print()
_by_vtc_tp = _with_pct(
    _po.groupby(["via_type_count", "type_pair"], as_index=False)["n_pairs"].sum(), by="via_type_count"
)
for vtc, grp in _by_vtc_tp.groupby("via_type_count"):
    print(f"=== via_type_count = {vtc}: type pairs ({grp['n_pairs'].sum():,} pairs) ===")
    print(grp.sort_values("n_pairs", ascending=False).drop(columns="via_type_count").to_string(index=False))
    print()
print("=== type-pair mix: % of each via_type_count's pairs (columns sum to 100) ===")
print(_by_vtc_tp.pivot(index="type_pair", columns="via_type_count", values="pct").fillna(0).to_string())
print()
print("=== per type pair: share linked by 2+ sources ===")
_tp = _po.assign(multi=_po["n_pairs"].where(_po["via_type_count"] >= 2, 0))
_tp = _tp.groupby("type_pair", as_index=False)[["n_pairs", "multi"]].sum()
_tp["pct_2plus"] = (100 * _tp["multi"] / _tp["n_pairs"]).round(3)
print(_tp.rename(columns={"multi": "n_2plus"}).sort_values("n_pairs", ascending=False).to_string(index=False))

# COMMAND ----------

# MAGIC %md ## Full table as CSV (paste back to save locally)

# COMMAND ----------

print(_po.sort_values(["via_type_count", "via_set", "n_pairs"], ascending=[True, True, False]).to_csv(index=False))
