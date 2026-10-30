# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph — connected components (Spark/GraphFrames)
# MAGIC
# MAGIC Runs after the build script. Reads `node_types`/`edges`, runs
# MAGIC `connectedComponents()`, and writes `result` (node/type/component/
# MAGIC component_size). **This notebook does ONLY that** -- for all downstream
# MAGIC analysis (per-type summary, type matrix, histograms, component-size
# MAGIC diagnostics, point-query helpers), see `IDGraph_CC_Analysis_Databricks.py`,
# MAGIC which reads this notebook's `result` output and does NOT need GraphFrames
# MAGIC installed at all -- splitting it out means the expensive, failure-prone
# MAGIC compute step only needs to run once (or whenever MAX_FANOUT/STAR_THRESHOLD/
# MAGIC the underlying edges change), while analysis can be iterated on freely
# MAGIC without ever touching this step again (`stage_dataframe` skips it anyway
# MAGIC if `result` already exists, but keeping them in separate notebooks means
# MAGIC you don't even need GraphFrames attached to the cluster you use for
# MAGIC day-to-day analysis).
# MAGIC
# MAGIC **Every join/groupBy/union goes through `@stage_dataframe`** (from
# MAGIC `staging_df.py`, imported below): each stage reads from and writes to its
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
from graphframes import GraphFrame
from staging_df import _path_exists, stage_dataframe


# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

BUILD_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph3/build_output/"
CC_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph3/cc_output/"
CHECKPOINT_PATH = "s3://data-science-research/ravi/IDGraph3/gf_checkpoints/"
RESULT_PATH = f"{CC_OUTPUT_PATH}/result"

spark.sparkContext.setCheckpointDir(CHECKPOINT_PATH)

# COMMAND ----------

# MAGIC %md ## Load build output, build the GraphFrame
# MAGIC Plain reads + renames -- not staged, since these aren't new computations,
# MAGIC just reading/relabeling what the build script already produced.

# COMMAND ----------

node_types = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/node_types")
edges = spark.read.parquet(f"{BUILD_OUTPUT_PATH}/edges")

vertices = node_types.withColumnRenamed("node_id", "id")
edges_gf = edges.withColumnRenamed("node_a", "src").withColumnRenamed("node_b", "dst")

g = GraphFrame(vertices, edges_gf)
print(f"{vertices.count():,} vertices, {edges_gf.count():,} edges")

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


result = result_with_size_fn(components, spark=spark, write_path=RESULT_PATH)

# One-time optimization for the point-query helpers / analysis notebook -- clusters
# rows by `component` within files so a lookup skips most of the table instead of
# scanning all ~4B rows.
spark.sql(f"OPTIMIZE delta.`{RESULT_PATH}` ZORDER BY (component)")

n_components = result.select("component").distinct().count()
print(f"{n_components:,} components over {result.count():,} nodes")

# COMMAND ----------

# MAGIC %md ## Done -- next steps
# MAGIC `result` is written to `RESULT_PATH` (Delta, Z-ORDERed by `component`).
# MAGIC All downstream analysis -- per-type summary, type matrix, histograms,
# MAGIC component-size diagnostics, point-query helpers -- lives in
# MAGIC `IDGraph_CC_Analysis_Databricks.py`, which reads this output directly and
# MAGIC does not need GraphFrames or this notebook re-run.
