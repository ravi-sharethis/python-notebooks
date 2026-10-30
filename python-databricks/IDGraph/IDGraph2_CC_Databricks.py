# Databricks notebook source
# MAGIC %md
# MAGIC # ID Graph -- connected components on IDGraph2 (sample, then full)
# MAGIC
# MAGIC Standalone GraphFrames smoke test / cost-calibration run against the ALREADY-
# MAGIC COMPUTED IDGraph2 output (`node_types`/`edges`), independent of the IDGraph3
# MAGIC rebuild (`IDGraph_Build_Databricks.py`), which still needs `RAW_PATH` filled in
# MAGIC and hasn't run yet. This answers two things before committing a large cluster
# MAGIC run against IDGraph3:
# MAGIC 1. Does GraphFrames actually work on this cluster (17.3 LTS / Spark 4.0.0 /
# MAGIC    Scala 2.13) -- unverified until this runs.
# MAGIC 2. How many iterations / how long does `connectedComponents()` actually take at
# MAGIC    real scale, so IDGraph3's cost can be estimated rather than guessed.
# MAGIC
# MAGIC **Important caveat**: IDGraph2's `edges` table predates the hybrid star/clique
# MAGIC rewrite -- it's the full-clique, no-`MAX_FANOUT` version that produced the 8TB
# MAGIC edges surprise (~117B edges, ~4.9B nodes; see chat). This is therefore a rough,
# MAGIC worst-case-ish mechanics/timing test, NOT a preview of IDGraph3's actual cost --
# MAGIC IDGraph3 (`MAX_FANOUT=64`, hybrid star) will have meaningfully fewer edges to
# MAGIC process than what this notebook feeds GraphFrames.
# MAGIC
# MAGIC **Run this notebook twice**: once with `SAMPLE = True` (fast, cheap, checks
# MAGIC mechanics and gives a rough timing signal), then with `SAMPLE = False` (the real
# MAGIC full-scale run) once the sample pass looks sane.

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

IDGRAPH2_BUILD_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph2/build_output/"
CC_OUTPUT_PATH = "s3://data-science-research/ravi/IDGraph2/cc_output/"
CHECKPOINT_PATH = "s3://data-science-research/ravi/IDGraph2/gf_checkpoints/"

spark.sparkContext.setCheckpointDir(CHECKPOINT_PATH)

# Run this notebook once with SAMPLE=True (smoke test + rough timing), review the
# output, then flip to SAMPLE=False and rerun for the real full-scale pass.
SAMPLE = True
SAMPLE_FILE_COUNT = 200  # number of edges/ parquet files to sample, not a row/node fraction
SAMPLE_SEED = 42

RUN_TAG = "sample" if SAMPLE else "full"

# COMMAND ----------

# MAGIC %md ## Load IDGraph2's node_types/edges
# MAGIC Sampling here is FILE-based (random subset of `edges`' underlying parquet
# MAGIC files), not a `.sample()`/join filter on the full tables -- a join-based filter
# MAGIC (build the sample from a small set of nodes, then join against the full `edges`
# MAGIC to find their edges) still forces Spark to scan every row of the full ~117B-row
# MAGIC / 8TB `edges` table to evaluate the join predicate, since there's no partition
# MAGIC pruning available on node_a/node_b. Reading a random slice of files instead
# MAGIC reads roughly SAMPLE_FILE_COUNT/total_files of the actual bytes -- the real
# MAGIC saving the sample pass is supposed to give you. Same caveat as the build
# MAGIC script's raw-file sampling: a node's edges can be split across files, so this
# MAGIC will fragment real components -- fine for a mechanics/timing smoke test, not for
# MAGIC anything analytical.
# MAGIC
# MAGIC For the same reason, full-table `.count()` on `node_types`/`edges` is skipped
# MAGIC entirely in SAMPLE mode -- even a `.count()` (parquet-footer-based, not a full
# MAGIC column scan) still means touching metadata across every file in the dataset,
# MAGIC which isn't needed just to run the sample pass.

# COMMAND ----------

if SAMPLE:
    import random

    edges_path = f"{IDGRAPH2_BUILD_OUTPUT_PATH}/edges"
    all_edge_files = [f.path for f in dbutils.fs.ls(edges_path) if f.path.endswith(".parquet")]
    rng = random.Random(SAMPLE_SEED)
    sample_files = rng.sample(all_edge_files, min(SAMPLE_FILE_COUNT, len(all_edge_files)))
    print(
        f"SAMPLE=True: reading {len(sample_files)}/{len(all_edge_files)} randomly-selected edges files "
        f"({100 * len(sample_files) / len(all_edge_files):.2f}%)"
    )
    run_edges = spark.read.parquet(*sample_files)

    # node_types is far smaller than edges (fewer rows, narrower schema) -- a full
    # scan of it (via this leftsemi join) is cheap relative to what we just avoided
    # doing to edges, and keeps the sample-mode vertex set from ballooning to the
    # full ~4.9B nodes (most of which the small edges sample never touches).
    node_types = spark.read.parquet(f"{IDGRAPH2_BUILD_OUTPUT_PATH}/node_types")
    referenced_nodes = run_edges.select(F.col("node_a").alias("node_id")).unionByName(
        run_edges.select(F.col("node_b").alias("node_id"))
    ).distinct()
    run_node_types = node_types.join(referenced_nodes, on="node_id", how="leftsemi")
else:
    run_node_types = spark.read.parquet(f"{IDGRAPH2_BUILD_OUTPUT_PATH}/node_types")
    run_edges = spark.read.parquet(f"{IDGRAPH2_BUILD_OUTPUT_PATH}/edges")
    print(f"IDGraph2 full size -- node_types: {run_node_types.count():,}, edges: {run_edges.count():,}")

vertices = run_node_types.withColumnRenamed("node_id", "id")
edges_gf = run_edges.withColumnRenamed("node_a", "src").withColumnRenamed("node_b", "dst")

g = GraphFrame(vertices, edges_gf)
n_vertices = vertices.count()
n_edges = edges_gf.count()
print(f"[{RUN_TAG}] graph: {n_vertices:,} vertices, {n_edges:,} edges")

# COMMAND ----------

# MAGIC %md ## Connected components -- timed
# MAGIC The timing only reflects real compute on a FIRST run at this `RUN_TAG`'s path --
# MAGIC `stage_dataframe` skips execution and just reads back existing output on a
# MAGIC rerun, so re-running with the same SAMPLE value will report a near-zero
# MAGIC elapsed time that means nothing about actual cost.

# COMMAND ----------

import time


@stage_dataframe(write_format="parquet")
def connected_components_fn(graph, spark=None, write_path=None):
    return graph.connectedComponents()


start = time.time()
components = connected_components_fn(g, spark=spark, write_path=f"{CC_OUTPUT_PATH}/{RUN_TAG}/components")
n_components_rows = components.count()  # forces materialization
elapsed = time.time() - start
print(f"[{RUN_TAG}] connectedComponents() -- {elapsed:.1f}s wall clock, {n_components_rows:,} rows")

# COMMAND ----------

# MAGIC %md ## Quick sanity summary
# MAGIC Not the full per-type/type-matrix suite -- that belongs to `IDGraph_CC_Databricks.py`
# MAGIC once IDGraph3 is actually ready. Just enough here to confirm the output looks
# MAGIC sane: component count, singleton share, largest components.

# COMMAND ----------

component_sizes = components.groupBy("component").count().withColumnRenamed("count", "component_size")
n_components = component_sizes.count()
n_singletons = component_sizes.filter(F.col("component_size") == 1).count()

print(
    f"[{RUN_TAG}] {n_components:,} components over {n_vertices:,} nodes "
    f"({n_singletons:,} singletons, {100 * n_singletons / n_components:.1f}% of components)"
)
print(f"[{RUN_TAG}] top 10 component sizes:")
component_sizes.orderBy(F.desc("component_size")).limit(10).show()

# COMMAND ----------

# MAGIC %md ## Done -- next steps
# MAGIC 1. If `SAMPLE=True` just ran: confirm GraphFrames ran without error and the
# MAGIC    timing looks reasonable, then set `SAMPLE=False` and rerun this notebook for
# MAGIC    the real full-scale pass.
# MAGIC 2. This notebook's `edges` source (IDGraph2) predates the hybrid star/clique
# MAGIC    rewrite and `MAX_FANOUT=64` -- treat full-scale timing here as a rough upper
# MAGIC    bound on cost, not IDGraph3's actual expected cost.
# MAGIC 3. Once IDGraph3's build script has a real `RAW_PATH` and has actually run,
# MAGIC    prefer `IDGraph_CC_Databricks.py` for the production CC pass + full summary
# MAGIC    stats (type_summary, type_matrix, histograms, point-query helpers).
