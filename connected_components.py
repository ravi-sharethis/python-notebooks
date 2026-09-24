"""
Standalone (no Spark) connected-components over the ID graph.

Nodes  = (type, value) identifiers, e.g. "TTD:abc123"
Edges  = two nodes are linked if they co-occur on rows sharing the same
         liid / individual_id5 / individual_id_tapad. Each edge carries a
         weight = number of distinct grouping keys (id5/tapad/liid values)
         that produced the link, i.e. how many independent times the two
         identifiers were seen tied together -- a higher weight is stronger
         evidence of a real link vs. a single coincidental co-occurrence.

DuckDB does the heavy out-of-core work directly over the parquet files;
scipy's sparse connected_components (C-optimized union-find) does the graph
traversal. No JVM/Spark required.

Perf note: an earlier version used window functions (MIN() OVER PARTITION)
which emit one output row per *input* row -- i.e. it materialized ~3x the
full base table before ever discarding the useless self-loop rows, and blew
through available disk for temp spill. This version filters down to only
the grouping keys that actually link >1 distinct node *before* ever forming
a pair, which cuts the working set by ~85% (most id5/tapad/liid values only
ever have a single associated node in the filtered type set).
"""

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = "--sample" in sys.argv
PARQUET_GLOB = (
    "./id5_tapad_li_joined_outer/part-0000[0-9]-*.parquet"  # 10/200 files (~5%)
    if SAMPLE
    else "./id5_tapad_li_joined_outer/*.parquet"
)

FILTER_TYPES = [
    "HARDWARE_IDFA",
    "HARDWARE_ANDROID_AD_ID",
    "HEM_MD5",
    "ID5_UID",
    "TTD",
    "ST",
]

OUT_PATH = "./id5_component_map_sample.parquet" if SAMPLE else "./id5_component_map.parquet"

con = duckdb.connect()
con.execute("SET preserve_insertion_order=false")
# Fail fast instead of silently filling the disk (bit us once already).
con.execute("SET max_temp_directory_size='100GiB'")
type_list = ", ".join(f"'{t}'" for t in FILTER_TYPES)

BASE_SQL = f"""
    SELECT type || ':' || value AS node_id, type,
           individual_id5, individual_id_tapad, liid
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE type IN ({type_list}) AND value IS NOT NULL
"""


def pairs_via(key_col: str, via_label: str) -> str:
    # Only keep grouping keys that actually tie together >1 distinct node --
    # this is the filter that keeps the self-join cheap (most keys are
    # singletons and would otherwise just produce useless self-loops).
    return f"""
    {via_label}_multi AS (
        SELECT {key_col}
        FROM base
        WHERE {key_col} IS NOT NULL
        GROUP BY {key_col}
        HAVING COUNT(DISTINCT node_id) > 1
    ),
    {via_label}_base AS (
        SELECT b.node_id, b.{key_col}
        FROM base b
        JOIN {via_label}_multi USING ({key_col})
    ),
    {via_label}_pairs AS (
        SELECT a.node_id AS node_a, b.node_id AS node_b,
               '{via_label}' AS via_type, a.{key_col} AS via_key
        FROM {via_label}_base a
        JOIN {via_label}_base b
          ON a.{key_col} = b.{key_col} AND a.node_id < b.node_id
    )
    """


FULL_SQL = f"""
WITH base AS ({BASE_SQL}),
{pairs_via("individual_id5", "id5")},
{pairs_via("individual_id_tapad", "tapad")},
{pairs_via("liid", "liid")},
all_pairs AS (
    SELECT * FROM id5_pairs
    UNION ALL
    SELECT * FROM tapad_pairs
    UNION ALL
    SELECT * FROM liid_pairs
)
SELECT node_a, node_b,
       COUNT(*) AS weight,
       COUNT(DISTINCT via_type) AS via_type_count
FROM all_pairs
GROUP BY node_a, node_b
"""

NODE_TYPES_SQL = f"""
WITH base AS ({BASE_SQL})
SELECT DISTINCT node_id, type FROM base
"""

print(f"[{'SAMPLE' if SAMPLE else 'FULL'}] scanning {PARQUET_GLOB} ...")

print("Computing node counts by type...")
node_types = con.execute(NODE_TYPES_SQL).df()
print(node_types["type"].value_counts().to_string())

print("Building weighted edge list...")
edges_df = con.execute(FULL_SQL).df()
print(f"{len(edges_df):,} distinct (node_a, node_b) edges")
print("weight distribution:")
print(edges_df["weight"].value_counts().sort_index().head(10).to_string())

# Full vertex set includes isolated nodes (no edges at all) so per-type
# "how many are connected" metrics are accurate, not just edge endpoints.
all_nodes = node_types["node_id"].to_numpy()
node_to_idx = {n: i for i, n in enumerate(all_nodes)}
n = len(all_nodes)

row = edges_df["node_a"].map(node_to_idx).to_numpy()
col = edges_df["node_b"].map(node_to_idx).to_numpy()
data = np.ones(len(row), dtype=np.int8)
graph = coo_matrix((data, (row, col)), shape=(n, n))

print("Running connected_components...")
n_components, labels = connected_components(graph, directed=False)
print(f"{n_components:,} components over {n:,} nodes")

result = node_types.copy()
result["component"] = labels
comp_sizes = result.groupby("component").size().rename("component_size")
result = result.join(comp_sizes, on="component")

print("\nPer-type connectivity (singleton = never linked to anything):")
summary = (
    result.assign(is_singleton=result["component_size"] == 1)
    .groupby("type")
    .agg(
        total_nodes=("node_id", "count"),
        singleton_nodes=("is_singleton", "sum"),
    )
)
summary["connected_nodes"] = summary["total_nodes"] - summary["singleton_nodes"]
summary["pct_connected"] = (100 * summary["connected_nodes"] / summary["total_nodes"]).round(2)
print(summary.to_string())

print("\nLargest components:")
print(comp_sizes.sort_values(ascending=False).head(10).to_string())

# Type x type edge breakdown: for each type, which counterpart types does it
# actually link to, and how often -- e.g. does ST mostly link to other ST
# nodes, or to TTD/hardware IDs?
type_map = node_types.set_index("node_id")["type"]
edges_df["type_a"] = edges_df["node_a"].map(type_map)
edges_df["type_b"] = edges_df["node_b"].map(type_map)
fwd = edges_df[["type_a", "type_b"]].rename(columns={"type_a": "from_type", "type_b": "to_type"})
bwd = edges_df[["type_b", "type_a"]].rename(columns={"type_b": "from_type", "type_a": "to_type"})
directed = pd.concat([fwd, bwd], ignore_index=True)

print("\nEdge partner breakdown by type (% of that type's edges going to each counterpart type):")
for t in FILTER_TYPES:
    sub = directed[directed["from_type"] == t]
    if sub.empty:
        continue
    counts = sub["to_type"].value_counts()
    pct = (100 * counts / counts.sum()).round(1)
    print(f"--- edges FROM {t} (n={len(sub):,}) ---")
    print(pd.DataFrame({"edge_count": counts, "pct": pct}).to_string())

result.to_parquet(OUT_PATH)
edges_df.to_parquet(OUT_PATH.replace("component_map", "edges"))
print(f"\nWrote {OUT_PATH} and edges parquet")
