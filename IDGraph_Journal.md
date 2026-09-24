# ID Graph Project Journal

Identity-resolution / ID-graph pipeline for ShareThis/Predactiv ad-tech data.
Nodes are `(type, value)` identifiers across six types — `HARDWARE_IDFA`,
`HARDWARE_ANDROID_AD_ID`, `HEM_MD5`, `ID5_UID`, `TTD`, `ST` — and edges connect
two nodes when they co-occur under a shared grouping key: `individual_id5`,
`individual_id_tapad`, or `liid`. Goal: merge these into "identities" (connected
components) while excluding untrustworthy bulk/bot evidence.

## 1. Timeline: local → Databricks

Started on a local machine (32GB RAM, 12 CPUs) against a ~2.29B-row dataset,
using DuckDB for out-of-core SQL and scipy's `connected_components` (C-optimized
union-find) for the graph traversal. This is `connected_components.py` and
`IDGraph_ConnectedComponents.ipynb`.

A new raw dataset arrived at ~9B rows, with `node_types` alone at **4,914,094,515
distinct nodes**. That node count rules out the local approach on its own — a
Python `NODE_INDEX` dict mapping every node to an integer, plus a scipy sparse
adjacency matrix, would need several hundred GB of RAM regardless of how small
`edges` gets. This forced a port to PySpark/Databricks with GraphFrames for the
actual `connectedComponents()` call, while keeping small aggregate outputs
(summaries, histograms) designed to come back down locally for visualization.

## 2. Core concepts

- **`weight`** (raw co-occurrence count) vs. **`via_type_count`** (how many of
  the 3 grouping-key *types* corroborate a pair) — `via_type_count` is the more
  reliable confidence signal; most edges have `via_type_count=1` in practice.
- **`MAX_FANOUT`** — per grouping-key-*value*, per via_type. If a key ties
  together more nodes than this, its evidence is excluded entirely from pair
  generation. This is the real "trust this evidence or not" / individual-vs-family
  lever. Does *not* cap final component size — nodes can still merge transitively
  through other, smaller keys (see Finding: transitive closure risk, below).
- **`STAR_THRESHOLD`** — pure performance/storage lever. Groups above this size
  (up to `MAX_FANOUT`) get a **star** (every member linked only to a chosen hub,
  `k-1` edges) instead of a full **clique** (`k(k-1)/2` edges). Does not change
  which nodes end up in the same component — GraphFrames/scipy only need *a
  path*, not direct adjacency. Cost: star sacrifices direct pairwise
  `weight`/`via_type_count` between two non-hub members of a large group.
- **`DENYLIST`** — value-level exclusion for confirmed bad actors, independent
  of fanout (belt-and-suspenders alongside `MAX_FANOUT`).
- **Hub priority** — a star's hub is chosen `HEM_MD5 > other HEM_* > ST >
  everything else` (hardware/cookie IDs, which churn most), since `HEM_MD5` is
  the most durable anchor. A naive `MIN(node_id)` approach was found to actually
  bias toward the *least* stable type, since `"HARDWARE_..."` sorts
  alphabetically before `"HEM_MD5"`.

## 3. Key findings

1. **`ST` is not a US state code** — it's a fixed-schema, protobuf-serialized
   Predactiv token (confirmed via base64/hex decoding).
2. **Fraud/bot cliques**, found via `plot_component()` visualization: an 8-node
   complete clique (`weight` up to 81,225, `via_type_count=1`, tapad-driven),
   and a 70-node all-`HARDWARE_ANDROID_AD_ID` clique driven by one `tapad` value
   fanning out to 70 distinct hardware IDs. That specific value
   (`No8xjSC1%2BF8tmUR6x2oK%2BwN5kOn4r6EQMogME9KNx4Ef6i2iPbTIaEY8POYimlBf`)
   **reappeared in the new dataset too** (fanout=1,557 in the latest audit) —
   confirmed persistent bad actor, now in `DENYLIST`.
3. **Transitive closure risk** — `connected_components` merges via *any* path,
   which is powerful but means excluding a specific large-fanout key doesn't
   prevent its members from merging anyway through a chain of other,
   individually-small, individually-legitimate-looking keys.
4. **Giant component** (found on an IDGraph2 sample CC run, pre-hybrid-star,
   `MAX_FANOUT=300`): one component covered **330,669,904 of 1,028,484,669**
   sampled nodes (~32%). Type breakdown of that component: TTD 43.2%,
   `HARDWARE_ANDROID_AD_ID` 23.1%, `HARDWARE_IDFA` 20.7%, `ST` 8.5%, `HEM_MD5`
   2.8%, `ID5_UID` 1.8% — dominated by the churniest types, reassuringly light
   on the more durable identifiers. **This is the central open validation
   question for IDGraph3**: does `MAX_FANOUT=64` + hybrid star actually shrink
   this, or does it persist via chains of individually-small groups?
5. **`liid` is messier than `tapad`**, contrary to where most scrutiny went —
   `liid` excludes ~7.7% of its groups at `MAX_FANOUT=64` vs. `tapad`'s ~0.55%,
   has 3 group-size-histogram anomaly zones (n≈13, 38, 52-55, the last nearly
   *doubling* before resuming decay) vs. `id5`'s 2 and `tapad`'s 0, and is the
   only via_type showing measurable within-via_type pair collapse (3.78%, vs.
   0% for `id5`/`tapad`).

## 4. Threshold decisions

- **`MAX_FANOUT=64`**: `id5`'s organic ceiling is ~32 (with an unexplained
  spike exactly at n=32, and a smaller one at n=16 — never fully resolved,
  worth double-checking these aren't query/rollup artifacts); `tapad`/`liid`
  values above ~100 started looking bot-like (100+ distinct nodes sharing one
  key reads as shared-default/bot, not a family) once actually reasoned
  through, not just eyeballed from curve shape. 64 sits between those anchors.
- **`STAR_THRESHOLD=20`** (not lowered to 10): the 11-20 size band alone holds
  ~11.5M `id5` groups — more aggregate clique cost than the entire 21-32 band —
  and is also the size range most likely to be genuine household/churn
  evidence worth preserving pairwise fidelity for.
- A single **global** `MAX_FANOUT` is a known compromise — `id5`, `tapad`, and
  `liid` have demonstrably different distribution shapes, so one number can't
  be exactly right for all three. Not yet split into per-via_type values.

## 5. Bugs found & fixed (Databricks pipeline)

1. **8TB `edges` blowup** — root cause: `O(k²)` full-clique cost summed across
   many *medium*-sized groups (not just extreme outliers) exploded total
   volume even with `MAX_FANOUT` bounding any single group. Fixed via the
   hybrid star/clique split.
2. **Pair canonicalization** — once hub selection stopped being purely
   alphabetical, `node_a` was no longer guaranteed `<` `node_b`, risking the
   same physical pair fragmenting into two `edges` rows (splitting its
   `weight`/`via_type_count`). Fixed via `F.least`/`F.greatest` in `pairs_fn`.
3. **Weight-inflation dedup bug** — `base` is an outer join across
   id5/tapad/liid; a node with one fixed key value but several different
   *other*-key values legitimately produced multiple `base` rows identical
   from one via_type's perspective, and the self-join multiplied every pair by
   however many such duplicate rows existed. Fixed via `filtered_base_fn`
   deduping to `(node_id, type, key_col)` per via_type. `fanout_fn` was always
   immune (uses `countDistinct`), which is why the group-size histograms used
   to set `MAX_FANOUT`/`STAR_THRESHOLD` remained trustworthy throughout.
4. **Unstaged diagnostic counts** — `pairs_via`'s denylist/excluded-count/
   group-count `.count()` calls ran as real Spark actions on *every* call,
   even when the downstream staged output already existed. Fixed via an
   `already_built` guard.
5. **Redundant union writes** — `all_pairs` and the denylist-candidates union
   were separately re-staging data that was already separately staged
   elsewhere. Fixed by replacing with lazy multi-path reads (metadata-only
   union, no new write).
6. **File fragmentation** — no stage ever called `.repartition()`; `edges`
   landed in 63,047 files for 1.7TB (vs. `via_keys`' 15,003 files for a third
   the size) — disproportionate even accounting for size, and it directly
   hurts compression and read performance. Fixed via explicit `repartition`
   targets (`edges_fn=8500`, `via_keys_fn=2600`, ~200MB/file). **Only affects
   future writes** — the existing 1.7TB `edges`/522.5GB `via_keys` predate
   this fix and would need deletion + rebuild to benefit.

## 6. Measured IDGraph3 dataset characteristics

(Full detail in the `idgraph3-dataset-characteristics` memory entry.)
Measured under `MAX_FANOUT=64`, `STAR_THRESHOLD=20`, current hub priority, and
the row-dedup fix.

| table | size | rows |
|---|---|---|
| `node_types` | 183.5GB | 4,914,094,515 |
| `via_keys` | 522.5GB | ~6.27B (histogram-derived estimate) |
| `all_pairs` (id5+tapad+liid pairs) | 454.9GB | 17,321,706,793 |
| `edges` | 1.7TB | ≤17,120,085,672 (upper bound; true count not yet measured, likely ~16-17B) |

Node type split: `ID5_UID` 32.1%, `TTD` 26.0%, `HARDWARE_ANDROID_AD_ID` 15.4%,
`HARDWARE_IDFA` 14.8%, `HEM_MD5` 9.2%, `ST` 2.4%.

Edge counts by via_type (`pair_rows` vs. `distinct_edges`, i.e. within-via_type
collapse): `tapad` 9,253,864,915 = 9,253,864,915 (0%); `id5` 2,736,742,603 =
2,736,742,603 (0%); `liid` 5,331,099,275 → 5,129,478,154 (3.78% collapse — the
only via_type showing any).

## 7. What each file owns

| File | Owns |
|---|---|
| `IDGraph_EDA.ipynb` | Earliest exploratory pass (pandas) over the raw joined ID data, before DuckDB was adopted. |
| `connected_components.py` | Standalone local (no Spark) production script for the *old* (~2.29B row) dataset: DuckDB builds the weighted edge list out-of-core; scipy runs `connected_components`. Superseded once node count made local processing infeasible. |
| `IDGraph_ConnectedComponents.ipynb` | The canonical, most-iterated local analysis notebook. DuckDB build cells (one connection per via_type — combining all three in one UNION-ALL query with shared CTEs caused a DuckDB planner hang), `run_cc()`, the per-type summary table, point-query helpers (`view_component`/`largest_components`/`component_of_node`/`component_of_node_pair`/`component_of_edge`), `plot_component()` visualization, type×type matrix, weight/via_type_count histograms, the "why we pair" writeup, and the 5 numbered findings. |
| `staging_df.py` | **Not authored here** — the user's own pre-existing utility providing the `@stage_dataframe` decorator (skip-if-output-exists resumability). Pasted inline at the top of both Databricks scripts so each is self-contained on a fresh cluster. |
| `IDGraph_Build_Databricks.py` | Production PySpark build script for IDGraph3. Owns: raw read (optional file-based `SAMPLE`), `base`/`node_types` (with `REUSE_BASE`/`REUSE_NODE_TYPES` to skip recompute from IDGraph2), fan-out audit + group-size histograms, the hybrid star/clique `pairs_via()` orchestration per via_type, hub selection, `hub_diversity_fn`/`denylist_candidates_fn` (review tooling for finding more bad actors), and `all_pairs`/`via_keys`/`edges` assembly. |
| `IDGraph_CC_Databricks.py` | Production GraphFrames CC + summary-stats script for IDGraph3. Owns: `connectedComponents()`, `result` (Delta, Z-ORDER by `component`), per-type summary/type-matrix/weight histograms mirroring the local notebook at Spark scale, the component-size diagnostics built specifically to validate the giant-component question, node/edge counts by type/via_type, and point-query helpers for the ~4B-row `result` table. |
| `IDGraph2_CC_Databricks.py` | Standalone, disposable GraphFrames smoke-test/cost-calibration script against IDGraph2's *already-computed* (pre-hybrid, `MAX_FANOUT=300`) output. Answered "does GraphFrames even work on this cluster" using existing data rather than waiting on the IDGraph3 rebuild — this is the run that first surfaced the giant-component finding. |

## 8. Open items / next steps

- **Currently blocked**: `IDGraph_CC_Databricks.py`'s `connectedComponents()`
  call is failing with `MetadataFetchFailedException` — an executor became
  completely unreachable mid-shuffle (connection refused, not just slow),
  most likely an OOM kill given the scale, possibly compounded by skew from
  high-degree hub nodes or a persisting giant component. Next diagnostic:
  check the Databricks cluster event log for that executor's removal reason,
  and check the node degree distribution for extreme outliers.
- Get the exact `edges.count()` for IDGraph3 (currently only bounded).
- **The central open question**: does `MAX_FANOUT=64` + hybrid star actually
  curb the giant component found on IDGraph2, once CC completes on IDGraph3?
- `liid`'s real tail beyond n=101 and its 3 histogram anomalies are
  uninvestigated — worth inspecting real groups near those sizes.
- `tapad`'s real tail beyond n=101 is also still unseen (`.show(100)`-truncated).
- Consider rebuilding `via_keys`/`edges` now to get the better file layout
  from the repartition fix (currently only helps future writes).
- Consider per-via_type `MAX_FANOUT` given the three via_types' clearly
  different distribution shapes.
