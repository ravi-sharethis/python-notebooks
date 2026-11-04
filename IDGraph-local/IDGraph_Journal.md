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
6. **First measurement, at `MAX_FANOUT=32`/`STAR_THRESHOLD=10`** (see §4 for
   why that's tighter than the originally-settled 64/20, which never
   completed CC). One component had **2,326,645,218 nodes — 47.3% of all
   4,914,094,515 nodes**, larger in relative terms than the ~32% found on the
   IDGraph2 sample. Its composition had also shifted: `ID5_UID`+`HEM_MD5`
   (the durable/trustworthy types) went from ~4.6% combined (old sample) to
   ~39.6% combined — no longer mostly churn/bot noise. At the time this
   looked like evidence that threshold tightening doesn't work — **see
   Finding 7, which overturns that conclusion.**
7. **Corrected: tightening DOES curb the giant component, non-linearly.** A
   second CC run at `MAX_FANOUT=16`/`STAR_THRESHOLD=6` shrank the giant
   component to **697,176,027 nodes — 14.19%** of all nodes (down from
   47.3% at 32/10). Only two real data points exist (32/10, 16/6), but the
   jump suggests a **percolation-like threshold effect** — moderate
   tightening (300ish→32) barely helped, aggressive tightening (32→16)
   caused real fragmentation. **The cost**: singleton rate nearly doubled,
   25.01%→**49.21%** — tightening strands far more nodes with zero links at
   all, a real tradeoff, not a free win. The 16/6 giant component's type mix
   also shifted: `TTD` rose to 35.85% (from 28.94%) while hardware types fell
   (`HARDWARE_ANDROID_AD_ID` 14.0%→10.48%, `HARDWARE_IDFA` 14.17%→8.35%) —
   hardware-driven bridging was cut much more effectively than TTD-driven
   bridging, suggesting `TTD` needs its own targeted fix regardless of where
   the threshold ends up. One specific pathology at 32/10 — near-total `TTD`
   dominance (79-81%) in the 257-1024 component-size range, breaking an
   otherwise smooth trend — **disappeared completely** at 16/6, concrete
   evidence `MAX_FANOUT` does targeted, not just uniform, work. See §8 for
   next steps; don't re-litigate "does tightening help" — it's resolved.

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
- **Update**: the 64/20 config above was too big for `connectedComponents()`
  to actually complete on the full dataset (16.95B edges — almost certainly
  the cause of the `MetadataFetchFailedException` shuffle failure). The
  pipeline was rerun at **`MAX_FANOUT=32`/`STAR_THRESHOLD=10`** (6.65B
  edges), which is what actually produced the CC results in §3 Finding 6 and
  §6b. This was a tractability concession, not a change in the reasoning
  above — the STAR_THRESHOLD=20 rationale (preserving fidelity for the 11-20
  band) still applies in principle, just couldn't be afforded in practice on
  the available cluster.

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
7. **`is_singleton` never persisted, surfaced by the notebook split** —
   `IDGraph_CC_Databricks.py` added `is_singleton` via `.withColumn()`
   *after* `result_with_size_fn` had already written `result` to Delta and
   read it back — a purely in-memory, lazy addition never written to disk.
   Invisible in the original single-file version (one continuous in-memory
   `result` object); surfaced once `IDGraph_CC_Analysis_Databricks.py` read
   `result` fresh from `RESULT_PATH` and `type_totals_fn` referenced a
   column that didn't exist there. Fixed by computing singleton directly
   from `component_size == 1` (a column that *is* persisted); the dead
   `.withColumn()` line was removed from the compute notebook.

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

**Note**: the table above is all under `MAX_FANOUT=64`/`STAR_THRESHOLD=20`.
That config was never actually run through CC successfully (see §4 update) —
`edges` at that config is 16,951,041,699 rows, 1.7TB. The config that
actually completed CC is `MAX_FANOUT=32`/`STAR_THRESHOLD=10`, with
**6,647,886,786 edges** over the same 4,914,094,515 nodes. `node_types` is
unaffected by MAX_FANOUT either way (always 4,914,094,515 / 183.5GB).

## 6b. IDGraph3 CC results (MAX_FANOUT=32/STAR_THRESHOLD=10)

Full results in the `idgraph3-current-status` memory entry. Headline: giant
component confirmed and worse than the IDGraph2 sample suggested — see §3
Finding 6. Supporting detail:

- **Singletons**: 1,228,969,090 nodes (25.0% of all nodes) are true
  singletons (never linked to anything) — the first real, unbiased singleton
  measurement (earlier sample-based attempts were structurally incapable of
  seeing singletons at all).
- **Component-size distribution**: smooth decay from n=2 (95,007,081) through
  the hundreds, then nothing until the single giant outlier at
  2,326,645,218 — same bimodal "blob + thin tail" shape as the IDGraph2
  sample, confirmed at full scale.
- **Per-type connectivity** (`pct_conn` = % of that type's nodes that are
  non-singleton): `ID5_UID` highest at 89.3%, `HEM_MD5` 83.7%, `ST` 69.7%,
  `TTD` 67.6%, `HARDWARE_ANDROID_AD_ID` 66.1%, `HARDWARE_IDFA` lowest at
  61.7%. `avg_deg_conn` (average degree among non-singleton nodes) ranges
  2.18 (`HARDWARE_IDFA`) to 7.24 (`HEM_MD5`) — modest across the board,
  which is what makes the giant component's existence a small-world/
  cross-type-linking phenomenon rather than a "everything has huge degree"
  one.
- **`via_type_count`**: 99.31% of edges are single-source (`via_type_count=1`),
  0.70% dual-source, 0.002% triple-source — confirms the earlier prediction
  that cross-via_type overlap is modest.
- **`weight`**: 98.37% of edges have `weight=1` (single co-occurrence
  instance) — most evidence is single-instance, not repeated.

## 7. What each file owns

**Location (2026-10-30)**: repo reorganized by project. ID graph local
side in `IDGraph-local/`: this journal, local notebooks (`IDGraph_EDA`,
`IDGraph_ConnectedComponents` + `connected_components.py`), saved CSV outputs
in `IDGraph-local/results/`, HTML reports in `IDGraph-local/reports/`.
Databricks notebooks in `python-databricks/IDGraph/`, shared `staging_df.py`
in `python-databricks/`. TTD pricing work in `TTD/`, other notebooks in
`misc/`. Older entries below that say a file is "in python-notebooks/" or
"in this directory" refer to these locations now.

| File | Owns |
|---|---|
| `IDGraph_EDA.ipynb` | Earliest exploratory pass (pandas) over the raw joined ID data, before DuckDB was adopted. |
| `connected_components.py` | Standalone local (no Spark) production script for the *old* (~2.29B row) dataset: DuckDB builds the weighted edge list out-of-core; scipy runs `connected_components`. Superseded once node count made local processing infeasible. |
| `IDGraph_ConnectedComponents.ipynb` | The canonical, most-iterated local analysis notebook. DuckDB build cells (one connection per via_type — combining all three in one UNION-ALL query with shared CTEs caused a DuckDB planner hang), `run_cc()`, the per-type summary table, point-query helpers (`view_component`/`largest_components`/`component_of_node`/`component_of_node_pair`/`component_of_edge`), `plot_component()` visualization, type×type matrix, weight/via_type_count histograms, the "why we pair" writeup, and the 5 numbered findings. |
| `staging_df.py` | **Not authored here** — the user's own pre-existing utility providing the `@stage_dataframe` decorator (skip-if-output-exists resumability). Lives at `python-databricks/staging_df.py`; since 2026-10-30 every Databricks notebook imports it (`sys.path.append(os.path.abspath(".."))` then `from staging_df import _path_exists, stage_dataframe`) instead of pasting it inline. `_path_exists` now gets `dbutils` via `pyspark.dbutils.DBUtils`, since a notebook's global `dbutils` isn't visible inside an imported module. |
| `IDGraph_Build_Databricks.py` | Production PySpark build script for IDGraph3. Owns: raw read (optional file-based `SAMPLE`), `base`/`node_types` (with `REUSE_BASE`/`REUSE_NODE_TYPES` to skip recompute from IDGraph2), fan-out audit + group-size histograms, the hybrid star/clique `pairs_via()` orchestration per via_type, hub selection, `hub_diversity_fn`/`denylist_candidates_fn` (review tooling for finding more bad actors), and `all_pairs`/`via_keys`/`edges` assembly. |
| `IDGraph_CC_Databricks.py` | Production GraphFrames CC script for IDGraph3 -- split (see chat) into compute-only + analysis-only notebooks so the expensive, failure-prone `connectedComponents()` step doesn't need rerunning every time a new diagnostic is added. Owns ONLY: loading `node_types`/`edges`, building the `GraphFrame`, running `connectedComponents()`, and writing `result` (Delta, Z-ORDER by `component`). Needs GraphFrames (`graphframes-py==0.12.1`) installed; nothing else does. |
| `IDGraph_CC_Analysis_Databricks.py` | All downstream analysis, reading `node_types`/`edges`/`via_keys`/the per-via_type `pairs` tables/`result` -- none of it computed by this notebook. Does NOT need GraphFrames installed (nothing here touches a `GraphFrame` object), so it can be iterated on freely without the CC notebook's dependency or cost. Owns: per-type summary/type-matrix/weight histograms mirroring the local notebook at Spark scale, the component-size diagnostics built specifically to validate the giant-component question (including `type_mix_by_size_bucket_fn`, which tracks how the type composition shifts across the same geometric size buckets used for the component-size histogram chart), node/edge counts by type/via_type, node distribution by (type, via_type) for singleton vs. plural nodes (with a built-in cross-check: CC-derived singleton counts vs. independently via_keys-derived ones, since "zero via_keys rows" and "CC singleton" are the same population by construction), and point-query helpers for the ~4B-row `result` table. |
| `IDGraph2_CC_Databricks.py` | Standalone, disposable GraphFrames smoke-test/cost-calibration script against IDGraph2's *already-computed* (pre-hybrid, `MAX_FANOUT=300`) output. Answered "does GraphFrames even work on this cluster" using existing data rather than waiting on the IDGraph3 rebuild — this is the run that first surfaced the giant-component finding. |

## 8. Open items / next steps

- **Resolved**: the `MetadataFetchFailedException` shuffle failure, via two
  changes landed in `IDGraph_CC_Databricks.py`: (1) switched GraphFrames
  install to `%pip install --force-reinstall graphframes-py==0.12.1`,
  confirmed working on Databricks Runtime 17.3 LTS/Spark 4.0.0/Scala 2.13 —
  resolves the GraphFrames/Spark-4.0 compatibility risk flagged early in the
  project and never verified until now; (2) `connectedComponents()` now
  called with `broadcastThreshold=-1`, disabling GraphFrames' internal
  broadcast-join optimization (default threshold 1M rows) — a plausible OOM
  source under skew, independent of edge count. Combined with dialing the
  config back to 32/10 (6.65B edges, down from 64/20's 16.95B), the rerun
  completed. **Not yet disambiguated** whether `broadcastThreshold` or the
  smaller edge count is what actually fixed it — worth testing `broadcastThreshold=-1`
  against the original 64/20 config if that config is ever revisited. A
  `spark.catalog.clearCache()`/`spark.conf.set(io.cache.enabled, false)`
  attempt was also tried and found unnecessary — left commented in the
  script as a tried-and-rejected marker.
- **Resolved, corrected**: does tightening `MAX_FANOUT`/`STAR_THRESHOLD` curb
  the giant component? **Yes, non-linearly** — 47.3% (32/10) → 14.19% (16/6).
  See §3 Finding 7. Don't re-litigate the earlier "no" conclusion (Finding 6)
  — it was based on only one data point and is superseded.
- **New decision needed**: only two points on the threshold/giant-component
  curve exist (32/10, 16/6), and 16/6's improvement came at the cost of
  nearly doubling the singleton rate (25.01%→49.21%). Decide: (a) test a
  point between or below these to map the curve properly, (b) whether 16/6's
  tradeoff is actually better than 32/10's given the real goal (avoiding
  over-merging vs. under-merging), (c) run the still-not-executed
  via_type_count>=2-filtered CC test and per-via_type CC test to check
  whether `TTD` specifically needs a targeted fix — its share of the giant
  component *rose* under tightening (28.94%→35.85%) while hardware types'
  fell, suggesting it's the more stubborn contributor.
- **New**: `size2_type_pair_breakdown_fn` (added to
  `IDGraph_CC_Analysis_Databricks.py`) has not been run yet — would settle
  the "~85% of size-2 components are pure ID5_UID-ID5_UID pairs" question
  directly instead of via the mathematical lower-bound inference currently
  backing it.
- **New**: the `id5` n=16 anomaly is now confirmed a THIRD independent time
  (raw histogram, 32/10 type-mix, 16/6 type-mix) — recurs at exactly size 16
  regardless of whether 16 is our own `MAX_FANOUT` boundary, pointing at an
  artifact intrinsic to `id5`'s upstream data generation. Worth an actual
  dedicated investigation now rather than continuing to just note it.
- `liid`'s real tail beyond n=101 and its 3 histogram anomalies are still
  uninvestigated — worth inspecting real groups near those sizes.
- `tapad`'s real tail beyond n=101 is also still unseen (`.show(100)`-truncated).
- Consider rebuilding `via_keys`/`edges` now to get the better file layout
  from the repartition fix (currently only helps future writes) — though
  note the currently-relevant `edges` is now built at whichever config is
  live (32/10 or 16/6), not the 64/20 one (16.95B rows) the repartition
  targets (8500/2600) were originally sized against; may need re-sizing.
- Consider per-via_type `MAX_FANOUT` given the three via_types' clearly
  different distribution shapes, and given `TTD` now looks like the
  specific lever worth isolating (see above).
- **Resolved (2026-10-20): recovered the full 16/6 component-size histogram**
  that a prior session (`f235666f-...`) pasted in full but only ever
  summarized into memory ("573 distinct ordinary sizes, zero gap") rather
  than preserving the raw per-size counts — a real gap, since that session
  ended (crashed) shortly after. Recovered by reading that session's own
  transcript file directly. Both configs' full per-size histograms (573
  ordinary sizes at 16/6, 276 at 32/10 — each independently verified to sum
  to the known exact node/component totals with zero gap) are now saved as
  `idgraph3_component_size_histogram_{16_6,32_10}.csv` alongside this
  journal, and used directly (not bounded/estimated) in the
  `idgraph3-threshold-report` artifact's "by number of components"
  distribution.
- **New finding, from the same recovered data: extreme single-via_type
  components exist at the very top of the "ordinary" (non-giant) size range,
  in both configs** — a nuance the bucket-averaged `257-512`/`513-1024`
  TTD-dominance pathology (§3 Finding 7) doesn't capture on its own. At
  32/10, the single largest non-giant component (size 617) is **100% TTD**
  — every one of its 617 nodes is a TTD cookie, with zero cross-type
  bridging. At 16/6, the single largest non-giant component (size 865) is
  **77.7% TTD** (672/865 nodes). Both sit in `513-1024`, the exact bucket
  where 32/10's bucket-*average* TTD share was reported as resolved/smooth
  at 16/6 — the average looking normal doesn't mean individual outlier
  components aren't still extreme. Full per-type breakdown of each config's
  top 5 largest components (giant + next 4) saved to
  `idgraph3_top5_components_by_type.csv`. Worth deciding whether this
  changes the read on "is the TTD pathology actually resolved."

## 9. The `idgraph3-threshold-report` artifact — structure & reproducibility

Live at https://claude.ai/code/artifact/c534d31d-b335-49f3-b465-9178336b9916
(private Claude artifact). The canonical, editable HTML source — the page
content only, without the viewer's injected `<head>`/frame-runtime wrapper —
is saved alongside this journal as `idgraph3_threshold_report.html`. To
update it in a future session: edit that file, then publish it to the same
artifact URL (Claude's `Artifact` tool, `action: "publish"`, passing `url`).
To pull a fresh copy of what's live instead (e.g. if it was edited from the
artifact's own share UI), use `action: "read"` on the URL, then strip the
single leading line (injected preamble/style/`<body>`) and the trailing
`</body></html>` — the real content is everything in between.

**Units convention**: every figure in the report is a NODE count unless
explicitly tagged COMPONENTS (a small pill next to the section heading, e.g.
`<span class="unit-tag">NODES</span>` / `<span class="unit-tag
components">COMPONENTS</span>`) — this was added after a reader mistook a
"giant component" value for a *count* of giant components (there's exactly
one, per config) rather than its node count. A mixed-unit table (the
per-type summary) gets a text legend instead of a single tag.

**Section-by-section structure, in order, with data source**:
1. Methodology — static prose + a units-convention callout box.
2. Headline numbers (`.stat-strip`, 4-wide grid) — total nodes; giant
   component % + exact node counts per config; singleton rate; total
   components + singleton/non-singleton split per config (singletons are
   ~83%/89% of all components — see §8); a full-width `HEM_MD5` breakdown
   card (singleton / ordinary / giant node counts + %, both configs — added
   because HEM_MD5 uniquely *flips* which group holds the majority of its
   nodes: giant-dominated 59.4% at 32/10, singleton-dominated 46.8% at 16/6,
   unlike every other type which just shifts gradually).
3. Key findings — static prose bullets, sourced from §3/§6b of this journal.
4. Nodes by type table — from the per-type summary paste (journal §3/§6b).
5. Component-size distribution, by nodes — `bucket_totals_{32_10,16_6}`
   arrays in the page's embedded `chartData`, SVG bar chart drawn in JS.
6. Component-size distribution, by components — `comp_lo/hi/exact_*`
   arrays in `chartData`; now fully exact for both configs, sourced from
   `idgraph3_component_size_histogram_{32_10,16_6}.csv` (see §8 — these are
   the real per-size histograms, not reconstructed).
7. Type mix by bucket (%), raw node counts by bucket, CDFs — from the
   `ws_*`/`wos_*`/type-mix-table pastes in the crashed session transcript
   (32/10 reconstructed within 0.01% from histogram + %, 16/6 exact).
8. Per-type summary, singleton/connected composition — from the
   `conn_identities`/`pct_conn`/etc. paste, same transcript.
9. Caveats — enumerates every approximation/reconstruction still in the
   report; keep this in sync whenever a "reconstructed" figure gets
   replaced with a real one (as happened for the component-count chart).

**CDF chart color/marker scheme**: the original 6 type colors had two
greens (`ID5_UID` `#4C8C6B`, `HEM_MD5` `#3E7C8C`) sitting next to each other
and were replaced using the `dataviz` skill's validated categorical palette
and `validate_palette.js`. A hard limit surfaced during this: that palette's
docs state only its first ~3 slots clear the strict `--pairs all` CVD gate
(all lines mutually visible, as in this 6-series CDF) — no selection of 6 of
its 8 hues passes cleanly in both light and dark. Final choice (best
available, light passes cleanly, dark needed a custom re-stepped violet to
clear the chroma floor): `ID5_UID`=blue, `TTD`=red, `AAID`=aqua,
`HARDWARE_IDFA`=yellow, `HEM_MD5`=violet, `ST`=green — defined as
`--series-*` CSS custom properties (light/dark/`[data-theme]`, same pattern
as the rest of the page's tokens) and referenced via `var()` directly in SVG
`stroke`/`fill` attributes. Residual CVD risk is compensated with a second,
color-independent channel: each type also gets a distinct marker shape
(circle/square/triangle/diamond/cross/star), drawn at 8 fixed bucket indices
per line and mirrored in the legend icons — this is the generally-applicable
fix when a chart needs more simultaneous categorical series than a validated
palette can cleanly separate on color alone.

**If rebuilding from scratch**: the three CSVs in this directory
(`idgraph3_component_size_histogram_32_10.csv`,
`idgraph3_component_size_histogram_16_6.csv`,
`idgraph3_top5_components_by_type.csv`) plus the per-type summary numbers in
§6b of this journal are sufficient to regenerate every number in the report
without needing the original Databricks pastes again.

## 10. Affinity crosswalk matching (32/10 + 16/6, DCR refresh 20261008)

Code lives in `IDGraph_CC_Analysis_Databricks.py`, "Affinity crosswalk
matching" section. One deliverable table, `affinity_enhanced_components`
(`{CC_OUTPUT_PATH}/affinity_enhanced_components`): exact replica of `result`
(all 6 types, same row count) plus `afs_individual_ids` (`array<string>` --
a node can map to more than one AFS id, gathered via `collect_set` rather
than silently fanning out the join) and `has_afs_id` (bool). Only
`HEM_MD5`/`HARDWARE_ANDROID_AD_ID`/`HARDWARE_IDFA` are joined against the
crosswalk (`all_id_to_afs_id`) -- it has no entries for `ID5_UID`/`TTD`/`ST`,
so those always come out with an empty array, never joined.

**Node-level match rates, first real run**:

| type | n_nodes | n_with_afs_id | pct |
|---|---:|---:|---:|
| HEM_MD5 | 453,949,154 | 89,884,356 | 19.80% |
| HARDWARE_IDFA | 727,086,362 | 48,713,684 | 6.70% |
| HARDWARE_ANDROID_AD_ID | 758,637,201 | 46,698,298 | 6.16% |
| TTD / ID5_UID / ST | (their totals) | 0 | 0% |

**Component-level targetability by size bucket**: three tables,
`component_targetability_{any,hem,maid}` (`any` = targetable via any
eligible type; `hem` = via `HEM_MD5`; `maid` = via `HARDWARE_ANDROID_AD_ID`
or `HARDWARE_IDFA`), saved as `idgraph3_component_targetability_{any,hem,
maid}.csv` in this directory (each sums to exactly 2,728,350,341 --
the known 16/6 total component count, cross-checked). Headline: **targetability
rises steadily with component size** -- 3.99% of singletons (`any`) vs.
>90% by the 65-128 bucket, up to 100% for the giant component. Internal
consistency holds exactly where it must: at size 1, `any = hem + maid`
precisely (a singleton can't be targetable via both at once); at every
larger bucket `any <= hem + maid`, with the gap being components targetable
via both simultaneously.

**The size-16 `id5` anomaly shows up a 4th time, in this new metric**:
`pct_targetable` (all three of any/hem/maid) breaks its otherwise smooth
rise exactly at size 16 (e.g. `any`: 36.42% at 15 -> **34.68%** at 16 ->
40.43% at 17), then recovers. This can't be a `MAX_FANOUT=16`-boundary
artifact specifically -- the same anomaly already showed up at size 16 in
the 32/10 run (`MAX_FANOUT=32`), so its location at 16 doesn't depend on
what `MAX_FANOUT` actually is. **However, the obvious mechanism (more
`ID5_UID` at size 16 crowding out the 3 eligible types) does not cleanly
explain the shape**: the eligible-type share (AAID+IDFA+HEM_MD5) declines
*monotonically* across 15->16->17 (39.84% -> 36.10% -> 33.90% per the
existing type-mix table), while targetability dips at 16 and *recovers* at
17 -- a monotonic composition shift can't produce a dip-then-recovery. Open:
whether size-16 components specifically have a higher rate of zero eligible-
type nodes at all (a distributional effect, not just an average-share
effect) would need a direct query against `component_affinity_flags`
to confirm -- not yet run.

**Also noted, smaller sample**: `pct_targetable` dips slightly at 513-1024
relative to 257-512/129-256 in all three tables (`any`: 99.22% -> 96.94%,
only ~3 of 98 components) -- plausibly connected to the known extreme-TTD
outlier components in that exact bucket (`idgraph3_top5_components_by_type.csv`
-- the size-865/617 components that are 77.7%/100% TTD, a type with zero
affinity coverage), but the sample is too small (98 components) to
distinguish from noise.

**Done**: the before/after component-size comparison is built --
https://claude.ai/code/artifact/2d73e95c-7e05-4e7e-b764-d3b87bd462f0
("How much of the graph is actually targetable?"), canonical source saved as
`idgraph3_affinity_targetability_report.html` in this directory. Headline:
only **4.80%** of all 2,728,350,341 components are targetable at all (`any`
definition); targetability rises from 3.99% at singletons to 100% for the
giant, but singletons are 88.6% of all components so they dominate the
absolute volume -- 2,321,821,874 of the 2,597,331,361 untargetable
components (89.4%) are singletons. Raw per-bucket numbers for all three
definitions (any/hem/maid) are in
`idgraph3_component_targetability_{any,hem,maid}.csv`. `hem` vs `maid`
trade the lead twice across the size range (maid leads at singletons, hem
leads at sizes 2-11, maid leads again from 12 through 257-512, hem edges
back ahead at the small-sample 513-1024 bucket) -- see the artifact for the
full per-bucket table.

**32/10 run added, 2026-10-25 -- the artifact above now compares both
configs.** Raw tables saved as
`idgraph3_component_targetability_{any,hem,maid}_32_10.csv` (each verified
to sum to exactly 1,478,773,504, the known 32/10 total component count).
Headline: **16/6 is more targetable than 32/10 at every single size
bucket**, not just in aggregate (4.80% vs 4.04% overall `any`; singletons
3.99% vs 2.96%) -- tightening the threshold config doesn't just reshape the
graph (the Threshold Comparison Report's subject), it directly improves how
much of it is reachable for affinity targeting.

**The 513-1024 TTD-outlier connection is now confirmed, not just
plausible.** At 32/10, the 513-1024 bucket is **0.00% targetable** under
all three definitions (0 of 6 components) and 257-512 craters to 13.33%
(`any`) -- vs. 96.94%/99.22% at 16/6. This is exactly the size range (and
exactly the config) where `idgraph3_top5_components_by_type.csv` already
identified specific near-giant components that are 100% TTD (size 617) or
TTD+ID5_UID only (size 782, no eligible-type nodes at all) -- structurally
incapable of matching the crosswalk regardless of match rate, since
`TTD`/`ID5_UID` have zero affinity coverage. It's also the same 257-512/
513-1024 range §3 Finding 7 already flagged as a TTD-dominance pathology at
32/10 that's "completely gone" at 16/6 -- three independent analyses
(type-mix pathology, top-5-components type breakdown, and now targetability)
now point at the same mechanism from three different angles.

**The size-16 `id5` anomaly holds the same shape under 32/10 too**
(`any`: 32.30% at 15 -> 29.06% at 16 -> 35.98% at 17), reinforcing it's a
property of the underlying `id5` data, not specific to either threshold
config -- consistent with everywhere else this anomaly has shown up.

## 11. MAX_FANOUT singleton sweep (2026-10-27)

`IDGraph_Fanout_Sweep_Databricks.py` predicts the singleton count for every
MAX_FANOUT at once, with no CC run: a node is non-singleton at fanout F iff
its smallest grouping-key group with `n_distinct >= 2` (across id5/tapad/liid)
is <= F. STAR_THRESHOLD is irrelevant (star and clique both connect every
member). Inputs: `{via}_filtered_base` + `{via}_fanout` from
`IDGraph5/build_output` (identical to IDGraph4's), `node_types` from
`IDGraph3/build_output`. First run hit Spark's "shuffle map stage with
indeterminate output" abort even at `shuffle.partitions=10000`; fixed by
staging each via_type's join separately before the final groupBy.

**Validated exactly**: predicted F=32 -> 1,228,969,090 and F=16 ->
2,418,307,483 singletons, matching both CC runs to the node. Raw output saved
as `idgraph_fanout_sweep_by_type.csv` (20 F values, per-type %) and
`idgraph_fanout_sweep_full.csv` (every F 2..1024 + uncapped).

**Findings:**
- **Singletons are almost entirely a MAX_FANOUT artifact.** Only 70,987,514
  nodes (1.44%) are unlinkable at any fanout. Of the singletons, 94% at 32
  and 97% at 16 are fanout-stranded.
- **Natural group sizes are large.** 94% of nodes are singletons even at
  F=2, and the median node's smallest group is ~16 (singleton rate crosses
  50% between F=15 and 16). Half of all nodes belong only to groups of 16+.
- **No knee below ~64.** The curve falls smoothly (25.0% at 32, 7.6% at 64,
  3.3% at 96, 2.1% at 128, 1.5% at 256); there's no singleton-based natural
  cutoff, so F has to be chosen against the giant-component side.
- **Per type**: ID5_UID links most easily (85.9% singleton at F=2 vs
  95-99% for others) but has the highest baseline (3.50%, ~55M of the 71M).
  HARDWARE_IDFA sits in the biggest groups (99.4% at F=2, 38.3% at 32).
  ST's baseline is ~0.01%.
- **Known histogram anomalies reappear as spikes in newly-linked nodes per
  F** (increment vs. mean of neighbors): F=16 (+4.9%, the id5 n=16 anomaly,
  5th independent appearance), F=32 (+6.6%, id5's natural ceiling), F=38
  (+2.9%, liid zone), and F=53 (+20%, followed by a sustained higher level
  through ~60 -- a step, i.e. a distinct population of groups starting at
  size 53; matches the liid 52-55 zone).

**Crosswalk HEM size (2026-10-29)**, `all_id_to_afs_id` refresh 20261008,
`Type=hem, Subtype=md5`: 1,734,351,089 rows, **211,954,481 distinct HEMs**,
88,228,090 distinct AFS individuals. So the overlap is 89,884,356 HEMs =
19.80% of ours (453.9M) and **42.4% of Affinity's**. ~8.2 rows per HEM
(heavy row duplication -- always countDistinct on this table); ~2.40 HEMs
per AFS individual on average.

**Crosswalk schema** (`all_id_to_afs_id`): `INDID, AFS_INDID, AFS_HHID,
AFS_INDIVIDUAL_ID, ID, Vendor, Type, Subtype`. Sample row: `INDID == ID`
(the HEM itself), `Vendor = liveintent`. `Vendor` is the provenance column
needed to test liid self-agreement; `AFS_HHID` is a household id, which
allows separating "same household, different person" from unrelated merges.
Meaning of `AFS_INDID` vs `AFS_INDIVIDUAL_ID` (we use the latter) not yet
confirmed.

## 12. Learning edge quality from Affinity labels (plan + Steps 0-1, 2026-10-30)

**Ground rule (user decision)**: the Affinity crosswalk is out-of-sample data.
It may be used to learn settings and to evaluate graphs, **never** to add
edges, merge nodes, or otherwise build the graph.

**Goal**: learn which of our links are trustworthy -- by source (via_type),
type pair, group size, and chaining -- turn that into build settings, and
prove a new graph is better on held-out labels.

### What is being tested
Every grouping key (one id5/tapad/liid value) claims all its members are one
person. For every pair of Affinity-matched nodes (HEM_MD5 / IDFA / AAID)
sharing a key, check whether Affinity agrees:

| Affinity says | label |
|---|---|
| same AFS individual (their `AFS_INDIVIDUAL_ID` sets overlap) | `agree` |
| different, both nodes' AFS individuals have >= 2 crosswalk IDs ("multi") | `strong_disagree` |
| different, at least one side's AFS individual has 1 crosswalk ID ("single") | `weak_disagree` |

Rationale for weak: a single-ID AFS individual may be one Affinity never
linked to anything, so its "different person" may mean "don't know". This is
an assumption -- tested by reporting `agree_rate_strong` (agree / (agree +
strong)) next to `agree_rate_all` (agree / all labeled).

TTD/ID5_UID/ST are never labeled directly (no crosswalk coverage). Key-level
quality learned on labeled members is assumed to carry over to a key's
unlabeled members; Step 2 tests TTD/ID5 as bridges directly.

### Data: nothing was lost to the star layout
`{via}_filtered_base` + `{via}_fanout` hold full key membership for every
key, including groups above MAX_FANOUT -- the star only changed which pairs
became edges. `edges` is useless here (via_type collapsed into a count).
`{via}_pairs`/`via_keys` are limited by star + cap. So all pairs are rebuilt
from filtered_base, independent of MAX_FANOUT/STAR_THRESHOLD: one run serves
every config.

### Steps
- **Step 0 (built)** -- node labels (`afs_ids`, `afs_class` single/multi,
  `split`); split by `xxhash64(SPLIT_SALT, AFS_INDIVIDUAL_ID) % 100 <
  TUNE_PCT` (70/30), so one person never lands in both; a node with several
  AFS ids takes its smallest one's split; pairs only within one split.
  Coverage check: AFS individuals by number of OUR nodes (only those with
  >= 2 produce `agree` pairs). Sanity: matched HEM 89,884,356 / IDFA
  48,713,684 / AAID 46,698,298.
- **Step 1 (built)** -- labeled pairs per via_type within keys of size
  2..1024 (`MAX_PAIR_GROUP_SIZE`); collapsed to one row per (via_type, pair)
  at its smallest group size; corroboration = which via_types link the pair
  (`via_set`, e.g. `liid+tapad`). Summaries **on the tune split only**: by
  via_type; via_type x type pair; via_type x type pair x group size; via_set
  x type pair.
- **Step 2 (not built)** -- chained pairs: matched pairs sharing no key but
  joined through one middle node (e.g. HEM -liid- TTD -tapad- IDFA), by
  source of each step, middle node type, group sizes; moderate sizes only.
  This is the giant-component mechanism and the direct TTD-bridge test.
- **Step 3 (not built)** -- score any built graph on the **holdout** split:
  recall (% same-person pairs in one component) and false-merge rate (% strong
  different-person pairs in one component), alongside singletons (fanout
  sweep) and targetability. Replaces the component-validation section briefly
  added to the analysis notebook on 2026-10-29 (removed).

### What the results feed (tuning levers)
1. Per-via_type caps instead of one MAX_FANOUT (largest size where agreement
   stays above a target; singleton cost from §11 sweep).
2. Type-pair rules (e.g. HEM-IDFA from tapad trusted to larger sizes than
   HEM-HEM); also star hub choice.
3. Edge weights = estimated agreement for (via_type, type pair, size,
   corroboration); then threshold edges before CC, or weighted community
   detection (Leiden) inside the giant.
4. Chain limits from Step 2 (e.g. no merges bridged through TTD) -- the
   long-pending TTD-specific fix.
5. Every candidate graph checked with Step 3 on holdout before adoption.

### Deferred
- Household (`AFS_HHID`): 3-way agree / same household / different household.
- liid self-agreement: Affinity's HEM links may come from LiveIntent
  (crosswalk `Vendor` column, e.g. `liveintent`), inflating liid. Check by
  re-scoring liid against non-liveintent crosswalk rows only. Main analysis
  doesn't need `Vendor`: our via_type comes from filtered_base.
- `AFS_INDID` vs `AFS_INDIVIDUAL_ID` meaning unconfirmed; we use the latter.

### Notebook: `IDGraph_Edge_Labels_Databricks.py`
Reads `{GRAPH_ROOT}/build_output/{via}_filtered_base|_fanout` (GRAPH_ROOT =
IDGraph5 by default; IDGraph4 identical), `IDGraph3/build_output/node_types`,
crosswalk refresh 20261008. Writes everything under
`{GRAPH_ROOT}/edge_labels/`: `cw_nodes`, `node_labels`, `node_label_summary`,
`afs_graph_coverage`, `labeled_members_{via}`, `labeled_pairs_{via}`,
`pairs_by_via`, `pair_corroboration`, `agreement_by_via_type_pair_size`,
`agreement_by_corroboration`. Every join is inside `@stage_dataframe`, each
via_type staged separately (lesson from the §11 indeterminate-stage abort),
so a crash resumes from the last finished stage. Standard Spark config block
at the top (shuffle.partitions 20000, AQE + coalesce + skew join, Delta
optimizeWrite). Verified end-to-end on toy data with local Spark
(2026-10-30); not yet run on Databricks.
**Never change `SPLIT_SALT`/`TUNE_PCT` after first results are seen.**

**Crosswalk working copy (2026-10-30)**: both notebooks that read the
crosswalk (`IDGraph_CC_Analysis_Databricks.py`, `IDGraph_Edge_Labels_Databricks.py`)
now read the user's copy `s3://data-science-research/ravi/afs_crosswalk_20261008`.
The prod/original path
(`s3://data-science-research/affinity_matching/DCR/refresh/{AFFINITY_RDATE}/crosswalk/all_id_to_afs_id`)
is kept commented directly below in each, to switch back for prod. Results
recorded above (§10 match rates, targetability) were produced from the
original path.

## 13. Edge-label results, Step 1, tune split (2026-10-30)

First Databricks run of `IDGraph_Edge_Labels_Databricks.py` (IDGraph5
build_output; group size = smallest shared key's n_distinct). Raw tables:
`results/idgraph_edge_labels_by_via_type_pair_size_tune.csv`,
`results/idgraph_edge_labels_corroboration_tune.csv` (per-via and
per-type-pair totals are sums of the size table; they match the printout).
"agree %" below = agree / (agree + strong_disagree). It is a lower bound on
precision (Affinity misses links too); compare cells relative to each other.

**No id5 rows at all.** id5 produced no labeled pairs. Not yet explained:
check `labeled_members_id5` / `labeled_pairs_id5` counts.

| via | pairs | agree % |
|---|---|---|
| liid | 893M | 2.2 |
| tapad | 84M | 42.1 |

**liid is good at small sizes and noise at large ones.** 97% of liid pairs
come from keys of size 49-128, where agreement is <1-7%. At size <=16:
HEM-IDFA ~62%, AAID-HEM ~53%, AAID-AAID 37-57%, IDFA-IDFA 33-38%,
AAID-IDFA ~23%. Steep drop from 17 onwards (HEM-IDFA 52 -> 39 -> 22 -> 7%
across 17-24 / 25-32 / 33-48 / 49-64). liid never produces HEM-HEM pairs
(one HEM per liid).

**tapad decays slowly, cliff at 129.** HEM-HEM strongest (77% at size 2-4,
still 42% at 65-128). IDFA-IDFA ~40-44% flat from 13 to 128. AAID-IDFA
(cross-OS) weakest, ~21% at every size. All type pairs collapse at 129+.

Cumulative share by cap (pairs kept / agrees kept / agree % of kept):

| cap | liid | tapad |
|---|---|---|
| 16 | 0.4% / 6% / 40.6 | 32% / 39% / 52.4 |
| 32 | 2.9% / 30% / 24.0 | 69% / 75% / 46.4 |
| 64 | 58% / 82% / 3.2 | 95% / 97% / 43.1 |
| 128 | 97% / 99% / 2.3 | ~100% / 100% / 42.1 |

**Corroboration is the strongest signal.** Pairs linked by both liid and
tapad: 60-75% agree across all type pairs (AAID-AAID 72.8, IDFA-IDFA 74.9,
HEM-IDFA 70.3), vs liid-only 1-12% and tapad-only 18-61%. Volume is small
(~2.9M pairs).

**Answers to the earlier questions:**
- HEM-IDFA vs HEM-HEM: depends on via. In tapad HEM-HEM (61%) beats
  HEM-IDFA (31%). In liid, HEM-IDFA is the best pair; there is no HEM-HEM.
- HEM(liid) vs HEM(tapad), same size: liid HEM-IDFA at <=16 is ~62% vs
  tapad 34-54%; liid AAID-HEM ~53% vs tapad 37-56%. Comparable or better
  for liid at small sizes; much worse at large sizes.
- Caveat: liid agreement may be inflated by LiveIntent-sourced crosswalk
  rows (Vendor check still deferred).

**Implications (candidate levers, to verify with Step 3 holdout):**
1. Per-via caps instead of one MAX_FANOUT: liid ~16 (maybe 24), tapad
   ~64-128. IDGraph5's global 16 is right for liid but throws away ~60% of
   tapad's agreeing pairs; raising tapad's cap should cut singletons at little
   precision cost (§11: median node's smallest group ~16).
2. Down-weight or drop tapad AAID-IDFA (cross-OS, ~21%: household/shared?).
3. Corroborated (liid+tapad) pairs deserve the highest weight.

## 14. Component quality vs Affinity: design for notebooks A + B (agreed 2026-11-04)

Simpler, component-level follow-up to §12-13 (pairs explain *why*; this
scores *outcomes*). Not built yet. Every join staged; standard config block;
Affinity used only to label/evaluate, never to split or merge a specific
component (§ Affinity = labels only).

### Definitions (agreed)
- **Class** per component (singletons included): join its nodes to the
  crosswalk, count distinct `AFS_INDIVIDUAL_ID`s. **0** = none matched,
  **1** = exactly one, **2** = more than one. Unmatched nodes (TTD, cookies)
  don't count against a match. A node mapping to 2 AFS ids just yields 2 -> class 2.
- **Over-merged** (component flag) = class 2.
- **Incomplete** (component flag) = some AFS person in this component also has
  IDs in another component.
- **Judgeable** = >= 2 matched IDs (class 2 was possible).
- **% over-merged** = over-merged / (class 1 + class 2) components; also an
  ID-weighted version.
- **% people split** = AFS people whose matched IDs fall in 2+ of our
  components / AFS people with 2+ IDs in our graph. Headline under-merge
  number; the incomplete flag carries it into per-component breakdowns.
- Percentages are flags counted over a group of components (whole graph or a
  breakdown cell); per component there are only flags.

### Notebook A -- foundation (`IDGraph_Component_Quality_Databricks.py`)
Inputs: GRAPH_ROOT (IDGraph5 default, any build) + its MAX_FANOUT; build
`via_keys`, `{via}_filtered_base`, `{via}_fanout`, `cc_output/result`;
`node_labels` from edge labels (graph-independent).
`SCORE_SPLIT = "tune"`: only AFS people in that split count as matches (others
treated as unmatched); `"holdout"` writes a separate output dir for final scoring.

Per component: class, over_merged, incomplete, judgeable, size, n matched IDs,
n AFS people, sources (via_types whose links built it), ID types (HEM only /
MAID only / HEM+MAID / neither; MAID = IDFA or AAID), largest group size per
source.

Per node token, stored as counts per component: **token = source set x ID
type** (e.g. `liid+tapad:HEM`, `tapad:TTD`), source set = sources whose links
include that node *in this component*. One token per node (no double counting
across sources), all ID types.

Singletons: source set from raw source data (`{via}_filtered_base`) + why
alone per source: alone in group / dropped by cap (+ dropped group size) /
denylisted. Every node comes from some source, so singletons always have one.

Per AFS person (with 2+ IDs in our graph): together vs split.

Tables: breakdowns by size, sources, ID types, sources x size (class shares,
% over-merged, % incomplete); singletons by source x reason; headline
% over-merged (component- and ID-weighted) and % people split.

### Notebook B -- models (`IDGraph_Component_Quality_Models_Databricks.py`)
Reads A's outputs so several models are compared on one foundation.
Two binary targets:
- over-merged: trains on judgeable components only when
  `OVERMERGE_TRAIN_JUDGEABLE_ONLY = True` (default; a component with 1 matched
  ID can never be class 2 and would teach "less Affinity data = good").
- incomplete: components with >= 1 matched ID (mainly singletons).

Models:
1. Lookup baseline: cell rate by sources x ID types x size bucket, thin cells
   fall back to coarser.
2. Bayesian naive Bayes: token rate =
   `(own over + PRIOR_STRENGTH * type_rate) / (own total + PRIOR_STRENGTH)`,
   type_rate = pooled empirical rate of the token's ID type (dominated by the
   biggest source automatically); unknown type -> global rate.
   `PRIOR_STRENGTH = 100`; optional beta-binomial fit. A token never seen
   (e.g. `id5+liid+tapad:AAID` = 0) sits at its type rate until data arrives.
   "most reliable source" mode dropped -- the prior covers it.
   Anti-saturation (a 500-node component would otherwise go to 0%/100%):
   token shares + size as a separate input, then calibration.
3. Logistic regression: log(size), source shares, ID-type shares, largest group
   size per source, singleton reason.
Evaluation: train tune, test holdout (A's holdout output): AUC, calibration
by 10% band, lift over the lookup baseline. Outputs: P(over-merged) and
P(incomplete) for every component, token rate table, model comparison.
TTD's effect IS learnable (TTD nodes sit in judged components) but scores for
TTD-only components are extrapolation.

### Use
- Compare graphs: run A on IDGraph4 / IDGraph5 / per-source-cap build; compare
  headline numbers.
- Break up clusters: what predicts high P(over-merged) -> stricter rules.
- Re-link singletons: high P(incomplete) + "dropped by cap" reasons -> which
  groups to re-add (e.g. tapad 17-64).
- Rules use only our own data; confirmed on holdout.
