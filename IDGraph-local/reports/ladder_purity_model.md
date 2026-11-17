# Component purity model (naive Bayes)

Status: built 2026-11-17 and run on the 15% sample (results: `IDGraph-local/results/idgraph5_component_quality_sample15_2026-11-09.md`, last section). Owner: Ravi.
Notebook: `python-databricks/IDGraph/IDGraph_Ladder_Purity_Databricks.py`.

## What it does

For every multi-node component of an ID graph, the model predicts **P(pure)**:
the probability that all the nodes in the component belong to one Affinity
person. We use it to compare graph versions V1 to V5 (see "Graph versions"),
including the many components Affinity can't check directly.

The model uses only graph information as input. Affinity is used to measure the
rates the model looks up (tune split) and to check the predictions (holdout
split). It is never used to build a graph.

## Terms

| Term | Meaning |
| --- | --- |
| Node | One ID in the graph, with a type: HEM, IDFA, AAID, TTD, ID5_UID or ST |
| Source | Where a link comes from: id5, liid or tapad |
| Link | Two nodes that share a key (a group of 2 to 16 nodes) in a source |
| Source set | Every source that links a given pair of nodes (e.g. `liid+tapad`), or every source that links a given node |
| Link class | A link's source set x the two node types it joins, e.g. `liid+tapad HEM-IDFA` |
| Node token | A node's source set x its type, e.g. `liid+tapad:HEM`, `tapad:TTD` |
| Component | A set of nodes connected by links; a singleton is a component of one node |
| Affinity person | A person in the Affinity crosswalk |
| Labelled node | A node that Affinity matches to exactly one Affinity person (only HEM, IDFA, AAID can be) |
| Judgeable component | A component with 2 or more labelled nodes, so we can tell whether it holds one Affinity person or several |
| Pure | A judgeable component whose labelled nodes all belong to one Affinity person |
| Tune / holdout | Affinity persons are split 70% / 30%; rates are measured on tune, predictions checked on holdout |

## Graph versions

Each version keeps the links of the one before, plus the next class of links:

| Version | Links kept |
| --- | --- |
| V1 | links seen in 2+ sources (any node types) |
| V2 | V1 + single-source classes with a pair-level same-Affinity-person rate of at least 60%: tapad HEM-HEM, liid HEM-IDFA, tapad IDFA-IDFA |
| V3 | V2 + liid IDFA-IDFA, liid AAID-HEM (50-60%) |
| V4 | V3 + the remaining rated single-source classes (below 50%) |
| V5 | every link: today's graph |

**Note: the ladder and this model use different link measures.** The ladder's
cut-offs were set with **pair-level** rates: the share of labelled pairs in a
link class that are the same Affinity person. Those exist only for links whose
two nodes can be labelled (HEM, IDFA, AAID), which is why every single-source
link touching TTD, ID5_UID or ST sits in V5. This model scores components with
**component-level** rates (below), which exist for every link class. We keep
the ladder as built for now; rebuilding it on the component-level measure is a
possible later step.

## Inputs: rates from notebook A (tune)

The model learns no rates of its own. Everything comes from notebook A, tune
split, judgeable components, per component size bucket (2, 3-4, 5-8, 9-16,
17-32, 33-64, 65-128, 129-1000).

| Rate | What it is | Notebook A table |
| --- | --- | --- |
| Size prior | share of judgeable components of that size that are pure | component summary |
| Link rate | share of judgeable components of that size containing a link of the class that are pure | pair table (`graph_pair_health`) |
| Node rate | share of judgeable components of that size containing the node token that are pure | token table (`token_health`) |

Link and node rates are looked up at the bucket matching the component's size.
If a cell has fewer than 50 judgeable components, the rate falls back to the
token's over-merge lift across all sizes x that size's over-merge rate.

Every link class and node token has a rate this way, including those touching
TTD, ID5_UID and ST: the rate comes from the labelled nodes around them, so the
nodes themselves don't need to be labelled.

**Assumption: all rates are measured on today's graph (V5).** Notebook A
measures them on the 15% sample of today's multi-node components, and the model
applies the same rates to V1 to V4. Link rates could instead be measured
directly on all labelled pairs, independent of any graph; node rates could be
measured on each version's own components. For now everything comes from
notebook A; the holdout check, reported per version, shows whether the rates
carry over.

## Which links count

Links are counted as the sources provided them, never expanded into every
possible pair of a group.

1. **Candidate links:** every pair of nodes that share a key in some source,
   labelled with all the sources that put them together. A pair inside both a
   tapad group and a liid group is one link of class `liid+tapad ...`, never
   two links.
2. **Pick n - 1 links** for a component of n nodes (the minimum that connects
   it), best-rated first. Links seen in 2+ sources usually rate highest, so they
   are picked first. When source groups don't overlap, this is k - 1 links for
   a group of k nodes.
3. **Each picked link adds its evidence once.**

Example: tapad provides the group {H1, I1, I2, A1} and liid provides the pair
{H1, I1}. Candidates: H1-I1 as `liid+tapad HEM-IDFA`, plus 5 tapad-only pairs.
Picked (4 nodes, 3 links): H1-I1 first, then the 2 best tapad links that join
I2 and A1. The second source appears once, in H1-I1's class.

## How a component is scored

Naive Bayes adds up evidence on the log-odds scale, where
logit(p) = ln(p / (1 - p)). Every rate is taken at the component's size.

1. **Start** at logit(size prior).
2. **Each picked link** adds logit(link rate) - logit(size prior).
3. **Each node** adds logit(node rate) - logit(size prior).
4. **Convert:** P(pure) = 1 / (1 + e^(-sum)). This is `p_raw`, the naive
   Bayes score.
5. **Recalibrate** (`p_pure`): the same naive Bayes score, passed through a
   small logistic calibration on (sum, log2 size). The rates are unchanged; the
   calibration only corrects over-confidence, because naive Bayes treats tokens
   as independent and big components pile up evidence. Then **check on
   holdout**: AUC and predicted vs actual share pure, per version.

Two diagnostic scores sit next to it:

| Score | Built from | Reads as |
| --- | --- | --- |
| `p_best` | start + the single strongest token (link or node) | optimistic ceiling: the best evidence the component has |
| `p_worst` | start + the weakest picked link | pessimistic floor: a component is only as good as the weakest link it needs |

One wrong link is enough to join two Affinity persons, so `p_worst` is the more
telling of the two.

## Worked examples

The rates below are illustrative, of the size notebook A reports. Real values
come from notebook A's tables.

### Example 1: a 4-node V2 component

Nodes: HEM h1, HEM h2, IDFA i1, IDFA i2. Size bucket 3-4, size prior 0.65.
Links: h1-i1 seen in liid and tapad; h1-h2 and i1-i2 seen in tapad. That is 3
links for 4 nodes, so all 3 are picked. Node source sets: h1 and i1 are linked
by liid and tapad, h2 and i2 by tapad only.

| Step | Rate at size 3-4 | Evidence |
| --- | --- | --- |
| Start: logit(0.65) | | +0.62 |
| link liid+tapad HEM-IDFA (h1-i1) | 0.72 | +0.33 |
| link tapad HEM-HEM (h1-h2) | 0.68 | +0.13 |
| link tapad IDFA-IDFA (i1-i2) | 0.66 | +0.04 |
| node liid+tapad:HEM (h1) | 0.63 | -0.09 |
| node liid+tapad:IDFA (i1) | 0.60 | -0.21 |
| node tapad:HEM (h2) | 0.68 | +0.13 |
| node tapad:IDFA (i2) | 0.64 | -0.04 |
| **Sum** | | **+0.91** |
| **P(pure)** = 1 / (1 + e^-0.91) | | **0.71** |
| p_best (start + strongest token, the two-source link +0.33) | | 0.72 |
| p_worst (start + weakest picked link, tapad IDFA-IDFA +0.04) | | 0.66 |

### Example 2: two sources beat one

A 2-node component, HEM and IDFA, size prior 0.70. The same pair, linked by
both sources or by tapad only:

| Step | Linked by liid and tapad | Linked by tapad only |
| --- | --- | --- |
| Start: logit(0.70) | +0.85 | +0.85 |
| Link | liid+tapad HEM-IDFA, rate 0.80: +0.54 | tapad HEM-IDFA, rate 0.66: -0.18 |
| HEM node | liid+tapad:HEM, rate 0.69: -0.05 | tapad:HEM, rate 0.71: +0.05 |
| IDFA node | liid+tapad:IDFA, rate 0.66: -0.18 | tapad:IDFA, rate 0.70: 0.00 |
| **Sum** | **+1.16** | **+0.71** |
| **P(pure)** | **0.76** | **0.67** |

The second source lifts the component from 0.67 to 0.76, through the link.

### Example 3: a link between unlabelled types, and why rates are size-adjusted

A 2-node component: HEM H1 and ST ST1, linked by tapad and liid. ST can't be
labelled, but the link class `liid+tapad HEM-ST` still has a component-level
rate: it comes from judgeable components containing such a link.

| Step | Rates at the component's size | Raw rates (all sizes mixed) |
| --- | --- | --- |
| Start | prior for 2-node components 0.70: +0.85 | prior over all sizes 0.46: -0.16 |
| link liid+tapad HEM-ST | rate at size 2 = 0.75: +0.25 | raw rate 0.237: -1.01 |
| node liid+tapad:HEM (H1) | rate at size 2 = 0.694: -0.03 | raw rate 0.246: -0.96 |
| node liid+tapad:ST (ST1) | rate at size 2 = 0.691: -0.04 | raw rate 0.227: -1.06 |
| **Sum** | **+1.03** | **-3.19** |
| **P(pure)** | **0.74** | **0.04** |

Raw rates mostly describe big components, which are rarely pure; used raw,
they would make a small 2-node component look 4% pure. Taking every rate at the
component's own size avoids counting size twice: tokens then only say whether
the component is purer or less pure than its size predicts.

## What is not scored

- **Singletons.** A one-node component can't hold two Affinity persons, so
  purity doesn't apply. Singletons are counted separately; whether they belong
  in another component is a different question (a planned reattach model).
- **Tokens with no rate** (a class or token notebook A never saw in a judgeable
  component) add no evidence.

## Output

Per version V1 to V5:

- the share of all multi-node components in each `p_pure` band (at least 0.9,
  0.7-0.9, 0.5-0.7, below 0.5), and the means of `p_pure`, `p_best`, `p_worst`
- holdout AUC of `p_pure`, `p_raw`, `p_best` and `p_worst` on holdout components
  with 2 labelled nodes, and the holdout calibration of `p_pure`

## Known limits

- Naive Bayes assumes tokens are independent. Tokens that usually appear
  together (e.g. `liid+tapad:HEM` nodes with `liid+tapad ...` links) double-count
  shared information; the size recalibration and the holdout check guard
  against over-confident scores.
- Component-level rates also reflect the other links a class tends to appear
  with.
- Node tokens are per source set (`liid+tapad:HEM`), so a node gives one token.
- All rates come from a 15% sample of today's multi-node components (V5).
- The ladder (V1 to V5) was built with pair-level rates, a different measure
  from the component-level rates used here.

## How it was checked

On toy data, every number was recomputed independently from notebook A's raw
tables and the ladder's pair lists: rates, the n - 1 picked links, nb_sum,
`p_best` and `p_worst` matched exactly.

## Where things live

- Rates: notebook A, `python-databricks/IDGraph/IDGraph_Component_Quality_Databricks.py`
  (component summary, pair table, token table).
- Graph versions: `python-databricks/IDGraph/IDGraph_Ladder_Databricks.py`.
- This model: `python-databricks/IDGraph/IDGraph_Ladder_Purity_Databricks.py`
  (writes under `IDGraph5_ladder/sample15/purity_v2/`).

## First results (15% sample, run 2026-11-17)

| Version | Multi-node components | Mean p_pure | Mean p_best / p_worst | Holdout AUC of p_pure |
| --- | --- | --- | --- | --- |
| V1 | 3.2M | 0.675 | 0.700 / 0.688 | 0.57 |
| V2 | 14.0M | 0.641 | 0.647 / 0.636 | 0.62 |
| V3 | 14.1M | 0.638 | 0.644 / 0.633 | 0.63 |
| V4 | 17.4M | 0.606 | 0.623 / 0.605 | 0.62 |
| V5 (today) | 46.5M | 0.611 | 0.628 / 0.607 | 0.65 |

- The ordering matches the measured purity (V1 > V2 ~ V3 > V4 ~ V5).
- Holdout AUC 0.64 overall: the model ranks components only modestly.
- No component scores above 0.70, the purity of 2-node components: scores stay
  close to the size prior, and links and nodes move them only a little.
- On holdout, predicted is about 0.15 below actual in every band: holdout holds
  30% of Affinity persons, so fewer over-merges are visible there. The ranking
  is unaffected.
