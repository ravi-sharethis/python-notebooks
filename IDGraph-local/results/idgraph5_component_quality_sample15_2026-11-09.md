# IDGraph5 (16/6) component quality: sample run, 2026-11-09

Run: 3-task job (A tune, A holdout, B). The header prints **15% of components**
(409,270,401 of 2,728,350,341 = 15.0%), so SAMPLE_PCT was 15, not 5.
The full raw output is in the chat transcript. Only the key tables are kept here.

## A: headline (tune)
- components 409,270,401; % over-merged (components) 47.11 (judgeable 95.97)
- % over-merged (IDs) 96.4; % incomplete 91.32; % people split 94.13

### by_size (tune)
| size | n | class1 | class2 | %over-merged | %incomplete |
|---|---|---|---|---|---|
| 1 | 362,757,488 | 5,647,716 | 4,485,898 | 44.27 | 92.63 |
| 2 | 16,262,615 | 91,730 | 59,990 | 39.54 | 82.97 |
| 3-4 | 12,397,270 | 265,923 | 208,180 | 43.91 | 84.27 |
| 5-8 | 9,373,258 | 512,105 | 512,227 | 50.01 | 86.69 |
| 9-16 | 6,345,121 | 624,056 | 786,198 | 55.75 | 89.09 |
| 17-32 | 1,623,775 | 253,022 | 363,523 | 58.96 | 89.37 |
| 33-64 | 419,928 | 78,613 | 181,135 | 69.73 | 91.79 |
| 65-128 | 78,909 | 10,566 | 57,040 | 84.37 | 95.36 |
| 129-1000 | 12,036 | 494 | 11,207 | 95.78 | 98.68 |
| giant | 1 (697,176,027 IDs) | 0 | 1 | 100 | 100 |

### by_id_types (tune)
| id_types | n | %class0 | %over-merged | %incomplete |
|---|---|---|---|---|
| HEM only | 33,180,782 | 86.41 | 24.59 | 82.72 |
| HEM+MAID | 8,028,934 | 57.08 | 55.85 | 88.11 |
| MAID only | 156,454,325 | 96.04 | 58.64 | 99.38 |
| neither | 211,606,360 | 100 | - | - |

## B: holdout AUC (LR)
| stratum | over_merged n / base / AUC | incomplete n / base / AUC |
|---|---|---|
| all | 3,032,135 / 0.949 / 0.882 | 6,349,113 / 0.894 / 0.742 |
| 1 | 1,920,790 / 1.000 / - | 4,342,519 / 0.915 / 0.792 |
| 2-4 | 132,021 / 0.849 / 0.741 | 284,421 / 0.811 / 0.628 |
| 5-16 | 655,720 / 0.850 / 0.633 | 1,195,099 / 0.848 / 0.579 |
| 17-128 | 315,081 / 0.889 / 0.584 | 517,016 / 0.867 / 0.560 |
| 129+ | 8,523 / 0.972 / 0.467 | 10,058 / 0.943 / 0.566 |

Calibration: lookup and LR are within about 3 points in every band. nb_sum is badly
calibrated below 0.8 (it predicts 0.48-0.66 where the actual rate is 0.93-1.0).

## Findings
1. **Single-node multi-person matches dominate class 2.** 4.49M of the 10.13M matched
   singletons are class 2: one node maps to 2+ AFS persons. That is 58.8% of MAID
   singletons and 24.0% of HEM singletons. This is Affinity multiplicity (shared
   devices, households), not graph merging. It inflates over_merged everywhere,
   including multi-node components that contain such a node.
2. **The giant component holds 91% of the IDs in matched components** (697.2M of
   766.2M). Excluding it, % over-merged (IDs) is 60.1% (41.5M / 69.1M), not 96.4%.
   NB token counts are node-weighted, so the giant drives every token rate to about 0.99.
3. **B's overall over_merged AUC (0.88) is inflated.** 63% of holdout rows are
   judgeable singletons, and all of them are positive. On multi-node components,
   LR AUC is 0.74 / 0.63 / 0.58 / 0.47 by stratum, and lift beyond size is weak.
4. **Multi-node components only:** class1 1.84M, class2 2.18M, so 54.3% over-merged.
   We can't split this into graph error and finding 1 until the labels are fixed.
5. **ID5_UID, ST and TTD never get labels on their own** (id5-only, ST-only and
   TTD-only components are 100% class 0), as expected.
6. **Most singletons are cap casualties.** source_group shows most singletons are
   nodes whose groups were dropped at sizes 17-128 (tapad, liid). Dropped HEM
   singletons are about 84% incomplete: the person has nodes elsewhere.

---
# Rerun 2026-11-10: single-person-node labels (commit 2092656), 15% sample

## A: tune, by_size
| size | class1 | class2 | %over-merged | %over-merged (judgeable) | %incomplete | %multi-AFS nodes |
|---|---|---|---|---|---|---|
| 1 | 5,647,716 | 0 | 0 | - | 83.4 | 44.3 |
| 2 | 97,234 | 3,116 | 3.1 | 30.0 | 70.2 | 35.0 |
| 3-4 | 301,022 | 21,528 | 6.7 | 35.2 | 71.2 | 34.6 |
| 5-8 | 624,795 | 79,206 | 11.3 | 43.5 | 73.4 | 35.7 |
| 9-16 | 816,652 | 145,093 | 15.1 | 52.6 | 76.1 | 38.2 |
| 17-32 | 349,575 | 77,874 | 18.2 | 58.4 | 76.1 | 39.1 |
| 33-64 | 133,073 | 60,329 | 31.2 | 73.5 | 78.8 | 40.2 |
| 65-128 | 26,246 | 30,119 | 53.4 | 87.9 | 84.1 | 41.2 |
| 129-1000 | 2,310 | 8,603 | 78.8 | 95.8 | 92.2 | 42.1 |
| giant | 0 | 1 | 100 | 100 | 100 | 41.5 |

- Totals: 5.1% of matched components over-merged (15.3% of multi-node ones,
  54.0% of judgeable ones). IDs over-merged excluding the giant: 23.1%
  (11.16M / 48.25M).
- id_types: HEM only 0.56%, MAID only 0.12%, HEM+MAID 16.7%. Over-merging happens
  where HEM and MAID are joined.
- Same size, different sources (9-16): liid 6.5%, id5+liid+tapad 11.5%,
  id5+tapad 12.3%, liid+tapad 14.2%, tapad 16.9%. At size 2: tapad 3.1%, liid 0%.
- About 35-44% of matched nodes map to 2+ AFS people at every size. That is a
  property of the Affinity crosswalk.

## B: holdout
- over_merged (judgeable, all multi-node): n 276,387, base 0.345. LR AUC 0.670
  overall; by stratum 0.516 (2-4), 0.581 (5-16), 0.666 (17-128), 0.648 (129+).
  All models over-predict by 0.1-0.2. This is the expected tune (70% of people)
  vs holdout (30%) gap in visible over-merges: tune judgeable rate 54% vs holdout 34.5%.
- incomplete: LR AUC 0.662 overall, 0.569 on 2+, 0.691 on singletons. Lookup and
  LR are well calibrated; nb_sum is not.
- Conclusion: little lift beyond size. The descriptive tables are the useful output.

## Rerun: source x group size, has-type, holdout (2026-11-10)
CSVs: `idgraph5_cq15_tune_source_group_size.csv`, `idgraph5_cq15_tune_has_type_size.csv`.

**Split-size correction checks out.** Tune and holdout over-merge rates differ only
because a split holds 70% vs 30% of people. If two nodes belong to different people,
both are visible with probability s^2; if they are the same person, with probability s.
So observed odds = true odds x s. Correcting:
- size-2 judgeable: tune 30.0% -> 38.0%, holdout 15.8% -> 38.4%
- all judgeable: tune 54.0% -> 62.7%, holdout 34.5% -> 63.7%

So about 38% of 2-node components whose two nodes each match one person hold two
different AFS people. That is likely household-level linking (devices and emails
shared in a home), not random error.

**Largest-group size vs over-merge, at the same component size (judgeable base):**
- tapad: rises with group size. 5-8 components: 30.9 (group 3), 34.3 (4),
  40.9 (5-6), 46.4 (7-8). 9-16 components: 42.9 (5-6), 44.9 (7-8), 51.3 (9-12),
  57.2 (13-16).
- liid: flat. 9-16 components: 49.9 / 50.1 / 50.3 / 48.5 / 51.6 / 53.6 / 52.1.
- id5: flat, slightly falling. 9-16 components: 50.3 -> 44.8.
- So a tighter cap is justified for tapad only (around 6-8). liid and id5 at 16
  show no per-group penalty.

**Presence of TTD / ID5_UID / ST at the same size (judgeable base):** no consistent
effect. Slightly lower below 17 nodes, slightly higher at 33-64, where the
comparison group is tiny. These types matter by growing components (they bridge
into the giant), not by making a given component worse.
- HEM-free multi-node components are mostly MAID-only; few are judgeable.

**Holdout headline:** 2.56% over-merged (judgeable 34.49%); IDs excluding the giant
13.09%; people split 90.87% (6.22M of 6.85M); multi-AFS nodes 41.6%.

**B LR coefficients (over_merged):** log2_size +0.69 and share_hem +0.67 are the
largest; source shares are negative relative to the baseline. Consistent with
"size plus HEM-MAID joins". The models add little; the A tables carry the signal.

**Correction (2026-11-10, while writing the report):** the tapad group-size effect is
weaker than stated above. It is clear only in 5-8-node components (30.9 -> 46.4).
At 9-16 nodes it's U-shaped (52.4 at group 3, 42.9 at 5-6, 57.2 at 13-16). From 17
nodes up, no source shows a group-size effect. So a tapad cap of about 8 is worth
testing, but expect a modest gain; component size is the strong signal. Also, on the
judgeable base the sources are similar at a fixed size (about 48-54% at 9-16 nodes).
The earlier "liid cleanest" reading was detection bias.
Full report: `IDGraph-local/reports/idgraph5_component_quality_report.html`.

## B: LR coefficients and NB token rates, tune (pasted 2026-11-12)
over_merged LR (intercept -2.179): log2_size +0.686, share_hem +0.669,
has_group_tapad +0.311, share_maid +0.223, has_hem_and_maid +0.146,
log2_max_group_liid +0.114, has_group_id5 +0.109, log2_max_group_tapad -0.100,
share_tapad -0.376, share_other -0.675, share_id5 -0.863, share_liid -0.954.

incomplete LR (intercept 1.616): share_maid +0.935, is_singleton +0.183,
log2_max_group_liid +0.108, log2_size +0.072, alone_tapad -0.475,
has_hem_and_maid -0.453, share_tapad -0.435, share_id5 -0.530, share_hem -0.848,
share_liid -1.224.

over_merged NB token rates (shrunk): tapad:* 0.54-0.60; id5:ID5_UID 0.64;
liid:* 0.71-0.78; liid+tapad:HEM 0.75, liid+tapad:IDFA 0.82, liid+tapad:AAID 0.83.
incomplete: dropped MAID singletons 0.93-0.96 (tapad, liid, both); dropped HEM
singletons 0.66-0.76; singleton[tapad:alone]:IDFA 0.94, :HEM 0.61;
dropped_size buckets 17-24 .. 129-256: 0.80-0.86 (flat); multi-node tokens 0.74-0.84.

Reading: token rates track component size (multi-source and liid nodes sit in
bigger components), so they rank liid+tapad nodes worst even though liid+tapad
pairs agree best at pair level (journal 13). Not usable as node/source weights.
LR, which holds size fixed, says the opposite for sources: more liid/id5 share
-> less over-merge than tapad share. MAID-heavy components are the incomplete ones;
dropped MAID singletons almost always belong to a person found elsewhere.
