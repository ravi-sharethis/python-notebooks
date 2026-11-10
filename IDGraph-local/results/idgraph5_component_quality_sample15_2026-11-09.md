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
