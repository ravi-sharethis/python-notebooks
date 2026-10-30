# TTD Pricing Pipeline — Reference & Journal

Working notes for the ShareThis × TTD segment-pricing pipeline: the Experiment 1
(−5pp) rollout, the Experiment 2 (±2.5pp) stratified test, and the $1,000+/mo
carve-out.

This file has two parts:

- **[Reference](#reference)** — the current state of understanding: what each
  notebook does, the state-machine vocabulary, known assumptions. Edited in
  place as understanding changes.
- **[Journal](#journal)** — a dated, append-only log of what was done, found,
  and decided. Never rewritten — if something in an old entry turns out to be
  wrong, a *later* entry says so; the old entry stays as the record of what was
  believed at the time.

When updating this file: fix Reference to match current reality, and add a new
dated entry under Journal rather than editing an old one.

---

## Reference

### Notebooks

| Notebook | Job | Reads | Writes |
|---|---|---|---|
| `TTD_Pricing_Simple_Experiment.ipynb` | Original Experiment 1 candidate construction: revenue-rank segments per provider, chunk into groups of ~3 for a matched-triplet design. | Ranked revenue export | `Price_Expt_Assignments2.csv` (historical, Global-only — confirmed via an earlier `nbs/` copy with an uncommented `to_csv` call) |
| `TTD3_Pricing.ipynb` | The original experiment's own DiD/ATE analysis: bucket-weighted treatment effects, the eCPM×Impressions revenue decomposition, and the −5pp/+5pp price-elasticity comparison. | `Price_Expt_Assignments2.csv`, a weekly TTD export, `Ranked_by_Revenue.csv` | Analysis only — no pipeline output files |
| `TTD_stratified_selection.ipynb` | Selects Group 1 (Experiment 1 holdout) and Group 2 (Experiment 2 candidates) each round; assigns each Group 2 member a ±2.5pp/control delta. | Prior round's `Price_Assign_*.csv`, fresh revenue pull, `base_price.csv` (via the prior round file's own `geo_rank`/`group` columns) | `holdout_list_*.csv`, `experiment2_candidates_*.csv` |
| `TTD_Price_Assign.ipynb` | Resolves one final PoM/delta per segment: bulk −5pp convergence, restore-segment correction, then the Experiment 2 overlay on top. | Fresh revenue pull, prior-round reference, the two files above | `Price_Assign_*.csv` (the live roster) |

`base_price.csv` is the odd one out: no notebook currently in the repo writes
it, and it isn't tracked in git. Its `geo_rank`/`group` columns are the
canonical, never-recomputed reference every other file joins against — see the
[2026-09-24](#2026-09-24) journal entry for what we could and couldn't
reconstruct about its origin.

### The state machine

Every segment's role each round is one of these labels:

| State | Meaning |
|---|---|
| `O` | Untouched — still at the provider's base PoM (0.15 for sharethis) |
| `T1 -0.05` | Bulk rollout: converged to the accepted −5pp target (0.10) |
| `T1 +0.05` | Historical +5pp test arm (Experiment 1's original 3-arm design; not part of the current rollout) |
| `C1` | Experiment 1 holdout — frozen at whatever price it already had |
| `T2 -0.025` / `T2 +0.025` | Experiment 2 treatment, direction-suffixed |
| `C2` | Experiment 2 in-group control (delta = 0) |

### Known assumptions

- **May vs. July revenue split is a design choice, not a re-validated one.**
  Confirmed correct by the user directly ("apply the T1 gate to May revenue, T2
  to July") after we found the inconsistency it was fixing. Not independently
  re-derived from first principles beyond that conversation.
- **`group` is only unique within `(provider, geo)`.** Numbering restarts at 0
  for every geo separately (verified: Global groups 0–339, US 0–276, Asia
  0–405, etc.). Every join on `group` in these notebooks already uses the
  composite key `(Third_Party_Data_Provider_Id, geo, group)` — assume the same
  anywhere new code touches it.
- **Missing revenue = $0, not "unknown".** Applied consistently across bucket
  assignment and every revenue-gated rule, so NaN comparisons (always False in
  pandas) don't silently exclude a segment from a rule by accident.
- **The 10-segment `restore_segments` list is complete and correct.** Taken as
  given each round (segments mistakenly bulk-treated despite being >$1,000/mo,
  reverted to base price) — never independently re-audited against the full
  universe for a possible 11th case.

---

## Journal

### 2026-08-07

**Round:** `assign_date` 2026-08-06, `week_start` 2026-08-10. Live file:
`Price_Assign_2026-08-06.csv`.

Fixes made to `TTD_stratified_selection.ipynb` and `TTD_Price_Assign.ipynb`
(commit `397aa01`):

- **Fixed — missing in-group control for Experiment 2.** Groups with exactly 2
  eligible members drew their delta via a plain 3-way random permutation
  truncated to 2 — only a 1-in-3 chance of including control. 61 of 261 T2
  segments had no in-group C2, breaking the matched-group design. Fixed by
  forcing one slot to control whenever exactly 2 members are eligible; verified
  0 of 216 after the fix.
- **Fixed — T1 rollout gate used the wrong revenue snapshot.** The bulk −5pp
  gate (`revenue <= $1,000`) was being evaluated against each round's
  freshly-pulled revenue, but the rollout is sticky by design — a segment's
  **May**-time revenue is what actually earned it T1 membership. Gating on the
  fresh (July) pull instead would re-litigate that decision for segments that
  simply grew past $1,000 since. Decoupled: T1 now gates on May revenue
  (preserved as `may_revenue`), while Experiment 2 eligibility and bucket
  labels keep using the fresh pull.
- **New — explicit $1,000+/mo carve-out for Experiment 2.** The blanket
  exclusion of revenue>$1,000 segments from Experiment 2 is reversed for this
  round, but only for segments that are both untouched (`state=='O'`) *and*
  have May revenue > $1,000 — the second condition matters: 2 of the first 9
  candidates found had May revenue under $1,000 and would have been
  double-perturbed by the ordinary bulk rule on top of the carve-out.
- **Fixed — bucket taxonomy reinstated to `[1000, 100, 10, 1]`.** Was
  `[300, 100, 10, 1]` ("1000+ retired"), which folded the carve-out's own
  population into the 100+ cell and made it unreportable as its own tier.
- **Fixed (prior session) — T1/T2 state-label bimodality.** Bare `T1`
  conflated the historical −5pp and +5pp arms; split into direction-suffixed
  labels. The bulk rule itself was generalized from a flat −5pp to a
  target-price convergence (`delta = target - PoM_old`), so a segment starting
  from any prior price lands on the same accepted target instead of drifting.

**Results so far:**

*Experiment 1 headline* (original test, Global, April/May data): −5pp cut →
+22.3% revenue, 95% CI [+5.6%, +34.7%] (relative), 957 segments tested across 3
arms. Driven by volume, not price pass-through — impressions rose
significantly in every bucket from 0+ to 100+; eCPM showed no consistent
directional movement.

*Price elasticity* (revenue Δ as % of each bucket's own control-baseline
revenue):

| Bucket | −5pp | +5pp | Asymmetry |
|---|---|---|---|
| 1+ | +156.0% | −11.1% | 14.0x |
| 10+ | +62.2% | −29.8% | 2.1x |
| 100+ | +73.1% | −50.4% | 1.45x |

A cut moves revenue more than an equal-sized increase hurts it, in every
bucket tested. (1+'s 14x is partly a scale artifact of its tiny $0.44/seg/wk
baseline — read the dollar terms alongside the percentage.) 0+ excluded (its
control baseline is $0, % undefined); 1000+ excluded as unreliable (n=5,
wildly unstable decomposition).

*Slide 5 secondary metrics* — same DiD/ATE methodology, extended to 10
metrics. Only two of five funnel stages are genuinely robust: Match %, Paid %,
Paid % Percentile (real, significant, consistent with revenue/impressions);
Relevance and Value nominally cross p<0.05 but with small effect sizes and
p-values close to the threshold (0.036, 0.044 vs. Match/Paid's 0.002/0.008) —
flagged as noise, not treated as wins; Selected % is not significant.

*Current live round state counts* (sharethis): `T1 -0.05`=2,982, `C1`=230,
`C2`=148, `T2 +0.025`=109, `T2 -0.025`=107, `O`=12. All 107 `T2 -0.025`
segments land at exactly PoM=7.5% (the $0.10 bulk-rollout base minus 2.5pp),
spread Global 38 / Asia 28 / US 22 / Unknown 16 / APAC 3.

**Deliverables:** `TTD_Price_Experiment.pptx` (9-slide executive deck), an HTML
mirror of the deck, standalone artifacts for the price-elasticity curve and an
ad-serving funnel view, and a price-elasticity section added to
`TTD3_Pricing.ipynb` (formula, worked example, asymmetry table, 3-panel plot).

**Open at end of entry:**
- Week-1 post-treatment analysis not yet built (needs a new notebook, not a
  reuse of `TTD3_Pricing.ipynb` as-is — see below).
- `TTD3_Pricing.ipynb` isn't a drop-in fit for the new round: it's scoped to
  Global only, reads the old `Price_Expt_Assignments2.csv` schema instead of
  the current `state` machine, assumes a symmetric ±5pp 3-arm design, and has
  no matched-group DiD. Plan is to port its reusable helpers (DiD math,
  bucket-weighting, the decomposition formula) into a new notebook rather than
  edit it in place, to keep it intact as the historical record slides 3–5
  still cite.
- Slide 9's logo mismatch: the user's own edit of that slide uses a different
  logo mark than slides 1–8; never resolved which is correct going forward.

---

### 2026-09-24

**Topic:** tracing the origin of `base_price.csv`, prompted by noticing
repeated `group` numbers across geos.

Confirmed the repeated-numbers observation is expected, not a bug: `group`
numbering restarts at 0 independently per geo (Global groups 0–339, US
0–276, Asia 0–405, Unknown 0–166, APAC 0–3 — all ~3 segments/group). Every
place `group` is used already keys on `(provider, geo, group)` for exactly
this reason.

Searched for the actual construction process: no commit history touches the
CSV (the `outputs/` directory isn't part of this git repo), no `.py`/`.ipynb`
anywhere under `/Users/ravirajan/code` writes it, and this session's own
memory had nothing on it. Cross-session transcript search across all local
Claude Code sessions (including archived) found only sessions that had
observed the file's timestamp via `ls -la` — none that built it.

A peer Claude Code session ("Pricing experiment T1/T2 rollout analysis")
identified the actual rank/chunk logic in `TTD_Pricing_Simple_Experiment.ipynb`
— confirmed directly:

- `reset_rev_rank()` resets revenue rank to 0 only when the provider changes
  (no geo-awareness at all).
- `group = floor((rank-1)/3)` chunks into triplets.
- An earlier saved copy of this notebook (`nbs/`, dated before the copy in
  this repo) has this exact logic filtered to `geo=='Global'`, with an
  *uncommented* `to_csv` writing straight to `Price_Expt_Assignments2.csv` —
  confirming this is genuinely what produced the original experiment's groups.
- A later copy of the same notebook (dated between the original run and
  `base_price.csv`'s July 24 appearance) shows the same `to_csv` redirected to
  a `.test` file, suggesting active adaptation without touching the original.

**Best-supported hypothesis, not confirmed:** `base_price.csv`'s multi-geo
coverage was produced by rerunning this same geo-blind logic once per geo
(editing the one `geo==` filter line each time) and concatenating the five
outputs — which would exactly reproduce the observed pattern (clean per-geo
numbering, same bare number reused across geos). The actual loop-and-concatenate
script was not found and is likely unrecoverable — not in git, not on disk, not
in any searchable session transcript.

**Status:** documented in Reference above; no code changed this entry.
