# Cluster granularity in the diversity-penalised AL extension (OAL arm)

**Complete, 2026-09-16 23:24.** Four settings x 20 languages, plus two gamma = 1.0 controls; 82 runs,
all exit 0, 80/80 setting-runs healthy (full session counts, no NaN), both controls bit-identical to
baseline OAL.

**Headline: at matched annotation effort, threshold 0.16 with gamma 0.93 beats baseline OAL in 15 of
20 languages, cutting mean cost by 32% (p = 0.0028), with the same result on all three metrics.**
This is the first significant result in the OAL arm - the arm where the extension had never worked.

Companion documents: `docs/other/CLUSTER_OBSERVATIONS.md` (cluster properties, selection rule),
`runs/reports/baseline_seed0__RESULTS.md` (the OAL vs OAL-CF reproduction), `runs/reports/dq_oal__summary.txt` and
`runs/reports/dq_oalcf__summary_*.txt` (the extension at the original threshold 0.25).

---

## 1. What is being varied

The extension scores each candidate clip in a session as

> **score(x) = u(x) · gamma^(n_c / q_c)**

`u` = least-confidence uncertainty, `c` = the clip's cluster, `n_c` = picks already taken from that
cluster this session, `q_c = k·sqrt(|c|) / sum_j sqrt(|c_j|)` = the cluster's share of the k = 8 query
budget. Clusters are formed per session by agglomerative clustering (cosine, average linkage,
`distance_threshold`) over the clips' own ECAPA embeddings - audio only, no labels, no model - so they
are identical for every target language, every gamma and both arms.

The published runs used `distance_threshold = 0.25`, chosen to keep cluster counts in a sensible
range, **not** to make clusters language-pure - and they are not: pooled homogeneity 0.28, one cluster
typically holding 73% of a session across several languages.

**Two dimensions are varied here**: cluster granularity (threshold) and penalty strength (gamma).
Because `q` shrinks as clusters shrink, raw gamma is not comparable across thresholds; the settings
were chosen to match the *intended* second-pick multiplier `m = gamma^(1/q)`:

| Setting | Threshold | q | gamma | Intended m | **Realised disturbance** |
|---|---|---|---|---|---|
| published | 0.25 | 4.30 | 0.8 | 0.95 | 1.31 of 8 queries, 0.7% u |
| mild | 0.18 | 0.95 | 0.95 | 0.95 | 1.21 of 8, 0.4% u |
| medium | 0.18 | 0.95 | 0.86 | 0.85 | 1.85 of 8, 1.2% u |
| mild | 0.16 | 0.47 | 0.975 | 0.95 | **0.98 of 8, 0.2% u** |
| **medium** | **0.16** | **0.47** | **0.93** | 0.85 | **1.30 of 8, 0.7% u** |

The realised-disturbance column is the important one, and it shows the calibration **under-corrected
at fine granularity**: 0.16 / gamma 0.975 came out *milder* in practice than 0.18 / gamma 0.95, despite
being matched on paper. 0.16 / gamma 0.93 is what actually lands at the same realised disturbance as
the published setting.

---

## 2. Cluster properties (no runs involved)

Pooled over all 463 sessions; per-corpus tables in `docs/other/CLUSTER_OBSERVATIONS.md` section 3b.

| | 0.25 | 0.20 | 0.18 | **0.16** |
|---|---|---|---|---|
| Clusters per session | 10.0 | 40.8 | 71.9 | **123.8** |
| Real clusters (>=8 clips) | 2.2 | 6.6 | 9.3 | 11.7 |
| Crumbs (<8 clips) | 7.8 | 34.2 | 62.6 | 112.1 |
| Clips sitting in crumbs | 2% | 10% | 17% | 29% |
| Largest cluster | 73% | 46% | 37% | 28% |
| **Homogeneity** | 0.28 | 0.59 | 0.74 | **0.84** |
| Completeness | 0.15 | 0.21 | 0.22 | 0.20 |
| V-measure | 0.12 | 0.27 | 0.31 | 0.30 |
| Clips in a >=90% pure cluster | 29% | 55% | 70% | 81% |

---

## 3. Results, OAL arm, 20 languages, seed 0

One-sided Wilcoxon on paired per-language scores, as in the paper's Section VI-G. Bonferroni over the
four tested settings gives a threshold of p < 0.0125.

### IMLM, Eq. 1 (baseline mean 0.1576)

| Setting | Wins | Mean | p | Damage | Benefit | Worst loss | sd |
|---|---|---|---|---|---|---|---|
| 0.25 / gamma 0.8 (published) | 11/20 | 0.1561 (-1%) | 0.449 | +0.708 | -0.736 | +0.329 (yo) | 0.113 |
| 0.18 / gamma 0.95 | 13/20 | 0.1418 (-10%) | 0.261 | +0.373 | -0.689 | +0.151 (cv) | 0.082 |
| 0.18 / gamma 0.86 | 9/20 | 0.1520 (-4%) | 0.649 | +0.481 | -0.593 | +0.177 (cv) | 0.086 |
| 0.16 / gamma 0.975 | 14/20 | 0.1352 (-14%) | 0.083 | +0.516 | -0.963 | +0.378 (or) | 0.123 |
| **0.16 / gamma 0.93** | **15/20** | **0.1065 (-32%)** | **0.0028 \*\*\*** | **+0.128** | **-1.149** | **+0.086 (hi)** | 0.085 |

### The same on the other two metrics

| Setting | IMLM repo scorer | p | DCF | p |
|---|---|---|---|---|
| 0.25 / gamma 0.8 | 11/20, -2% | 0.449 | 11/20, -1% | 0.435 |
| 0.18 / gamma 0.95 | 13/20, -10% | 0.249 | 12/20, -11% | 0.249 |
| 0.18 / gamma 0.86 | 9/20, -4% | 0.622 | 9/20, -4% | 0.649 |
| 0.16 / gamma 0.975 | 13/20, -14% | 0.088 | 13/20, -15% | 0.108 |
| **0.16 / gamma 0.93** | **15/20, -33%** | **0.0028 \*\*\*** | **14/20, -35%** | **0.0028 \*\*\*** |

The best setting is best on **every** dimension at once: win count, mean, p-value, total damage, total
benefit, worst single loss, and spread. That did not happen for any other setting.

### 3.1 Where the gain comes from

Annotation effort is **identical by construction** - 962 AL queries per run in every setting, and no
corrective feedback in this arm. The improvement is pure labelling efficiency:

| | Baseline OAL | 0.16 / gamma 0.93 |
|---|---|---|
| Miss rate (FNR) | 0.1453 | **0.0952** (-34%) |
| False-alarm rate (FPR) | 0.0090 | **0.0048** (-47%) |
| AL queries | 962 | 962 |

Both error types fall, for the same expert effort.

### 3.2 The dose-response is threshold-dependent

| Threshold | Does the stronger push (m = .85) beat the milder one (m = .95)? |
|---|---|
| 0.18 | **No** - better in 6/20, p = 0.96 |
| 0.16 | **Yes** - better in 14/20, p = 0.049 |

This kills the simple rule "the cheapest intervention always wins", which held at threshold 0.25 (mild
> medium > harsh) and with OAL-CF. Strength and granularity **interact**.

The realised-disturbance column in section 1 explains it: at 0.16 the intended m = 0.95 calibration
under-shot, producing only 0.98 changed queries per session; gamma 0.93 restores ~1.3, the same level
as the published setting and as 0.18 / gamma 0.95.

**And that comparison is the cleanest statement of the finding:**

> At matched realised disturbance (~1.3 of 8 queries changed, ~0.7% of uncertainty given up), moving
> from ~10 clusters per session to ~122 takes the result from **-1% (p = 0.45)** to **-32%
> (p = 0.0028)**.

Disturbance alone explains nothing - the published setting disturbs exactly as much and achieves
nothing. **Granularity is what converts a fixed amount of disturbance into a large gain.**

### 3.3 Per language, IMLM Eq. 1

| Corpus | Lang | Baseline | 0.18/g.95 | 0.18/g.86 | 0.16/g.975 | **0.16/g.93** |
|---|---|---|---|---|---|---|
| AC | bas | 0.5767 | 0.5468 (-5%) | 0.5426 (-6%) | 0.4045 (-30%) | **0.3083 (-47%)** |
| AC | ha | 0.3101 | 0.0284 (-91%) | 0.0420 (-86%) | 0.0238 (-92%) | 0.1711 (-45%) |
| AC | yo | 0.2998 | 0.3465 (+16%) | 0.1678 (-44%) | 0.1146 (-62%) | **0.0470 (-84%)** |
| CR | hy-AM | 0.0352 | 0.0257 (-27%) | 0.0263 (-25%) | 0.0332 (-5%) | **0.0139 (-61%)** |
| CR | ckb | 0.0560 | 0.0783 (+40%) | 0.0383 (-32%) | 0.0485 (-13%) | 0.0360 (-36%) |
| CR | cv | 0.1380 | 0.2885 (+109%) | 0.3149 (+128%) | 0.1649 (+20%) | **0.1149 (-17%)** |
| CR | kmr | 0.1913 | 0.1587 (-17%) | 0.2575 (+35%) | 0.1321 (-31%) | 0.1390 (-27%) |
| CR | ky | 0.1633 | 0.1346 (-18%) | 0.1663 (+2%) | 0.1046 (-36%) | 0.1015 (-38%) |
| CR | sr | 0.0310 | 0.0178 (-43%) | 0.0319 (+3%) | 0.0309 (-0%) | 0.0274 (-12%) |
| CR | tt | 0.2217 | 0.1169 (-47%) | 0.1210 (-45%) | 0.1258 (-43%) | **0.0912 (-59%)** |
| SEA | yue | 0.3555 | 0.3414 (-4%) | 0.3337 (-6%) | 0.3594 (+1%) | 0.3177 (-11%) |
| SEA | id | 0.0635 | 0.0387 (-39%) | 0.1067 (+68%) | 0.0175 (-72%) | 0.0533 (-16%) |
| SEA | vi | 0.0528 | 0.0563 (+7%) | 0.0650 (+23%) | 0.0752 (+42%) | 0.0730 (+38%) |
| SA | cnh | 0.0198 | 0.0191 (-4%) | 0.0287 (+45%) | 0.0165 (-17%) | 0.0260 (+31%) |
| SA | dv | 0.0567 | 0.0963 (+70%) | 0.0821 (+45%) | 0.0419 (-26%) | 0.0605 (+7%) |
| SA | hi | 0.1487 | 0.2152 (+45%) | 0.2236 (+50%) | 0.2020 (+36%) | 0.2351 (+58%) |
| SA | ml | 0.1796 | 0.0687 (-62%) | 0.1767 (-2%) | 0.1625 (-9%) | 0.1050 (-42%) |
| SA | mr | 0.0880 | 0.0721 (-18%) | 0.0813 (-8%) | 0.0728 (-17%) | 0.0669 (-24%) |
| SA | or | 0.1009 | 0.0787 (-22%) | 0.1332 (+32%) | 0.4786 (+374%) | 0.1121 (+11%) |
| SA | skr | 0.0627 | 0.1069 (+70%) | 0.0997 (+59%) | 0.0943 (+50%) | **0.0304 (-52%)** |

Hindi is the one language harmed by every setting (+36% to +58%). Odia swings from +374% at
0.16 / gamma 0.975 to +11% at 0.16 / gamma 0.93 - a reminder of how violent single-run swings are on
this task.

---

## 4. Reading

1. **Cluster granularity, not penalty strength, is the lever.** At matched realised disturbance,
   finer clusters turn a null result into a 32% mean reduction.
2. **The penalty needs real structure to allocate over.** At 0.25 there are ~2 substantial clusters
   per session, so "spread the queries" means almost nothing; at 0.16 there are ~12, plus crumbs.
3. **The crumb hypothesis is refuted as a one-directional cost.** 0.16 has 29% of each session in
   sub-8-clip clusters, and it is the best setting - once the strength is right. Crumbs add variance
   (see Odia at gamma 0.975), not a systematic penalty.
4. **Strength and granularity interact**, so gamma must be re-tuned whenever the threshold changes.
   The `gamma = m**q` recipe is a useful first guess but under-corrects at fine granularity; the
   realised disturbance (queries changed per session, uncertainty given up) is the quantity to match,
   and it is measurable from `al_diversity.csv` without labels.
5. **Cluster-quality metrics did not predict performance.** Completeness and V peak at 0.18; the best
   setting is 0.16. Internal scores give clean provenance for choosing a threshold, nothing more.

---

## 4a. Why does it work? (investigated 2026-09-17)

**It is not the mechanism the paper's OAL-CF relies on.** OAL-CF wins by raising the target class's
share of the adaptation pool (paper Fig. 9). The extension does not: pool target prevalence is
**identical across every setting** - 2.3% pooled over 20 languages, 2.8% on Caucasus, from threshold
0.25 through 0.16 and at every gamma. The AL budget is also fixed at 8 queries per session by
construction. So the same number of labels, with the same class balance, produce a better classifier.

**What changes is both error types, in the exact order of the settings' results:**

| Setting | Misses / session | False alarms / session | Final FNR | Final FPR | IMLM |
|---|---|---|---|---|---|
| baseline | 2.06 | 6.25 | 0.1453 | 0.0090 | 0.1576 |
| 0.25 / g0.80 | 2.14 | 5.21 | 0.1438 | 0.0075 | 0.1561 |
| 0.18 / g0.95 | 1.90 | 4.75 | 0.1303 | 0.0069 | 0.1418 |
| 0.16 / g0.975 | 1.70 | 4.86 | 0.1237 | 0.0070 | 0.1352 |
| **0.16 / g0.93** | **1.49** | **3.33** | **0.0952** | **0.0048** | **0.1065** |

**Where the gain is made: early.** Share of a run's final misses incurred in the first 25% of
sessions is 59% (baseline) and 62% (0.16/g0.93); on `cr/tt` the baseline has already accumulated 795
of its eventual 997 misses by session 15, against 253 for 0.16/g0.93. Per language, the share of the
*change* in misses that occurs in the first quarter of the stream:

- **Caucasus: 44-125%** (shares >100% mean the advantage is won early and partly given back later).
  This is where the effect is real, and it is an early-sessions effect.
- **African: 0% for all three languages** - bas/ha/yo improve *late*, consistent with the
  regression-to-the-mean reading of that corpus (its baseline runs collapse in late bursts).
- **Hindi, the consistent loser: 0% early**; it degrades steadily and ends +257 misses.

**Reading.** On Caucasus the extension appears to buy a better-conditioned classifier in the opening
sessions - when the pool is tiny and a single wasted query costs a lot - and that advantage persists
rather than compounding. That is consistent with finer clusters preventing the first few sessions
from spending all 8 queries inside one confusable mass, and with the cheap-intervention result
(section 3.2): the benefit comes from *where* the early queries land, not from how many are moved.

**What could not be tested.** Whether the extension queries more target-language clips, finds them
earlier, or spreads across more languages is **not recoverable from archived runs**: upstream's
`save_active_files` does not record which clips were queried (`docs/other/DEVIATIONS.md`, "al_samples.txt
cannot identify the queried clips"). Testing those hypotheses needs a re-run with query logging added
to `utils/diverse_query.py`.

## 5. Limits

- **Exploratory, not confirmatory.** Four settings were tried after seeing the 0.25 results, and the
  thresholds were motivated by cluster measures scored against the **test stream's** own language
  labels. p = 0.0028 survives Bonferroni over the four settings, but the honest family is larger.
- **One model seed, one stream order.** This is the single most important gap: a 15/20, p = 0.0028
  result on one seed must be replicated before it is claimed. Odia moving +374% -> +11% between two
  neighbouring settings shows the scale of run-to-run swing.
- **This is the OAL arm.** Which cuts both ways: the extension had never worked here, so the effect is
  surprising - but the paradigm that matters is OAL-CF, and that is untested at these thresholds.
- **Partial results mislead.** An earlier 16-of-20 read of 0.16 / gamma 0.975 suggested it was a
  failure; the final four languages reversed it. Nothing on this task should be interpreted before
  African Continent lands.

## 5a. Pre-declared prediction for the OAL-CF test (written 2026-09-17, BEFORE the runs)

The OAL arm is a screen. The confirmatory test is the same setting on the paradigm that matters, and
this prediction is recorded before that batch starts so the test is not another exploration:

**Setting: threshold 0.16, gamma 0.93, OAL-CF, 20 languages, seed 0.** Launched with
`bash tools/legacy/run_dq_oalcf_thresh_cpu.sh 0.16:0.93` -> `runs/dq_oalcf_thresh/`.

Predictions, in order of how strongly the OAL screen supports them:
  1. **Caucasus improves broadly against baseline OAL-CF: at least 5 of 7 languages.** This is the
     primary prediction - CR is where the screen found a real effect (7/7, -37%, p = 0.0078) and its
     baseline is not pathological.
  2. Pooled over 20 languages, the extension keeps beating baseline OAL-CF at roughly the rate seen at
     threshold 0.25 (15/20), with a **better mean** than that setting's (0.0763 -> 0.0762, i.e. flat).
  3. African improves again (it improves under nearly every perturbation - regression to the mean, not
     evidence).
  4. South Asia and Southeast Asia show no benefit.

Falsification: if Caucasus drops below 5/7, the OAL screen did not transfer to the CF arm and the
granularity result should be reported as OAL-only.

Caveat carried into the test: corrective feedback makes the model more confident, so the same gamma
moves fewer queries (measured at threshold 0.25: 0.69 changed queries per session with CF vs 1.31
without). gamma 0.93 was kept unchanged rather than compensated, because in the CF arm the mildest
setting was the one that won at 0.25. The realised disturbance will be read off `al_diversity.csv`.

## 5b. Outcome of the pre-declared OAL-CF test (2026-09-17 10:02) - PRIMARY PREDICTION FALSIFIED

`bash tools/legacy/run_dq_oalcf_thresh_cpu.sh 0.16:0.93` -> `runs/reports/dq_oalcf_thresh__summary.txt`. 20/20 runs
healthy; gamma = 1.0 control passed (bit-identical to baseline OAL-CF).

| Prediction (section 5a) | Result | Verdict |
|---|---|---|
| 1. **CR improves in >= 5/7** vs baseline OAL-CF (primary) | **3/7, mean +0%** (p = 0.59) | **Falsified** |
| 2. Pooled win rate near 15/20, better mean than 0.25/gamma 0.8 | **9/20**, mean 0.0763 -> 0.0732 (-4%), p = 0.58 | Falsified on wins; mean marginally better |
| 3. AC improves again | **0/3, +84%** (bas +215%) | Falsified (opposite) |
| 4. SA and SEA flat | SA 3/7 +6% (flat); **SEA 3/3 -41%** (p = 0.125, the floor for n = 3) | SA as predicted; SEA not |

Other metrics agree: repo scorer 10/20 (p = 0.49), DCF 8/20 (p = 0.61). Against baseline **OAL** the
setting is 15/20, p = 0.0024 - but that is corrective feedback doing the work (baseline OAL-CF vs OAL is
16/20, p = 0.0024), not the extension.

**Per corpus, both OAL-CF settings of the extension against baseline OAL-CF (IMLM Eq. 1):**

| Corpus | 0.25 / gamma 0.8 | **0.16 / gamma 0.93** |
|---|---|---|
| AC | 1/3, +400% | 0/3, +84% |
| CR | 5/7, -12% | **3/7, +0%** |
| SEA | 3/3, -37% | 3/3, -41% |
| SA | 6/7, -18% | **3/7, +6%** |
| **Pooled** | **15/20, flat mean, p = 0.016** | **9/20, -4%, p = 0.58** |

Realised disturbance: **0.74 changed queries per session** (vs 1.30 for the same setting with OAL),
confirming that corrective feedback roughly halves how much a given gamma moves the query set.

**Conclusion, as the pre-declaration required:** the OAL screen did **not** transfer to the CF arm.
The granularity result (0.16 / gamma 0.93: 15/20, p = 0.0028 with OAL; Caucasus 7/7, p = 0.0078) must be
reported as **OAL-only**. With corrective feedback, the published clustering (threshold 0.25,
gamma 0.8) remains the better setting - it lost its Caucasus and South Asia gains when clusters were made
finer. The two arms prefer different operating points, which fits the earlier observation that the
optimal intervention level is arm-dependent. Southeast Asia benefits in both CF settings (-37%, -41%)
while it was null with OAL, and African is hurt in both CF settings while it "improved" with OAL -
consistent with the African OAL gains having been regression to the mean on a broken baseline, since
the African OAL-CF baseline (0.025) leaves no such room.

## 5c. Exploratory: 0.18 / gamma 0.95 with OAL-CF (declared before launch, 2026-09-17)

**This run is exploratory mapping, not a test.** The confirmatory test (5a/5b) has already been run and
failed. This run only locates where the OAL-CF optimum sits between threshold 0.25 and 0.16.

- Launched with `bash tools/legacy/run_dq_oalcf_thresh_cpu.sh 0.18:0.95` -> `runs/dq_oalcf_thresh/`
  (the 0.16 / gamma 0.93 summary is preserved as `summary_t16_g93.txt`).
- gamma 0.95 is expected to give ~0.6 changed queries per session (1.21 with OAL, roughly halved by
  corrective feedback), close to the 0.69 at which 0.25 / gamma 0.8 won.
- **Expectation from the existing data:** between 0.16's 9/20 and 0.25's 15/20, since at matched
  intervention the coarser clustering won with corrective feedback.
- **Decision rule, fixed now:** this changes the recommended OAL-CF setting only if it beats
  0.25 / gamma 0.8 on win count (more than 15/20 against baseline OAL-CF) **and** reaches p < 0.017
  (Bonferroni over the three OAL-CF settings tried: 0.25/0.8, 0.16/0.93, 0.18/0.95). Anything less is
  reported as a point on the curve.

**Outcome (2026-09-17 20:54, 20/20 runs healthy, gamma = 1.0 control at 0.18 bit-identical): a point on
the curve; the recommendation does not change.** Summary: `runs/reports/dq_oalcf_thresh__summary_t18_g95.txt`.

| vs baseline OAL-CF | better | mean | p |
|---|---|---|---|
| IMLM Eq. 1 | 10/20 | 0.0763 -> 0.0755 (-1%) | 0.31 |
| IMLM repo | 10/20 | 0.1093 -> 0.1087 | 0.39 |
| DCF | 12/20 | 0.0473 -> 0.0464 | 0.35 |

- Per corpus (Eq. 1): AC 0/3 +131% (bas +339%), CR 3/7 +2%, SEA 2/3 -13%, SA 5/7 -14% (or +160%).
- Realised disturbance 0.61 changed queries/session, uncertainty given up 0.3%, as expected.
- Head to head (Eq. 1, two-sided Wilcoxon): vs 0.25 / gamma 0.8 better in 6/20, p = 0.20; vs 0.16 /
  gamma 0.93 better in 8/20, p = 0.65.
- The expectation (between 9/20 and 15/20) held only at its bottom end: 10/20. OAL-CF across threshold:
  0.25 -> 15/20 (p = 0.016), 0.18 -> 10/20, 0.16 -> 9/20. With corrective feedback, finer clusters do not
  help; 0.25 / gamma 0.8 remains the OAL-CF setting (and it is itself single-seed and borderline).
- CR, the corpus where tighter clusters helped OAL (7/7), is flat again (3/7), matching 5b.

**Session-by-session, 0.18/0.95 vs 0.25/0.8 (OAL-CF)**, table in
`runs/reports/dq_oalcf_thresh__session_diff_t18_vs_t25.csv`:
- The two runs diverge from session 0 in all 20 languages (different queries -> different pool sizes ->
  different models); there is no later "switch-on" point.
- The net Eq. 1 difference is concentrated: the top 3 of 83-159 sessions carry >= 75% of it in 17/20
  languages, often one session (bas s31, or s1, vi s7, kmr s16/s29, yue s1/s42).
- 13/20 decisive sessions are target bursts; the rest are false-alarm spikes. Misses dominate Eq. 1
  (pooled misses 2287 -> 2637; false alarms 10443 -> 13119 but weighted by 1/N).
- A burst is a cascade: the model after AL flags some targets, CF confirms them (cf_tp), the refit
  catches the rest. When the foothold fails the burst is lost (kmr s16: 0.18 run cf_tp 0, 184/190 missed;
  0.25 run cf_tp 73, 110 missed). In the decisive session itself the extension changed only 0-3 queries,
  so outcomes are set by the accumulated trajectory, not by that session's selection.
- On the first target burst of each language, 0.18 is worse in 8, better in 9, tied in 3: no systematic
  direction. The difference between the settings looks like trajectory noise decided at a few bursts,
  which a model-seed control would quantify.

## 6. Next

1. **Seed replication of 0.16 / gamma 0.93 on OAL** (20 runs, ~5.5 h). Cheapest possible test of the
   headline. Everything below depends on it.
2. **0.16 / gamma 0.93 with OAL-CF** (20 runs, ~20 h) - the paradigm that matters. Note this supersedes
   the earlier plan to test 0.18 with OAL-CF.
3. Optional: 0.14 / gamma ~0.90, to check whether the trend continues past 0.16 or turns over.
4. The general model-seed control, still outstanding for every result in this project.

## 7. Reproduce

    bash tools/legacy/run_dq_oal_thresh_cpu.sh 0.18:0.95 0.16:0.975   # 42 runs incl. gamma=1 controls
    # the medium pair (0.18:0.86, 0.16:0.93) was appended to the running queue
    python tools/cluster_threshold_sweep.py > runs/analysis/cluster_analysis/threshold_sweep.txt
    python tools/plot_cluster_languages.py --corpus ac

Run directories: `runs/dq_oal_thresh/output/<corpus>_oal_t<NN>_g<NN>/<lang>`, logs in `logs/`,
progress in `progress.log`. `summary.txt` is the batch's own summary and covers only the two settings
it was launched with; this file supersedes it.
