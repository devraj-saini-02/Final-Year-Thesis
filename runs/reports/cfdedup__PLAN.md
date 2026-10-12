# Corrective-feedback deduplication: plan and pre-declared success rule

Written 2026-10-10, BEFORE any full run of the method. Code: `oalcf-main/utils/cf_dedup.py`,
entry point `oalcf-main/run_cfdedup.py`, unit tests `oalcf-main/tests/test_cf_dedup.py`.

## The method

In OAL-CF the expert reviews every clip the classifier flags after each session's fit. Instead,
group the session's flagged clips by ECAPA cosine distance (agglomerative, average linkage,
threshold T, the same clusterer as the diversity extension), have the expert label ONE clip per
group (the one the model is most confident is target), and copy that label to the rest. Wrong
copied labels enter the training pool as they would with a real expert. T = 0 is the control
(no grouping, bit-identical to baseline OAL-CF).

## Why (measured before this plan, `seed 1`, 6 languages, offline on baseline trajectories)

- CF supplies 72% of all expert labels in OAL-CF; the extension targets that channel.
- At T = 0.10: 30% fewer CF labels, 0.18% of copied labels wrong (0.8% of false alarms labelled
  target). At 0.12: 49% fewer, 0.62% wrong (2.9% of false alarms). From 0.14 the false-alarm error
  climbs steeply (6.4% at 0.14, 9.3% at 0.16, 33% at 0.25).
- If every copied label is right, the model trains on exactly the same data, so accuracy is
  unchanged by construction; only wrong copies can move it.

## Which metric shows the saving

The paper's Eq. 1 does not charge confirmations of correct detections, so it cannot show a CF
saving. The dissertation's Eq. 3.8, `(N_adapt + N_fp)/N + N_fn/N_target` with N_adapt = samples
labelled for adaptation, charges every CF label; we read N_adapt as **clips the expert labelled**
(the thesis derives the metric from expert time per clip). `tools/score_runs.py` reports
`IMLM_thesis` (Eq. 3.8) and `IMLM_repo_expert` (the released scorer with pool size replaced by
expert labels) next to the unchanged `IMLM_eq1` and `IMLM_repo`.

## First run: T = 0.10, seed 2, OAL-CF, 20 languages

Seed 2, not seed 1, because seed 1 produced the offline estimates above; seed 2 is out of sample.
`bash tools/run_experiment.sh --algo oalcf --seeds 2 --cfdedup 0.10`, launched only if the
three smoke runs (seed 1, African: T = 0 control, ha and yo at 0.10) pass every check.

**Predictions, fixed now:**
1. Expert CF labels fall by 15-45% pooled over the 20 languages (offline: 30%).
2. Copied-label error below 1% of CF labels; false alarms labelled target below 2%.
3. Accuracy not harmed: IMLM_eq1 mean within +-5% of baseline OAL-CF (seed 2), and no
   significant difference (two-sided Wilcoxon p > 0.05). Languages whose run never copies a
   wrong label must be bit-identical to baseline.
4. IMLM_thesis lower than baseline in at least 15/20 languages.

**Reading:** 1-4 all hold -> run seeds 0, 1, 3, 4 (about 40 h) before any claim; a single seed is
a first look, not a result. Prediction 3 fails -> the copied errors cost accuracy; inspect where
(cf_dedup.csv, wrong labels per session) before trying T = 0.08 or another representative rule.
Prediction 1 fails low -> the saving does not generalise beyond the seed it was measured on.

Not decided yet: T = 0.12, and the representative rule (`--cfd_rep confident|uncertain|medoid`).
Both should be chosen on seed-1 data only, never on the seeds used to confirm.

## Outcome, seed 2 (2026-10-10 01:33 -> 11:21, 20/20 runs exit 0, all full length)

`runs/cf_dedup/oalcf_cfd0.1/seed_2/`; comparisons in `runs/comparisons/cf_dedup/` and
`runs/reports/cfdedup__OVERNIGHT_STATUS.txt` (written before the 2026-10-10 layout change, so it shows old paths).
All four predictions hold.

| prediction | result | |
|---|---|---|
| 1. CF expert labels -15..-45% | 51,581 -> 36,287, **-29.7%** (offline estimate on seed 1: -30%); per language 3% (sr) to 68% (mr) | holds |
| 2. copy error < 1%, false alarms labelled target < 2% | **0.22%** of CF labels; **0.74%** of false alarms | holds |
| 3. Eq. 1 within +-5%, two-sided p > 0.05; zero-error languages bit-identical | 0.0554 -> 0.0580 (**+4.65%**, just inside), p = 0.66; **9/9** zero-error languages identical | holds (mean only just) |
| 4. IMLM_thesis better in >= 15/20 | **16/20, -9%, p = 0.0024** (IMLM_repo_expert 16/20, p = 0.0018) | holds |

- Eq. 1: 5 better, 6 worse, 9 identical. Changes come from a few wrong copies sending a run down a
  different trajectory, in both directions: cv -44%, yo -45%, yue +61%, or +38% (one wrong label),
  dv +18%. DCF 3/20 better, +9%, p = 0.88 (two-sided n.s.).
- Wrong copies concentrate in three languages: tt 39, hi 32, yue 25 (yue from session 0, and 12 of
  them targets labelled non-target). Any rule to protect such cases (e.g. no grouping in early
  sessions) must be chosen on seed-1 data only; seed 2 is now a confirmation seed.
- Next, as planned: the same setting, unchanged, on seeds 0, 1, 3, 4 (~40 h) before any claim.
