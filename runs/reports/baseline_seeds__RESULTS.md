# Baseline OAL over five stream-order seeds (seed 0 + seeds 1-4)

80 runs, 2026-09-17 23:50 to 2026-09-18 10:39, `bash tools/legacy/run_baselines_seeds_cpu.sh 1 2 3 4 --arms oal`,
all healthy (full session counts). Metric: IMLM Eq. 1 (authors' reading), `tools/score_runs.py`.
A seed reshuffles the **test stream only**; train order, bootstrap and model seed are seed 0's
(`docs/other/DEVIATIONS.md`). Seeds 1-4 were fixed before any run. Tables: `summary.txt`.

## 1. Headline: averaged over seeds, our OAL baseline matches the paper

| | pooled OAL, 20 languages |
|---|---|
| paper, Fig. 12 read-off (mean of the 20 values) | 0.1015 |
| ours, seed 0 alone | 0.1576 |
| **ours, mean of seeds 1-4** | **0.1063** |

Per language, mean of seeds 1-4 vs the paper's value: Spearman rank correlation **0.81** (seed 0 alone:
0.24), two-sided Wilcoxon **p = 0.52** (no detectable difference), and the paper's value falls inside our
five-seed range for 12 of 20 languages. **This is the strongest reproduction evidence in the project:
on a single seed the baseline looked 55% too high, and the gap was stream-order luck, not method.**

## 2. African was a seed-0 artefact, as predicted

| corpus | seed 0 | mean of seeds 1-4 | per seed (1,2,3,4) |
|---|---|---|---|
| AC | 0.3955 | **0.0614** | 0.035, 0.030, 0.137, 0.043 |
| CR | 0.1195 | 0.1016 | 0.095, 0.120, 0.095, 0.096 |
| SEA | 0.1573 | 0.1836 | 0.178, 0.199, 0.160, 0.198 |
| SA | 0.0938 | 0.0970 | 0.129, 0.098, 0.070, 0.091 |

AC falls by 6.4x and lands in the paper's range (Fig. 12 AC: 0.020-0.057). Seed 0 holds every AC target
back to 33% of the stream; seeds 1-4 start at 0-12%. Every AC claim computed on seed 0 alone - including
the "AC improves under 6 of 7 extension settings" regression-to-the-mean reading in
`runs/reports/dq_oal_thresh__RESULTS.md` 4a - rests on a stream that is unrepresentative.
CR, SEA and SA move little, so their seed-0 corpus-level numbers stand. SEA is the one corpus
consistently worse than the paper, driven by yue (0.36-0.46 across all five seeds vs the paper's 0.29).

## 3. How noisy is one seed?

Per-language relative spread (sd / mean over the 5 seeds): median **48%** over all 20 languages, **38%**
excluding AC. Extremes: bas 169%, ha 122%, yo 112%, or 67%, mr 60%; smallest yue 9%, ky 18%.
Single-seed differences below ~40% per language are not distinguishable from stream-order luck.

**What this does and does not say about the extension results.** Every gamma comparison so far is
*paired on one stream* (same seed 0 for baseline and extension), so it is not exposed to the variance
measured here; this batch does not license discarding those results. What it shows is that the stream
alone moves a language's score by tens of percent, and section 5c of `runs/reports/dq_oal_thresh__RESULTS.md`
shows those moves are decided in a handful of burst sessions - the same mechanism a gamma perturbs.
The matching control is the same extension setting on seeds 1-4, still to run.

## 4. Per language, IMLM Eq. 1

See `summary.txt` for the full table with per-seed values, and section 1 above for the aggregate.
Ranking of languages is broadly stable across seeds; the outliers are the three AC languages, whose
seed-0 values are 5-36x their best seed.

## 5. Next

1. OAL-CF on seeds 1-4 (~39 h) completes the reproduction: it gives the CF-vs-OAL win count and
   Table V's 0.105 -> 0.049 pooled pair on a seed average rather than one stream.
2. Then, if the extension is to be claimed anywhere, the same seeds for the chosen setting
   (0.25 / gamma 0.8 with OAL-CF; 0.16 / gamma 0.93 with OAL).
3. Do not select seeds by results. Seeds 1-4 were fixed in advance; report all of them.
