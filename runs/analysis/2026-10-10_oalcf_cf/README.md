# 2026-10-10: why the diversity extension fails with OAL-CF, and the CF-deduplication idea

Data and checks behind the analysis of 2026-10-10. Scripts are in `tools/analysis/`; the plan and
pre-declared success rule for the new method are in `runs/reports/cfdedup__PLAN.md`.

## Contents

| path | what | made by |
|---|---|---|
| `probe/<arm>_<corpus>_<lang>_s1.pkl` | Counterfactual probe of 12 BASELINE runs (seed 1; ha yo ky vi skr ml; OAL and OAL-CF). Per session: every candidate clip's uncertainty, true label, locale, pre- and post-fit p(target), cluster ids at thresholds 0.25/0.18/0.16/0.10, the baseline's 8 AL picks, the picks the diversity extension WOULD make at 0.25/0.8, 0.18/0.95, 0.16/0.93, 0.10/0.97, and the clips corrective feedback labelled. Every replay was checked bit-identical to its archived `runs/baseline/<arm>/seed_1/` run (`probe/batch_progress.log`). | `tools/analysis/cf_probe.py` |
| `tables/runs_table.csv` | One row per archived run (831 at the time): scores, label composition (AL / CF), AL target hits derived from the pool's class balance, extension diagnostics. | `tools/analysis/build_runs_table.py` |
| `tables/probe_sessions.csv`, `tables/probe_swaps.csv` | Per-session summary of the probe, and every clip the extension would swap in / out. | `tools/analysis/analyze_probe.py` |
| `tables/cf_dedup_sessions.csv`, `tables/cf_dedup_clusters.csv` | Offline CF deduplication at 10 thresholds (0.02-0.25) on the OAL-CF probe runs: expert labels needed, wrong copied labels, cluster sizes. | `tools/analysis/cf_dedup_offline.py` |
| `cf_dedup_tables.txt` | **The CF-dedup tables in one text file**: per threshold (0.02-0.25) the number of clusters, cluster sizes, predicted labels saved (and listening hours), wrong copies by type, per-language breakdowns, and prediction vs the real seed-2 runs. | `tools/analysis/cf_dedup_tables.py` |
| `cfdedup_smoke/` | Checks on the first three real runs of the CF-dedup code (seed 1, African); the runs themselves are in `runs/cf_dedup/oalcf_cfd0.1/seed_1/` and `runs/cf_dedup/oalcf_cfd0/seed_1/`. `verify.txt` has one over-strict check marked FAIL (fixed, see `verify2.txt`: all pass). | `tools/analysis/verify_cfdedup_smoke.py` |

`p_tgt_post` for a clip the baseline queried by AL comes from a model trained on that clip, so it is
not a fair "would CF have caught it" measure for AL picks; use `p_tgt_pre` for those.

## Findings (seed 1 unless stated)

**The diversity extension cannot help OAL-CF.**
- CF gives 96% of the target labels in OAL-CF; AL queries 4% (all 100 baseline runs).
- 85% of any OAL-CF score difference is missed targets, and the misses are confident: median rank
  171 of ~720 by p(target), 0% inside the uncertainty top-8 (both arms). No 8-query rule reaches them;
  even "query unflagged neighbours of flagged clips" reaches ~10%.
- The extension swaps a big-cluster pick (median cluster 136-148 clips) for a crumb (6-7 clips), with
  no gain in targets, identically in OAL and OAL-CF.
- Its one causal local effect, measured at the first divergent session over 623 run pairs: about 5-7
  fewer false alarms that session (p ~ 0.04). In OAL-CF that removes hard negatives CF would have
  labelled, and false alarms barely count in Eq. 1.
- Outcome differences are trajectory noise: runs that changed only 4-16 queries in total still end a
  median +-17% from baseline; 70% of the difference sits in 3 burst sessions.
- On seeds 1-4 alone the extension does not help OAL either (0.16/0.93: 12/20, -4%, p = 0.25); the
  OAL "success" came from seed 0's badly converged baseline. OAL-CF on seeds 1-4: 6-7/20, +13-17%.

**CF deduplication (one expert label per cluster of near-duplicate CF clips).**
- Offline, 6 OAL-CF runs, 6,726 CF labels: at 0.10, 30% fewer expert labels, 0.18% wrong copies
  (0.8% of false alarms labelled target); at 0.12, 49% / 0.62% (2.9%); from 0.14 the false-alarm error
  climbs (6.4%, 9.3% at 0.16, 33% at 0.25).
- Real runs at 0.10: ha 42% fewer CF labels, accuracy identical (bit-identical scores);
  yo 61% fewer, but 7 wrong copies in session 27 led to a missed burst in session 71
  (104 of 105 targets), Eq. 1 0.022 -> 0.137. Control (threshold 0) bit-identical to baseline.
- Eq. 1 does not charge CF confirmations, so the saving shows in the dissertation's Eq. 3.8
  (`IMLM_thesis`) and the released scorer with expert labels (`IMLM_repo_expert`), both in
  `tools/score_runs.py`.
- Seed 2, all 20 languages, at 0.10: launched 2026-10-10 01:33 -> `runs/cf_dedup/oalcf_cfd0.1/seed_2/`;
  status in `runs/reports/cfdedup__OVERNIGHT_STATUS.txt`.

## Reproduce

    python tools/analysis/build_runs_table.py          # tables/runs_table.csv (+ sessions.pkl, not archived)
    python tools/analysis/analyze_probe.py             # probe summary + tables/probe_*.csv
    python tools/analysis/cf_dedup_offline.py          # tables/cf_dedup_*.csv
    python tools/analysis/verify_cfdedup_smoke.py      # checks on the archived smoke runs
