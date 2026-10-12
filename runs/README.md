# runs/

Every result in the project, grouped by **what was run**, then by seed, then by language.
See `GUIDE.md` in the repo root for how to start runs and compare them.

```
runs/
  baseline/                       the paper's methods, as published
      oal/                            OAL: active learning only
      oalcf/                          OAL-CF: active learning + corrective feedback
          seed_0/ ... seed_4/             one folder per stream-order seed
              progress.log                    when each run of this batch started and finished
              cr_tt/                          one run: <corpus>_<language>
                  scores.csv                      per-session results; every reported number comes from here
                  command.txt                     the exact command that produced it
                  state_dict.pt                   final model weights
                  train.log                       training output
                  al_samples.txt                  (upstream file; cannot identify the queried clips, see DEVIATIONS)
  diversity_extension/            our first extension: cluster-quota diversity penalty on AL queries
      oal_t0.16_g0.93/                setting = <algo>_t<threshold>_g<gamma>; gamma 1 = control
          seed_N/<corpus>_<lang>/         same files as above, plus al_diversity.csv (clusters/queries per session)
  cf_dedup/                       our second extension: one expert label per group of near-duplicate CF clips
      oalcf_cfd0.1/                   setting = <algo>_cfd<threshold>; threshold 0 = control
          seed_N/<corpus>_<lang>/         same files as above, plus cf_dedup.csv (CF clips, expert labels,
                                          wrong copies per session)
  comparisons/                    saved comparisons (report.txt + table.csv), by the category of the
      baseline/  diversity_extension/  cf_dedup/      setting being tested; never overwritten
  analysis/
      cluster_analysis/               cluster maps and threshold tables (docs/other/CLUSTER_OBSERVATIONS.md)
      2026-10-10_oalcf_cf/            why the diversity extension fails with OAL-CF; CF-dedup measurements
  reports/                        write-ups (*RESULTS.md), plans with pre-declared success rules (*PLAN.md),
                                  summaries, status logs
  figures/
      base_paper_reproduction/    the paper's figures and tables, reproduced (tools/paper_figures.py)
      convergence/                every session's error curves, per language (tools/plot_convergence.py)
      oal_gcf/                    OAL-CF vs OAL-GCF, our CF-deduplication method
                                  (tools/analysis/plot_gcf_figures.py)
  INVENTORY.md                    every run, its score and location (auto-generated; do not edit)
```

- Corpora: `ac` African Continent, `cr` Caucasus Region, `sea` Southeast Asia, `sa` South Asia.
- Session counts, as a health check: ac 83, cr 159, sea 126, sa 95 rows in `scores.csv`.
- Do not rename these folders by hand: `tools/experiments.py` is the one place that knows the layout
  (`run_path()`, `seed_dir()`), and every tool finds runs through it. Specs as typed on the command
  line: `oal`, `oalcf`, `oal@0.16/0.93` (diversity extension), `oalcf+cfd0.1` (CF dedup).
- `tools/run_experiment.sh` writes new runs straight into this layout and, when a batch ends, refreshes
  `INVENTORY.md` and saves the comparisons against the matching baseline into `comparisons/<category>/`.

## Layout history

The layout changed twice; run contents were moved untouched and verified byte-for-byte each time.

| date | from | to | log |
|---|---|---|---|
| 2026-09-18 | `runs/<batch>/output/<corpus>_<arm>.../<lang>` | `runs/seed_N/<setting>/<corpus>_<lang>` | `tools/migrate_runs_layout.py` |
| 2026-10-10 | `runs/seed_N/<setting>/<corpus>_<lang>` | `runs/<category>/<setting>/seed_N/<corpus>_<lang>` | `reports/layout_migration_2026-10-10.txt` |

Older write-ups quote older paths. To translate a 2026-09-18-style path:

| old | new |
|---|---|
| `runs/seed_N/oal_baseline/`, `runs/seed_N/oalcf_baseline/` | `runs/baseline/oal/seed_N/`, `runs/baseline/oalcf/seed_N/` |
| `runs/seed_N/<algo>_t<T>_g<G>/` | `runs/diversity_extension/<algo>_t<T>_g<G>/seed_N/` |
| `runs/seed_N/oalcf_cfd<T>/` | `runs/cf_dedup/oalcf_cfd<T>/seed_N/` |
| `runs/cluster_analysis/` | `runs/analysis/cluster_analysis/` |
| `runs/comparisons/<name>/` | `runs/comparisons/<category>/<name>/` |

Report files are named `<original batch>__<file>`, e.g. `reports/dq_oal_thresh__RESULTS.md` (the
diversity-extension write-up), `reports/baseline_seeds__RESULTS.md` (the multi-seed reproduction),
`reports/cfdedup__PLAN.md` (the CF-dedup plan and its outcomes).
