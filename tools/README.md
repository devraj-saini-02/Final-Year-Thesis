# tools/

Start with `GUIDE.md` in the repo root. You normally need only the first two scripts here.

## Run experiments and compare results

| script | what it does |
|---|---|
| `run_experiment.sh` | **Runs everything.** Pick algorithm, seeds, and optionally the extension's threshold and gamma. `bash tools/run_experiment.sh` with no arguments prints usage. Existing runs are skipped unless you pass `--force`, which replaces them. |
| `compare_runs.py` | **Compares two settings** across the 20 languages, with the paper's Wilcoxon test. `--list` shows what exists. Saves every comparison to its own folder in `runs/comparisons/<category>/` (`report.txt` + `table.csv`, never overwritten). Metrics: `IMLM_eq1` (default), `IMLM_thesis`, `IMLM_repo`, `IMLM_repo_expert`, `DCF`, `FNR`, `FPR`. |
| `run_inventory.py` | Regenerates `runs/INVENTORY.md`: every run, its score, its location. Run it after any new batch. |
| `experiments.py` | The shared index of finished runs (old and new folder layouts). `compare_runs.py` uses it; run it directly to list everything. |
| `score_runs.py` | Scores run directories one by one (both IMLM formulas, DCF, FNR, FPR). Use it when you want raw numbers for specific folders. |
| `run_cpu.py` | Forces CPU and launches the authors' `run.py` or `run_dq.py`. `run_experiment.sh` calls it; call it directly only for one-off runs. |
| `summarize_seeds.py` | Multi-seed reproduction table: OAL vs OAL-CF per seed, per language, next to the paper's published values. |

| `migrate_runs_layout.py` | One-time move of `runs/` into the seed-first layout (2026-09-18). Kept for the record; nothing to run. |
| `migrate_runs_by_method.py` | One-time move of `runs/` into the current method-first layout (2026-10-10), verified byte-for-byte. Kept for the record; nothing to run. |

## analysis/

Scripts behind `runs/analysis/2026-10-10_oalcf_cf/` (why the diversity extension fails with OAL-CF, and the
CF-deduplication measurements): `cf_probe.py` (counterfactual replay of a baseline run, bit-identical),
`analyze_probe.py`, `build_runs_table.py`, `cf_dedup_offline.py`, `cf_dedup_tables.py` (the threshold
tables), `verify_cfdedup_smoke.py`. See that folder's README.

`analysis/plot_gcf_figures.py` makes every OAL-CF vs OAL-GCF figure in `runs/figures/oal_gcf/` (12 figures,
including the method diagram and the single-session cluster maps): `python tools/analysis/plot_gcf_figures.py`.

`run_experiment.sh --algo oalcf --seeds N --cfdedup T` runs the CF-deduplication extension
(`oalcf-main/run_cfdedup.py`); compare with `compare_runs.py oalcf oalcf+cfd0.1 --metric IMLM_thesis`
(and `IMLM_eq1`). Plan and success rule: `runs/reports/cfdedup__PLAN.md`.

## Figures and cluster analysis

| script | what it does |
|---|---|
| `paper_figures.py` | Reproduces the paper's per-language comparison figure from our runs (`--seed N`, default 0). |
| `plot_convergence.py` | Error over sessions, to see how fast a run converges (`--seed N`, default 0). |
| `plot_session_clusters.py` | Draws the clusters inside a session. |
| `plot_cluster_languages.py` | Cluster maps coloured by language, two thresholds side by side. |
| `cluster_threshold_sweep.py` | Cluster quality (purity, sizes) across thresholds. Produced the tables in `docs/other/CLUSTER_OBSERVATIONS.md`. |

## corpus/

One-time scripts that built `data/`: finding which Common Voice locales the paper used, downloading,
assembling splits, extracting features, building stream orders. **Do not re-run them.** The corpora are
already built, and rebuilding would change which clips are in which split, invalidating every result.
Kept for provenance and for the methods section of the paper.

## legacy/

Superseded batch scripts, one per experiment we ran, plus a few diagnostics. `run_experiment.sh` now
does all of this in one place. They are kept because each documents exactly how a published result was
produced, and their write-ups in `runs/*/RESULTS.md` refer to them by name. Nothing new should use them.

Worth knowing about:

- `repro_mps_batchnorm_grad.py` — the minimal demonstration that Apple's GPU backend computes wrong
  gradients for this model. The reason everything runs on CPU.
- `verify_g1.sh` — checks that the extension with gamma = 1.0 reproduces the baseline bit for bit.
- `run_baselines_seeds_cpu.sh` — produced `runs/seed_1..4/`; `run_experiment.sh` replaces it.
