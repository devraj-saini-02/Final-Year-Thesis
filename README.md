# OAL-GCF 

Reproducing *The Value of Corrective Feedback in the Online Active Learning Paradigm*
(Lindsey et al., IEEE TPAMI 2026) on spoken language verification, 20 target languages across four
corpora, and testing two extensions of our own: a diversity penalty on query selection, and
**OAL-GCF** (grouped corrective feedback), which cuts the expert's corrective-feedback labelling.

**Running the experiments? Read [GUIDE.md](GUIDE.md).** It covers how to run an experiment and how to compare
results, and assumes no knowledge of the code. To rebuild the data, read [DATA.md](DATA.md).

```bash
source venv/bin/activate
python tools/compare_runs.py --list                     # everything that has been run
python tools/compare_runs.py oal oalcf                  # does corrective feedback help?
python tools/compare_runs.py oalcf oalcf+cfd0.1 --seeds 0 1 2 3 4 --metric IMLM_thesis   # does OAL-GCF help?
bash  tools/run_experiment.sh                           # usage for starting new runs
```

## Sources

- **Base paper:** M. Lindsey et al., *The Value of Corrective Feedback in the Online Active Learning
  Paradigm*, IEEE TPAMI, 2026. https://doi.org/10.1109/TPAMI.2025.3639522
- **Original code** (the paper's footnote 1), which we started from: https://github.com/markrl/oalcf.
  `oalcf-main/` is that repository; its first commit is the code as downloaded, and every change we
  made is a separate commit on top.
- **Author's thesis:** M. Lindsey, *Online Active Learning with Corrective Feedback: A New Paradigm
  Applied to Streaming Audio*, PhD thesis (ECE), Carnegie Mellon University. We use its Eq. 3.8 as the
  `IMLM_thesis` metric.
- **Data:** Mozilla Common Voice 17.0 (Amharic from 26.0). See [DATA.md](DATA.md).

The PDFs are not in this repository: get the paper from its DOI and the thesis from Carnegie Mellon's
library, and put them in `docs/sources/`.

## Our results are included

`runs/` holds every result we produced: `scores.csv` and `command.txt` per run, plus the comparisons,
reports and figures. They are here for reference, so you can check our numbers without re-running
anything. Model weights (`state_dict.pt`), training logs and `al_samples.txt` are left out (27 GB).

**To re-run a setting yourself, first delete our copy of it**, e.g.
`rm -r runs/cf_dedup/oalcf_cfd0.1/seed_2/`, or pass `--force`. `tools/run_experiment.sh` skips any
run that already has a `scores.csv`, so without that step it would do nothing and you would be
reading our numbers, not yours.

## Map

| path | what |
|---|---|
| [GUIDE.md](GUIDE.md) | how to run and compare experiments, and where the project stands — **start here** |
| [DATA.md](DATA.md) | how the four corpora were built, what is included, how to rebuild them |
| [runs/INVENTORY.md](runs/INVENTORY.md) | every run that exists, what it scored, where it lives (auto-generated) |
| [tools/README.md](tools/README.md) | what each script is for |
| `docs/` | `sources/`: put the paper and thesis PDFs here (not included, see Sources). `other/`: `REPRO_BRIEF.md`, `DEVIATIONS.md`, `CLUSTER_OBSERVATIONS.md` |
| `data/` | the four corpora; only the stream-order files are included, the rest is rebuilt per [DATA.md](DATA.md) |
| `oalcf-main/` | the authors' released code, as a git repo, with our fixes as visible commits; our extensions are `run_dq.py` and `run_cfdedup.py` |
| `runs/` | results by category (`baseline/`, `diversity_extension/`, `cf_dedup/`), then setting, seed, language; write-ups in `runs/reports/`, figures in `runs/figures/` — see [runs/README.md](runs/README.md) |

## Where it stands

- **The reproduction holds.** Averaged over stream-order seeds 1–4, baseline OAL scores 0.1063 against
  the paper's 0.1015. Over five seeds corrective feedback beats OAL in 20/20 languages (paper: 17/20).
  See `runs/reports/baseline_seeds__RESULTS.md`.
- **Extension 1 (diversity penalty) does not hold up across seeds**, and the reason is measured:
  `runs/analysis/2026-10-10_oalcf_cf/README.md`
- **Extension 2, OAL-GCF, cuts corrective-feedback labelling by 30%** over five seeds, improves IMLM by
  6% on the equations that charge expert labels (18/20 languages, p = 0.002), and shows no significant
  accuracy change. See `runs/comparisons/cf_dedup/oalcf_vs_cfdedup_seeds0-4/SUMMARY.txt` and the
  figures in `runs/figures/oal_gcf/`.
- Open directions are listed at the end of [GUIDE.md](GUIDE.md).

Everything trains on CPU: the Apple GPU backend computes wrong gradients for this model
(`docs/other/DEVIATIONS.md`). Scripts enforce this; do not bypass them.

macOS (Apple Silicon) throughout. Start scripts with `bash`, not zsh — see the macOS notes in
[GUIDE.md](GUIDE.md).
