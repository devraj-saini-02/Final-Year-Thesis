# GUIDE — how to run experiments and compare results

Written for whoever takes over this project. No prior knowledge of the code is
assumed, and nothing here needs an AI assistant: two commands do everything, one to run an experiment
and one to compare two experiments.

## Cheat sheet

```bash
cd oalcf-repro                    # the repository root, wherever it is
source venv/bin/activate

python tools/compare_runs.py --list                  # what has already been run
python tools/compare_runs.py oal oalcf               # compare two settings (saved automatically)
bash   tools/run_experiment.sh                       # usage for starting runs

# start a long batch so it survives closing the Terminal
nohup bash tools/run_experiment.sh --algo oalcf --seeds 1 2 3 4 > /dev/null 2>&1 &
tail -f runs/baseline/oalcf/seed_1/progress.log      # watch it
pgrep -fl run_cpu.py                                 # is anything running?
pkill -f run_experiment.sh; pkill -f run_cpu.py      # stop everything (safe: re-run to resume)
```

Nothing is destructive: re-issuing a command skips runs that already exist, and only `--force`
overwrites. If you get stuck, every command prints its own usage when run with no arguments.

If you only read two sections, read **[Part 1](#part-1--run-an-experiment)** and
**[Part 2](#part-2--compare-results)**.

---

## What this project is

We are reproducing a published paper — *The Value of Corrective Feedback in the Online Active Learning
Paradigm* (Lindsey et al., IEEE TPAMI, 2026; PDF in `docs/sources/`) — and testing two extensions of our own.

The task, **spoken language verification**: a stream of audio clips arrives in sessions of 720 clips.
One language is the *target*; the model must flag clips of that language. After each session an expert
labels a few clips and the model retrains. There are 20 target languages across 4 corpora.

Three things vary in our experiments:

| | what it means |
|---|---|
| **Algorithm** | **OAL** = the expert labels 8 clips the model is least sure about, each session. **OAL-CF** = the same, plus the expert corrects every clip the model flagged as target ("corrective feedback"). OAL-CF is the paper's contribution. |
| **Seed** | Which order the clips arrive in. The paper used one unrecorded order; we generate ours with seeds 0–4. A seed changes only the order, never which clips exist. |
| **Extension 1: diversity (ours)** | Cluster each session's clips, give each cluster a quota, and penalise picking many of the 8 AL queries from the same cluster. Controlled by **threshold** (how finely to cluster) and **gamma** (how hard to penalise). Did not hold up across seeds. |
| **Extension 2: OAL-GCF (ours)** | *Grouped corrective feedback.* Group the clips the model flags for corrective feedback into near-duplicates; the expert labels **one clip per group** and that label is copied to the rest. Controlled by one **threshold** (how similar clips must be to share a label). About 30% fewer expert labels. |

Lower scores are always better. They are error/effort costs, not accuracy.

---

## Setup (once)

```bash
cd oalcf-repro                    # the repository root, wherever it is
source venv/bin/activate          # Python 3.11 with everything installed
python tools/experiments.py       # prints every finished run; if this works, you are set up
```

On a fresh clone there is no `venv/` and no corpus yet. Create the environment with
`python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`, then build
the data as described in [DATA.md](DATA.md) (rule 4 below is about the machine where it already exists).

Hard rules for this machine:

1. **Everything trains on CPU.** Apple's GPU backend (MPS) computes wrong gradients for this model and
   silently produces garbage. Every script here forces CPU. Never add `--gpus 1` or run `run.py`
   directly without `tools/run_cpu.py`.
2. **Use `bash`, not `zsh`, to start scripts** (`bash tools/run_experiment.sh ...`).
3. **Two runs at a time** is the tested setting. More does not go faster on this machine.
4. **Do not re-download or rebuild the corpora.** `data/` holds 43 GB of audio and features that took
   days to assemble. Nothing you run for experiments writes to `data/`.

### macOS notes

Everything here was built and tested on macOS (Apple Silicon), so it should behave the same on your
Mac. A few specifics worth knowing:

- **The shell.** Your Terminal is zsh, but the scripts are bash, and macOS ships the old bash 3.2.
  Always start them with `bash tools/...` (or `/bin/bash tools/...`). Both are tested. Running one
  with `zsh` or `sh` will fail in confusing ways.
- **Sleep.** The scripts hold the machine awake with `caffeinate -ims`, tied to the script's own
  process, so it ends when the batch ends. Your **display still sleeps** — that is expected and does
  not stop anything. Closing the laptop lid does, so leave it open, and keep it plugged in.
- **Long batches.** Start them with `nohup ... &` (shown above) so closing the Terminal window does
  not kill them.
- **No `timeout` command** on macOS by default; do not add it to scripts. Use the built-in resume
  instead: stop with `pkill`, then re-issue the same command.
- **Check what is running:** `pgrep -fl run_cpu.py` (the training processes) or
  `pgrep -fl run_experiment.sh` (the batch). `pgrep -c` does not work here.
- **Python** comes from `venv/` in the repo root; activate it with `source venv/bin/activate`. Do not
  install a different Python or upgrade PyTorch — the CPU/MPS finding in `docs/other/DEVIATIONS.md` is tied
  to this version.

---

## Part 1 — Run an experiment

One command. Answer four questions and fill them in.

### Step 1: which algorithm?

| you want | use |
|---|---|
| active learning only (the weaker baseline) | `--algo oal` |
| active learning + corrective feedback (the paper's method) | `--algo oalcf` |

### Step 2: which seeds?

Seeds are stream orders, 0 to 4. Seed 0 is the original; 1–4 were added later.

- **Testing something quickly:** one seed, `--seeds 0`.
- **Making a claim you will write in the paper:** all five, `--seeds 0 1 2 3 4`.

One seed is not enough evidence on its own. A language's score moves by a median of 48% between seeds
(`runs/reports/baseline_seeds__RESULTS.md`), so a difference seen on a single seed can be pure luck.

Never choose which seeds to report after seeing the results. Decide the list first, run it, report all
of it.

### Step 3: baseline or our extension?

| you want | add |
|---|---|
| the algorithm exactly as published | nothing |
| extension 1, diversity (OAL or OAL-CF) | `--diverse --threshold T --gamma G` |
| extension 2, OAL-GCF (only with `--algo oalcf`) | `--cfdedup T` (we use **0.10**; 0 = no grouping = baseline OAL-CF) |

### Step 4: if using extension 1, which threshold and gamma?

**Threshold** sets cluster size. Lower = smaller, purer clusters.

| threshold | clusters per session | language purity | notes |
|---|---|---|---|
| 0.25 | ~10 | 0.28 | one huge cluster plus crumbs |
| 0.18 | ~71 | 0.74 | balanced |
| 0.16 | ~122 | 0.84 | finest setting tried with OAL |

**Gamma** sets penalty strength: 1.0 = no penalty at all (identical to the baseline), lower = stronger.
Note the same gamma bites harder at a lower threshold, because clusters are smaller.

Settings already tested, so you do not need to repeat them (IMLM Eq. 1 vs the same algorithm's baseline):

| algorithm | threshold | gamma | seeds | result |
|---|---|---|---|---|
| OAL | 0.16 | 0.93 | 0–4 | 14/20, −12%, p = 0.035, but carried by seed 0 (15/20, −32%); **seeds 1–4 alone: 12/20, −4%, p = 0.25** |
| OAL | 0.18 | 0.95 | 0–4 | 10/20, +3%, p = 0.59 (seed 0 alone had said 13/20, −10%) |
| OAL | 0.25 | 0.8 | 0 | 11/20, −1%, not significant |
| OAL-CF | 0.25 | 0.8 | 0 | 15/20, p = 0.016 — **never re-run on seeds 1–4** |
| OAL-CF | 0.16 | 0.93 | 0–4 | 7/20, +9%, p = 0.96 |
| OAL-CF | 0.10 | 0.97 | 0–4 | 6/20, +19%, p = 0.93 |
| OAL-CF | 0.18 / 0.12 | 0.95 / 0.9 | 0 | 10/20 and 10/20, not significant |

In short, extension 1 does not survive multiple seeds; why is in `runs/analysis/2026-10-10_oalcf_cf/README.md`.
Full list with every gamma: `python tools/experiments.py`.

### Step 4b: if using OAL-GCF, which threshold?

**0.10** is what was run on all five seeds: about 30% fewer corrective-feedback labels with 0.2% of labels
copied wrongly. Higher thresholds save more but copy more wrong labels (0.12: ~49% saved, 0.6% wrong;
0.16: ~78%, 1.7% wrong), measured offline in `runs/analysis/2026-10-10_oalcf_cf/cf_dedup_tables.txt`.
A new threshold should be tried on seed 1 first and confirmed on the other seeds.

### Putting it together

```bash
# baseline OAL-CF on seeds 1 and 2
bash tools/run_experiment.sh --algo oalcf --seeds 1 2

# our extension with OAL, threshold 0.16, gamma 0.93, on all five seeds
bash tools/run_experiment.sh --algo oal --seeds 0 1 2 3 4 --diverse --threshold 0.16 --gamma 0.93

# OAL-GCF (grouped corrective feedback) at threshold 0.10, on all five seeds
bash tools/run_experiment.sh --algo oalcf --seeds 0 1 2 3 4 --cfdedup 0.10

# quick test: one corpus only (Caucasus is the most informative single corpus)
bash tools/run_experiment.sh --algo oal --seeds 0 --corpora cr --diverse --threshold 0.16 --gamma 0.93
```

Optional: `--corpora ac cr sea sa` (default all four), `--jobs N` (default 2).

### How long it takes

Per seed, all 20 languages, two runs at a time:

| algorithm | time per seed | why |
|---|---|---|
| OAL | **~2.5 h** | one training pass per session |
| OAL-CF | **~10 h** | a second training pass per session, and a growing pool |
| OAL-GCF | **~10 h** | same as OAL-CF |

So five seeds of OAL is about half a day; five seeds of OAL-CF is about two days. The extensions cost
the same as their algorithm — clustering adds seconds, not hours.

Per corpus, roughly: Caucasus 60%, South Asia 20%, Southeast Asia 15%, African 5% of the total.

### While it runs

```bash
tail -f runs/<category>/<setting>/seed_<N>/progress.log     # e.g. runs/cf_dedup/oalcf_cfd0.1/seed_1
```

- The machine stays awake by itself. Your screen still sleeps, which is fine. **Keep it plugged in**;
  the batch stops if the battery dies.
- You can close the terminal. Runs keep going only if you started them detached (below).
- **Started detached** (survives closing the terminal):
  ```bash
  nohup bash tools/run_experiment.sh --algo oal --seeds 1 > /dev/null 2>&1 &
  ```
- **To stop everything:** `pkill -f run_experiment.sh; pkill -f run_cpu.py`
- **To resume:** run the exact same command again. Finished runs are skipped automatically.

**It never repeats work, and it never silently overwrites.** Before queueing, the script checks every
results folder and drops runs that already exist. Ask for something fully done and it says
`Nothing left to run`; ask for a half-finished setting and it queues only what is missing. So it is
always safe to re-issue a command.

If you *want* to redo a run — say you changed something in the code and the old result is stale — add
`--force`. That deletes the existing run folders for the settings you asked for and runs them again:

```bash
bash tools/run_experiment.sh --algo oal --seeds 0 --diverse --threshold 0.16 --gamma 0.93 --force
```

Without `--force` an existing result is always kept. This is deliberate: an OAL-CF seed costs about
10 hours, and an accidental repeat would overwrite finished work.

Results land in `runs/<category>/<setting>/seed_<N>/<corpus>_<language>/`, where the category is
`baseline/` (oal, oalcf), `diversity_extension/` or `cf_dedup/`. When the batch ends, its comparisons
against the matching baseline are saved automatically to `runs/comparisons/<category>/`.

---

## Part 2 — Compare results

```bash
python tools/compare_runs.py <A> <B> --seeds 0
```

B is what you are testing; A is what you compare it against. How to name a setting:

| spec | means |
|---|---|
| `oal` | baseline OAL |
| `oalcf` | baseline OAL-CF |
| `oal@0.16/0.93` | OAL + extension 1 (diversity), threshold 0.16, gamma 0.93 |
| `oalcf@0.25/0.8` | OAL-CF + extension 1 (diversity), threshold 0.25, gamma 0.8 |
| `oalcf+cfd0.1` | OAL-GCF: OAL-CF + extension 2 (grouped corrective feedback), threshold 0.10 |

### The comparisons worth making

```bash
# 1. Does corrective feedback help? (the paper's central claim)
python tools/compare_runs.py oal oalcf --seeds 0

# 2. Does an extension help the algorithm it is added to?
python tools/compare_runs.py oal oal@0.16/0.93 --seeds 0 1 2 3 4
python tools/compare_runs.py oalcf oalcf+cfd0.1 --seeds 0 1 2 3 4 --metric IMLM_thesis   # OAL-GCF

# 3. Is one extension setting better than another?
python tools/compare_runs.py oal@0.18/0.95 oal@0.16/0.93 --seeds 0
```

`python tools/compare_runs.py --list` shows every setting and seed that exists. For the full picture —
every run, what it scored, and where it sits on disk — see `runs/INVENTORY.md`. It refreshes itself at
the end of every batch; regenerate it by hand any time with `python tools/run_inventory.py`.

If you ask for seeds that only one of the two settings has, the tool says so and uses only the seeds
both have. Baseline OAL and OAL-CF, OAL-GCF, and four extension-1 settings exist on all five seeds;
the other extension-1 settings on seed 0 only (`python tools/compare_runs.py --list`).

### Where comparisons are saved

Every comparison is written to `runs/comparisons/<category>/` automatically (the category of the
setting being tested), one folder per comparison:

```
runs/comparisons/
  baseline/oal_vs_oalcf_seeds0_IMLM_eq1/
      report.txt     the full report, as printed
      table.csv      the per-language table
  diversity_extension/oal_vs_oal_t0.16_g0.93_seeds0-1-2-3-4_IMLM_eq1/
  cf_dedup/oalcf_vs_oalcf_cfd0.1_seeds2_IMLM_thesis/
```

The folder name records both settings, the seeds and the metric. Run the same comparison again and the
first folder keeps its plain name while the new one appears beside it as `..._1`, then `..._2`, so an
earlier result is never overwritten. Add `--no-save` to print without saving, or `--csv PATH` to put the
table somewhere specific as well.

### Reading the output

```
   oal@0.16/0.93 better in 15/20 languages
   mean 0.1576 -> 0.1065 (-32%)
   one-sided Wilcoxon (is oal@0.16/0.93 lower?): p = 0.0028   SIGNIFICANT at 0.05
```

- **Win count** — how many of the 20 languages improved. This is the paper's own headline format
  ("OAL-CF is better for 17 of 20 languages").
- **Mean** — averaged across languages. It can be dragged around by one disaster, so read it together
  with the win count, never alone.
- **Wilcoxon p** — the paper's significance test (§VI-G). It asks: if the setting made no difference,
  how often would we see a result this lopsided? Below 0.05 is the usual bar.
- **Per corpus** — the four regions separately. A pooled result driven by one corpus is not a general
  finding; ours mostly come from the Caucasus.

### Metrics

Default is `IMLM_eq1`, the paper's Equation 1 — the only metric comparable to the paper's figures. It
mixes error cost and expert effort into one number.

Others via `--metric`: `IMLM_repo` (the released code's slightly different formula), `DCF` (weighted
error rate), `FNR` (missed targets), `FPR` (false alarms). A result that holds on all of them is worth
more than one that appears in only one.

For OAL-GCF use **`IMLM_thesis`** (the author's thesis Eq. 3.8) or **`IMLM_repo_expert`**: they are the
only IMLM versions that charge the expert for every corrective-feedback label, so they are the ones that
can show a labelling saving. Eq. 1 counts confirming a correct detection as free, so for OAL-GCF it
measures accuracy only. Report it alongside, to show accuracy is unchanged.

### Rules for honest comparison

1. **Compare one seed against the same seed.** The tool does this for you. It also averages each
   language over seeds first, so five seeds never masquerade as 100 independent results.
2. **Wait for all 20 languages.** We twice read a partial result as the opposite of what it turned out
   to be — the last few languages flipped it.
3. **Write down what you expect before you run it.** `runs/reports/dq_oal_thresh__RESULTS.md` §5a–5c shows the
   format: state the prediction, then record what happened, including when it failed.
4. **Testing many settings inflates significance.** Testing 4 settings needs roughly p < 0.0125 rather
   than 0.05 (divide 0.05 by the number tried).
5. **The extensions' off-switch must reproduce the baseline exactly** (gamma 1.0 for extension 1,
   threshold 0 for OAL-GCF). It is a free correctness check on the extension code; it has always passed.

---

## Where things are

```
oalcf-repro/
  GUIDE.md            this file
  DATA.md             how the corpora were built and how to rebuild them
  docs/
    sources/          the paper and the author's PhD thesis (PDFs, not in git: links in README.md)
    other/
      REPRO_BRIEF.md  findings from reading the paper and the released code
      DEVIATIONS.md   every place we differ from the paper, and why
      CLUSTER_OBSERVATIONS.md   cluster behaviour, threshold/gamma tables
      email_to_author*.txt      correspondence with the paper's author (not in git)
  data/               the four corpora: audio, features, stream orders (43 GB, do not modify;
                      only the order files are in git, see DATA.md)
  oalcf-main/         the authors' released code (a git repo; our fixes are visible as commits)
  tools/              scripts you run (see tools/README.md); analysis and figure scripts in tools/analysis/
  runs/               results by category, setting, seed and language (see runs/README.md)
```

Results and write-ups:

| folder | what it holds |
|---|---|
| `runs/baseline/` | the paper's methods as published: `oal/`, `oalcf/` |
| `runs/diversity_extension/` | our first extension (cluster-quota AL), one folder per threshold/gamma |
| `runs/cf_dedup/` | our second extension (corrective-feedback deduplication) |
| `runs/comparisons/` | saved comparisons, grouped by the same three categories |
| `runs/reports/` | write-ups and summary tables (the `RESULTS.md` and `__PLAN.md` files) |
| `runs/analysis/` | cluster maps and threshold tables (`cluster_analysis/`), dated analyses with their data |
| `runs/figures/` | `base_paper_reproduction/` (the paper's figures, reproduced), `convergence/` (per-run error curves), `oal_gcf/` (OAL-CF vs our OAL-GCF) |

Inside a setting, one folder per seed (`seed_0` … `seed_4`), and inside that one folder per run named
`<corpus>_<language>`, holding `scores.csv`, `command.txt`, `state_dict.pt` and `train.log` (plus
`al_diversity.csv` or `cf_dedup.csv` for the extensions). The weights and logs are not in git, so a
fresh clone has only the scores. Full map: `runs/README.md`.

The write-ups worth reading:
- `runs/reports/baseline_seeds__RESULTS.md` — the reproduction (OAL over five seeds)
- `runs/comparisons/cf_dedup/oalcf_vs_cfdedup_seeds0-4/SUMMARY.txt` — OAL-GCF vs OAL-CF, five seeds, all metrics
- `runs/reports/cfdedup__PLAN.md` — OAL-GCF's plan, the success rule fixed before running, and outcomes
- `runs/analysis/2026-10-10_oalcf_cf/README.md` — why extension 1 fails with OAL-CF, and how OAL-GCF was designed
- `runs/reports/dq_oal_thresh__RESULTS.md` — extension 1 (diversity), seed-0 write-up
- `runs/reports/baseline_seed0__RESULTS.md` — the original 40 baseline runs

`tools/compare_runs.py` finds runs through `tools/experiments.py`, so you never need to type these paths.

---

## Where the project stands

1. **The reproduction works.** Averaged over seeds 1–4, our baseline OAL scores 0.1063 against the
   paper's 0.1015 — within 5%, with per-language rank correlation 0.81. On seed 0 alone it looked 55%
   too high, which was one unlucky stream order, not a bug. (`runs/reports/baseline_seeds__RESULTS.md`)
2. **Corrective feedback reproduces too:** over five seeds OAL-CF beats OAL in **20/20** languages
   (IMLM Eq. 1 −44%, DCF −50%); on seed 0 alone 16/20 (the paper says 17/20).
3. **Extension 1 (diversity) does not hold up.** Its best seed-0 results faded on seeds 1–4, and with
   OAL-CF every setting tried is null or worse. The reason is measured: the errors that dominate the
   score are confident misses that no choice of 8 queries reaches, and corrective feedback supplies 96%
   of the target labels anyway. (`runs/analysis/2026-10-10_oalcf_cf/README.md`)
4. **Extension 2, OAL-GCF, works on the cost the expert actually pays.** Over five seeds it cuts
   corrective-feedback labelling by **29.9%** (29.7–30.2% on every seed; ~109 of 364 listening hours) and
   improves IMLM by **6%** on the two equations that charge expert labels (thesis Eq. 3.8 and repo
   scorer with expert labels: **18/20 languages, p = 0.002**). Accuracy shows **no significant change**
   (Eq. 1 +4.6%, DCF +6%, two-sided p ≥ 0.4). The weak spots are Cantonese (yue), worse on 3 of 5 seeds,
   and one Yoruba run; both trace to a few wrong copied labels, mostly early in the stream.
   (`runs/comparisons/cf_dedup/oalcf_vs_cfdedup_seeds0-4/SUMMARY.txt`, figures in `runs/figures/oal_gcf/`)

### Open directions

- **Safeguards for OAL-GCF**, aimed at its wrong copies: no grouping in the first ~10% of sessions
  (about half of all wrong copies happen there), and never copying a "not the target" verdict. Design
  them on seed 1, confirm on the others, as in `runs/reports/cfdedup__PLAN.md`.
- **OAL-GCF at threshold 0.12 or 0.14**: larger saving, more wrong copies; worth measuring for real.
- **Extension 1 at OAL-CF 0.25/0.8 on seeds 1–4** (~40 h): its only good OAL-CF result was never re-tested.
- **Corrective-feedback cap.** The paper labels every flagged clip (what we run); the thesis's baseline
  caps corrective feedback at 16 clips per session. Which one the paper's numbers used is unresolved.
- **Model-seed noise control.** Every comparison varies the stream order only; the model's own random
  seed was never varied, so how much of a per-language difference is training noise is unmeasured.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `no order file for seed N` | that seed was never generated. See "Adding a seed" below. |
| `Nothing to compare` | that setting/seed was never run. `python tools/compare_runs.py --list`. |
| A run says `exit 1` in progress.log | read `runs/<category>/<setting>/seed_<N>/failed/<corpus>_<lang>.log`. Re-running the same command retries only the failed run. |
| `command not found: caffeinate` | you are not on macOS; delete that line from the script. |
| Comparison prints `INCOMPLETE` | fewer than 20 languages finished. Wait; partial reads have misled us before. |
| Something looks wrong in a result | check `runs/.../progress.log` for a failed run, and check session counts: 83 (African), 159 (Caucasus), 126 (Southeast Asia), 95 (South Asia). |

### Adding a seed

Seeds 0–4 exist. To add seed 5, from the repo root:

```bash
cd oalcf-repro                    # the repository root, wherever it is
source venv/bin/activate
ROOT=$PWD
SEED=5
for C in AfricanContinentLID CaucasusRegionLID SoutheastAsiaLID SouthAsiaLID; do
  WORK=$(mktemp -d)/$C
  mkdir -p "$WORK/orders"
  (cd "$WORK" && python "$ROOT/oalcf-main/utils/org_cluster.py" \
       --feat_dir "$ROOT/data/$C/ecapalang" --seed $SEED --splits test)
  cp "$WORK/orders/cluster_order_test" "$ROOT/data/$C/orders/seed${SEED}_order_test"
  cp "$ROOT/data/$C/orders/cluster_order_train" "$ROOT/data/$C/orders/seed${SEED}_order_train"
done
python tools/experiments.py      # sanity check that nothing else changed
```

Build it in a scratch directory as shown: `org_cluster.py` writes `orders/` relative to wherever you
are, and running it inside a corpus folder would overwrite seed 0's files. The train order is copied,
not regenerated, so every seed shares one bootstrap and only the arrival order changes
(`docs/other/DEVIATIONS.md`).

---

## One-line glossary

- **Session** — a block of 720 clips. The model retrains after each.
- **Query / budget** — the 8 clips per session the expert labels. Fixed at 8 everywhere here.
- **Corrective feedback (CF)** — the expert also corrects every clip the model flagged as target.
- **Bootstrap** — the 8 pre-labelled clips the model starts from.
- **IMLM** — the paper's cost metric, error plus expert effort. Lower is better.
- **DCF / FNR / FPR** — weighted error rate / missed targets / false alarms.
- **Target language** — the one language being detected in that run; 20 in total.
- **Threshold, gamma** — extension 1's two settings: cluster size and penalty strength.
- **OAL-GCF** — extension 2: grouped corrective feedback, one expert label per group of near-duplicate
  flagged clips (spec `oalcf+cfd0.1`, folder `runs/cf_dedup/`).
- **IMLM_thesis / IMLM_repo_expert** — the IMLM versions that charge every expert label, including
  corrective feedback; the right metrics for OAL-GCF.
