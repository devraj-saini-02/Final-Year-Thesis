# INVENTORY — every run in this repo

Generated 2026-10-12 by `python tools/run_inventory.py`. Re-run it after any new batch; do not edit by hand.

**932 runs**, 26 settings, seeds 0–4. Metric below is IMLM_eq1 (the paper's Eq. 1), lower is better.

One run = one target language, one algorithm, one seed: 20 languages across four corpora (African 3, Caucasus 7, Southeast Asia 3, South Asia 7).

**Health:** 932/932 runs have their full session count (African 83, Caucasus 159, Southeast Asia 126, South Asia 95).

## 1. What exists (languages finished, out of 20)

| setting | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | total |
|---|---|---|---|---|---|---|
| `oal` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oal@0.16/1` | 1 | — | — | — | — | 1 |
| `oal@0.16/0.975` | **20** | — | — | — | — | 20 |
| `oal@0.16/0.93` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oal@0.18/1` | 1 | — | — | — | — | 1 |
| `oal@0.18/0.95` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oal@0.18/0.86` | **20** | — | — | — | — | 20 |
| `oal@0.2/0.9` | — | 3 | — | — | — | 3 |
| `oal@0.25/1` | 3 | — | — | — | — | 3 |
| `oal@0.25/0.8` | **20** | — | — | — | — | 20 |
| `oal@0.25/0.5` | **20** | — | — | — | — | 20 |
| `oal@0.25/0.2` | **20** | — | — | — | — | 20 |
| `oalcf` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oalcf+cfd0` | — | 1 | — | — | — | 1 |
| `oalcf@0.1/0.97` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oalcf@0.1/0.9` | — | **20** | — | — | — | 20 |
| `oalcf+cfd0.1` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oalcf@0.12/0.9` | **20** | — | — | — | — | 20 |
| `oalcf@0.16/1` | 1 | — | — | — | — | 1 |
| `oalcf@0.16/0.93` | **20** | **20** | **20** | **20** | **20** | 100 |
| `oalcf@0.18/1` | 1 | — | — | — | — | 1 |
| `oalcf@0.18/0.95` | **20** | — | — | — | — | 20 |
| `oalcf@0.25/1` | 1 | — | — | — | — | 1 |
| `oalcf@0.25/0.8` | **20** | — | — | — | — | 20 |
| `oalcf@0.25/0.5` | **20** | — | — | — | — | 20 |
| `oalcf@0.25/0.2` | **20** | — | — | — | — | 20 |

`oal` = active learning only. `oalcf` = plus corrective feedback (the paper's method). `@t/g` = our extension at cluster threshold t and decay gamma g. **gamma 1.0 runs are controls**: the penalty is inert, so they must reproduce the baseline bit for bit, and they did at every threshold.

## 2. How each setting scored

Each extension setting is compared against its own baseline (`oal` or `oalcf`) on the seeds where both exist, one row per setting. 'better' counts languages that improved.

| setting | seeds used | better | mean before → after | p | clusters/session | queries changed | uncertainty given up |
|---|---|---|---|---|---|---|---|
| `oal@0.16/1` | 0 | control | penalty inert; reproduces the baseline exactly (1 run checked) | | | | |
| `oal@0.16/0.975` | 0 | 14/20 | 0.1576 → 0.1352 (-14%) | 0.0825 | 121 | 1.08 of 8 | 0.3% |
| `oal@0.16/0.93` | 0,1,2,3,4 | 14/20 | 0.1165 → 0.1027 (-12%) | 0.0348 ** | 121 | 1.41 of 8 | 1.1% |
| `oal@0.18/1` | 0 | control | penalty inert; reproduces the baseline exactly (1 run checked) | | | | |
| `oal@0.18/0.95` | 0,1,2,3,4 | 10/20 | 0.1165 → 0.1204 (+3%) | 0.5938 | 70 | 1.31 of 8 | 0.7% |
| `oal@0.18/0.86` | 0 | 9/20 | 0.1576 → 0.1520 (-4%) | 0.6494 | 70 | 1.90 of 8 | 1.9% |
| `oal@0.2/0.9` | incomplete (3/20 languages) | | | | | | |
| `oal@0.25/1` | 0 | control | penalty inert; reproduces the baseline exactly (3 runs checked) | | | | |
| `oal@0.25/0.8` | 0 | 11/20 | 0.1576 → 0.1561 (-1%) | 0.4492 | 10 | 1.35 of 8 | 0.7% |
| `oal@0.25/0.5` | 0 | 13/20 | 0.1576 → 0.1246 (-21%) | 0.1559 | 10 | 2.13 of 8 | 2.3% |
| `oal@0.25/0.2` | 0 | 11/20 | 0.1576 → 0.1449 (-8%) | 0.2979 | 10 | 2.58 of 8 | 5.6% |
| `oalcf+cfd0` | incomplete (1/20 languages) | | | | | | |
| `oalcf@0.1/0.97` | 0,1,2,3,4 | 6/20 | 0.0656 → 0.0780 (+19%) | 0.9336 | 448 | 0.35 of 8 | 0.2% |
| `oalcf@0.1/0.9` | 1 | 7/20 | 0.0721 → 0.0732 (+2%) | 0.9232 | 449 | 0.50 of 8 | 0.5% |
| `oalcf+cfd0.1` | 0,1,2,3,4 | 6/20 | 0.0656 → 0.0686 (+5%) | 0.7339 |  | |  |
| `oalcf@0.12/0.9` | 0 | 10/20 | 0.0763 → 0.0795 (+4%) | 0.4782 | 310 | 0.64 of 8 | 0.8% |
| `oalcf@0.16/1` | 0 | control | penalty inert; reproduces the baseline exactly (1 run checked) | | | | |
| `oalcf@0.16/0.93` | 0,1,2,3,4 | 7/20 | 0.0656 → 0.0717 (+9%) | 0.9587 | 121 | 0.94 of 8 | 0.9% |
| `oalcf@0.18/1` | 0 | control | penalty inert; reproduces the baseline exactly (1 run checked) | | | | |
| `oalcf@0.18/0.95` | 0 | 10/20 | 0.0763 → 0.0755 (-1%) | 0.3108 | 70 | 0.80 of 8 | 0.6% |
| `oalcf@0.25/1` | 0 | control | penalty inert; reproduces the baseline exactly (1 run checked) | | | | |
| `oalcf@0.25/0.8` | 0 | 15/20 | 0.0763 → 0.0762 (-0%) | 0.0164 ** | 10 | 0.86 of 8 | 0.7% |
| `oalcf@0.25/0.5` | 0 | 11/20 | 0.0763 → 0.0744 (-3%) | 0.3506 | 10 | 1.31 of 8 | 2.4% |
| `oalcf@0.25/0.2` | 0 | 10/20 | 0.0763 → 0.0941 (+23%) | 0.7392 | 10 | 1.88 of 8 | 5.7% |

** = significant at 0.05. Read these with care (GUIDE.md, 'Where the project stands'): a seed-0-only row is one stream order and has repeatedly proved misleading.

## 3. Baselines per corpus (mean over the seeds each has)

| algorithm | seeds | African Continent | Caucasus Region | Southeast Asia | South Asia | all 20 |
|---|---|---|---|---|---|---|
| oal | 0,1,2,3,4 | 0.1282 | 0.1052 | 0.1784 | 0.0964 | 0.1165 |
| oalcf | 0,1,2,3,4 | 0.0317 | 0.0649 | 0.1099 | 0.0618 | 0.0656 |

For reference, the paper's own values average 0.1015 over the 20 languages (`docs/other/REPRO_BRIEF.md` section 7).

## 4. Where a run lives on disk

```
runs/<category>/<setting>/seed_<seed>/<corpus>_<language>/
```

For example:

```
runs/baseline/oal/seed_0/cr_tt      # oal, seed 0, Caucasus Tatar
runs/diversity_extension/oal_t0.16_g0.93/seed_0/cr_tt      # oal@0.16/0.93, seed 0, Caucasus Tatar
```

Each run folder holds `scores.csv` (per-session results, the source of every number above), `command.txt` (the exact command), `state_dict.pt` (final weights), `train.log`, and for extension runs `al_diversity.csv` (clusters and queries per session).

## 5. Language codes

- **African Continent** (`ac`, 83 sessions): `bas`, `ha`, `yo`
- **Caucasus Region** (`cr`, 159 sessions): `hy-AM`, `ckb`, `cv`, `kmr`, `ky`, `sr`, `tt`
- **Southeast Asia** (`sea`, 126 sessions): `yue`, `id`, `vi`
- **South Asia** (`sa`, 95 sessions): `cnh`, `dv`, `hi`, `ml`, `mr`, `or`, `skr`

## 6. Regenerate / explore

```bash
python tools/run_inventory.py                       # rewrite this file
python tools/compare_runs.py --list                 # same coverage, printed
python tools/compare_runs.py oal oal@0.16/0.93 --seeds 0 1 2 3 4   # any comparison
```
