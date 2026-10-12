# SLV baselines: OAL vs OAL-CF, 20 languages, seed 0 (CPU)

Run 2026-09-12 20:59 to 2026-09-13 09:28 (12.5 h, two jobs at a time) with
`bash tools/legacy/run_baselines_cpu.sh`. **40/40 runs healthy** (`health.txt`): every session
completed, no NaN, finite final weights, all on CPU.

**Setup.** Corpora as built in `docs/other/DEVIATIONS.md` (cv-corpus-17.0; CR/SEA/SA backgrounds
constructed with filler). Stream order seed 0. Paper settings (Table IV, `--auto_weight
--auto_mult 10 --min_delta 1e-3`), 8 AL queries per session, uncapped CF for OAL-CF
(`--sim_type fpstps`), `--nworkers 0`, training on CPU via `tools/run_cpu.py` (MPS BatchNorm
bug, see DEVIATIONS). Each run's command is in `output/<corpus>_<arm>/<lang>/command.txt`.

**Scoring.** Both scorers come from the same `scores.csv` (no extra runs):
- **IMLM, Eq. 1:** the paper's formula as the authors' own plotting code computes it (each
  false positive counted once, CF-confirmed true positives not charged, bootstrap = 8).
  Default in `tools/score_runs.py`. The default was switched to this reading after these
  results had been seen; see DEVIATIONS "Scoring". The literal reading is also reported.
- **IMLM, repo scorer:** `scoring/print_basic_results.py`, unchanged upstream.

Paper values are read off Fig. 12 by eye (±0.005, `docs/other/REPRO_BRIEF.md` §7).

---

## Headline: Tables V and VII

| Metric (mean over 20 languages) | OAL | OAL-CF | OAL-CF lower in | One-sided Wilcoxon p |
|---|---|---|---|---|
| **IMLM, Eq. 1 (authors' reading)** | 0.1576 | **0.0763** | **16/20** | **2.4e-3** |
| IMLM, Eq. 1 (literal) | 0.1576 | 0.0837 | 13/20 | 1.8e-2 |
| IMLM, repo scorer | 0.1658 | 0.1093 | 10/20 | 2.0e-1 (not significant) |
| DCF | 0.1112 | 0.0473 | 15/20 | 1.8e-3 |
| Paper (Table V / VII, SLV) | 0.105 | 0.049 | 17/20 (Fig. 12 caption) | IMLM 1.8e-3, DCF 1.8e-3 |

- **Under the paper's Eq. 1, the central claim reproduces.** OAL-CF beats OAL in 16 of 20
  languages, significant at the 99% level, with the mean IMLM halved (0.158 -> 0.076, -52%;
  the paper's Table V is -53%). DCF agrees independently (15/20, p = 1.8e-3). Our absolute
  means are about 50% above the paper's. The paper does not say how Table V pools, so the
  comparison of plain means is itself an assumption.
- **Under the repo scorer it does not.** OAL-CF wins only 10 of 20 and the test is not
  significant, because that scorer charges every CF-confirmed true positive as labelling work.
  The headline therefore depends on the scorer. Both were committed to before the results.

---

## Per language: IMLM, Eq. 1 (authors' reading), OAL -> OAL-CF

| Corpus | Lang | Ours | Paper Fig. 12 | Ours | Paper | Agrees? |
|---|---|---|---|---|---|---|
| AC | bas | 0.577 -> 0.021 | 0.042 -> 0.036 | CF wins | CF wins | yes |
| AC | ha | 0.310 -> 0.026 | 0.020 -> 0.038 | CF wins | **CF loses** | **no** |
| AC | yo | 0.300 -> 0.028 | 0.057 -> 0.012 | CF wins | CF wins | yes |
| CR | hy-AM | 0.035 -> 0.023 | 0.032 -> 0.028 | CF wins | ≈ tie | yes (tie) |
| CR | ckb | 0.056 -> 0.056 | 0.100 -> 0.030 | CF wins (-0.4%) | CF wins | yes |
| CR | cv | 0.138 -> 0.136 | 0.305 -> 0.210 | CF wins (-1.4%) | CF wins | yes |
| CR | kmr | 0.191 -> 0.130 | 0.185 -> 0.025 | CF wins | CF wins | yes |
| CR | ky | 0.163 -> 0.064 | 0.130 -> 0.070 | CF wins | CF wins | yes |
| CR | sr | 0.031 -> 0.014 | 0.018 -> 0.026 | CF wins | **CF loses** | **no** |
| CR | tt | 0.222 -> 0.111 | 0.087 -> 0.033 | CF wins | CF wins | yes |
| SEA | yue | 0.356 -> 0.259 | 0.290 -> 0.185 | CF wins | CF wins | yes |
| SEA | id | 0.064 -> 0.031 | 0.013 -> 0.013 | CF wins | ≈ tie | yes (tie) |
| SEA | vi | 0.053 -> 0.094 | 0.083 -> 0.110 | CF loses | CF loses | yes |
| SA | cnh | 0.020 -> 0.018 | 0.070 -> 0.065 | CF wins (-8%) | CF wins | yes |
| SA | dv | 0.057 -> 0.085 | 0.073 -> 0.035 | **CF loses** | CF wins | **no** |
| SA | hi | 0.149 -> 0.179 | 0.183 -> 0.045 | **CF loses** | CF wins | **no** |
| SA | ml | 0.180 -> 0.067 | 0.093 -> 0.080 | CF wins | CF wins | yes |
| SA | mr | 0.088 -> 0.073 | 0.017 -> 0.017 | CF wins | ≈ tie | yes (tie) |
| SA | or | 0.101 -> 0.036 | 0.098 -> 0.087 | CF wins | CF wins | yes |
| SA | skr | 0.063 -> 0.074 | 0.135 -> 0.035 | **CF loses** | CF wins | **no** |

Direction agrees with the paper for 15 of 20 (paper ties counted as agreeing). All scorers,
DCF, FNR and FPR per language: `runs/figures/base_paper_reproduction/tables.md` and `.../per_language.csv`.

---

## Observations

1. **Five disagreements with the paper.**
   - Two of the paper's three named CF losses (Hausa, Serbian) come out as CF wins. The third,
     Vietnamese, is reproduced.
   - Three of our four CF losses are in SA (Dhivehi, Hindi, Saraiki). SA is 46% filler. That is
     a hypothesis, not a finding: an offline check before the reset found filler making up only
     12-34% of these targets' false alarms, below its 46% share of the stream (Hindi's main
     confusers are Urdu and Punjabi). Testing it needs the geographic-only SA variant.
   - Our four CF losses are not near-ties: +78% (vi), +50% (dv), +21% (hi), +18% (skr). The
     near-ties are among the wins: ckb (-0.4%), cv (-1.4%), cnh (-8%).
2. **AC OAL is far worse than the paper's** (0.30-0.58 against 0.02-0.06). Fig. 4 shows why:
   in the seed-0 order, AC's rare targets arrive in a few dense bursts, and OAL, with 8 queries
   per session, cannot adapt within a burst. That inflates CF's advantage on AC and flips Hausa
   to a CF win. A stream-order effect, not the MPS bug.
3. **Class balance (Fig. 9) matches the paper's description.** OAL-CF's training pool settles
   at about 50-60% target while OAL's stays near 10%, in every corpus. AC's OAL-CF pool is near
   0% until about session 26 because no AC target arrives before then (Fig. 4).
4. **Absolute values scatter around the paper's.** Some land on it (hy-AM, ky); some are above
   (tt, ml OAL, mr, id) or below (cv, cnh). Expected with reconstructed corpora and one order.

---

## Figures (`runs/figures/base_paper_reproduction/`, convergence in `runs/figures/convergence/`)

| File | Paper | Notes |
|---|---|---|
| `fig04_stream_order.png` | Fig. 4 | AC test stream, target positions: Dirichlet (seed 0, used) vs a uniform shuffle (display only) |
| `fig07_trends.png` | Fig. 7 | Postquential IMLM, prequential IMLM and their difference per session, mean over the 20 languages. The mean steps where a corpus's stream ends (sessions 83, 95, 126, dotted) |
| `fig08_oalcf_pre_post.png` | Fig. 8 | OAL-CF prequential vs postquential IMLM and DCF. The paper's post-CF curve is not recorded by the repo |
| `fig09_prevalence.png` | Fig. 9 | Target prevalence in the adaptation pool per session, per corpus |
| `fig12_imlm_eq1.png` | Fig. 12 | Final IMLM (Eq. 1) per language, OAL vs OAL-CF |
| `fig12_imlm_repo.png` | — | Same with the repo scorer |
| `fig12_vs_paper.png` | — | Ours next to the paper's Fig. 12 read-off |
| `convergence/<corpus>_<lang>.png` | — | Every session: cumulative DCF, IMLM (Eq. 1) and IMLM (repo), OAL vs OAL-CF |
| `convergence/<corpus>_mean.png` | — | Same, averaged over the corpus's languages |

Not produced: Fig. 6 (needs the budget sweep), Fig. 8's post-CF curve (not recorded by the
repo), Fig. 10 (needs reverse and random-order runs), Fig. 11 (VTD, out of scope).

---

## Caveats

- One stream order (seed 0). Decided 2026-09-13 not to add seeds for now; point 2 above shows
  the order can matter.
- Reconstructed corpora: AC identified (5% larger background, Amharic from release 26.0);
  CR/SEA/SA constructed with 10%/42%/46% filler. See DEVIATIONS.
- CPU numerics are not bit-identical to the authors' CUDA runs.
- Paper per-language values are chart read-offs (±0.005).

## Reproduce

    bash tools/legacy/run_baselines_cpu.sh                      # the 40 runs (resumable), about 12.5 h
    python tools/score_runs.py runs/seed_0/output/*/*               # both scorers
    python tools/score_runs.py --literal runs/seed_0/output/*/*     # literal Eq. 1
    python tools/paper_figures.py                        # runs/figures/base_paper_reproduction/
    python tools/plot_convergence.py                     # runs/figures/convergence/

Files here: `output/` (the 40 runs: `scores.csv`, `command.txt`, `state_dict.pt`,
`al_samples.txt`), `logs/` (per-run logs), `scores.csv` / `scores.txt` (Eq. 1 authors' reading
+ repo scorer), `scores_eq1_literal.csv`, `health.txt`, `progress.log`.
