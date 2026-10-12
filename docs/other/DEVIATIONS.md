# Deviations from the paper / released code

Append-only log. Each entry: what changed, why, and what it should/shouldn't affect.

> **Reset, 2026-09-12.** Every run made before this date trained on Apple MPS with PyTorch
> 2.3.0, whose BatchNorm backward is wrong for this model (see "Training device" below). All
> of those results were deleted, and the entries that interpreted them were removed with them.
> The corpus, feature and order-file entries below are unaffected: none of them depend on
> training. All results from here on come from CPU runs.

---

## Training device: CPU, not MPS (PyTorch 2.3.0 MPS BatchNorm gradient bug)

**Finding.** On PyTorch 2.3.0 the MPS backend computes wrong weight and bias gradients for a
`BatchNorm1d` whose input does not require grad. The model's first layer, `bn_layer`
(`src/model.py`, applied to the raw ECAPA features), is exactly that case, so every MPS training
step fed that layer a wrong gradient.

**Evidence.**
- Minimal repro, independent of this repo: `tools/legacy/repro_mps_batchnorm_grad.py`. Random data
  through BN -> Linear -> BN -> LeakyReLU -> Linear: the MPS first-BN gradients were wrong in
  30/30 calls (|bias grad| about 2 against CPU's about 2e-10). With `input.requires_grad_(True)`,
  the only change, they were wrong in 0/30.
- A training step saved from inside a run and replayed from identical model state, optimizer
  state and batch gave a first-BN bias gradient of about 1e-10 on CPU and 0.15 to 52 on MPS,
  varying between repetitions. Inside long MPS runs the same gradient reached 1.2e19 and then
  inf; Adam turns an infinite gradient into NaN weights (inf/inf), after which the model
  predicts non-target for everything. That happened in 19 of 20 OAL-CF runs and 0 of 20 OAL
  runs. OAL-CF trains 4-5x more steps on larger pools, so it had more exposure. The non-NaN
  runs were affected too: `bn_layer` drifted to bias 0.4-0.8 and weights -0.45 to 2.36, where
  a correct run keeps them near 0 and 1.
- Running the same configuration on CPU gave finite weights throughout, with the first-layer
  bias staying below 4e-4.

**Change.** All training runs go through `tools/run_cpu.py`, which resolves every device request
to CPU and appends `--gpus 0` (so `command.txt` records it). `--gpus 0` alone is not enough: it
moves only the Lightning Trainer, while the query-strategy, corrective-feedback and drift helpers
call `utils.device.get_device()` with no argument and would still use MPS. No repo code changed.
The Commit A MPS portability changes stay in the repo but are not used for results. Decision
(2026-09-12): keep the wrapper rather than a repo portability commit, so no repo code changes
and nothing about the method can be affected.

**Cost.** Measured on the same Armenian OAL-CF sessions, CPU takes 1.18x the MPS time per
session. Two CPU jobs in parallel produced `scores.csv` files bit-identical to a solo run, at
1.31x the throughput of one job.

**What it affects.** Numerics only, not the method. CPU results are not bit-identical to the
authors' CUDA runs (neither would MPS be), but they are correct. Side effect:
`torch.set_flush_denormal(True)` (paper Sec. V-D, Commit B) now applies, since it controls the
CPU's floating-point mode and I believe has no effect on GPU kernels.

---

## Scoring: the repo's IMLM differs from the paper's Eq. 1 (both are reported)

`scoring/print_basic_results.py:19` computes

    IMLM = (N_fp + N_pool) / N_nontarget + N_fn / N_target

where `N_pool` is the whole final adaptation pool: bootstrap + AL queries + every CF-labelled
sample, **including CF-confirmed true positives**. The paper's Eq. 1 (Sec. IV-A) is

    IMLM = (N_ann + N_fp) / N + N_fn / N_target

with `N` the total number of samples in the stream and `N_ann` = "bootstrap samples, AL queries,
and false positives". Confirming a correct detection is the expert's normal work and is not
charged. The repo's formula matches the dissertation's Eq. 3.6 (`N_adapt` = all adaptation
samples), not the paper.

Eq. 1 has two readings for OAL-CF. Read literally, a CF-corrected false positive counts twice:
once in `N_ann` as an annotation, once in `N_fp` as an error. The paper's wording in Sec. II
("every false positive error, including those that are later corrected by the expert, is
counted equal to an AL query") suggests once.

**What it affects.** Only how results are scored. The OAL arm is barely affected. For OAL-CF on
a high-prevalence target, the repo's formula charges thousands of true-positive confirmations
as labelling, and that can reverse the OAL vs OAL-CF ordering. On a CPU validation run of
Armenian (deleted in the reset) the two scorers gave opposite verdicts.

**Decision (2026-09-12, human): report both, computed from the same runs.** No extra runs are
needed, because every term of both formulas is already in `scores.csv`.
`tools/score_runs.py` prints `IMLM_repo` (identical to `print_basic_results.py`, which is left
untouched) and `IMLM_eq1`. Compare against Fig. 12 only via `IMLM_eq1`.

**Update (2026-09-13, human): `IMLM_eq1` defaults to the authors' reading.** The authors' own
plotting code (`plotting/trend_plot.py`, postquential branch) computes Eq. 1 as
`(N_fp + N_boot + N_AL) / N + N_fn / N_target`: every false positive counted once, including
those CF corrects, and CF-confirmed true positives not charged. `--literal` gives the literal
reading (CF-corrected false positives also in `N_ann`, so counted twice). **Timing, for the
record:** the authors' formula was found before any baseline had finished, but the default was
switched after all 40 baseline results had been seen. Both readings and the repo scorer are
reported for every run (`runs/baseline/*/seed_0/`), so no reading is hidden.

**One correction to the authors' code:** `trend_plot.py` takes the bootstrap size as
`n_samples[0] - n_al[0]`, which for OAL-CF also counts session 0's CF labels (e.g. 297 instead
of 8). We use the true bootstrap of 8 (paper Sec. V-D). This changes IMLM by at most about 0.003.

---

## Framing note (added when CR/SEA/SA work began): identification vs. construction

AC's corpus was **identified**: African Continent is a real, bounded region, a plausible
candidate locale list existed, and the evidence (per-language checksums, release ranking)
converged on one answer within ~4-5%. CR, SEA, and SA are different in kind, not degree:
the paper's region labels are not strict geography (CR's own named targets include
Serbian [Balkans] and Kyrgyz [Central Asia]; SA's include Hakha Chin [`cnh`, Myanmar]), no
public code or supplementary material defines the bundles, and the background locale sets
required to hit Table I's totals undershoot every geographically-plausible candidate list
by 10-45% — an order of magnitude worse than AC's gap. With 100+ un-placed locales in the
release, subset-sum over them is unfalsifiable by the standard set in the AC entry below
("subset-sum will always find something that hits one window").

**Consequence:** AC's background is *identified* (real locales, evidence-based). CR/SEA/SA's
backgrounds are *constructed*: geographic core (the locales that plausibly belong, same
method as AC) topped up by a seeded random draw from the remaining catalog to reach Table
I's total sample count and session count, holding the paper-specified target locales and
their exact counts untouched. This preserves prevalence (which enters DCF directly as the
prior) and corpus size, at the cost of the non-target identity being partly arbitrary — a
different, larger, and explicitly-labeled uncertainty than AC's.

The original plan treated composition as a sensitivity axis with three variants per corpus
(geographic-only, geographic+fill, random-fill). **Only geographic+fill has been built.** The
other two variants do not exist yet. Construction details: "CR/SEA/SA corpus construction" below.

---

## CR/SEA/SA corpus construction (geographic core + seeded filler)

Built by `tools/corpus/assemble_cr_sea_sa_corpus.py` and `tools/corpus/fill_cr_sea_sa.py` from the
`fsicoli/common_voice_17_0` mirror (the same release as AC). Test splits use Common Voice's own
split assignment.

- **Geographic core:** whole test splits of every locale that plausibly belongs to the region,
  targets included (lists in `assemble_cr_sea_sa_corpus.py`).
- **Filler:** a seeded draw (`random.Random(0)`) from a shared pool of seven out-of-region
  locales (`eo eu hu tr nl fa ar`, 83,576 test clips), sized to bring each corpus to Table I's
  test total exactly. The clips are symlinked into each corpus's `test/`. Draws are independent
  per corpus, so a filler clip can appear in more than one corpus.

| | CR | SEA | SA |
|---|---|---|---|
| Test clips (= Table I exactly) | 114,480 | 90,720 | 68,400 |
| Geographic core | 102,976 | 52,812 | 37,207 |
| Filler (share of corpus) | 11,504 (10.0%) | 37,908 (41.8%) | 31,193 (45.6%) |
| Targets vs Table II | 7/7 match to 2 d.p. | each about +1% | 6/7 within 1%; `or` 696 vs about 1,005 (entry below) |
| Est. test hours (Table II) | 148.5 (162) | 112.7 (146) | 95.0 (119) |

The hours are **estimates**, computed from each locale's release-wide `avgDurationSecs` in the
cv-dataset metadata, not measured from the clips. For comparison, AC comes to 80.5 h (82). The
SEA and SA shortfalls of about 20% are consistent with the constructed backgrounds differing
from the paper's.

**`al_samples.txt` cannot identify the queried clips (upstream bug, found 2026-09-17).**
`src/dataset.py:140-147` (`save_active_files`) writes, for each active base index `ii`,
the filename `feat_files[int(ii / samples_per_batch)]` and the number `ii % samples_per_batch`.
The first term indexes the *file list* by a batch number, so it names whichever file happens to sit
at position `floor(ii/720)` rather than the clip actually selected; the second is a position inside a
session, not a session number. Checked on `cr_oal/tt`: all 1,280 lines name Armenian files, the 161
distinct names all resolve into the first 200 entries of the **train** order (the bootstrap head), and
0 of 1,280 reconstructed indices land inside the test stream. The mapping is lossy, so **which clips a
run queried is not recoverable from any archived run** - only counts are (`n_al` in `scores.csv`,
`al_diversity.csv` for cluster-level behaviour). Not fixed: `run.py` and `src/dataset.py` are kept
byte-identical to upstream, and the file is not used by any scoring path. Any future experiment that
needs query identities must log them separately in `utils/diverse_query.py`.

**Toki Pona in the SEA core (found 2026-09-14).** The SEA core list
(`tools/corpus/assemble_cr_sea_sa_corpus.py:35`) includes `tok`, Toki Pona, a constructed language with
no region, with no recorded reason. It contributes 2,179 test clips (2.4% of SEA), and its
`train.tsv` is in the SEA train-order pool. So SEA's out-of-region share is 44.2%
(37,908 filler + 2,179 tok), not the 41.8% in the table. Left in place: removing it would change
the SEA stream and invalidate every SEA run, and it acts as one more background locale, like the
filler. The SEA core is also broad in a documented way: it spans East Asia (ja, ko, mn, zh-CN,
zh-TW) as well as Southeast Asia.

**Train order pool.** Each corpus's train order is built from the full `train.tsv` of its target
+ core locales **plus all seven filler locales** (`tools/corpus/build_train_filelists.py`), not the
drawn filler subset: CR 852,434, SEA 489,583, SA 470,814 filenames (Table I train 666,720 /
1,363,680 / 797,040). Only the bootstrap head is extracted (`tools/corpus/assemble_crseasa_bootstrap.py`):
CR 89, SEA 43, SA 89 train clips. That includes the context clips at the train/test seam and the
array-wraparound positions that `get_sample_with_context` loads even though they are later zeroed.
So some bootstrap non-target samples are filler-language clips.

**Checks (2026-09-12).** 0 decode failures in every corpus. Every order entry has a feature file;
no non-finite, all-zero or duplicate embeddings; no broken symlinks; no train/test clip overlap.
ECAPA's own language head classifies the stored features of its known languages correctly (e.g.
ha 95%, hy 92%, all seven filler locales 51-95%), so features match their audio and labels.

---

## SA target list: `or` (Odia) does not match Table II at any Common Voice release

Table II states Odia at 1.47% of the SA test split (~1005 of 68,400) — identical to
Saraiki's stated 1.47%. Leave-one-out over SA's 7 target checksums against cv-corpus-17.0
shows 6 of 7 match within <1% (cnh 0.53%, dv 0.59%, hi 0.45%, ml 0.71%, mr 0.40%, skr
0.50%) while `or` alone is off by 30.75% (696 actual vs. 1005 wanted). Checked `or`'s test
count across all 25 Common Voice releases with a `datasets/scripted-speech` entry (2020-2026):
it never once reaches within 12% of 1005 (closest: 881 at 19.0, 12.3% off; every other release
is 30%+ off, monotonically worse moving away from 17.0 in either direction). There is no release
at which Odia matches its stated Table II percentage.

**Conclusion: this is not a release-identification problem, it's a target-list problem
specific to `or`.** Most likely explanation: a transcription artifact in Table II (Odia's
neighboring row, Saraiki, shares the identical "1.47" and does match CV cleanly at 1010/68400
= 1.477%; Odia's actual CV17.0 rate is 696/68400 = 1.02%, suggesting the true figure may have
been closer to 1.0% and something produced a duplicate/copied digit). No alternative Common
Voice locale code for Odia exists to swap in.

**Decision: stay on cv-corpus-17.0 for SA** (same snapshot as AC/CR/SEA — switching to 19.0
for a 12.4%-vs-30.8% improvement driven by one mismeasured locale would abandon a real,
strong signal — 6/7 targets within 1% — for a worse one, and would break having one CV
snapshot across all four SLV corpora, which matters since CV regenerates splits
non-deterministically between releases). Keep all 7 SA target locales including `or`; its
real CV17.0 rate (1.02%) is used as its actual weight in the constructed corpus rather than
the paper's stated 1.47%. Recorded as an unresolved anomaly in the paper's own Table II, not
a bug in this reproduction.

---

## Stream order is seeded (was unseeded upstream)

`utils/org_cluster.py` called `np.random.shuffle` and `np.random.dirichlet` with no seed
(original lines 37-38, 44, 54-55). Every invocation produced a different stream order, and
§VI-E of the paper shows order materially affects results, so the paper's exact ordering
cannot be recovered regardless of what we do.

**Change:** added `--seed` (default `0`), applied via `np.random.seed`/`random.seed` before
the shuffle/dirichlet draws, printed and written to `orders/seed.txt`.

**Why this is acceptable:** seeding does not change the distribution the paper specifies
(Dirichlet concentration `delta=1`, `dirichlet_numchunks = number of locales`,
`min_size_thresh=10` — none of these appear in the paper either, they're read from the
source). It only makes a single run reproducible against itself.

**What it affects:** the exact session-by-session order of clips in every run — a nuisance
variable, not a parameter the paper's numbers were tuned against. Do not read small
differences from the paper's Fig. 12 numbers as failures on this basis.

Note: `org_cluster.py` builds its file list with an unsorted `glob`, so the same seed
reproduces the same order only for the same directory listing. All current order files
were built once and are reused; do not regenerate them.

---

## AC seed order files 1-10: test order reshuffled, train order fixed at seed 0

`data/AfricanContinentLID/orders/seed{1..10}_order_test` are the test stream reshuffled under
seeds 1-10. `seed{1..10}_order_train` are **byte-identical copies** of seed 0's
`cluster_order_train`, so every seed shares seed 0's 8-sample bootstrap and only the order the
test stream is presented in varies. Select with `--order_file seed{N}`.

**Why:** reshuffling the train order too would very likely move the non-target bootstrap anchor
into a different background locale. `rw` (Kinyarwanda, 1,003,028 train clips, multi-GB shards) is
the largest risk, and it would need new train-shard downloads per seed, with no way to know the
size until the order exists. The paper does not say bootstrap composition must co-vary with
stream order. **Record this as a deviation in any multi-seed result.**

**CR / SEA / SA seed order files 1-4 (built 2026-09-17), same scheme.** `data/{CaucasusRegionLID,
SoutheastAsiaLID,SouthAsiaLID}/orders/seed{1..4}_order_test` are the test stream reshuffled with
`org_cluster.py --seed N --splits test`; `seed{1..4}_order_train` are byte copies of seed 0's
`cluster_order_train`. They were built in a scratch directory (org_cluster.py writes `orders/`
relative to the cwd), so no seed-0 file was touched. Checks: every seed file holds exactly the seed-0
clip set; rebuilding seed 0 reproduces all existing order files byte for byte, and AC seed 1 too.
Note for rebuilding: the CR/SEA/SA seed-0 orders were built in one call with `--splits train,test`
(`--train_list_file train_filenames.txt`), so their test order depends on the random state left by
the train shuffle; `--splits test` alone does NOT reproduce them. AC seed 0 was built test-only.
The seeds 1-4 were fixed in advance (not chosen by looking at the orders). First target position,
as % of the stream, for the record: AC seed 0 32-34%, seeds 1-4 0-12%; CR seed 2 has cv/kmr/sr
at 19-20% (other seeds 0-8%); SEA and SA 0-9% in every seed.
Batch script ready but **not run**: `bash tools/legacy/run_baselines_seeds_cpu.sh 1 2 3 4` (160 runs,
~50 h), summary `tools/summarize_seeds.py`.

---

## AC bootstrap composition (external, train-derived, `--boot_in` not passed)

Computed from `orders/cluster_order_train` (1,278,122 lines) with the same
`int(max(idxs)/720)+1` logic as `dataset.py`: `n_boot_batches` = **6** for bas and ha (a
4,320-sample train prefix) and **23** for yo (16,560). Only the 8 bootstrap indices (first 4
non-target + first 4 target) plus their ±3-sample context windows are ever loaded, which is why
only 43 train `.npy` files exist. One of them required downloading a `kab` (Kabyle) train shard
(~1 GB), because seed 0's shared non-target anchor (train-order positions 0-3) falls in `af`/`kab`.

---

## Operational: use `--nworkers 0` for every run (not a paper deviation)

Not a fidelity change — `--nworkers` isn't a paper hyperparameter, it's a PyTorch DataLoader
setting. Recorded here because it changes how every run must be launched.

**Finding:** with the default `--nworkers 4`, `run.py` was measured taking 15-30s per epoch
(one spike to ~4 min) on a trivial single-batch, <30-sample session fit — clearly dominated
by fixed overhead, not compute. Root cause: `reset_trainer()` recreates the `Trainer` (and
therefore its `DataLoader`s) on every session, and PyTorch Lightning spins up `nworkers` new
worker subprocesses on every single `fit`/`test` call rather than reusing them. With
`patience=15` on real data that can mean dozens of epochs per fit, so the respawn cost is
paid dozens of times per session.

Re-ran the identical 10-session synthetic corpus with `--nworkers 0`: all 10 sessions
completed in **11 seconds**, sub-second per epoch. Confirms the earlier multi-hour pace was
a DataLoader-worker-respawn artifact, not the model or data pipeline being slow.

**Consequence:** pass `--nworkers 0` on every `run.py` invocation. This is purely an
engineering flag — it does not touch the algorithm, the data, or any paper-specified
hyperparameter, so it carries no scientific risk.

---

## Corpus: release and locale list are unspecified in the paper, both inferred

Neither the paper nor the thesis states which Common Voice release or which locale set
makes up "African Continent" — checked exhaustively (repo has no corpus-build script;
thesis Table 4.4 is identical to the paper's Table II, no fuller locale list anywhere in
either document, no CV version/date cited in either bibliography — reference is the
generic 2019 Ardila et al. corpus paper in both).

**Release:** `tools/corpus/find_ac_corpus.py` ranked all 52 numbered CV releases by how closely
their per-locale test-split counts match Table II's bas/ha/yo prevalence (0.95/1.16/1.75%,
i.e. ≈540/660/995 clips out of 56,880). `cv-corpus-16.1` (2023-12-06) and `cv-corpus-17.0`
(2024-03-15) are statistically indistinguishable on this checksum (worst per-language error
0.3% vs 0.4%; every other release tested is ≥1.4%, most far worse). **17.0 was chosen for
availability** — `mozilla-foundation` only hosts `common_voice_13_0` and `common_voice_17_0`
on Hugging Face; 16.1 isn't published there. 13.0 fails Stage 1 badly (≥1.4% error on this
checksum) and was never a candidate on merits. The audio was fetched from the third-party
`fsicoli/common_voice_17_0` mirror; its byte-identity with Mozilla's official release was not
verified.

**Locale list:** the paper's rule is "languages found in" the African Continent region,
never an enumerated list. No subset of any plausible candidate locale set reproduces
Table I's exact 56,880-clip test total in either release (closest: +2,420 over, in 16.1) —
the total is dominated by four large locales (Swahili, Kinyarwanda, Luganda, Kabyle) whose
sizes (12k-16k clips each) are far coarser than the 720-clip session granularity, so no
combination lands in that narrow a window. Chose not to force a fit by dropping any of
these on the grounds it would make the reproduction unfalsifiable (any 4% discrepancy could
be "explained away" post hoc by cutting whichever locale closes the gap). Used the full set
of 15 African-continent locales with actual data in the release (`bas ha yo ig sw rw lg am
af kab zgh dyu ti tig tw`) instead, accepting a ~4% larger background pool than the paper's.

Cross-check supporting this call: bas+ha+yo alone sum to 2,201/56,880 = 3.87% in Table II's
own terms; in the chosen 15-locale set at both 16.1 and 17.0 they sum to 2,201/59,040 (post-
truncation) ≈ 3.73%, and the per-language rates individually (bas 0.92%, ha 1.12%, yo
1.69% at 17.0) land inside 0.9-1.8% -- the paper's stated regime for rare-target selection.
Agreement on the *language* checksum to within a few hundredths of a percent, while only the
*background* count is off, points at CV split regeneration between the authors' download and
this metadata snapshot (plus the exact 79×720 truncation implying the authors also cut to a
whole number of sessions) rather than a wrong locale or release choice.

**What it affects:** exact non-target class balance and total stream length; IMLM/DCF are
cost-per-sample metrics, not pooled totals, so this does not on its own predict a directional
bias toward either paradigm. Record as a deviation in the Phase 7 write-up; do not read small
numeric differences from Fig. 12 as failures on this basis alone.

(Session-count and prevalence figures above were superseded once `am` was added from a
different release — see next entry for the final, correct numbers.)

---

## Corpus: `am` (Amharic) sourced from a different release than the other 14 locales

`am` does not exist on `fsicoli/common_voice_17_0` (confirmed 404 on both `transcript/am/*`
and `audio/am/test` — not an empty directory, the paths don't resolve at all). The official
`mozilla-foundation/common_voice_17_0` repo on Hugging Face is empty (README only; Mozilla
moved distribution to Mozilla Data Collective, Oct 2025) so there was no gated-but-available
fallback either.

**Source used:** Mozilla Data Collective, `Common Voice Scripted Speech 26.0 - Amharic`
(`cv-corpus-26.0-2026-06-12`), fetched via the official `datacollective` Python SDK. MDC
bundles every split (train/dev/test/other/invalidated) plus all transcripts into one archive
per language with no split-level access (confirmed by inspecting the SDK's `download_dataset`
signature and source — no split/manifest parameter exists) — unlike `fsicoli`'s per-split
blobs. For `am` this is harmless: the whole bundle is 62 MB. Only the 281 rows listed in that
archive's `test.tsv` were extracted to disk; train/dev/other/invalidated audio inside the same
archive was left in the `.tar.gz`, never extracted.

(This bundling is also why the corpus didn't simply move to 26.0 entirely: Kinyarwanda's
26.0 archive is 61.4 GB with no way to fetch only its test split, confirmed directly via the
API before deciding against it.)

**Consequence — `am` is ~2.25 years newer than the other 14 locales.** Its test-split size
grew from 205 clips (17.0, per the official CV release metadata) to 281 clips (26.0) over
that period — continuous crowdsourced growth, consistent with the release-identification
argument above (later releases move further from Table II, not closer). `am` is a background
locale, not a target (`bas`/`ha`/`yo` are all sourced from 17.0, untouched by this), so it
doesn't affect the checksum that identified the release.

**Final assembled corpus (all 15 locales, 14 from 17.0 + am from 26.0):**

| | clips |
|---|---|
| raw test total | 59,778 |
| sessions after truncation (÷720) | 83 |
| clips used / dropped | 59,760 / 18 |

bas 541 (0.905%), ha 661 (1.106%), yo 999 (1.672%) of 59,760 — still inside the paper's
stated 0.9-1.8% rare-target regime; these supersede the 82-session/59,040-clip figures in the
previous entry, which were computed before `am` was added.

**How the 18 are dropped:** not on disk. All 59,778 clips are in `test/`, `ecapalang/test/` and
the order file. `dataset.py` floors to 83 whole sessions, so the last 18 entries of the seed-0
order are never presented: 17 `lg` and 1 `yo`. Yoruba is therefore effectively 998 clips in
use. The paper does not say how it truncated.

---
