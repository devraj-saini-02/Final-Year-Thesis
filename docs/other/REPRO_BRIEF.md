# OAL-CF Reproduction Brief — AC-SLV, OAL vs OAL-CF

Working brief for Claude Code. Written after reading the paper (incl. Tables I/II/IV/V
and Figs 6/12, which do not survive PDF text extraction) and the full `oalcf-main` source.

Everything below marked **[read]** comes from the paper or the source directly.
Everything marked **[inferred]** is reasoning that should be checked, not trusted.

---

## 0. Scope

Reproduce OAL vs OAL-CF on Spoken Language Verification, African Continent corpus only.
Targets: `bas`, `ha`, `yo`. PML out of scope.

Working rules in force: fix what blocks the paper's method; don't change the method;
portability and fidelity fixes in separate commits; paper beats README; don't tune toward
the paper's numbers; don't delete dead code; don't download datasets unprompted.

---

## 1. Verified against the paper

| Item | Source | Status |
|---|---|---|
| AC test split = 56,880 utterances | Table I **[read]** | Printed, not derived. = 79 × 720 exactly |
| AC train 982,800 / val 51,840 | Table I **[read]** | 1365 and 72 sessions |
| AC duration = 82 hours | Table II **[read]** | ⇒ ~5.19 s mean clip. Third checksum |
| bas 0.95% / ha 1.16% / yo 1.75% | Table II **[read]** | ≈ 540 / 660 / 995 test clips |
| Adam, lr 1e-4, wd 1e-5, batch 256, patience 15, margin 1e-3 | Table IV **[read]** | `params.py` matches all but margin |
| ψ = 0.09 | Eq. 5 **[read]** | `--xent_weight` default 0.09 ✓ |
| 1024 hidden / 2 comp layers / 3 class layers / 256 embed | Fig. 5 **[read]** | `params.py` defaults match ✓ |
| context = 3 either side (7 total) | §V-B **[read]** | `--context` default 3 ✓ |
| AL budget 8, bootstrap 8, 720/session | §V-D **[read]** | `params.py` defaults match ✓ |

**11 of the 12 cells in Table I are exact multiples of 720** (AC 1365/72/79 sessions;
SEA 1894/156/126; SA 1107/92/95; CR train 926, test 159 — only CR val, 112,719, is not).
The corpora were truncated to whole sessions **at construction time**, not by the code.
See §4.

---

## 2. Established findings — confirmed in source

| # | Finding | Confirmation |
|---|---|---|
| 1 | triton is unused | No import anywhere in the tree ✓ |
| 2 | `utils/ddm.py` crashes on Mac | `ddm.py:23,28` — unconditional `.cuda()`. Called every session from `run.py:194` ✓ |
| 3 | `--auto_weight --auto_mult 10` required | `params.py:57,59` — defaults `False`/`1.0`. `utils/utils.py:161-162` implements Eq. 4 exactly, with `mult` on the target class ✓ |
| 4 | Don't pass `--max_fb_samples` | `params.py:117` — default `None` = no limit. Just omit the flag ✓ |
| 5 | `--feat_root` must contain `LID` | `src/dataset.py:20` — `elif 'LID' in params.feat_root` ✓ |
| 6 | `org_cluster.py` hardcodes `wavlm/` | `utils/org_cluster.py:12` ✓ |
| 7 | `--min_delta` typed int | `params.py:33` — `type=int, default=0`. Table IV wants 1e-3 ✓ |
| 11 | CF second fit reuses stale class weights | `run.py:237` fits with no `update_xent` call between `transfer_samples` (230) and `fit` (237). Algorithm as written — **do not fix** ✓ |
| 12 | `euclidean_distance` has an unguarded `sqrt` | `utils/utils.py:39-41`: `sqrt(sum(diff²))` with no epsilon, so an exactly-zero pair distance gives a NaN gradient. Latent: not triggered in practice (closest pair seen 0.012). Upstream code — **do not fix**. The paper's Eq. 3 specifies cosine distance, `Dcos ∈ [0,1]`, not Euclidean ✓ |
| 13 | Scorer ≠ paper's Eq. 1 | `scoring/print_basic_results.py:19` charges the whole adaptation pool (incl. CF-confirmed true positives) as annotations, divided by non-targets. Eq. 1 charges bootstrap + AL + false positives, divided by all samples. Matches the dissertation's Eq. 3.6 instead. The authors' own `plotting/trend_plot.py` implements Eq. 1 with each false positive counted once; `tools/score_runs.py` follows that by default. See `docs/other/DEVIATIONS.md` ✓ |
| 14 | MPS BatchNorm gradient bug | In PyTorch 2.3.0, not the repo: wrong weight/bias gradients for a BatchNorm whose input does not require grad, which is `bn_layer`. All training runs on CPU via `tools/run_cpu.py`. Repro: `tools/legacy/repro_mps_batchnorm_grad.py`. See `docs/other/DEVIATIONS.md` ✓ |

Minor correction to the carried-forward notes: finding #8 cites Table IV for 720
samples/session. Table IV does not contain it — it is in §III-A and §V-D prose. The
number is right.

---

## 3. What changes the plan (new)

### 3.1 The filename convention is decoded

`org_cluster.py:15` does `ff.split('_')[2]` on a path like `ecapalang/test/<name>.npy`;
`dataset.py:283` does `line.split('_')[2]` on the order-file line. Both resolve the
locale from the **third underscore-delimited field**. That matches Common Voice's native
clip naming, `common_voice_{locale}_{id}.mp3`. **[inferred, high confidence]**

So the required on-disk layout is:

```
<corpus_root>/                       # must contain the literal substring "LID"
  ecapalang/{train,dev,test}/common_voice_{locale}_{id}.npy
  orders/cluster_order_{train,dev,test}
```

and `--feat_root <corpus_root>/ecapalang/`. `dataset.py:278` computes the corpus root as
`feat_root/..`, so the trailing path element must be the feature name.

Guard: this parse breaks if any path component before the basename contains `_`. Keep
directory names underscore-free.

### 3.2 Corpus construction is not in the repo

There is no Common Voice → `AfricanContinentLID/` script anywhere in the tree. The
feature extractors assume a directory of MP3s already split into `train/dev/test`. So the
unreleased parts are not just "which release" and "which locales" — they also include the
split assembly and the truncation to a whole number of sessions.

This is the largest reproduction gap. It has to be rebuilt from the paper's rule
("languages found in the region", using CV's own splits) plus the three checksums in §1.

### 3.3 The stream order is not reproducible

`org_cluster.py` calls `np.random.shuffle` and `np.random.dirichlet` with **no seed**
(lines 37-38, 44, 54-55). Every invocation produces a different stream order. §VI-E of the
paper shows order materially affects results.

Consequence: the paper's exact ordering cannot be recovered. Recommended handling —
seed the script, record the seed, and treat order as a nuisance variable. Seeding does not
change the distribution the paper specifies, so this deviates from the released code but
not from the paper. It must still be written down as a deviation.

Dirichlet parameters, for the record: `delta = 1` (concentration),
`dirichlet_numchunks = number of locales in the corpus`, `min_size_thresh = 10`. None of
these appear in the paper.

**One order file serves all three languages.** The order is built over the whole corpus;
`--lid_target` only decides which locale maps to label 1. So `bas`, `ha`, and `yo` runs
share a single stream, and so does every point in the budget sweep. Generate it once.

### 3.4 Bootstrap needs more of the train split than expected

`dataset.py:330-349` (`load_external_bootstrap`) does the following:

1. Builds a second `BaseLidData` on `env_name='train'` — this **requires
   `orders/cluster_order_train` to exist and list the entire train split**.
2. Scans forward from index 0 until it has 4 target + 4 non-target.
3. Sets `n_boot_batches = floor(max(idx)/720) + 1`.
4. **Prepends the first `n_boot_batches × 720` train entries** to the test file list.

So features are needed for the first `n_boot_batches × 720` train clips, not for 8 clips.
With ~1% target prevalence in a class-clustered order, that could be anywhere from 720 to
several thousand. It is knowable exactly once the train order file exists.

`next_batch` (`dataset.py:128-129`) offsets session windows by `n_boot_batches`, and
`n_batches` is fixed at 79 before the prepend, so the session accounting stays correct.

There is a bootstrapping circularity to break: `org_cluster.py` globs `*.npy`, so the
train order file cannot be built until train features exist, but we only want train
features for the head of that order file. Resolution: build the train order from the MP3
file list (or the CV TSV) rather than the `.npy` list. The order file content is
identical either way — it is derived from filenames only.

### 3.5 Only `ddm.py` actually crashes

Every other CUDA call is guarded by `if torch.cuda.is_available():`
(`query_strategies.py:189,215,281,300,308`; `corrective_feedback.py:88`). On a Mac those
guards are False, so the model and data stay on CPU — **correct, just slow**. PyTorch
Lightning returns the module to CPU during teardown after each `fit`/`test`, so there is
no device-mismatch error.

The real portability work is `run.py:94-95`:

```python
accelerator='gpu' if params.gpus>0 else 'cpu',
devices=[0],
```

`devices=[0]` is invalid for MPS — it needs `devices=1`.

**Update (2026-09-12):** MPS runs without crashing but computes wrong gradients for the
model's first BatchNorm layer (finding #14). All training now runs on CPU via
`tools/run_cpu.py`. The MPS portability work stays in the repo but is not used for results.

### 3.6 `torch.set_flush_denormal(True)` is missing from the code

§V-D of the paper says denormals are flushed to zero. The call appears **nowhere in the
repo**. Add it (fidelity commit).

Caveat: on Apple Silicon `torch.set_flush_denormal(True)` may return `False` — the
underlying control is x86 MXCSR. Check the return value and log it. If it returns False,
that is an unfixable divergence from the paper on this hardware and belongs in the
write-up, since it applies across 79 × 2 fits per run.

### 3.7 Feature I/O will dominate runtime

`dataset.py:314-328` (`get_sample_with_context`) issues **7 separate `np.load()` calls per
`__getitem__`**. Per run that is roughly 7 × 720 × (pre-test + fit epochs + AL scoring +
post-test) × 79 sessions — on the order of 10⁶ file opens.

The whole AC test feature set is 56,880 × 256 × 4 B ≈ **58 MB**. Caching every `.npy` in
a dict at dataset construction is bit-identical and turns a multi-hour run into a
multi-minute one. Strongly recommended, as a separate performance commit.

### 3.8 Small operational notes

- `run.py:52` uses `os.mkdir`, not `makedirs`. `--run_name ac/bas` requires `output/ac/`
  and `checkpoints/ac/` to exist first. The shipped scripts do this; a bare `python run.py`
  does not.
- `--sim_type fpstps` ⇒ OAL-CF. Omitting `--sim_type` (default `None`) ⇒ OAL.
- `--lid_target` selects the language (`params.py:159`, default `ha`).
  `scripts/run_aclid_parallel.sh` never sets it, so it silently runs Hausa only.
- `--load_best` defaults False, so the model carried forward is the last-epoch state, not
  the checkpointed best. Early stopping monitors `train/loss` (`params.py:27`).
- `n_batches = int(len(ds)/720)` (`dataset.py:26`) floors, so a partial final session is
  dropped. With 56,880 nothing is lost.

---

## 4. Corpus identification — do this on metadata first

Do **not** download audio yet. Common Voice ships per-locale TSVs; the split row counts
alone are enough to solve both unknowns.

For each candidate CV release, and for the set of African locales in it, search for a
locale subset where:

1. `floor(sum(test rows) / 720) × 720 == 56,880`, i.e. raw test total in [56,880, 57,599]
2. `bas / ha / yo` land at 0.95% / 1.16% / 1.75% of 56,880 (≈ 540 / 660 / 995 clips)
3. total test duration ≈ 82 h (⇒ ~5.19 s mean clip)
4. train total ≈ 982,800 and val ≈ 51,840, under the same truncation rule

Constraint worth checking early **[inferred]**: AC train is 982,800 utterances ≈ 1,365 h.
Kinyarwanda (`rw`) alone has exceeded that in CV releases since roughly CV 9. So either
the release is early, or `rw` is excluded from AC, or splits were capped. Verify against
real release statistics rather than trusting this recollection.

Only after a candidate locale list + release is pinned should audio be downloaded, and
then only for: the full test split, plus the head of the train split identified in §3.4.

---

## 5. Work order

### Commit A — portability (no behaviour change)

1. `requirements.txt`: drop `triton==2.3.0`.
2. `run.py:94-95`: device selection that resolves cuda → mps → cpu, with `devices=1` for
   MPS. Keep `--gpus` semantics.
3. `utils/ddm.py:23,28`: replace unconditional `.cuda()` with the shared device helper.
4. `utils/query_strategies.py`, `utils/corrective_feedback.py`: same helper in place of the
   `torch.cuda.is_available()` guards, so MPS is used instead of falling back to CPU.
5. `utils/extract_ecapa.py:18,38`: `run_opts={"device": ...}` and the `.cuda()` call.

Verification: `--debug` run must complete without a device error.

### Commit B — paper fidelity

1. `src/params.py:33`: `--min_delta` `type=int` → `type=float`. Leave the default at 0;
   pass `--min_delta 1e-3` on the command line so the diff stays minimal and the paper
   value is visible in `command.txt`.
2. `run.py`: add `torch.set_flush_denormal(True)` near line 26, and log the return value.

### Commit C — SLV wiring

1. `utils/org_cluster.py:12`: `wavlm` → `ecapalang`. Make it a parameter rather than a
   second hardcode if that is cheap.
2. Add an explicit seed, printed and written alongside the order file.
3. Allow the train order to be built from a filename list rather than a `.npy` glob
   (§3.4 circularity).

### Commit D — performance

In-memory feature cache in `BaseLidData`. Must be verified bit-identical against the
uncached path on a small slice before it is trusted.

### Then

4. Corpus identification (§4), metadata only.
5. Smoke run: one language, `--debug`, then a truncated run of a handful of sessions,
   both paradigms.
6. Baselines: 3 languages × 2 paradigms at budget 8.
7. Budget sweep — scope to be decided.

---

## 6. Baseline command shape

```
python run.py \
  --run_name ac_oalcf/bas \
  --env_name test \
  --lid_target bas \
  --feat_root <ROOT>/AfricanContinentLID/ecapalang/ \
  --sim_type fpstps \
  --auto_weight --auto_mult 10 \
  --min_delta 1e-3
```

Drop `--sim_type fpstps` for the OAL arm. Do not pass `--max_fb_samples`. `--n_queries`
defaults to 8 and is the sweep variable. Create `output/ac_oalcf/` and
`checkpoints/ac_oalcf/` first.

---

## 7. Expectations, so divergence is readable

- Table V (0.105 / 0.049) is pooled across all four SLV corpora. AC-only will not match it.
- Fig. 12 per-language, read off the plot: bas ≈ 0.042 → 0.036 (CF wins),
  ha ≈ 0.020 → 0.038 (**CF loses**), yo ≈ 0.057 → 0.012 (CF wins). §VII states the Hausa
  loss explicitly. Reproducing the Hausa loss is success, not failure. (Corrected
  2026-09-12: the first read-off had ha ≈ 0.028 → 0.033 and yo ≈ 0.065 → 0.032.)
  All 20 languages are in the table below.

### Fig. 12 read-off, all 20 languages (IMLM, OAL → OAL-CF)

Read by eye from the figure rendered at 170 dpi; treat each value as ±0.005. The caption says
OAL-CF has lower IMLM for 17 of 20; §VII names the exceptions as Hausa, Serbian and Vietnamese.

| Corpus | Lang | OAL | OAL-CF | | Corpus | Lang | OAL | OAL-CF |
|---|---|---|---|---|---|---|---|---|
| AC | bas | 0.042 | 0.036 | | SA | cnh | 0.070 | 0.065 |
| AC | ha | 0.020 | 0.038 | | SA | dv | 0.073 | 0.035 |
| AC | yo | 0.057 | 0.012 | | SA | hi | 0.183 | 0.045 |
| CR | ckb | 0.100 | 0.030 | | SA | ml | 0.093 | 0.080 |
| CR | cv | 0.305 | 0.210 | | SA | mr | 0.017 | 0.017 (≈ tie) |
| CR | hy-AM | 0.032 | 0.028 | | SA | or | 0.098 | 0.087 |
| CR | kmr | 0.185 | 0.025 | | SA | skr | 0.135 | 0.035 |
| CR | ky | 0.130 | 0.070 | | SEA | id | 0.013 | 0.013 (≈ tie) |
| CR | sr | 0.018 | 0.026 | | SEA | vi | 0.083 | 0.110 |
| CR | tt | 0.087 | 0.033 | | SEA | yue | 0.290 | 0.185 |

Which IMLM formula produced the figure is not stated beyond Eq. 1; the repo's scorer differs
from Eq. 1 (finding #13), so compare against these values only with the Eq. 1 scorer.
- Fig. 6's "OAL-CF's worst budget beats OAL's best" is a 20-language pooled claim with a
  margin of roughly 10% by eye. It may not survive restriction to three languages.
- Wilcoxon at n=20 across languages has no n=3 equivalent.
- Basaa is not in VoxLingua107 (§V-B lists the 9 that are), so the ECAPA embedding has no
  Basaa in its training data. Expect it to behave differently from ha and yo. Do not fix.
- The stream order differs from the paper's by construction (§3.3).
