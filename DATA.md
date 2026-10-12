# DATA — how the four corpora were built

The paper evaluates on four spoken-language-verification corpora cut from Mozilla Common Voice:
African Continent (AC), Caucasus Region (CR), Southeast Asia (SEA) and South Asia (SA). It does not
publish them, and it names neither the Common Voice release nor the background locales. This file says
how we rebuilt them, what this repository includes, and what you have to download yourself.
Every decision below is argued in full in [docs/other/DEVIATIONS.md](docs/other/DEVIATIONS.md).

## What is in the repository and what is not

| | in the repo? | why |
|---|---|---|
| Stream-order files, `data/<corpus>/orders/` (`cluster_order_test`, `cluster_order_train`, `seed1..4_order_test`, `seed.txt`) | **yes** (~120 MB) | They fix exactly which clip arrives in which session. They cannot be regenerated reliably (below), so they are published. |
| Bootstrap train-clip lists (`data/{CR,SEA,SA}/bootstrap_needed_files.txt`, `docs/other/ac_bootstrap_train_files.txt`) | **yes** | the few train clips the model's 8-sample bootstrap needs |
| Audio (`test/`, `train/`, `*.mp3`) | no (~43 GB) | download from Common Voice (steps below) |
| ECAPA features (`ecapalang/`, one `.npy` per clip) | no | computed from the audio in step 4 |
| `seed1..4_order_train` | no | byte copies of `cluster_order_train`: `cp cluster_order_train seedN_order_train` |

`data/` is git-ignored apart from the files marked "yes". The training code expects each corpus at
`data/<CorpusName>LID/` (the name must contain `LID`), with `ecapalang/{train,test}/*.npy` and
`orders/` inside it.

## The corpora

| corpus | directory | target languages | test clips | sessions of 720 |
|---|---|---|---|---|
| AC | `AfricanContinentLID` | bas, ha, yo | 59,778 (59,760 used) | 83 |
| CR | `CaucasusRegionLID` | hy-AM, ckb, cv, kmr, ky, sr, tt | 114,480 | 159 |
| SEA | `SoutheastAsiaLID` | yue, id, vi | 90,720 | 126 |
| SA | `SouthAsiaLID` | cnh, dv, hi, ml, mr, or, skr | 68,400 | 95 |

Each corpus streams the **test split** of every locale in it, in the order fixed by its order file.
The run scores one target language at a time: clips of that locale are "target", all others are not.
One order file per corpus serves all of its target languages.

**Release.** Common Voice **17.0**, which we picked by matching the paper's Table I/II clip counts
against the metadata of every release (`tools/corpus/find_ac_corpus.py`, `find_corpus.py`, run on a
clone of https://github.com/common-voice/cv-dataset). We downloaded the audio from the ungated
Hugging Face mirror [`fsicoli/common_voice_17_0`](https://huggingface.co/datasets/fsicoli/common_voice_17_0)
(CC0, no token needed). We did not check that the mirror is byte-identical to Mozilla's release.
The one exception is **Amharic (`am`)**, which the mirror lacks: it comes from
Common Voice **26.0** on the [Mozilla Data Collective](https://datacollective.mozillafoundation.org)
(free account, API key; we keep it in `.env` as `MDC_API_KEY`, which is never committed).

**Locales.**
- **AC**: all 15 African locales with data in 17.0: `bas ha yo ig sw rw lg am af kab zgh dyu ti tig tw`.
  No subset matches Table I's total exactly, so the corpus is about 4% larger than the paper's.
- **CR / SEA / SA**: a geographic core of whole test splits, plus a seeded draw of filler clips
  that brings each corpus to exactly Table I's test total. The paper's own background locales are
  unknown, so these backgrounds are *constructed*, not identified.
  - CR core: `hy-AM ckb kmr cv ky sr tt az ka os ab ba sah mhr mrj myv mdf tk ug kk uz`
  - SEA core: `yue id vi th lo zh-CN zh-TW zh-HK nan-tw ja ko mn tok`
  - SA core: `cnh dv hi ml mr or skr bn pa-IN sat as ta te ne-NP ur`
  - filler pool: `eo eu hu tr nl fa ar`, drawn with `random.Random(0)` (CR 10%, SEA 42%, SA 46% filler)

**Truncation.** Runs use whole sessions of 720 clips, so any remainder at the end of the order is never
streamed (18 clips in AC; 0 in the others). Nothing is deleted on disk.

## Rebuilding from scratch

Run everything from the repository root with the project's virtual environment active, unless a step
says otherwise. The fetch scripts print a dry-run plan with sizes, and download only with `--go`.

1. **Download the test audio and the manifests** (test audio for every locale above, plus `train.tsv`
   manifests; train audio is not needed beyond step 5):
   ```bash
   python tools/corpus/fetch_ac_corpus.py --go          # AC, 14 locales from the mirror
   python tools/corpus/fetch_cr_sea_sa_corpus.py --go   # CR/SEA/SA cores + filler pool
   ```
   For `am`, download "Common Voice Scripted Speech 26.0 – Amharic" from the Mozilla Data Collective
   into `data/AfricanContinentLID/raw/mdc_am/` (path and file name in `tools/corpus/assemble_ac_corpus.py`).

2. **Lay the audio out flat**, as `<corpus>/test/common_voice_<locale>_<id>.mp3`. The code reads the
   locale from the file name, so the naming matters:
   ```bash
   python tools/corpus/assemble_ac_corpus.py
   python tools/corpus/assemble_cr_sea_sa_corpus.py
   python tools/corpus/fill_cr_sea_sa.py        # seeded filler draw, symlinked into each corpus
   ```

3. **Use the published order files.** Copy `data/<corpus>/orders/` from this repository, then add the
   train-order copies for seeds 1–4 (`cp cluster_order_train seedN_order_train`). Do not regenerate the
   orders: `oalcf-main/utils/org_cluster.py --seed` is deterministic only for the same directory
   listing, and the CR/SEA/SA seed-0 orders were built in one call over train and test together.
   If you add clips, or need a new seed, see "Adding a seed" in [GUIDE.md](GUIDE.md).

4. **Extract ECAPA features** (model [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa),
   as in the paper), from inside each corpus directory:
   ```bash
   mkdir -p data/{AfricanContinentLID,CaucasusRegionLID,SoutheastAsiaLID,SouthAsiaLID}/ecapalang
   cd data/AfricanContinentLID && python ../../oalcf-main/utils/extract_ecapa.py && cd -
   bash tools/corpus/run_cr_sea_sa_extract.sh      # CR, SEA, SA (edit ROOT at its top first)
   ```
   It processes `train/`, `dev/` and `test/` and skips clips that already have a feature file, so it is
   safe to run again.
   Our extraction logged 0 decode failures in every corpus. A failure count above 0 means the
   corpus differs from ours.

5. **Fetch the bootstrap train clips.** Each run trains its first model on 8 clips from the *train*
   order: the first 4 target and 4 non-target clips, each with ±3 neighbours. So only a handful of
   train clips are ever loaded (AC 43, CR/SEA/SA 40–80), but Common Voice serves train audio as one
   archive per locale, so whole train shards have to be downloaded (about 13 GB for CR/SEA/SA):
   ```bash
   python tools/corpus/fetch_crseasa_bootstrap.py --go
   python tools/corpus/assemble_crseasa_bootstrap.py   # then run step 4 again for train/
   ```
   For AC the clips were gathered by hand; they are listed in `docs/other/ac_bootstrap_train_files.txt`
   (one needs the `kab` train shard, about 1 GB). Put them in `data/AfricanContinentLID/train/` and run
   step 4 again.

6. **Check** before trusting a run: the session count in a run's `scores.csv` must be 83 / 159 / 126 / 95,
   and every name in an order file must have a matching `.npy` under `ecapalang/`.

`tools/corpus/run_crseasa_pipeline*.sh` are the scripts we actually ran for steps 4–5 and the first
baselines. Their paths are hard-coded to the original machine, so read them before you run them.
`tools/corpus/make_synth_corpus.py` builds a small synthetic corpus (`data/synthLID/`) for checking that
the code runs at all, without any downloads.
