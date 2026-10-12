#!/usr/bin/env python3
"""
Fetch the narrow slice of Mozilla Common Voice 17.0 needed for the AC-SLV
reproduction, from the ungated community mirror fsicoli/common_voice_17_0
(CC0, no auth token). The official mozilla-foundation/common_voice_17_0 repo
on Hugging Face is empty (2.88 kB, README only) -- distribution moved to
Mozilla Data Collective in October 2025. Do not use it.

Repo layout (per-file blobs, not a single downloadable archive):
    audio/{locale}/{split}/{locale}_{split}_{n}.tar
    transcript/{locale}/{split}.tsv

Fetches, for each of the 14 African-continent locales identified in
docs/other/DEVIATIONS.md ("Corpus: release and locale list are unspecified..."):
    - test split audio (.tar shards) -- full, this is what all reported
      metrics are computed on
    - train.tsv and test.tsv manifests only -- NO train audio. rw alone
      has a ~1M-utterance train split (~61 GB); we only need enough of it,
      once the real order file is built, for the 8-sample bootstrap.

am (Amharic) is deliberately excluded: confirmed 404 on both transcript and
audio paths on this mirror (not merely an empty directory). To be sourced
separately from Mozilla Data Collective later -- do not fetch or substitute
it here. Any OTHER locale printing MISSING is a real problem, not expected.

Usage:
    python tools/fetch_ac_corpus.py            # dry run: prints the full
                                                # file list and byte total,
                                                # downloads nothing
    python tools/fetch_ac_corpus.py --go        # actually downloads, but
                                                # aborts if the dry-run
                                                # total exceeds LIMIT_GB
"""

import sys
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "fsicoli/common_voice_17_0"
REPO = f"datasets/{REPO_ID}"
OUT = "data/AfricanContinentLID/raw"
LOCALES = ["bas", "ha", "yo", "ig", "sw", "rw", "lg",
           "af", "kab", "zgh", "dyu", "ti", "tw", "tig"]
MANIFESTS = ["test.tsv", "train.tsv"]
LIMIT_GB = 2.7


def main():
    fs = HfFileSystem()
    plan = []  # (repo-relative path, bytes)

    for loc in LOCALES:
        for m in MANIFESTS:
            p = f"{REPO}/transcript/{loc}/{m}"
            try:
                plan.append((p[len(REPO) + 1:], fs.info(p)["size"]))
            except (FileNotFoundError, EntryNotFoundError):
                print(f"  MISSING {loc}/{m}", file=sys.stderr)
        d = f"{REPO}/audio/{loc}/test"
        try:
            for e in fs.ls(d, detail=True):
                if e["type"] == "file" and e["name"].endswith(".tar"):
                    plan.append((e["name"][len(REPO) + 1:], e["size"]))
        except (FileNotFoundError, EntryNotFoundError):
            print(f"  MISSING {d}", file=sys.stderr)

    for path, size in plan:
        print(f"{size:>13,}  {path}")
    total = sum(s for _, s in plan)
    print(f"\n{len(plan)} files, {total / 1e9:.2f} GB -> {OUT}/")

    if "--go" not in sys.argv:
        print("dry run -- rerun with --go to download")
        sys.exit()

    if total / 1e9 > LIMIT_GB:
        sys.exit(f"ABORT: {total / 1e9:.2f} GB exceeds {LIMIT_GB} GB limit")

    for path, size in plan:
        print(f"[{size / 1e6:>8.1f} MB] {path}")
        hf_hub_download(REPO_ID, filename=path, repo_type="dataset",
                         local_dir=OUT)


if __name__ == "__main__":
    main()
