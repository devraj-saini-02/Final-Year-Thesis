#!/usr/bin/env python3
"""
Fetch the test-split audio (+ train/test manifests) needed to construct
CR, SEA, and SA -- the three remaining SLV corpora -- from the same ungated
mirror used for AC (fsicoli/common_voice_17_0, release chosen for AC on
Table I/II checksum grounds; see docs/other/DEVIATIONS.md for why CR/SEA/SA reuse
that release too).

Unlike AC, these corpora's backgrounds are CONSTRUCTED, not identified (see
docs/other/DEVIATIONS.md "Framing note"): the 17 target locales are paper-specified
and confident; everything else is a geographic-core-plus-seeded-fill pool
used to hit Table I's totals. This script fetches the audio POOL (targets +
geographic core + a shared filler set) once; the specific geo / geo_fill /
random *compositions* (which subset of the pool counts as "this corpus's
background" under which seed) are built afterward from local metadata only,
no further downloads needed -- so the same downloaded pool serves all three
corpora and all three composition variants.

Usage:
    python tools/fetch_cr_sea_sa_corpus.py            # dry run, prints plan + byte total
    python tools/fetch_cr_sea_sa_corpus.py --go        # actually downloads
"""
import sys
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "fsicoli/common_voice_17_0"
REPO = f"datasets/{REPO_ID}"
OUT = "data/CrSeaSa/raw"
LIMIT_GB = 15.0

TARGETS = {
    "CR": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
    "SEA": ["yue", "id", "vi"],
    "SA": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"],
}
GEO_CORE = {
    "CR": ["hy-AM", "ckb", "kmr", "cv", "ky", "sr", "tt", "az", "ka", "os",
           "ab", "ba", "sah", "mhr", "mrj", "myv", "mdf", "tk", "ug", "kk", "uz"],
    "SEA": ["yue", "id", "vi", "th", "lo", "zh-CN", "zh-TW", "zh-HK",
            "nan-tw", "ja", "ko", "mn", "tok"],
    "SA": ["cnh", "dv", "hi", "ml", "mr", "or", "skr", "bn", "pa-IN",
           "sat", "as", "ta", "te", "ne-NP", "ur"],
}
# Shared filler pool: sized to cover the combined ~80.6k-clip shortfall
# (CR 11.5k + SEA 37.9k + SA 31.2k) across all three corpora with margin,
# for the geo_fill and random composition variants. Deliberately excludes
# the very largest locales (ca/en/de/fr/es/it, 15-16k clips each) to keep
# the download bounded -- these seven sum to ~83.6k clips on their own.
FILLER = ["eo", "eu", "hu", "tr", "nl", "fa", "ar"]

ALL_LOCALES = sorted(set(sum(GEO_CORE.values(), [])) | set(FILLER))
MANIFESTS = ["test.tsv", "train.tsv"]


def main():
    fs = HfFileSystem()
    plan = []

    print(f"Locales in pool: {len(ALL_LOCALES)}")
    print("  targets:", sum(len(v) for v in TARGETS.values()), "| geo-core union:",
          len(set(sum(GEO_CORE.values(), []))), "| filler:", len(FILLER))
    print()

    for loc in ALL_LOCALES:
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
