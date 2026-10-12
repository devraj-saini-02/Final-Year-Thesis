#!/usr/bin/env python3
"""
Fetch the train-split shards CR/SEA/SA's load_external_bootstrap needs, per
the same circularity noted for AC in docs/other/DEVIATIONS.md ("AC: bas/yo OAL FNR
are structurally similar..." entry): only the first 4 target + 4 non-target
indices of each target language's train order, +/-3 context, are ever loaded
by activate_samples -- not the full n_boot_batches * 720 prefix.

Common Voice only serves train audio as one tar per locale (no row-level
fetch), so getting a handful of clips per locale still means downloading
that locale's whole train shard -- same situation as AC's `kab` download.

LOCALES below is the union, across all three corpora and all 17 target
languages, of every locale that load_external_bootstrap's bootstrap window
actually touches (computed by scanning each corpus's real orders/
cluster_order_train -- see the scan this list was derived from). 13.39 GB
total, confirmed via a dry-run HfFileSystem listing before any bytes moved.

Usage:
    python tools/fetch_crseasa_bootstrap.py            # dry run
    python tools/fetch_crseasa_bootstrap.py --go        # actually downloads
"""
import os
import sys
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "fsicoli/common_voice_17_0"
REPO = f"datasets/{REPO_ID}"
OUT = "data/CrSeaSa/raw"
LIMIT_GB = 15.0

LOCALES = sorted(set(
    "ckb cv eo hu hy-AM kk kmr ky myv os sr tt ug uz "
    "id ja mn nan-tw vi yue "
    "bn cnh dv hi ml mr nl or pa-IN skr tr "
    "ab ar eu ta".split()
))
# ab/ar/eu/ta added after the first pass: get_sample_with_context loads (then
# zeroes) a +/-context window at every activated index, including the very
# first test session (bleeds backward across the train/test seam) and the
# very last one (wraps forward via `idx = ii % max_idx` back to the start of
# the array). Both bleeds needed locales beyond the original 8-raw-index scan
# -- see docs/other/DEVIATIONS.md.


def main():
    fs = HfFileSystem()
    plan = []

    for loc in LOCALES:
        d = f"{REPO}/audio/{loc}/train"
        try:
            for e in fs.ls(d, detail=True):
                if e["type"] == "file" and e["name"].endswith(".tar"):
                    plan.append((e["name"][len(REPO) + 1:], e["size"]))
        except (FileNotFoundError, EntryNotFoundError):
            print(f"  MISSING {d}", file=sys.stderr)

    have, todo = [], []
    for path, size in plan:
        (have if os.path.exists(os.path.join(OUT, path)) else todo).append((path, size))

    for path, size in plan:
        tag = "cached" if (path, size) in have else "new"
        print(f"{size:>13,}  {path}  [{tag}]")
    total = sum(s for _, s in plan)
    new_total = sum(s for _, s in todo)
    print(f"\n{len(plan)} files, {total / 1e9:.2f} GB total "
          f"({len(have)} cached, {len(todo)} new = {new_total / 1e9:.2f} GB to fetch) -> {OUT}/")

    if "--go" not in sys.argv:
        print("dry run -- rerun with --go to download")
        sys.exit()

    if new_total / 1e9 > LIMIT_GB:
        sys.exit(f"ABORT: {new_total / 1e9:.2f} GB to fetch exceeds {LIMIT_GB} GB limit")

    for path, size in todo:
        print(f"[{size / 1e6:>8.1f} MB] {path}")
        hf_hub_download(REPO_ID, filename=path, repo_type="dataset",
                         local_dir=OUT)


if __name__ == "__main__":
    main()
