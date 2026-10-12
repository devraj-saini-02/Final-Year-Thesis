#!/usr/bin/env python3
"""
Extract exactly the train clips CR/SEA/SA's bootstrap needs (see
tools/fetch_crseasa_bootstrap.py) from the downloaded locale train shards,
into each corpus's own flat data/{corpus}/train/ directory -- same layout
as the test split, same common_voice_{locale}_{id}.mp3 naming.

Replicates load_external_bootstrap's own scan (dataset.py:330-349) against
each corpus's real orders/cluster_order_train, per target language: first 4
target + 4 non-target indices, +/-3 context window around each.

Two boundary cases beyond the raw 8 indices also need real features, both
because get_sample_with_context (dataset.py:315-329) loads a clip's context
window *before* zeroing it at a session edge -- the load still happens even
though the value gets discarded:

  1. The train/test seam. The bootstrap prefix (train_ds.feat_files[:n_boot
     _batches*720]) is prepended directly onto the test stream, so the first
     test session's leading edge bleeds backward into the LAST `context`
     entries of the train prefix (this is what crashed the first pass over
     hy-AM: common_voice_hu_37177198, at position n_boot_batches*720-3).
  2. The array wraparound. get_sample_with_context indexes via
     `idx = ii % max_idx` with `max_idx = len(self)-1`. Since every SLV test
     split here is an exact multiple of 720 (no dropped tail), the very last
     test session's trailing edge overflows past the end of the array and
     wraps to absolute positions 1..context -- i.e. lines[1:context+1] of
     the train order file, for every language, regardless of n_boot_batches.

Usage: python tools/assemble_crseasa_bootstrap.py
"""
import glob
import os
import tarfile

RAW = "data/CrSeaSa/raw"
CONTEXT = 3
SAMPLES_PER_BATCH = 720

CORPORA = {
    "CaucasusRegionLID": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
    "SoutheastAsiaLID": ["yue", "id", "vi"],
    "SouthAsiaLID": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"],
}


def locale_of(name):
    return name.split("_")[2]


def bootstrap_window(lines, target):
    n_target = n_nontarget = 0
    idxs = []
    for i, name in enumerate(lines):
        is_target = (locale_of(name) == target)
        if is_target and n_target < 4:
            n_target += 1
            idxs.append(i)
        elif not is_target and n_nontarget < 4:
            n_nontarget += 1
            idxs.append(i)
        if n_target >= 4 and n_nontarget >= 4:
            break
    n_boot_batches = (max(idxs) // SAMPLES_PER_BATCH) + 1
    boundary = n_boot_batches * SAMPLES_PER_BATCH
    needed = set()
    for i in idxs:
        for j in range(max(0, i - CONTEXT), min(len(lines), i + CONTEXT + 1)):
            needed.add(lines[j])
    # boundary case 1: train/test seam
    for j in range(max(0, boundary - CONTEXT), boundary):
        needed.add(lines[j])
    return needed


def main():
    for corpus, targets in CORPORA.items():
        order_path = os.path.join("data", corpus, "orders", "cluster_order_train")
        out_dir = os.path.join("data", corpus, "train")
        os.makedirs(out_dir, exist_ok=True)

        with open(order_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        needed_by_locale = {}
        # boundary case 2: array wraparound (`idx = ii % max_idx`) -- the very
        # last test session's trailing edge overflows past the array end and
        # wraps to absolute positions 1..CONTEXT, same for every language.
        for j in range(0, CONTEXT + 1):
            needed_by_locale.setdefault(locale_of(lines[j]), set()).add(lines[j])
        for target in targets:
            for name in bootstrap_window(lines, target):
                needed_by_locale.setdefault(locale_of(name), set()).add(name)

        print(f"=== {corpus}: {sum(len(v) for v in needed_by_locale.values())} clips across {len(needed_by_locale)} locales ===")
        n_written = 0
        for loc, names in sorted(needed_by_locale.items()):
            # a locale's train split may be split across multiple shards
            # (e.g. uz_train_0.tar, uz_train_1.tar) -- search all of them
            tar_paths = sorted(glob.glob(os.path.join(RAW, "audio", loc, "train", f"{loc}_train_*.tar")))
            if not tar_paths:
                print(f"  MISSING shard(s) for {loc}, skipping {len(names)} clip(s)")
                continue
            wanted_files = {f"{n}.mp3" for n in names}
            found = set()
            for tar_path in tar_paths:
                if not wanted_files - found:
                    break
                with tarfile.open(tar_path) as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        basename = os.path.basename(member.name)
                        if basename not in wanted_files or basename in found:
                            continue
                        found.add(basename)
                        out_path = os.path.join(out_dir, basename)
                        if os.path.exists(out_path):
                            continue
                        member.name = basename
                        tf.extract(member, path=out_dir)
                        n_written += 1
            missing = wanted_files - found
            if missing:
                print(f"  {loc}: {len(missing)} clip(s) not found in any shard: {sorted(missing)}")
        print(f"  {n_written} new clips -> {out_dir}/")


if __name__ == "__main__":
    main()
