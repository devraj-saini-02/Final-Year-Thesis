#!/usr/bin/env python3
"""
Seeded fill: draw enough filler clips (from data/CrSeaSaFiller/test/, the
7-locale shared pool) to bring each of CR/SEA/SA up to Table I's test total,
and symlink them into that corpus's own test/ directory (extract_ecapa.py
needs the mp3s physically globbable from the corpus root; dataset.py later
needs the resulting .npy features to live under that corpus's own
ecapalang/test/, so everything has to actually live in-tree, not just be
referenced).

Draws are independent per corpus (same seed, sequential draws from the full
pool) -- overlap of the same filler clip across two different corpora is
fine, these are three separate experiments, not one shared dataset.

Usage: python tools/fill_cr_sea_sa.py
"""
import glob
import os
import random

FILLER_DIR = "data/CrSeaSaFiller/test"
SHORTFALL = {
    "CaucasusRegionLID": 11504,
    "SoutheastAsiaLID": 37908,
    "SouthAsiaLID": 31193,
}
SEED = 0


def main():
    pool = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FILLER_DIR, "*.mp3")))
    print(f"filler pool: {len(pool)} clips")

    rng = random.Random(SEED)
    for corpus, n_needed in SHORTFALL.items():
        draw = rng.sample(pool, n_needed)
        test_dir = os.path.join("data", corpus, "test")
        n_linked, n_skipped = 0, 0
        for fname in draw:
            src = os.path.abspath(os.path.join(FILLER_DIR, fname))
            dst = os.path.join(test_dir, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                n_skipped += 1
                continue
            os.symlink(src, dst)
            n_linked += 1
        print(f"{corpus}: drew {len(draw)}, linked {n_linked}, already present {n_skipped}")


if __name__ == "__main__":
    main()
