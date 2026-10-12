#!/usr/bin/env python3
"""
Build train_filenames.txt per corpus (CR/SEA/SA) from already-downloaded
train.tsv manifests (metadata only, no audio) -- same purpose as AC's
train_filenames.txt: lets org_cluster.py build the full train order without
needing the (982,800-scale) train audio downloaded, breaking the
load_external_bootstrap circularity per docs/other/DEVIATIONS.md.

Each corpus's train order draws from its own target+geo-core locales plus
all 7 shared filler locales (the same locale pool used for that corpus's
test-split geo_fill composition, at the locale level -- train order doesn't
need to mirror the exact test-split filler *clip* draw, only plausible
non-target/target audio to seed the 8-sample bootstrap).

Usage: python tools/build_train_filelists.py
"""
import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

RAW_TRANSCRIPT = "data/CrSeaSa/raw/transcript"
FILLER = ["eo", "eu", "hu", "tr", "nl", "fa", "ar"]

CORPUS_LOCALES = {
    "CaucasusRegionLID": ["hy-AM", "ckb", "kmr", "cv", "ky", "sr", "tt", "az", "ka",
                           "os", "ab", "ba", "sah", "mhr", "mrj", "myv", "mdf", "tk",
                           "ug", "kk", "uz"] + FILLER,
    "SoutheastAsiaLID": ["yue", "id", "vi", "th", "lo", "zh-CN", "zh-TW", "zh-HK",
                          "nan-tw", "ja", "ko", "mn", "tok"] + FILLER,
    "SouthAsiaLID": ["cnh", "dv", "hi", "ml", "mr", "or", "skr", "bn", "pa-IN",
                      "sat", "as", "ta", "te", "ne-NP", "ur"] + FILLER,
}


def main():
    for corpus, locales in CORPUS_LOCALES.items():
        out_path = os.path.join("data", corpus, "train_filenames.txt")
        n = 0
        with open(out_path, "w") as out:
            for loc in locales:
                tsv_path = os.path.join(RAW_TRANSCRIPT, loc, "train.tsv")
                with open(tsv_path) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        out.write(row["path"] + "\n")
                        n += 1
        print(f"{corpus}: {n} train filenames -> {out_path}")


if __name__ == "__main__":
    main()
