#!/usr/bin/env python3
"""
Flatten the raw fetched CR/SEA/SA audio (tools/fetch_cr_sea_sa_corpus.py,
fsicoli/common_voice_17_0) into per-corpus flat mp3 test/ directories, same
layout as AC:

    data/CaucasusRegionLID/test/common_voice_{locale}_{id}.mp3
    data/SoutheastAsiaLID/test/common_voice_{locale}_{id}.mp3
    data/SouthAsiaLID/test/common_voice_{locale}_{id}.mp3

Only test-split audio was fetched (same reasoning as AC: train is handled
later, targeted-only, once each corpus's order file establishes
n_boot_batches). train/ and dev/ are created empty here -- dev is PML-only
(out of scope), train gets filled in a later phase.

Each corpus gets its own target + geographic-core locales (unambiguous,
mutually exclusive). The 7 filler locales (eo, eu, hu, tr, nl, fa, ar) are
shared across corpora for the geo_fill/random composition variants (built
later, from local metadata -- no further download) and are extracted once
into a separate shared pool directory rather than duplicated per corpus.

Usage: python tools/assemble_cr_sea_sa_corpus.py
"""

import os
import tarfile

RAW = "data/CrSeaSa/raw"

CORPORA = {
    "CaucasusRegionLID": ["hy-AM", "ckb", "kmr", "cv", "ky", "sr", "tt", "az", "ka",
                           "os", "ab", "ba", "sah", "mhr", "mrj", "myv", "mdf", "tk",
                           "ug", "kk", "uz"],
    "SoutheastAsiaLID": ["yue", "id", "vi", "th", "lo", "zh-CN", "zh-TW", "zh-HK",
                          "nan-tw", "ja", "ko", "mn", "tok"],
    "SouthAsiaLID": ["cnh", "dv", "hi", "ml", "mr", "or", "skr", "bn", "pa-IN",
                      "sat", "as", "ta", "te", "ne-NP", "ur"],
}
FILLER = ["eo", "eu", "hu", "tr", "nl", "fa", "ar"]
FILLER_OUT = "data/CrSeaSaFiller/test"


def extract_locale(loc, out_dir):
    tar_path = os.path.join(RAW, "audio", loc, "test", f"{loc}_test_0.tar")
    n = 0
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".mp3"):
                continue
            member.name = os.path.basename(member.name)
            tf.extract(member, path=out_dir)
            n += 1
    return n


def main():
    grand_total = 0
    for corpus, locales in CORPORA.items():
        out = os.path.join("data", corpus)
        test_out = os.path.join(out, "test")
        os.makedirs(test_out, exist_ok=True)
        os.makedirs(os.path.join(out, "train"), exist_ok=True)
        os.makedirs(os.path.join(out, "dev"), exist_ok=True)
        print(f"=== {corpus} ===")
        total = 0
        for loc in locales:
            n = extract_locale(loc, test_out)
            print(f"  {loc:<8} {n:>6} clips")
            total += n
        print(f"  -> {total} clips in {test_out}/\n")
        grand_total += total

    print("=== shared filler pool ===")
    os.makedirs(FILLER_OUT, exist_ok=True)
    filler_total = 0
    for loc in FILLER:
        n = extract_locale(loc, FILLER_OUT)
        print(f"  {loc:<8} {n:>6} clips")
        filler_total += n
    print(f"  -> {filler_total} clips in {FILLER_OUT}/")

    print(f"\n{grand_total} clips across 3 corpora + {filler_total} filler clips "
          f"= {grand_total + filler_total} total")


if __name__ == "__main__":
    main()
