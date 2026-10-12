#!/usr/bin/env python3
"""
Flatten the raw fetched Common Voice data (14 locales from fsicoli/common_voice_17_0,
1 locale -- am -- from Mozilla Data Collective's Common Voice 26.0) into the single flat
mp3 directory run.py's pipeline expects:

    data/AfricanContinentLID/test/common_voice_{locale}_{id}.mp3

Only the test split is assembled here -- train audio was never fetched (see
docs/other/DEVIATIONS.md), only train.tsv manifests, which stay in data/AfricanContinentLID/raw/
for the order-building step later.

Sources per locale:
    - 14 locales (bas ha yo ig sw rw lg af kab zgh dyu ti tw tig): one tar per locale at
      data/AfricanContinentLID/raw/audio/{locale}/test/{locale}_test_0.tar, already
      correctly named common_voice_{locale}_{id}.mp3 inside -- extract and flatten.
    - am: single bundled archive covering all splits at
      data/AfricanContinentLID/raw/mdc_am/*.tar.gz, clips/ mixes every split together --
      filter to just the filenames listed in test.tsv before extracting.

Usage: python tools/assemble_ac_corpus.py
"""

import csv
import io
import os
import tarfile

RAW = "data/AfricanContinentLID/raw"
OUT = "data/AfricanContinentLID/test"

FSICOLI_LOCALES = ["bas", "ha", "yo", "ig", "sw", "rw", "lg",
                    "af", "kab", "zgh", "dyu", "ti", "tw", "tig"]

MDC_AM_ARCHIVE = os.path.join(
    RAW, "mdc_am", "common-voice-scripted-speech-26-0-amhari-9d4926a1.tar.gz")
MDC_AM_PREFIX = "cv-corpus-26.0-2026-06-12/am"


def extract_fsicoli_locale(loc):
    tar_path = os.path.join(RAW, "audio", loc, "test", f"{loc}_test_0.tar")
    n = 0
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".mp3"):
                continue
            basename = os.path.basename(member.name)
            member.name = basename  # strip the "{loc}_test_0/" prefix
            tf.extract(member, path=OUT)
            n += 1
    return n


def extract_mdc_amharic():
    with tarfile.open(MDC_AM_ARCHIVE) as tf:
        tsv_member = tf.getmember(f"{MDC_AM_PREFIX}/test.tsv")
        tsv_text = tf.extractfile(tsv_member).read().decode()
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        test_filenames = {row["path"] for row in reader}

        n = 0
        for member in tf.getmembers():
            if not member.isfile() or not member.name.startswith(f"{MDC_AM_PREFIX}/clips/"):
                continue
            basename = os.path.basename(member.name)
            if basename not in test_filenames:
                continue  # train/dev/other/invalidated -- skip, audio never wanted
            member.name = basename
            tf.extract(member, path=OUT)
            n += 1
    return n, len(test_filenames)


def main():
    os.makedirs(OUT, exist_ok=True)

    total = 0
    for loc in FSICOLI_LOCALES:
        n = extract_fsicoli_locale(loc)
        print(f"{loc:<6} {n:>6} clips")
        total += n

    n, expected = extract_mdc_amharic()
    flag = "" if n == expected else f"  MISMATCH (test.tsv lists {expected})"
    print(f"{'am':<6} {n:>6} clips{flag}")
    total += n

    print(f"\n{total} clips total -> {OUT}/")


if __name__ == "__main__":
    main()
