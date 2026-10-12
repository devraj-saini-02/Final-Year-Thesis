#!/usr/bin/env python3
"""
Identify which Common Voice release and which African locale set produce the
paper's AC-SLV corpus, using release metadata only. No audio downloads.

Input: a clone of https://github.com/common-voice/cv-dataset
       (text only, small). The per-release files live in datasets/.

    git clone https://github.com/common-voice/cv-dataset
    python find_ac_corpus.py --datasets cv-dataset/datasets

Checksums, all from the paper:
  Table I   AC test  = 56,880   train = 982,800   val = 51,840
  Table II  bas 0.95%  ha 1.16%  yo 1.75%  of the test split
            (~540 / 660 / 995 clips), corpus duration 82 hours

Splits are truncated to a whole number of 720-sample sessions (11 of the 12
cells in Table I are exact multiples of 720), so a candidate matches if its
RAW total floors to the published figure -- i.e. test in [56880, 57599].
"""

import argparse
import glob
import json
import os
import re

TARGET_TEST = 56_880
TARGET_TRAIN = 982_800
TARGET_VAL = 51_840
SESSION = 720
TARGET_HOURS = 82.0

# Per-language test-clip counts implied by Table II.
LANG_TARGETS = {"bas": 0.0095, "ha": 0.0116, "yo": 0.0175}

# Candidate African locales in Common Voice. The paper gives only the rule
# ("languages found in" the region), never the list, so this is a superset to
# search over -- not a claim about what the authors used. Edit freely.
AFRICAN = [
    "bas", "ha", "yo", "ig", "sw", "rw", "lg", "wo", "am", "ff", "zu", "nso",
    "tn", "ts", "ve", "nr", "ss", "xh", "af", "ary", "arz", "kab", "taq",
    "tzm", "shi", "zgh", "sn", "ln", "bm", "dyu", "om", "so", "ti", "mos",
    "ee", "tw", "luo", "kea", "nqo", "ak", "sg", "ny", "st",
]


def load_releases(datasets_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(datasets_dir, "cv-corpus-*.json"))):
        name = os.path.basename(path)[:-5]
        try:
            with open(path) as f:
                blob = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ! skipping {name}: {e}")
            continue
        locales = blob.get("locales")
        if not isinstance(locales, dict):
            continue
        out[name] = locales
    return out


def bucket(loc_blob, key):
    b = loc_blob.get("buckets") or {}
    v = b.get(key)
    return int(v) if isinstance(v, (int, float)) else None


def release_sort_key(name):
    m = re.search(r"cv-corpus-(\d+)(?:\.(\d+))?", name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def stage1(releases, tol):
    """Rank releases by how close bas/ha/yo test counts are to Table II."""
    want = {k: TARGET_TEST * p for k, p in LANG_TARGETS.items()}
    rows = []
    for name, locales in releases.items():
        got, err = {}, []
        for lang, target in want.items():
            if lang not in locales:
                got[lang] = None
                err.append(1.0)
                continue
            n = bucket(locales[lang], "test")
            got[lang] = n
            err.append(1.0 if not n else abs(n - target) / target)
        rows.append((max(err), name, got))
    rows.sort(key=lambda r: (r[0], release_sort_key(r[1])))
    return rows, want


def totals(locales, subset, key):
    return sum(bucket(locales[l], key) or 0 for l in subset if l in locales)


def hours(locales, subset):
    h = 0.0
    for l in subset:
        if l not in locales:
            continue
        n = bucket(locales[l], "test") or 0
        avg = locales[l].get("avgDurationSecs")
        if isinstance(avg, (int, float)):
            h += n * avg
    return h / 3600.0


def subset_sums(locales, pool, lo, hi, forced, max_solutions):
    """Exact subset-sum on test counts. Locales in `forced` are always in."""
    base = totals(locales, forced, "test")
    if base > hi:
        return []
    optional = [(l, bucket(locales[l], "test") or 0)
                for l in pool if l in locales and l not in forced]
    optional = [(l, n) for l, n in optional if 0 < n <= hi]
    optional.sort(key=lambda x: -x[1])

    sols, cur = [], []
    remaining = [0] * (len(optional) + 1)
    for i in range(len(optional) - 1, -1, -1):
        remaining[i] = remaining[i + 1] + optional[i][1]

    def rec(i, total):
        if len(sols) >= max_solutions:
            return
        if lo <= total <= hi:
            sols.append((total, sorted(forced) + sorted(cur)))
            return
        if i >= len(optional) or total > hi or total + remaining[i] < lo:
            return
        l, n = optional[i]
        cur.append(l)
        rec(i + 1, total + n)
        cur.pop()
        rec(i + 1, total)

    rec(0, base)
    return sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="path to cv-dataset/datasets")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="relative tolerance on per-language test counts")
    ap.add_argument("--top", type=int, default=6,
                    help="how many releases to examine in detail")
    ap.add_argument("--max-solutions", type=int, default=15)
    args = ap.parse_args()

    releases = load_releases(args.datasets)
    if not releases:
        raise SystemExit(f"no cv-corpus-*.json found under {args.datasets}")
    print(f"loaded {len(releases)} releases\n")

    rows, want = stage1(releases, args.tol)

    print("STAGE 1 -- per-language test counts vs Table II")
    print(f"  wanted: " + "  ".join(f"{k} ~{v:.0f}" for k, v in want.items()))
    print(f"  {'release':<32} {'bas':>7} {'ha':>7} {'yo':>7}   worst err")
    for err, name, got in rows[:args.top * 2]:
        cells = "".join(f"{(got[l] if got[l] is not None else '-'):>8}"
                        for l in ("bas", "ha", "yo"))
        flag = "  <<<" if err <= args.tol else ""
        print(f"  {name:<32}{cells}   {err:8.1%}{flag}")

    print("\nSTAGE 2 -- corpus totals for the candidate releases")
    lo, hi = TARGET_TEST, TARGET_TEST + SESSION - 1
    for err, name, got in rows[:args.top]:
        locales = releases[name]
        present = [l for l in AFRICAN if l in locales]
        t_test = totals(locales, present, "test")
        t_train = totals(locales, present, "train")
        t_dev = totals(locales, present, "dev")
        print(f"\n  {name}   ({len(present)} African locales present)")
        print(f"    all-African test  {t_test:>9,}   "
              f"want [{lo:,}, {hi:,}]   "
              f"{'MATCH' if lo <= t_test <= hi else f'off by {t_test - lo:+,}'}")
        print(f"    all-African train {t_train:>9,}   want ~{TARGET_TRAIN:,}")
        print(f"    all-African dev   {t_dev:>9,}   want ~{TARGET_VAL:,}")
        print(f"    all-African hours {hours(locales, present):>9.1f}   "
              f"want ~{TARGET_HOURS}")

        if not (lo <= t_test <= hi):
            sols = subset_sums(locales, present, lo, hi,
                               forced=set(LANG_TARGETS),
                               max_solutions=args.max_solutions)
            if not sols:
                print("    no locale subset hits the test window")
            else:
                print(f"    {len(sols)} subset(s) hit the test window:")
                for total, subset in sols[:args.max_solutions]:
                    tr = totals(locales, subset, "train")
                    dv = totals(locales, subset, "dev")
                    hr = hours(locales, subset)
                    print(f"      test {total:,}  train {tr:,}  dev {dv:,}  "
                          f"{hr:.1f} h")
                    print(f"        {' '.join(subset)}")

    print("\nJudge candidates on ALL FOUR checksums, not just the test total --")
    print("subset-sum will always find something that hits one window.")


if __name__ == "__main__":
    main()
