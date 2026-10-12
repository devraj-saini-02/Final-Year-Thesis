#!/usr/bin/env python3
"""
Generalized version of find_ac_corpus.py for the other three SLV corpora
(CR, SEA, SA). Same method: use Common Voice release metadata only (no audio)
to find which release + locale set reproduces a corpus's Table I/II checksums.

Input: a clone of https://github.com/common-voice/cv-dataset
    git clone https://github.com/common-voice/cv-dataset
    python tools/find_corpus.py --region CR --datasets cv-dataset/datasets/scripted-speech

Checksums are all read from the paper (Tables I and II). Splits are truncated
to a whole number of 720-sample sessions before publication, so a candidate
matches the test checksum if its RAW total floors into [target, target+719].
"""

import argparse
import glob
import json
import os
import re

SESSION = 720

# region: (test, train, val, hours, {abbr: fraction_of_test}, candidate_superset)
REGIONS = {
    "CR": dict(
        test=114_480, train=666_720, val=112_719, hours=162.0,
        lang_targets={
            "hy-AM": 0.0374, "ckb": 0.0460, "cv": 0.0113, "kmr": 0.0341,
            "ky": 0.0141, "sr": 0.0134, "tt": 0.0433,
        },
        candidates=[
            "hy-AM", "ckb", "kmr", "cv", "ky", "sr", "tt",
            "az", "ka", "os", "ab", "ba", "sah", "mhr", "mrj", "myv", "mdf",
            "tk", "ug", "kk", "uz", "tg",
        ],
    ),
    "SEA": dict(
        test=90_720, train=1_363_680, val=112_320, hours=146.0,
        lang_targets={"yue": 0.0286, "id": 0.0397, "vi": 0.0139},
        candidates=[
            "yue", "id", "vi", "th", "lo", "my", "km", "ms", "tl", "ceb",
            "jv", "su", "zh-CN", "zh-TW", "zh-HK", "nan-tw",
            "ja", "ko", "mn", "tok",
        ],
    ),
    "SA": dict(
        test=68_400, train=797_040, val=66_240, hours=119.0,
        lang_targets={
            "cnh": 0.0111, "dv": 0.0322, "hi": 0.0459, "ml": 0.0103,
            "mr": 0.0255, "or": 0.0147, "skr": 0.0147,
        },
        candidates=[
            "cnh", "dv", "hi", "ml", "mr", "or", "skr",
            "bn", "pa-IN", "sat", "as", "ta", "te", "ne-NP", "ur", "gu",
            "kn", "sd",
        ],
    ),
}


def load_releases(datasets_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(datasets_dir, "cv-corpus-*.json"))):
        name = os.path.basename(path)[:-5]
        if "delta" in name or "singleword" in name:
            continue
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


def stage1(releases, lang_targets, target_test, tol):
    want = {k: target_test * p for k, p in lang_targets.items()}
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
    ap.add_argument("--region", required=True, choices=sorted(REGIONS))
    ap.add_argument("--datasets", required=True,
                    help="path to cv-dataset/datasets/scripted-speech")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--max-solutions", type=int, default=15)
    args = ap.parse_args()

    cfg = REGIONS[args.region]
    releases = load_releases(args.datasets)
    if not releases:
        raise SystemExit(f"no cv-corpus-*.json found under {args.datasets}")
    print(f"[{args.region}] loaded {len(releases)} releases\n")

    rows, want = stage1(releases, cfg["lang_targets"], cfg["test"], args.tol)

    langs = list(cfg["lang_targets"])
    print("STAGE 1 -- per-language test counts vs Table II")
    print("  wanted: " + "  ".join(f"{k} ~{v:.0f}" for k, v in want.items()))
    header = "  " + f"{'release':<32}" + "".join(f"{l:>9}" for l in langs) + "   worst err"
    print(header)
    for err, name, got in rows[:args.top * 2]:
        cells = "".join(f"{(got[l] if got[l] is not None else '-'):>9}" for l in langs)
        flag = "  <<<" if err <= args.tol else ""
        print(f"  {name:<32}{cells}   {err:8.1%}{flag}")

    print("\nSTAGE 2 -- corpus totals for the candidate releases")
    lo, hi = cfg["test"], cfg["test"] + SESSION - 1
    for err, name, got in rows[:args.top]:
        locales = releases[name]
        present = [l for l in cfg["candidates"] if l in locales]
        t_test = totals(locales, present, "test")
        t_train = totals(locales, present, "train")
        t_dev = totals(locales, present, "dev")
        print(f"\n  {name}   ({len(present)}/{len(cfg['candidates'])} candidate locales present)")
        print(f"    candidate-set test  {t_test:>9,}   "
              f"want [{lo:,}, {hi:,}]   "
              f"{'MATCH' if lo <= t_test <= hi else f'off by {t_test - lo:+,}'}")
        print(f"    candidate-set train {t_train:>9,}   want ~{cfg['train']:,}")
        print(f"    candidate-set dev   {t_dev:>9,}   want ~{cfg['val']:,}")
        print(f"    candidate-set hours {hours(locales, present):>9.1f}   want ~{cfg['hours']}")

        if not (lo <= t_test <= hi):
            sols = subset_sums(locales, present, lo, hi,
                                forced=set(langs),
                                max_solutions=args.max_solutions)
            if not sols:
                print("    no locale subset (within candidate superset) hits the test window")
            else:
                print(f"    {len(sols)} subset(s) hit the test window:")
                for total, subset in sols[:args.max_solutions]:
                    tr = totals(locales, subset, "train")
                    dv = totals(locales, subset, "dev")
                    hr = hours(locales, subset)
                    print(f"      test {total:,}  train {tr:,}  dev {dv:,}  {hr:.1f} h")
                    print(f"        {' '.join(subset)}")

    print("\nJudge candidates on ALL FOUR checksums, not just the test total --")
    print("subset-sum will always find something that hits one window.")


if __name__ == "__main__":
    main()
