#!/usr/bin/env python3
"""
Multi-seed summary of the paper reproduction: baseline OAL vs OAL-CF, 20 SLV languages, stream-order
seeds 0..N (seed 0 = runs/baseline_seed0; seeds >= 1 from tools/run_baselines_seeds_cpu.sh).

A seed changes only the TEST stream order (`--order_file seed{N}`); the train order, hence the
bootstrap, and the model seed are those of seed 0 (docs/other/DEVIATIONS.md, AC seed order files).

Works on partial results: a seed is used for the pooled tables only when all 40 of its runs are done.

Usage (from anywhere): python tools/summarize_seeds.py [--seeds 0 1 2 3 4]
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from score_runs import score  # noqa: E402

LANGS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
         "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
# Fig. 12 read-off (docs/other/REPRO_BRIEF.md section 7), IMLM OAL -> OAL-CF, +-0.005
PAPER = {"bas": (0.042, 0.036), "ha": (0.020, 0.038), "yo": (0.057, 0.012), "ckb": (0.100, 0.030),
         "cv": (0.305, 0.210), "hy-AM": (0.032, 0.028), "kmr": (0.185, 0.025), "ky": (0.130, 0.070),
         "sr": (0.018, 0.026), "tt": (0.087, 0.033), "cnh": (0.070, 0.065), "dv": (0.073, 0.035),
         "hi": (0.183, 0.045), "ml": (0.093, 0.080), "mr": (0.017, 0.017), "or": (0.098, 0.087),
         "skr": (0.135, 0.035), "id": (0.013, 0.013), "vi": (0.083, 0.110), "yue": (0.290, 0.185)}


def run_dir(seed, c, arm, lang, _idx={}):
    """Baseline run directory for one arm/seed/language, wherever it lives (tools/experiments.py)."""
    if not _idx:
        from experiments import index
        _idx.update(index())
    return _idx.get((arm, "baseline", None, None, seed), {}).get((c, lang))


def p1(a, b):  # one-sided Wilcoxon, H1: a < b
    d = np.array(a) - np.array(b)
    return wilcoxon(a, b, alternative="less").pvalue if np.any(d != 0) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--metric", default="IMLM_eq1", choices=["IMLM_eq1", "IMLM_repo", "DCF"])
    args = ap.parse_args()
    m = args.metric
    order = [(c, l) for c in LANGS for l in LANGS[c]]

    res, bad = {}, []  # res[seed][(c,l,arm)] = metric
    for s in args.seeds:
        res[s] = {}
        for c, l in order:
            for arm in ("oal", "oalcf"):
                d = run_dir(s, c, arm, l)
                if d is None:
                    continue
                r = score(d)
                if r["sessions"] != SESS[c]:
                    bad.append(f"s{s} {c}_{arm}/{l}: {r['sessions']}/{SESS[c]} sessions")
                    continue
                res[s][(c, l, arm)] = r[m]
    full = [s for s in args.seeds if len(res[s]) == 40]

    print(f"Paper reproduction over stream-order seeds, metric {m}")
    print("A seed reshuffles the test stream only; bootstrap and model seed are fixed (DEVIATIONS.md).\n")
    print("runs done per seed: " + ", ".join(f"s{s} {len(res[s])}/40" for s in args.seeds))
    if bad:
        print("INCOMPLETE/UNHEALTHY: " + "; ".join(bad))
    print()

    print("== Per seed: OAL-CF vs OAL (paper: 17/20, Table V 0.105 -> 0.049, one-sided Wilcoxon)")
    for s in full:
        a = [res[s][(c, l, "oalcf")] for c, l in order]
        b = [res[s][(c, l, "oal")] for c, l in order]
        w = sum(x < y for x, y in zip(a, b))
        print(f"   seed {s}: CF better in {w:2d}/20 | mean OAL {np.mean(b):.4f} -> OAL-CF {np.mean(a):.4f} "
              f"({100 * (np.mean(a) / np.mean(b) - 1):+.0f}%) | p = {p1(a, b):.4f}")
    if len(full) >= 2:
        a = [np.mean([res[s][(c, l, "oalcf")] for s in full]) for c, l in order]
        b = [np.mean([res[s][(c, l, "oal")] for s in full]) for c, l in order]
        w = sum(x < y for x, y in zip(a, b))
        print(f"   seed-mean ({len(full)} seeds): CF better in {w}/20 | mean OAL {np.mean(b):.4f} -> "
              f"OAL-CF {np.mean(a):.4f} ({100 * (np.mean(a) / np.mean(b) - 1):+.0f}%) | p = {p1(a, b):.4f}")
        wins = [sum(res[s][(c, l, "oalcf")] < res[s][(c, l, "oal")] for c, l in order) for s in full]
        print(f"   win count across seeds: min {min(wins)}, max {max(wins)}, mean {np.mean(wins):.1f}")

    seeds_any = [s for s in args.seeds if res[s]]
    print(f"\n== Per language, mean (sd) over available seeds; paper Fig. 12 read-off for reference")
    print(f"   {'':4s}{'lang':6s} {'n':>2s} {'OAL':>15s} {'OAL-CF':>15s} {'CF wins':>8s}   "
          f"{'paper OAL':>9s} {'paper CF':>8s}  per-seed OAL-CF")
    for c, l in order:
        ss = [s for s in seeds_any if (c, l, "oal") in res[s] and (c, l, "oalcf") in res[s]]
        if not ss:
            continue
        o = np.array([res[s][(c, l, "oal")] for s in ss])
        f = np.array([res[s][(c, l, "oalcf")] for s in ss])
        sd = lambda x: f"({x.std(ddof=1):.3f})" if len(x) > 1 else "(  -  )"
        po, pf = PAPER[l]
        print(f"   {c:4s}{l:6s} {len(ss):2d} {o.mean():.4f} {sd(o)} {f.mean():.4f} {sd(f)} "
              f"{int((f < o).sum()):>4d}/{len(ss)}   {po:9.3f} {pf:8.3f}  "
              + " ".join(f"s{s}:{v:.3f}" for s, v in zip(ss, f)))


if __name__ == "__main__":
    main()
