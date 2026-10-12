"""Summarize the diversity extension (run_dq.py) with OAL-CF for one or more corpora.

Usage: python tools/summarize_dq_oalcf.py --corpus cr --gammas 0.8 0.5 0.2
       python tools/summarize_dq_oalcf.py --corpus ac cr sea --gammas 0.8 0.5 0.2   (pooled)

Compares each gamma with OAL-CF against baseline OAL-CF, against the same gamma with OAL
(runs/dq_oal), and against baseline OAL, paired per language (one-sided Wilcoxon, as in the
paper's Section VI-G). Only the gammas passed are read, so a partial batch can be summarized. The
significance threshold stays Bonferroni over the 3 planned gammas (0.8 / 0.5 / 0.2), p < 0.0167,
whichever subset is summarized.

--arm/--root exist so the script can be exercised on the OAL extension runs (runs/dq_oal).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/tools")
from score_runs import score  # noqa: E402

LANGS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
         "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
TAG = {"0.8": "dq080", "0.5": "dq050", "0.2": "dq020"}
ALPHA = 0.05 / 3

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", nargs="+", required=True, choices=list(LANGS))
ap.add_argument("--gammas", nargs="+", required=True, choices=list(TAG))
ap.add_argument("--arm", default="oalcf", choices=["oal", "oalcf"])
ap.add_argument("--root", default=f"{ROOT}/runs/seed_0")
a = ap.parse_args()

pairs = [(c, l) for c in a.corpus for l in LANGS[c]]
n = len(pairs)
A, B, Q = a.root, f"{ROOT}/runs/seed_0", f"{ROOT}/runs/seed_0"
LOGS = f"{os.path.dirname(A)}/logs"
tags = {g: TAG[g] for g in a.gammas}
arm = a.arm.upper().replace("CF", "-CF")


def new(t, c, l):
    return f"{A}/{c}_{a.arm}_{t}/{l}"


print(f"Diversity-penalized AL with {arm}, {' + '.join(c.upper() for c in a.corpus)} ({n} languages), "
      f"seed 0, CPU; gamma {' / '.join(a.gammas)}\n")
print("== health")
bad = 0
for g, t in tags.items():
    for c, l in pairs:
        d, log = new(t, c, l), f"{LOGS}/{c}_{a.arm}_{t}_{l}.log"
        rows = sum(1 for _ in open(f"{d}/scores.csv")) - 1 if os.path.exists(f"{d}/scores.csv") else 0
        txt = open(log, errors="ignore").read().replace("\r", "\n") if os.path.exists(log) else ""
        nan = sum(1 for x in txt.splitlines() if "test/loss" in x and "nan" in x.lower()) // 2
        ok = rows == SESS[c] and nan == 0 and "used: False" in txt and os.path.exists(f"{d}/state_dict.pt")
        if not ok:
            bad += 1
            print(f"  CHECK gamma={g} {c}/{l}: sessions {rows}/{SESS[c]}, NaN {nan}")
print(f"  {len(tags) * n - bad}/{len(tags) * n} runs OK\n")
if bad:
    sys.exit("stopping: fix the runs above before comparing")


def compare(title, base_dir):
    print(f"== {title}")
    for g, t in tags.items():
        for metric, key in (("IMLM Eq.1 (authors)", "IMLM_eq1"), ("IMLM repo", "IMLM_repo"), ("DCF", "DCF")):
            x = np.array([score(new(t, c, l))[key] for c, l in pairs])
            y = np.array([score(base_dir(t, c, l))[key] for c, l in pairs])
            p = wilcoxon(x, y, alternative="less").pvalue if np.any(x != y) else 1.0
            print(f"  gamma={g}  {metric:20s} better in {int((x < y).sum())}/{n}, worse in {int((x > y).sum())}/{n} | "
                  f"mean {y.mean():.4f} -> {x.mean():.4f} | p = {p:.4f}{'  *' if p < ALPHA else ''}")
    print()


print(f"One-sided Wilcoxon (the gamma {arm} arm lower). Significant (*) only if p < {ALPHA:.4f} "
      f"(Bonferroni over 3 planned gammas). With n = {n} the smallest possible p is {0.5 ** n:.4f}.\n")
if a.arm == "oalcf":
    compare("gamma OAL-CF vs baseline OAL-CF (does the extension help OAL-CF?)", lambda t, c, l: f"{B}/{c}_oalcf/{l}")
    compare("gamma OAL-CF vs gamma OAL (does CF still help under the extension?)", lambda t, c, l: f"{Q}/{c}_oal_{t}/{l}")
compare(f"gamma {arm} vs baseline OAL", lambda t, c, l: f"{B}/{c}_oal/{l}")

if a.arm == "oalcf":
    print("== annotation volume (mean per run): AL / CF true positives / CF false positives")
    dirs = {"baseline": lambda c, l: f"{B}/{c}_oalcf/{l}"}
    for g, t in tags.items():
        dirs[f"gamma={g}"] = (lambda t: lambda c, l: new(t, c, l))(t)
    for name, f in dirs.items():
        s = [score(f(c, l)) for c, l in pairs]
        print(f"  {name:10s} AL {np.mean([x['AL'] for x in s]):6.0f} | CF TP {np.mean([x['CF_tp'] for x in s]):6.0f} | "
              f"CF FP {np.mean([x['CF_fp'] for x in s]):6.0f}")
    print()

print("== how much each gamma changed the selection (al_diversity.csv, mean per session)")
for g, t in tags.items():
    df = pd.concat([pd.read_csv(f"{new(t, c, l)}/al_diversity.csv") for c, l in pairs])
    print(f"  gamma={g}: queries shared with plain uncertainty top-8 {df.overlap_with_runpy.mean():.2f}/8 | "
          f"distinct clusters among the 8 {df.distinct_clusters_picked.mean():.2f} | "
          f"clusters per session {df.n_clusters.mean():.1f}")

print("\n== per language, IMLM Eq.1 (authors' reading), OAL / OAL-CF")
print(f"  {'':4s}{'lang':6s} {'baseline':>17s} | " + " | ".join(f"{'gamma ' + g:>17s}" for g in tags))
for c, l in pairs:
    cells = [f"{score(f'{B}/{c}_oal/{l}')['IMLM_eq1']:.4f} / {score(f'{B}/{c}_oalcf/{l}')['IMLM_eq1']:.4f}"]
    for t in tags.values():
        oal = score(f"{Q}/{c}_oal_{t}/{l}")["IMLM_eq1"]
        cf = score(new(t, c, l))["IMLM_eq1"] if a.arm == "oalcf" else float("nan")
        cells.append(f"{oal:.4f} / {cf:.4f}")
    print(f"  {c:4s}{l:6s} " + " | ".join(cells))
