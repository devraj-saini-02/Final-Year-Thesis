#!/usr/bin/env python3
"""
Per-session convergence graphs for the SLV baselines, OAL vs OAL-CF.

For every language: cumulative DCF, cumulative IMLM by the paper's Eq. 1 (authors' reading)
and cumulative IMLM by the repo scorer, after each session. For every corpus: the same, averaged
over its languages (all runs in a corpus have the same number of sessions).

Every session is plotted: scores.csv has one row per session and nothing is subsampled. Each
point is the cumulative value up to and including that session, so the last point equals the
run's final score in tools/score_runs.py.

Usage: python tools/plot_convergence.py [--seed N] [--out OUT_DIR]
  reads the baseline runs of stream-order seed N (default 0): runs/baseline/<arm>/seed_N/<corpus>_<lang>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_runs import score  # noqa: E402
from experiments import run_path  # noqa: E402
from paper_figures import CORPORA, ARMS, trends  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = [("dcf", "Cumulative DCF"), ("imlm_eq1", "Cumulative IMLM (Eq. 1)"), ("imlm_repo", "Cumulative IMLM (repo scorer)")]


def curves(run_dir):
    s = pd.read_csv(os.path.join(run_dir, "scores.csv"))
    t = trends(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        repo = (s["fps"].cumsum() + s["n_samples"]) / s["ns"].cumsum() + np.nan_to_num(s["fns"].cumsum() / s["ps"].cumsum())
    c = dict(dcf=s["cum_dcf"].to_numpy(), imlm_eq1=t["post_imlm"], imlm_repo=repo.to_numpy())
    final = score(run_dir)
    assert abs(c["imlm_eq1"][-1] - final["IMLM_eq1"]) < 1e-9 and abs(c["imlm_repo"][-1] - final["IMLM_repo"]) < 1e-9
    return c


def plot(curve_by_arm, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, lab) in zip(axes, PANELS):
        for arm, (name, col) in ARMS.items():
            y = curve_by_arm[arm][key]
            ax.plot(np.arange(len(y)), y, color=col, lw=1, label=f"{name} (final {y[-1]:.4f})")
        ax.set_xlabel("Session")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="stream-order seed of the baseline runs")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/figures/convergence"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for c, (cname, langs) in CORPORA.items():
        per_lang = {l: {arm: curves(run_path(arm, "baseline", None, None, args.seed, c, l)) for arm in ARMS} for l in langs}
        for l in langs:
            plot(per_lang[l], f"{cname} / {l}: OAL vs OAL-CF, every session", os.path.join(args.out, f"{c}_{l}.png"))
            n += 1
        mean = {arm: {k: np.mean([per_lang[l][arm][k] for l in langs], axis=0) for k, _ in PANELS} for arm in ARMS}
        plot(mean, f"{cname}: mean over {len(langs)} languages, every session", os.path.join(args.out, f"{c}_mean.png"))
        n += 1
    print(f"wrote {n} convergence plots to {args.out}")


if __name__ == "__main__":
    main()
