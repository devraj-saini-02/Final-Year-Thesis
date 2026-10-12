#!/usr/bin/env python3
"""
Definitive plots for the cluster-quota diversity experiment.

  score(x) = u(x) * gamma**(n_c/q_c),  greedy, agglomerative cosine clustering.
  gamma = 1.0 is baseline OAL (verified bit-identical to unmodified upstream).

gamma = 0.5 is deliberately EXCLUDED: only 3 of its 20 runs completed.

Outputs to runs/dq_plots/:
  <corp>_<lang>.png      per language, 4 metrics (IMLM, DCF, FNR, FPR)
  <corp>_aggregate.png   per corpus, mean over its languages
  pooled_paper_style.png all 20 languages pooled -- the paper's Fig. 7 method
  summary_imlm.png       per-language IMLM change, sorted

Usage: python tools/plot_dq_final.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN = os.path.join(ROOT, "oalcf-main")
OUT = os.path.join(ROOT, "runs", "dq_plots")
os.makedirs(OUT, exist_ok=True)

CORP = [("ac", "AC", ["bas", "ha", "yo"]),
        ("cr", "CR", ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"]),
        ("sea", "SEA", ["yue", "id", "vi"]),
        ("sa", "SA", ["cnh", "dv", "hi", "ml", "mr", "or", "skr"])]
ARCH = {"cr": "cr_oal_precap", "sea": "sea_oal_precap", "sa": "sa_oal_precap"}
NAME = {"bas": "Basaa", "ha": "Hausa", "yo": "Yoruba", "hy-AM": "Armenian",
        "ckb": "Sorani Kurdish", "cv": "Chuvash", "kmr": "Kurmanji Kurdish",
        "ky": "Kyrgyz", "sr": "Serbian", "tt": "Tatar", "yue": "Cantonese",
        "id": "Indonesian", "vi": "Vietnamese", "cnh": "Hakha Chin",
        "dv": "Dhivehi", "hi": "Hindi", "ml": "Malayalam", "mr": "Marathi",
        "or": "Odia", "skr": "Saraiki"}
GAMMAS = [1.0, 0.9, 0.7]
STYLE = {1.0: dict(color="#8d8d8d", lw=2.4, ls="--", label="OAL (baseline)"),
         0.9: dict(color="#2e4057", lw=1.7, ls="-", label="$\\gamma$=0.9"),
         0.7: dict(color="#d1495b", lw=1.7, ls="-", label="$\\gamma$=0.7")}
PANELS = [("cum_imlm", "Cumulative IMLM"), ("cum_dcf", "Cumulative DCF"),
          ("cum_fnr", "Cumulative FNR"), ("cum_fpr", "Cumulative FPR")]


def path(c, l, g):
    if g == 1.0:
        return (f"{MAIN}/output/dq_g100/ac_{l}" if c == "ac"
                else f"{MAIN}/output/{ARCH[c]}/{l}")
    return f"{MAIN}/output/dq_g{int(round(g*100)):03d}/{c}_{l}"


def load(c, l, g):
    d = pd.read_csv(os.path.join(path(c, l, g), "scores.csv"))
    cf, cn = d.fps.cumsum(), d.fns.cumsum()
    cs, cp = d.ns.cumsum(), d.ps.cumsum().replace(0, np.nan)
    d["cum_imlm"] = (cf + d.n_samples) / cs + cn / cp
    d["cum_fnr"] = cn / cp
    d["cum_fpr"] = cf / cs
    return d


def final_imlm(c, l, g):
    d = pd.read_csv(os.path.join(path(c, l, g), "scores.csv"))
    n = d.n_samples.values[-1]
    return (d.fps.sum() + n) / d.ns.sum() + d.fns.sum() / d.ps.sum()


def per_language():
    for c, C, langs in CORP:
        for l in langs:
            D = {g: load(c, l, g) for g in GAMMAS}
            b, g9, g7 = (final_imlm(c, l, g) for g in GAMMAS)
            fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
            for a, (k, nm) in zip(ax, PANELS):
                for g in GAMMAS:
                    a.plot(D[g][k].values, **STYLE[g])
                a.set_xlabel("Session"); a.set_ylabel(nm)
                a.set_title(nm); a.grid(alpha=.25)
            ax[0].legend(fontsize=8)
            fig.suptitle(f"{C} / {l} — {NAME[l]}     "
                         f"IMLM: OAL {b:.4f} | $\\gamma$=0.9 {g9:.4f} ({(g9-b)/b*100:+.1f}%) "
                         f"| $\\gamma$=0.7 {g7:.4f} ({(g7-b)/b*100:+.1f}%)", fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{OUT}/{c}_{l}.png", dpi=110)
            plt.close(fig)


def per_corpus():
    for c, C, langs in CORP:
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
        for g in GAMMAS:
            Ds = [load(c, l, g) for l in langs]
            n = min(len(d) for d in Ds)
            for a, (k, nm) in zip(ax, PANELS):
                a.plot(np.nanmean([d[k].values[:n] for d in Ds], axis=0), **STYLE[g])
        for a, (k, nm) in zip(ax, PANELS):
            a.set_xlabel("Session"); a.set_ylabel(f"{nm} (mean)")
            a.set_title(nm); a.grid(alpha=.25)
        ax[0].legend(fontsize=8)
        fig.suptitle(f"{C} corpus — mean over {len(langs)} target languages", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{c}_aggregate.png", dpi=110)
        plt.close(fig)


def pooled():
    """Paper Sec VI-C: cumulative over ALL languages; shorter corpora stop
    contributing after their last session."""
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for g in GAMMAS:
        S = [load(c, l, g) for c, _, langs in CORP for l in langs]
        T = max(len(d) for d in S)
        acc = {k: np.zeros(T) for k in ["fps", "fns", "ns", "ps", "nann"]}
        for d in S:
            n = len(d)
            acc["fps"][:n] += d.fps.values; acc["fns"][:n] += d.fns.values
            acc["ns"][:n] += d.ns.values;  acc["ps"][:n] += d.ps.values
            acc["nann"][:n] += np.diff(np.concatenate([[0], d.n_samples.values]))
        cf, cn, cs = np.cumsum(acc["fps"]), np.cumsum(acc["fns"]), np.cumsum(acc["ns"])
        cp, ca = np.cumsum(acc["ps"]), np.cumsum(acc["nann"])
        cp = np.where(cp == 0, np.nan, cp)
        fnr, fpr = cn / cp, cf / cs
        for a, v in zip(ax, [(cf + ca) / cs + fnr, .75 * fnr + .25 * fpr, fnr, fpr]):
            a.plot(v, **STYLE[g])
    for a, (_, nm) in zip(ax, PANELS):
        a.set_xlabel("Session"); a.set_ylabel(nm); a.set_title(nm); a.grid(alpha=.25)
    ax[0].legend(fontsize=8)
    fig.suptitle("Pooled over all 20 SLV target languages (paper Sec. VI-C methodology)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pooled_paper_style.png", dpi=120)
    plt.close(fig)


def summary():
    rows = []
    for c, C, langs in CORP:
        for l in langs:
            b = final_imlm(c, l, 1.0)
            rows.append((f"{l}", C, (final_imlm(c, l, 0.9) - b) / b * 100,
                         (final_imlm(c, l, 0.7) - b) / b * 100))
    rows.sort(key=lambda r: min(r[2], r[3]))
    lab = [f"{r[0]} ({r[1]})" for r in rows]
    y = np.arange(len(rows)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - w/2, [np.clip(r[2], -100, 300) for r in rows], w,
            color="#2e4057", label="$\\gamma$=0.9")
    ax.barh(y + w/2, [np.clip(r[3], -100, 300) for r in rows], w,
            color="#d1495b", label="$\\gamma$=0.7")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=9)
    ax.set_xlabel("IMLM change vs baseline OAL (%)   — negative is better, clipped at +300%")
    ax.set_title("Per-language IMLM change, all 20 SLV languages")
    ax.legend(); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/summary_imlm.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    per_language(); per_corpus(); pooled(); summary()
    print(f"wrote {len(os.listdir(OUT))} files to {OUT}")
