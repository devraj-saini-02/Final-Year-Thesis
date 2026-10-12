#!/usr/bin/env python3
"""
Paper-style figures and tables for the SLV baselines (OAL vs OAL-CF), built from every run's
per-session scores.csv (one row per session; every session is used, nothing is subsampled).

Per-session curves use the authors' own formulas from oalcf-main/plotting/trend_plot.py
(cumulative postquential / prequential IMLM and DCF), with one correction: the bootstrap is
the true 8 samples (paper Sec. V-D). trend_plot.py takes n_samples[0] - n_al[0], which for
OAL-CF also counts session 0's CF labels as bootstrap. Final values equal
tools/score_runs.py's IMLM_eq1 (authors' reading).

Outputs (mirroring the paper):
  fig04_stream_order.png    Fig. 4   target positions in the AC test stream: Dirichlet (used) vs uniform
  fig07_trends.png          Fig. 7   postquential IMLM, prequential IMLM and their difference over sessions
  fig08_oalcf_pre_post.png  Fig. 8   OAL-CF prequential vs postquential IMLM and DCF
                                     (the paper's post-CF curve is not recorded by the repo)
  fig09_prevalence.png      Fig. 9   target prevalence in the adaptation pool over sessions, per corpus
  fig12_imlm_eq1.png        Fig. 12  final IMLM per language, OAL vs OAL-CF (Eq. 1, authors' reading)
  fig12_imlm_repo.png                the same with the repo scorer
  fig12_vs_paper.png                 ours next to the paper's Fig. 12 (read-off, +-0.005)
  tables.md, per_language.csv        Table V (means across languages), Table VII (Wilcoxon), per language

Usage: python tools/paper_figures.py [--seed N] [--out OUT_DIR]
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
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_runs import score  # noqa: E402
from experiments import run_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = {
    "ac": ("Africa", ["bas", "ha", "yo"]),
    "cr": ("Caucasus", ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"]),
    "sea": ("SE Asia", ["yue", "id", "vi"]),
    "sa": ("S Asia", ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]),
}
# Fig. 12 read-off (docs/other/REPRO_BRIEF.md section 7), IMLM (OAL, OAL-CF), +-0.005
PAPER = {"bas": (.042, .036), "ha": (.020, .038), "yo": (.057, .012), "ckb": (.100, .030),
         "cv": (.305, .210), "hy-AM": (.032, .028), "kmr": (.185, .025), "ky": (.130, .070),
         "sr": (.018, .026), "tt": (.087, .033), "cnh": (.070, .065), "dv": (.073, .035),
         "hi": (.183, .045), "id": (.013, .013), "ml": (.093, .080), "mr": (.017, .017),
         "or": (.098, .087), "skr": (.135, .035), "vi": (.083, .110), "yue": (.290, .185)}
ARMS = {"oal": ("OAL", "tab:blue"), "oalcf": ("OAL-CF", "tab:orange")}


def run_dir(seed, corpus, arm, lang):
    """runs/baseline/<arm>/seed_N/<corpus>_<lang> (tools/experiments.py knows the layout)."""
    return run_path(arm, "baseline", None, None, seed, corpus, lang)


def trends(s):
    """Cumulative per-session metrics (authors' trend_plot.py formulas, true bootstrap)."""
    g = lambda c: s[c].to_numpy(dtype=float)
    boot = g("n_samples")[0] - g("n_al")[0] - g("cf_tp")[0] - g("cf_fp")[0]
    n_all = np.cumsum(g("pre_ps") + g("pre_ns"))
    n_pos, n_neg = np.cumsum(g("pre_ps")), np.cumsum(g("pre_ns"))
    al = np.cumsum(g("n_al"))
    with np.errstate(divide="ignore", invalid="ignore"):
        fnr = np.nan_to_num(np.cumsum(g("fns")) / n_pos)
        fpr = np.nan_to_num(np.cumsum(g("fps")) / n_neg)
        pre_fnr = np.nan_to_num(np.cumsum(g("pre_fns")) / n_pos)
        pre_fpr = np.nan_to_num(np.cumsum(g("pre_fps")) / n_neg)
    post = (np.cumsum(g("fps")) + boot + al) / n_all + fnr
    pre = (np.cumsum(g("pre_fps")) + boot + al - g("n_al")[0]) / n_all + pre_fnr
    return dict(post_imlm=post, pre_imlm=pre, diff_imlm=pre - post,
                post_dcf=0.75 * fnr + 0.25 * fpr, pre_dcf=0.75 * pre_fnr + 0.25 * pre_fpr,
                p_target=g("p_target") * 100)


def ragged_mean(arrays):
    """Mean at each session over the languages whose run reaches that session."""
    m = np.full((len(arrays), max(len(a) for a in arrays)), np.nan)
    for i, a in enumerate(arrays):
        m[i, :len(a)] = a
    return np.nanmean(m, axis=0), (~np.isnan(m)).sum(axis=0)


def mark_dropouts(ax, counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i - 1]:
            ax.axvline(i - 0.5, color="grey", lw=0.6, ls=":")


def fig04(out):
    order = [l.strip() for l in open(os.path.join(ROOT, "data/AfricanContinentLID/orders/cluster_order_test")) if l.strip()]
    order = order[: (len(order) // 720) * 720]            # the 83 whole sessions actually presented
    loc = np.array([n.split("_")[2] for n in order])
    uniform = loc[np.random.default_rng(0).permutation(len(loc))]  # display only, not used by any run
    colors = {"yo": "tab:green", "ha": "tab:orange", "bas": "tab:blue"}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.2), sharex=True)
    for ax, seq, title in ((axes[0], loc, "Dirichlet"), (axes[1], uniform, "Uniform")):
        for row, lang in enumerate(["bas", "ha", "yo"]):
            ax.eventplot(np.where(seq == lang)[0], lineoffsets=row, linelengths=0.8, colors=colors[lang], linewidths=0.4)
        ax.set_yticks([0, 1, 2], ["bas", "ha", "yo"])
        ax.set_ylabel(title)
    axes[1].set_xlabel("Sample index")
    axes[1].set_xlim(0, len(loc))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig04_stream_order.png"), dpi=300)
    plt.close(fig)


def fig07_08(T, out):
    langs = [(c, l) for c in CORPORA for l in CORPORA[c][1]]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6), sharex=True)
    for ax, key, lab in zip(axes, ["post_imlm", "pre_imlm", "diff_imlm"],
                            ["Postquential\nIMLM", "Prequential\nIMLM", "Pre-post IMLM\ndifference"]):
        for arm, (name, col) in ARMS.items():
            m, n = ragged_mean([T[(c, l, arm)][key] for c, l in langs])
            ax.plot(m, color=col, label=name, lw=1)
        mark_dropouts(ax, n)
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Session")
    axes[-1].set_xlim(-1, len(n))
    fig.suptitle("Mean over the 20 SLV languages (dotted lines: a corpus's stream ends)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig07_trends.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.5), sharex=True)
    for ax, (pre, post, lab) in zip(axes, [("pre_imlm", "post_imlm", "IMLM"), ("pre_dcf", "post_dcf", "DCF")]):
        for key, name, col in ((pre, "Prequential", "tab:blue"), (post, "Postquential", "tab:orange")):
            m, n = ragged_mean([T[(c, l, "oalcf")][key] for c, l in langs])
            ax.plot(m, color=col, label=name, lw=1)
        mark_dropouts(ax, n)
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Session")
    axes[-1].set_xlim(-1, len(n))
    fig.suptitle("OAL-CF, mean over the 20 SLV languages (post-CF evaluation is not recorded by the repo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig08_oalcf_pre_post.png"), dpi=300)
    plt.close(fig)


def fig09(T, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))
    for ax, (c, (title, langs)) in zip(axes.flat, CORPORA.items()):
        for arm, (name, col) in ARMS.items():
            m, _ = ragged_mean([T[(c, l, arm)]["p_target"] for l in langs])
            ax.plot(m, color=col, label=name, lw=1)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Target prevalence (%)")
    for ax in axes[1]:
        ax.set_xlabel("Session")
    axes[1, 0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig09_prevalence.png"), dpi=300)
    plt.close(fig)


def fig12(S, out):
    rows = [sorted(CORPORA["ac"][1] + CORPORA["cr"][1], key=str.lower),
            sorted(CORPORA["sa"][1] + CORPORA["sea"][1], key=str.lower)]
    for key, fname, ylab in (("IMLM_eq1", "fig12_imlm_eq1.png", "IMLM (Eq. 1)"),
                             ("IMLM_repo", "fig12_imlm_repo.png", "IMLM (repo scorer)")):
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5))
        for ax, langs in zip(axes, rows):
            x = np.arange(len(langs))
            for j, (arm, (name, col)) in enumerate(ARMS.items()):
                ax.bar(x + (j - 0.5) * 0.4, [S[(l, arm)][key] for l in langs], 0.4, color=col, label=name)
            ax.set_xticks(x, langs)
            ax.set_ylabel(ylab)
        axes[0].legend()
        axes[1].set_xlabel("Language")
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, langs in zip(axes, rows):
        x = np.arange(len(langs))
        w = 0.2
        ax.bar(x - 1.5 * w, [S[(l, "oal")]["IMLM_eq1"] for l in langs], w, color="tab:blue", label="OAL (ours)")
        ax.bar(x - 0.5 * w, [S[(l, "oalcf")]["IMLM_eq1"] for l in langs], w, color="tab:orange", label="OAL-CF (ours)")
        ax.bar(x + 0.5 * w, [PAPER[l][0] for l in langs], w, color="tab:blue", alpha=0.35, hatch="//", label="OAL (paper)")
        ax.bar(x + 1.5 * w, [PAPER[l][1] for l in langs], w, color="tab:orange", alpha=0.35, hatch="//", label="OAL-CF (paper)")
        ax.set_xticks(x, langs)
        ax.set_ylabel("IMLM (Eq. 1)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("Language")
    fig.suptitle("Ours vs the paper's Fig. 12 (paper values read off the figure, +-0.005)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig12_vs_paper.png"), dpi=300)
    plt.close(fig)


def tables(S, SL, out):
    recs = []
    for c, (_, langs) in CORPORA.items():
        for l in langs:
            o, f, ol, fl = S[(l, "oal")], S[(l, "oalcf")], SL[(l, "oal")], SL[(l, "oalcf")]
            recs.append(dict(corpus=c, lang=l,
                             oal_imlm_eq1=o["IMLM_eq1"], oalcf_imlm_eq1=f["IMLM_eq1"],
                             oal_imlm_eq1_literal=ol["IMLM_eq1"], oalcf_imlm_eq1_literal=fl["IMLM_eq1"],
                             oal_imlm_repo=o["IMLM_repo"], oalcf_imlm_repo=f["IMLM_repo"],
                             oal_dcf=o["DCF"], oalcf_dcf=f["DCF"], oal_fnr=o["FNR"], oalcf_fnr=f["FNR"],
                             oal_fpr=o["FPR"], oalcf_fpr=f["FPR"],
                             paper_oal=PAPER[l][0], paper_oalcf=PAPER[l][1]))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, "per_language.csv"), index=False, float_format="%.4f")

    metrics = [("imlm_eq1", "IMLM, Eq. 1 (authors' reading)"), ("imlm_eq1_literal", "IMLM, Eq. 1 (literal)"),
               ("imlm_repo", "IMLM, repo scorer"), ("dcf", "DCF"), ("fnr", "FNR"), ("fpr", "FPR")]
    lines = ["# SLV baselines, seed 0: tables", "",
             "Generated by `tools/paper_figures.py` from the 40 runs' `scores.csv`. Means are plain",
             "averages over the 20 target languages; the paper does not say how its Table V pools.", "",
             "## Table V: mean over the 20 SLV languages", "",
             "| Paradigm | " + " | ".join(n for _, n in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for arm, name in (("oal", "OAL"), ("oalcf", "OAL-CF")):
        lines.append(f"| {name} | " + " | ".join(f"{df[f'{arm}_{k}'].mean():.4f}" for k, _ in metrics) + " |")
    lines += ["", "Paper Table V (pooled, SLV): IMLM 0.105 (OAL) and 0.049 (OAL-CF).", "",
              "## Table VII: is OAL-CF lower than OAL? One-sided Wilcoxon signed-rank, n = 20", "",
              "| Metric | OAL-CF lower in | p-value |", "|---|---|---|"]
    for k, n in metrics[:4]:
        a, b = df[f"oal_{k}"].to_numpy(), df[f"oalcf_{k}"].to_numpy()
        lines.append(f"| {n} | {int((b < a).sum())}/20 | {wilcoxon(b, a, alternative='less').pvalue:.2e} |")
    lines += ["", "Paper Table VII (SLV, OAL-CF vs OAL): IMLM p = 1.8e-3, DCF p = 1.8e-3.", "",
              "## Per language: IMLM (OAL -> OAL-CF)", "",
              "| Corpus | Lang | Eq. 1 (authors) | Eq. 1 (literal) | Repo scorer | DCF | Paper Fig. 12 |",
              "|---|---|---|---|---|---|---|"]
    for r in recs:
        f2 = lambda a, b: f"{r[a]:.4f} -> {r[b]:.4f}"
        lines.append(f"| {r['corpus'].upper()} | {r['lang']} | {f2('oal_imlm_eq1', 'oalcf_imlm_eq1')} | "
                     f"{f2('oal_imlm_eq1_literal', 'oalcf_imlm_eq1_literal')} | {f2('oal_imlm_repo', 'oalcf_imlm_repo')} | "
                     f"{f2('oal_dcf', 'oalcf_dcf')} | {r['paper_oal']:.3f} -> {r['paper_oalcf']:.3f} |")
    open(os.path.join(out, "tables.md"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="stream-order seed of the baseline runs")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/figures/base_paper_reproduction"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    T, S, SL = {}, {}, {}
    for c, (_, langs) in CORPORA.items():
        for l in langs:
            for arm in ARMS:
                d = run_dir(args.seed, c, arm, l)
                T[(c, l, arm)] = trends(pd.read_csv(os.path.join(d, "scores.csv")))
                S[(l, arm)] = score(d)
                SL[(l, arm)] = score(d, literal=True)
                assert abs(T[(c, l, arm)]["post_imlm"][-1] - S[(l, arm)]["IMLM_eq1"]) < 1e-9, (c, l, arm)

    fig04(args.out)
    fig07_08(T, args.out)
    fig09(T, args.out)
    fig12(S, args.out)
    tables(S, SL, args.out)
    print(f"wrote figures and tables to {args.out}")


if __name__ == "__main__":
    main()
