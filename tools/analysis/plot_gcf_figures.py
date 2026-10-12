#!/usr/bin/env python3
"""Figures comparing OAL-CF with OAL-GCF (grouped corrective feedback; run spec oalcf+cfd0.1).
Every value is a mean over stream-order seeds 0-4 (seeds are never compared with each other).
Only the IMLM equations that charge the expert for corrective-feedback labels are plotted: the
dissertation's Eq. 3.8 (IMLM_thesis) and the released scorer with expert labels (IMLM_repo_expert).
Layouts follow the paper: Fig. 12 (per-language grouped bars, two rows) and Fig. 7 (trends over time).

    python tools/analysis/plot_gcf_figures.py          -> runs/figures/oal_gcf/*.png

Palette (validated with the dataviz skill's validator, light surface #fcfcfb, all pairs):
violet #6250d6 = OAL-CF, aqua #1baf7a = OAL-GCF. Aqua is below 3:1 contrast with the surface, so key
values are direct-labelled in ink colours; the error part of stacked bars also carries a 45-degree
texture so the two parts never rely on lightness alone.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from experiments import ALL_LANGS, CORPUS_NAME, LANGS, SESSIONS, run_path  # noqa: E402
from score_runs import score  # noqa: E402

OUT = os.path.join(ROOT, "runs/figures/oal_gcf")
SEEDS = range(5)
CF_NAME, GCF_NAME = "OAL-CF", "OAL-GCF"
VIOLET, AQUA = "#6250d6", "#1baf7a"
VIOLET_T, AQUA_T = "#c5beef", "#a6e3cc"          # tints of the same hues, for the error part of stacks
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
ARMS = [("base", CF_NAME, VIOLET, VIOLET_T), ("gcf", GCF_NAME, AQUA, AQUA_T)]
METRICS = {"IMLM_thesis": "IMLM (thesis Eq. 3.8)", "IMLM_repo_expert": "IMLM (repo scorer, expert labels)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "text.color": INK, "hatch.color": SURFACE, "hatch.linewidth": 1.0,
})


# ---------------------------------------------------------------------------- data
def run_dirs(c, l, s):
    return {"base": run_path("oalcf", "baseline", None, None, s, c, l),
            "gcf": run_path("oalcf", "cfdedup", 0.1, None, s, c, l)}


def expert_cf_per_session(arm, d, sc):
    """CF labels the expert gave each session: all CF clips for OAL-CF, one per group for OAL-GCF."""
    if arm == "base":
        return (sc.cf_tp + sc.cf_fp).to_numpy(float)
    return pd.read_csv(os.path.join(d, "cf_dedup.csv")).expert_labels.to_numpy(float)


def load():
    rows, curves = [], {}
    for c, l in ALL_LANGS:
        for s in SEEDS:
            for arm, d in run_dirs(c, l, s).items():
                sc = pd.read_csv(os.path.join(d, "scores.csv"))
                r = score(d)
                cfe = expert_cf_per_session(arm, d, sc)
                n = (sc.pre_ps + sc.pre_ns).sum()
                nt = sc.pre_ps.sum()
                ann = 8 + sc.n_al.sum() + cfe.sum()
                rows.append(dict(corpus=c, lang=l, seed=s, arm=arm, IMLM_thesis=r["IMLM_thesis"],
                                 IMLM_repo_expert=r["IMLM_repo_expert"], cf_expert=cfe.sum(),
                                 part_labels=ann / n, part_fa=sc.fps.sum() / n, part_miss=sc.fns.sum() / nt))
                # cumulative IMLM (thesis Eq. 3.8) after each session, post- and prequential
                N = np.cumsum(sc.pre_ps + sc.pre_ns).to_numpy(float)
                NT = np.cumsum(sc.pre_ps).to_numpy(float)
                lab = 8 + np.cumsum(sc.n_al.to_numpy(float) + cfe)          # labels given up to and incl. session t
                lab_prev = np.concatenate([[8.0], lab[:-1]])                  # labels available before session t
                with np.errstate(divide="ignore", invalid="ignore"):
                    miss_post = np.nan_to_num(np.cumsum(sc.fns.to_numpy(float)) / NT)
                    miss_pre = np.nan_to_num(np.cumsum(sc.pre_fns.to_numpy(float)) / NT)
                post = (np.cumsum(sc.fps.to_numpy(float)) + lab) / N + miss_post
                pre = (np.cumsum(sc.pre_fps.to_numpy(float)) + lab_prev) / N + miss_pre
                curves[(c, l, s, arm)] = (post, pre)
    return pd.DataFrame(rows), curves


def lang_means(D, metric):
    m = D.groupby(["corpus", "lang", "arm"], sort=False)[metric].mean().unstack("arm")
    return m.loc[[(c, l) for c, l in ALL_LANGS]]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(os.path.join(OUT, name), ROOT))


def legend_arms(ax, **kw):
    handles = [Patch(color=col, label=name) for _, name, col, _ in ARMS]
    return ax.legend(handles=handles, ncols=2, handlelength=1.2, handletextpad=0.5, **kw)


# ---------------------------------------------------------------------------- Fig. 12 analogue
def fig_per_language(D, metric, fname):
    m = lang_means(D, metric)
    a, b = m["base"].to_numpy(), m["gcf"].to_numpy()
    p = wilcoxon(b, a, alternative="less").pvalue
    rows = [[k for k in m.index if k[0] in ("ac", "cr")], [k for k in m.index if k[0] in ("sea", "sa")]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6))
    for ax, keys in zip(axes, rows):
        x = np.arange(len(keys))
        for k, (arm, name, col, _) in enumerate(ARMS):
            v = m.loc[keys, arm].to_numpy()
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=col, edgecolor=SURFACE, linewidth=1.5, label=name)
        ymax = float(m.loc[keys].to_numpy().max())
        for xi, key in zip(x, keys):
            ch = 100 * (m.loc[key, "gcf"] / m.loc[key, "base"] - 1)
            top = max(m.loc[key, "base"], m.loc[key, "gcf"])
            ax.text(xi, top + ymax * 0.03, f"{ch:+.0f}%", ha="center", va="bottom", fontsize=8.5,
                    color=INK if ch < 0 else MUTED, fontweight="semibold" if ch < 0 else "normal")
        ax.set_xticks(x, [f"{l}\n{c.upper()}" for c, l in keys])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("IMLM")
        ax.set_ylim(0, ymax * 1.18)
    better = int((b < a).sum())
    axes[0].set_title("IMLM per target language, mean of 5 stream orders (lower is better)", pad=24)
    legend_arms(axes[0], loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.1)
    axes[1].set_xlabel(f"OAL-GCF lower for {better} of 20 languages; mean {a.mean():.4f} to {b.mean():.4f} "
                       f"({100 * (b.mean() / a.mean() - 1):+.1f}%), one-sided Wilcoxon p = {p:.4f}", fontsize=9)
    fig.tight_layout()
    save(fig, fname)


# ---------------------------------------------------------------------------- Fig. 7 analogue
def fig_trends(curves):
    T = max(SESSIONS.values())
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for ax, k, ylabel in zip(axes, (0, 1), ("postquential IMLM\n(thesis Eq. 3.8)", "prequential IMLM\n(thesis Eq. 3.8)")):
        for arm, name, col, _ in ARMS:
            stack = np.full((len(ALL_LANGS) * len(SEEDS), T), np.nan)
            i = 0
            for c, l in ALL_LANGS:
                for s in SEEDS:
                    v = curves[(c, l, s, arm)][k]
                    stack[i, :len(v)] = v
                    i += 1
            mean = np.nanmean(stack, axis=0)
            ax.plot(np.arange(T), mean, color=col, lw=2, label=name)
            ax.text(T - 1, mean[-1], f" {mean[-1]:.3f}", va="bottom" if arm == "base" else "top", fontsize=8.5,
                    color=INK2)
        for end in sorted(set(SESSIONS.values()))[:-1]:
            ax.axvline(end - 0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, T * 1.06)
    for corpus, n in SESSIONS.items():
        if n < T:
            axes[0].text(n - 0.5, axes[0].get_ylim()[1], f"{corpus.upper()} ends ", ha="right", va="top",
                         fontsize=7.5, color=MUTED)
    axes[1].set_xlabel("session")
    axes[0].set_title("IMLM trends across the stream, mean over 20 languages and 5 stream orders", pad=24)
    legend_arms(axes[0], loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.1)
    fig.tight_layout()
    save(fig, "fig3_imlm_trends.png")


# ---------------------------------------------------------------------------- IMLM split into its parts
def fig_components(D):
    D = D.assign(part_errors=D.part_fa + D.part_miss)
    g = D.groupby(["corpus", "arm"], sort=False)[["part_labels", "part_errors"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    corpora = list(LANGS)
    x = np.arange(len(corpora))
    for k, (arm, name, col, tint) in enumerate(ARMS):
        xs = x + (k - 0.5) * 0.36
        lab = g.xs(arm, level="arm").loc[corpora, "part_labels"].to_numpy()
        err = g.xs(arm, level="arm").loc[corpora, "part_errors"].to_numpy()
        ax.bar(xs, lab, width=0.34, color=col, edgecolor=SURFACE, linewidth=1.5)
        ax.bar(xs, err, bottom=lab, width=0.34, color=tint, edgecolor=SURFACE, linewidth=1.5, hatch="///")
        for xi, l_, e_ in zip(xs, lab, err):
            ax.text(xi, l_ / 2, f"{l_:.3f}", ha="center", va="center", fontsize=8, color=SURFACE, fontweight="semibold")
            ax.text(xi, l_ + e_ + 0.002, f"{l_ + e_:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [CORPUS_NAME[c] for c in corpora])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("IMLM (thesis Eq. 3.8)")
    handles = [Patch(color=VIOLET, label=f"{CF_NAME}: expert-label part"),
               Patch(color=AQUA, label=f"{GCF_NAME}: expert-label part"),
               Patch(facecolor=VIOLET_T, hatch="///", edgecolor=SURFACE, label=f"{CF_NAME}: error part"),
               Patch(facecolor=AQUA_T, hatch="///", edgecolor=SURFACE, label=f"{GCF_NAME}: error part")]
    ax.legend(handles=handles, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.1)
    ax.set_title("Where the IMLM saving comes from: the expert-label part shrinks in every corpus;\n"
                 "the error part rises only in African Continent and Southeast Asia (yo, yue)", pad=44)
    ax.text(0, -0.16, "Mean over languages and 5 stream orders. IMLM = (expert labels + false alarms) / N + misses / N_target.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig4_imlm_components_by_corpus.png")


# ---------------------------------------------------------------------------- labelling effort
def fig_cf_labels(D):
    g = D.groupby(["corpus", "lang", "arm"], sort=False).cf_expert.mean().unstack("arm")
    g["saved"] = 1 - g.gcf / g.base
    g = g.sort_values("saved", ascending=False)
    a, b = g.base.to_numpy(), g.gcf.to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    y = np.arange(len(g))
    ax.hlines(y, b, a, color=GRID, lw=3, zorder=1)
    ax.scatter(a, y, s=46, color=VIOLET, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=CF_NAME)
    ax.scatter(b, y, s=46, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=4, label=GCF_NAME)
    ax.set_yticks(y, [f"{l}  ({c.upper()})" for c, l in g.index])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    xmax = a.max()
    for yi, sv in zip(y, g.saved):
        ax.text(xmax * 1.04, yi, f"−{100 * sv:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax * 1.16)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    tot = 1 - D[D.arm == "gcf"].cf_expert.sum() / D[D.arm == "base"].cf_expert.sum()
    ax.set_title(f"Corrective-feedback labels the expert gives per run: {100 * tot:.0f}% fewer with OAL-GCF", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xlabel("expert CF labels per run (mean of 5 stream orders)")
    ax.text(1.0, -0.1, "right column: CF labels saved", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, "fig5_expert_cf_labels_per_language.png")


def fig_cumulative_labels():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, c in zip(axes, LANGS):
        series = {}
        for arm, name, col, _ in ARMS:
            cum = []
            for l in LANGS[c]:
                for s in SEEDS:
                    d = run_dirs(c, l, s)[arm]
                    sc = pd.read_csv(os.path.join(d, "scores.csv"))
                    cum.append(8 + np.cumsum(sc.n_al.to_numpy(float) + expert_cf_per_session(arm, d, sc)))
            series[arm] = np.mean(cum, axis=0)
        x = np.arange(1, len(series["base"]) + 1)
        ax.fill_between(x, series["gcf"], series["base"], color=AQUA, alpha=0.12, lw=0)
        for arm, name, col, _ in ARMS:
            ax.plot(x, series[arm], color=col, lw=2, label=name)
            ax.text(x[-1], series[arm][-1], f" {series[arm][-1]:,.0f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(f"{CORPUS_NAME[c]}  (−{100 * (1 - series['gcf'][-1] / series['base'][-1]):.0f}%)", fontsize=10.5)
        ax.set_xlabel("session"); ax.set_xlim(1, len(x) * 1.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_ylabel("cumulative expert labels\n(bootstrap + AL + CF)")
    axes[0].legend(loc="upper left", handlelength=1.4)
    fig.suptitle("Expert labelling effort over the stream (mean over languages)", x=0.01, ha="left",
                 fontweight="semibold", fontsize=12)
    fig.tight_layout()
    save(fig, "fig6_cumulative_expert_labels.png")


def fig_tradeoff():
    S = pd.read_csv(os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf/tables/cf_dedup_sessions.csv"))
    g = S.groupby("t").sum(numeric_only=True)
    g = g[g.index >= 0.08]
    saved, fa_err = 100 * (1 - g.asked / g.n_cf), 100 * g.fp_as_t / g.n_fp
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(saved, fa_err, color=GRID, lw=2, zorder=1)
    ax.scatter(saved, fa_err, s=46, color=VIOLET, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    sel = g.index == 0.10
    ax.scatter(saved[sel], fa_err[sel], s=150, facecolor="none", edgecolor=AQUA, linewidth=2.2, zorder=4)
    for t, xv, yv in zip(g.index, saved, fa_err):
        ax.annotate(f"{t:.2f}", (xv, yv), textcoords="offset points", xytext=(-6, 7), ha="right", fontsize=8.5,
                    color=INK if t == 0.10 else INK2, fontweight="semibold" if t == 0.10 else "normal")
    ax.annotate("used: 0.10\n30% saved, 0.8% of false alarms mislabelled", (saved[sel].iloc[0], fa_err[sel].iloc[0]),
                textcoords="data", xytext=(8, 18), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("CF labels saved (%)")
    ax.set_ylabel("false alarms given a wrong 'target' label (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, None)
    ax.set_title("Grouping threshold: labels saved vs labelling error (offline measurement)", fontsize=11)
    save(fig, "fig7_threshold_tradeoff.png")



# ---------------------------------------------------------------------------- method diagram
def fig_method():
    from matplotlib.patches import FancyBboxPatch, Circle
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, text, edge=MUTED, fill=SURFACE, lw=1.2, size=9, weight="normal", color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=lw, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1, rad=0.0, color=MUTED):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, connectionstyle=f"arc3,rad={rad}"))

    top = [("Session t\n720 clips arrive", ), ("Prequential\ntest", ), ("AL: expert labels\nthe 8 most\nuncertain clips", ),
           ("Adapt\nthe model", ), ("Postquential\ntest", ), ("Model flags clips\nas target\n(CF candidates)", )]
    w, h, y0, gap = 13.2, 9.5, 36, 3.1
    xs = [1 + i * (w + gap) for i in range(len(top))]
    for x, (t,) in zip(xs, top):
        box(x, y0, w, h, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w + 0.5, y0 + h / 2, b - 0.5, y0 + h / 2)
    ax.text(1, 49.2, "One session of online active learning with corrective feedback", fontsize=12,
            fontweight="semibold", color=INK, va="top")

    # the corrective-feedback step: OAL-CF vs OAL-GCF
    cf_x, cf_w = 24, 25
    gcf_x, gcf_w = 54, 44
    by, bh = 4, 21
    box(cf_x, by, cf_w, bh, "", edge=VIOLET, lw=2)
    ax.text(cf_x + cf_w / 2, by + bh - 2.6, "OAL-CF", ha="center", fontsize=11, fontweight="semibold", color=VIOLET)
    ax.text(cf_x + cf_w / 2, by + bh / 2 - 1.5, "Expert listens to\nevery flagged clip:\nconfirms targets,\ncorrects false alarms",
            ha="center", va="center", fontsize=9, color=INK, linespacing=1.4)
    box(gcf_x, by, gcf_w, bh, "", edge=AQUA, lw=2)
    ax.text(gcf_x + 1.5, by + bh - 2.6, "OAL-GCF (ours)", ha="left", fontsize=11, fontweight="semibold", color=AQUA)
    ax.text(gcf_x + 1.5, by + bh / 2 - 1.5,
            "1. Group near-duplicate flagged\n    clips (ECAPA cosine distance < 0.10)\n2. Expert labels one clip per group\n"
            "3. Its label is copied to the rest",
            ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)
    # mini illustration of groups
    groups = [((86.5, 18.5), [(85.2, 19.4), (87.6, 19.6), (86.0, 17.3), (87.9, 17.6)]),
              ((94.0, 17.5), [(93.0, 18.2), (95.0, 16.9)]),
              ((92.5, 10.5), [(91.3, 11.2), (93.6, 11.3), (92.6, 9.4)]),
              ((85.5, 10.0), [(85.5, 10.0)])]
    for (gx, gy), pts in groups:
        if len(pts) > 1:
            ax.add_patch(Circle((gx, gy), 2.7, facecolor="none", edgecolor=MUTED, lw=1.0, ls="--"))
        for k, (px, py) in enumerate(pts):
            ax.add_patch(Circle((px, py), 0.6, facecolor=AQUA if k == 0 else SURFACE, edgecolor=AQUA, lw=1.4, zorder=3))
    ax.add_patch(Circle((80.6, 6.0), 0.6, facecolor=AQUA, edgecolor=AQUA, lw=1.4))
    ax.text(81.7, 6.0, "expert labels it", va="center", fontsize=7.5, color=INK2)
    ax.add_patch(Circle((90.4, 6.0), 0.6, facecolor=SURFACE, edgecolor=AQUA, lw=1.4))
    ax.text(91.5, 6.0, "label copied", va="center", fontsize=7.5, color=INK2)

    # flagged clips go to either CF step
    fx = xs[-1] + w / 2
    arrow(fx + 2, y0 - 0.6, fx + 2, by + bh + 0.6, color=AQUA)
    arrow(fx - 2, y0 - 0.6, cf_x + cf_w - 2, by + bh + 0.6, rad=0.12, color=VIOLET)
    # both feed the second adaptation, then the next session
    ad_x, ad_w = 1, 15
    box(ad_x, by + 5, ad_w, 11, "Adapt again on\nall labels,\nthen session t+1", size=9)
    arrow(cf_x - 0.6, by + 10.5, ad_x + ad_w + 0.6, by + 10.5, color=VIOLET)
    arrow(gcf_x + 4, by - 0.6, ad_x + ad_w / 2 + 3, by + 4.4, rad=-0.18, color=AQUA)
    arrow(ad_x + ad_w / 2, by + 16.6, xs[0] + w / 2, y0 - 0.6, color=MUTED)
    save(fig, "fig8_method_diagram.png")


# ---------------------------------------------------------------------------- saving vs IMLM change
def fig_saving_vs_imlm(D):
    lab = D.groupby(["corpus", "lang", "arm"], sort=False).cf_expert.mean().unstack("arm")
    im = lang_means(D, "IMLM_thesis")
    saved = 100 * (1 - lab.gcf / lab.base)
    ch = 100 * (im["gcf"] / im["base"] - 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axhspan(ax.get_ylim()[0] if False else -100, 0, color=AQUA, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, lw=0.9)
    worse = ch > 0
    ax.scatter(saved[~worse], ch[~worse], s=70, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="IMLM lower with OAL-GCF")
    ax.scatter(saved[worse], ch[worse], s=70, marker="D", color="#e87ba4", edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="IMLM higher with OAL-GCF")
    nudge = {"kmr": (-7, 6, "right"), "ckb": (7, -11, "left"), "ml": (-7, -10, "right"), "bas": (7, -9, "left"),
             "hi": (7, -2, "left"), "dv": (3, 7, "left"), "skr": (-7, 4, "right"), "tt": (6, 4, "left")}
    for (c, l), xv, yv in zip(saved.index, saved, ch):
        dx, dy, ha = nudge.get(l, (6, 4, "left"))
        ax.annotate(l, (xv, yv), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK2)
    ax.set_xlim(0, max(saved) * 1.12)
    ax.set_ylim(min(ch) * 1.25, max(ch) * 1.5)
    ax.set_xlabel("corrective-feedback labels saved (%)")
    ax.set_ylabel("change in IMLM (%)  (below 0 = better)")
    ax.set_title("Every language saves labels; IMLM improves for 18 of 20", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=0.1, handletextpad=0.3)
    save(fig, "fig9_saving_vs_imlm_per_language.png")


# ---------------------------------------------------------------------------- saving vs session load
def fig_saving_vs_load():
    rows = []
    for c, l in ALL_LANGS:
        for s in SEEDS:
            cf = pd.read_csv(os.path.join(run_dirs(c, l, s)["gcf"], "cf_dedup.csv"))
            rows.append(cf[cf.n_cf > 0][["n_cf", "expert_labels"]])
    S = pd.concat(rows)
    bins = [0, 5, 10, 25, 50, 100, 200, 500, 10 ** 6]
    names = ["1-5", "6-10", "11-25", "26-50", "51-100", "101-200", "201-500", "500+"]
    S["bin"] = pd.cut(S.n_cf, bins, labels=names)
    g = S.groupby("bin", observed=True).agg(n=("n_cf", "size"), cf=("n_cf", "sum"), ex=("expert_labels", "sum"))
    g["saved"] = 100 * (1 - g.ex / g.cf)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(g))
    ax.bar(x, g.saved, width=0.62, color=AQUA, edgecolor=SURFACE, linewidth=1.5)
    for xi, v in zip(x, g.saved):
        ax.text(xi, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK, fontweight="semibold")
    ax.set_xticks(x, [f"{nm}\n{n:,} sessions" for nm, n in zip(names, g.n)]); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("clips flagged for corrective feedback in the session")
    ax.set_ylabel("CF labels saved (%)")
    ax.set_ylim(0, max(g.saved) * 1.18)
    ax.set_title("The saving is largest where corrective feedback is most expensive")
    save(fig, "fig10_saving_vs_session_load.png")


# ---------------------------------------------------------------------------- target share in the pool (paper Fig. 9)
def fig_pool_prevalence():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, c in zip(axes, LANGS):
        for arm, name, col, _ in ARMS:
            v = np.mean([pd.read_csv(os.path.join(run_dirs(c, l, s)[arm], "scores.csv")).p_target.to_numpy()
                         for l in LANGS[c] for s in SEEDS], axis=0)
            ax.plot(np.arange(len(v)), 100 * v, color=col, lw=2, label=name)
        ax.axhline(50, color=MUTED, lw=0.8, ls=":")
        ax.set_title(CORPUS_NAME[c], fontsize=10.5)
        ax.set_xlabel("session")
    axes[0].set_ylabel("target share of the\nadaptation pool (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right", handlelength=1.4)
    fig.suptitle("Class balance of the training pool is unchanged by grouping (mean over languages)", x=0.01, ha="left",
                 fontweight="semibold", fontsize=12)
    fig.tight_layout()
    save(fig, "fig11_target_share_in_pool.png")



# ---------------------------------------------------------------------------- one session, made visible
def fig_session_clusters(corpus="ac", lang="yo", session=26, threshold=0.10, fname="fig12_cf_session_clusters.png"):
    """The corrective-feedback clips of one session, grouped exactly as OAL-GCF groups them.
    Session chosen among the probed runs as the clearest large example (347 flagged clips, biggest saving,
    no wrong copy). The probe replays baseline OAL-CF, so the flagged set is the one OAL-CF sent to the expert.
    Layout: t-SNE (cosine, perplexity 30) - checked to keep 95% of grouped clips next to a clip of their own
    group (MDS 34%, PCA 29%). Groups are formed in the full 256-d embedding space, not on the map."""
    import pickle, types
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE
    sys.path.insert(0, os.path.join(ROOT, "oalcf-main"))
    from src.dataset import BaseLidData
    from utils.diverse_query import cluster_session
    corpdir = {"ac": "AfricanContinentLID", "cr": "CaucasusRegionLID", "sea": "SoutheastAsiaLID", "sa": "SouthAsiaLID"}
    langname = {"yo": "Yoruba", "ha": "Hausa", "ky": "Kyrgyz", "vi": "Vietnamese", "ml": "Malayalam", "skr": "Saraiki"}
    probe = os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf/probe", f"oalcf_{corpus}_{lang}_s1.pkl")
    rec = [r for r in pickle.load(open(probe, "rb"))["sessions"] if r["session"] == session][0]
    prm = types.SimpleNamespace(feat_root=os.path.join(ROOT, "data", corpdir[corpus], "ecapalang") + "/", env_name="test",
                                order_file="seed1", lid_target=lang, reverse_order=False, bootstrap=8,
                                samples_per_batch=720, context=3)
    ds = BaseLidData(prm); ds.load_external_bootstrap()
    pos = {b: i for i, b in enumerate(rec["base"])}
    idx = np.array([pos[b] for b in rec["cf_base"]])
    lab, post = rec["label"][idx], rec["p_tgt_post"][idx]
    Z = np.stack([ds._load_feat(ds.feat_files[b].format("ecapalang")) for b in rec["cf_base"]]).astype(np.float64)
    groups, ng = cluster_session(Z, threshold)
    sizes = np.bincount(groups)
    reps = {g: int(np.flatnonzero(groups == g)[np.argmax(post[groups == g])]) for g in range(ng)}
    is_rep = np.zeros(len(lab), bool); is_rep[list(reps.values())] = True
    wrong_mask = np.array([lab[k] != lab[reps[groups[k]]] for k in range(len(lab))])
    # ---- layout: groups placed by t-SNE of their centroids, then separated so outlines never overlap;
    #      clips inside a group laid out by that group's own PCA, scaled to fit its circle.
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    cent = np.stack([Zn[groups == g].mean(axis=0) for g in range(ng)])
    P = TSNE(n_components=2, metric="cosine", perplexity=min(15, ng - 1), init="pca", random_state=0).fit_transform(cent)
    P = (P - P.mean(axis=0)) / np.ptp(P, axis=0).max() * 100
    rad = 1.6 + 1.25 * np.sqrt(sizes)
    for _ in range(600):                                   # push overlapping circles apart
        moved = False
        for a in range(ng):
            d = P - P[a]
            dist = np.hypot(d[:, 0], d[:, 1])
            need = rad + rad[a] + 0.8
            clash = (dist < need) & (np.arange(ng) != a)
            if clash.any():
                moved = True
                for b in np.flatnonzero(clash):
                    v = d[b] / (dist[b] + 1e-9) if dist[b] > 1e-9 else np.array([1.0, 0.0])
                    push = (need[b] - dist[b]) / 2
                    P[b] += v * push; P[a] -= v * push
        if not moved:
            break
    XY = np.zeros((len(lab), 2))
    for g in range(ng):
        m = np.flatnonzero(groups == g)
        if len(m) == 1:
            XY[m[0]] = P[g]; continue
        loc = Zn[m] - Zn[m].mean(axis=0)
        u, sv, vt = np.linalg.svd(loc, full_matrices=False)
        xy = loc @ vt[:2].T if len(m) > 2 else np.array([[-1.0, 0.0], [1.0, 0.0]])
        xy = xy / (np.abs(xy).max() + 1e-12) * (rad[g] - 1.3)
        XY[m] = P[g] + xy

    TARGET, FALSE = AQUA, "#e87ba4"
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    for g in range(ng):
        col = TARGET if lab[reps[g]] == 1 else FALSE
        ax.add_patch(plt.Circle(P[g], rad[g], facecolor=col, alpha=0.12 if sizes[g] > 1 else 0.05, edgecolor="none", zorder=1))
        ax.add_patch(plt.Circle(P[g], rad[g], facecolor="none", edgecolor=col, lw=1.1 if sizes[g] > 1 else 0.7,
                                ls="-" if sizes[g] > 1 else ":", zorder=2))
        if sizes[g] >= 8:
            ax.text(P[g][0], P[g][1] + rad[g] - 0.9, f"{sizes[g]}", ha="center", va="top", fontsize=8.5,
                    color=INK, fontweight="semibold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    for cls, col, mk in ((1, TARGET, "o"), (0, FALSE, "^")):
        sel = lab == cls
        ax.scatter(XY[sel & ~is_rep, 0], XY[sel & ~is_rep, 1], s=22, marker=mk, facecolor=SURFACE, edgecolor=col,
                   linewidth=1.2, zorder=3)
        ax.scatter(XY[sel & is_rep, 0], XY[sel & is_rep, 1], s=48, marker=mk, color=col, edgecolor=INK, linewidth=0.7,
                   zorder=4)
    if wrong_mask.any():
        ax.scatter(XY[wrong_mask, 0], XY[wrong_mask, 1], s=150, facecolor="none", edgecolor=INK, lw=1.4, zorder=6)
        if wrong_mask.sum() <= 3:
            for k in np.flatnonzero(wrong_mask):
                what = (f"false alarm grouped with real {langname.get(lang, lang)} clips:\nit inherits the label "
                        f"'{langname.get(lang, lang)}' (a wrong copy)") if lab[k] == 0 else \
                       ("real clip grouped with false alarms:\nit inherits the label 'not the target' (a wrong copy)")
                ax.annotate(what, XY[k], xytext=(XY[k][0] + 16, XY[k][1] + 24), fontsize=8.5, color=INK,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9), zorder=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=GRID))
    lo, hi = (P - rad[:, None]).min(axis=0), (P + rad[:, None]).max(axis=0)
    ax.set_xlim(lo[0] - 2, hi[0] + 2); ax.set_ylim(lo[1] - 2, hi[1] + 4)
    name = langname.get(lang, lang)
    handles = [Patch(facecolor=TARGET, alpha=0.6, label=f"real {name} clip ({int(lab.sum())})")]
    if (lab == 0).any():
        handles.append(Patch(facecolor=FALSE, alpha=0.6,
                             label=f"false alarm: misclassified as {name} ({int((1 - lab).sum())})"))
    if wrong_mask.any():
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=12,
                              markeredgewidth=1.4, label=f"wrong copy: group label is wrong for this clip ({int(wrong_mask.sum())})"))
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=INK, markeredgewidth=0.6,
                       markersize=7.5, label=f"the expert labels it ({ng})"),
                Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.3,
                       markersize=6.5, label=f"label copied, no listening needed ({len(lab) - ng})"),
                Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=13,
                       label="circle: one group (number = clips in it)"),
                Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=9,
                       markeredgewidth=0.7, linestyle="none", label="dotted circle: a single clip (group of one)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=0.2)
    ax.set_title(f"{name}, session {session}: clips sent for corrective feedback, grouped by OAL-GCF",
                 pad=74 + 18 * max(0, (len(handles) - 5 + 1) // 2), fontsize=12)
    ax.text(0.0, -0.03, f"{len(lab)} clips flagged in this session ({int(lab.sum())} real {name}, "
            f"{int((1 - lab).sum())} false alarm{'s' if (1 - lab).sum() != 1 else ''}) fall into {ng} groups: {int((sizes >= 2).sum())} groups of "
            f"2 or more (largest {sizes.max()} clips) and {int((sizes == 1).sum())} single clips. "
            f"The expert labels {ng} clips instead of {len(lab)}: {100 * (1 - ng / len(lab)):.0f}% saved, "
            f"{int(wrong_mask.sum())} wrong cop{'ies' if wrong_mask.sum() != 1 else 'y'}.\nGroups: ECAPA cosine distance < {threshold:.2f} in the full embedding space. "
            "Layout: groups placed by t-SNE of their centres (similar groups nearer) and pushed apart so circles do not overlap;\n"
            "clips inside a circle show that group's own spread, scaled to fit. Circle area grows with the number of clips.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", linespacing=1.5)
    save(fig, fname)


def main():
    for f in os.listdir(OUT) if os.path.isdir(OUT) else []:       # this script owns the folder's figures
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    D, curves = load()
    fig_per_language(D, "IMLM_thesis", "fig1_imlm_thesis_per_language.png")
    fig_per_language(D, "IMLM_repo_expert", "fig2_imlm_repo_expert_per_language.png")
    fig_trends(curves)
    fig_components(D)
    fig_cf_labels(D)
    fig_cumulative_labels()
    fig_tradeoff()
    fig_method()
    fig_saving_vs_imlm(D)
    fig_saving_vs_load()
    fig_pool_prevalence()
    fig_session_clusters("ac", "yo", 73, fname="fig12_cf_session_clusters.png")
    fig_session_clusters("ac", "yo", 26, fname="fig12b_cf_session_clusters_burst.png")


if __name__ == "__main__":
    main()
