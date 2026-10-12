#!/usr/bin/env python3
"""
Analysis + convergence graphs for the cluster-quota diversity AL sweep.

Compares score(x) = u(x) * gamma**(n_c/q_c) against baseline OAL across all 20
SLV target languages.  gamma=1.0 is the matched control (penalty identically 1).

Outputs:
  runs/DQ_RESULTS.md                     full tables
  runs/dq_plots/{corpus}_{lang}.png      per-language convergence, all gammas
  runs/dq_plots/{corpus}_aggregate.png   per-corpus mean convergence
  runs/dq_plots/pooled.png               pooled across all 20 languages
  runs/dq_plots/gamma_curve.png          pooled IMLM/DCF vs gamma

Usage: python tools/plot_dq.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN = os.path.join(ROOT, "oalcf-main")
PLOTS = os.path.join(ROOT, "runs", "dq_plots")
os.makedirs(PLOTS, exist_ok=True)

CORPORA = {
    "ac":  ("AfricanContinentLID", ["bas", "ha", "yo"]),
    "cr":  ("CaucasusRegionLID", ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"]),
    "sea": ("SoutheastAsiaLID", ["yue", "id", "vi"]),
    "sa":  ("SouthAsiaLID", ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]),
}
GAMMAS = [0.5, 0.7, 0.9, 1.0]
COLORS = {0.5: "#9b2226", 0.7: "#d1495b", 0.9: "#2e4057", 1.0: "#8d8d8d"}


def tag(g):
    return "g%03d" % int(round(g * 100))


ARCHIVE = {"cr": "cr_oal_precap", "sea": "sea_oal_precap", "sa": "sa_oal_precap"}


def run_dir(g, short, lang):
    """gamma=1.0 is baseline OAL.  For CR/SEA/SA the archived precap runs are
    that baseline (verified bit-identical to gamma=1.0); AC's stream changed, so
    AC uses freshly-run dq_g100 baselines."""
    if g == 1.0 and short in ARCHIVE:
        p = os.path.join(MAIN, "output", ARCHIVE[short], lang)
        if os.path.isfile(os.path.join(p, "scores.csv")):
            return p
    return os.path.join(MAIN, "output", f"dq_{tag(g)}", f"{short}_{lang}")


def load(path):
    f = os.path.join(path, "scores.csv")
    if not os.path.isfile(f):
        return None
    d = pd.read_csv(f)
    if len(d) < 2:
        return None
    d["cum_fps"] = d["fps"].cumsum()
    d["cum_fns"] = d["fns"].cumsum()
    d["cum_ns"] = d["ns"].cumsum()
    d["cum_ps"] = d["ps"].cumsum()
    d["cum_fnr"] = d["cum_fns"] / d["cum_ps"].replace(0, np.nan)
    d["cum_fpr"] = d["cum_fps"] / d["cum_ns"].replace(0, np.nan)
    # n_samples is already a running cumulative annotation count
    d["cum_imlm"] = ((d["cum_fps"] + d["n_samples"]) / d["cum_ns"]
                     + d["cum_fns"] / d["cum_ps"].replace(0, np.nan))
    return d


def final(d):
    r = d.iloc[-1]
    fps, fns = d["fps"].sum(), d["fns"].sum()
    ns, ps = d["ns"].sum(), d["ps"].sum()
    nad = r["n_samples"]
    return dict(dcf=r["cum_dcf"], fnr=fns / ps if ps else np.nan,
                fpr=fps / ns if ns else np.nan,
                imlm=(fps + nad) / ns + (fns / ps if ps else np.nan),
                fps=fps, fns=fns, ns=ns, ps=ps, nad=nad, sessions=len(d))


def main():
    data = {}          # (g, short, lang) -> df
    for g in GAMMAS:
        for short, (_, langs) in CORPORA.items():
            for l in langs:
                d = load(run_dir(g, short, l))
                if d is not None:
                    data[(g, short, l)] = d

    have = sorted({g for (g, _, _) in data})
    print(f"loaded {len(data)} runs; gammas present: {have}")

    # ---------- per-language convergence ----------
    for short, (_, langs) in CORPORA.items():
        for l in langs:
            present = [g for g in GAMMAS if (g, short, l) in data]
            if len(present) < 2:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
            for g in present:
                d = data[(g, short, l)]
                style = dict(color=COLORS[g], lw=2.2 if g == 1.0 else 1.6,
                             ls="--" if g == 1.0 else "-",
                             label=f"$\\gamma$={g}" + (" (baseline)" if g == 1.0 else ""))
                ax[0].plot(d["cum_dcf"].values, **style)
                ax[1].plot(d["cum_imlm"].values, **style)
            for a, nm in zip(ax, ["Cumulative DCF", "Cumulative IMLM"]):
                a.set_xlabel("Session"); a.set_ylabel(nm); a.grid(alpha=.25)
            ax[0].set_title(f"{short}/{l} — DCF"); ax[1].set_title(f"{short}/{l} — IMLM")
            ax[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, f"{short}_{l}.png"), dpi=120)
            plt.close(fig)

    # ---------- per-corpus aggregate ----------
    for short, (_, langs) in CORPORA.items():
        present = [g for g in GAMMAS
                   if all((g, short, l) in data for l in langs)]
        if len(present) < 2:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
        for g in present:
            n = min(len(data[(g, short, l)]) for l in langs)
            dcf = np.mean([data[(g, short, l)]["cum_dcf"].values[:n] for l in langs], axis=0)
            iml = np.mean([data[(g, short, l)]["cum_imlm"].values[:n] for l in langs], axis=0)
            style = dict(color=COLORS[g], lw=2.2 if g == 1.0 else 1.6,
                         ls="--" if g == 1.0 else "-",
                         label=f"$\\gamma$={g}" + (" (baseline)" if g == 1.0 else ""))
            ax[0].plot(dcf, **style); ax[1].plot(iml, **style)
        for a, nm in zip(ax, ["Cumulative DCF", "Cumulative IMLM"]):
            a.set_xlabel("Session"); a.set_ylabel(f"{nm} (mean over languages)"); a.grid(alpha=.25)
        ax[0].set_title(f"{short} corpus — DCF"); ax[1].set_title(f"{short} corpus — IMLM")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f"{short}_aggregate.png"), dpi=120)
        plt.close(fig)

    # ---------- pooled ----------
    allkeys = [(s, l) for s, (_, ls) in CORPORA.items() for l in ls]
    pooled = {}
    for g in GAMMAS:
        if not all((g, s, l) in data for s, l in allkeys):
            continue
        agg = {k: 0 for k in ["fps", "fns", "ns", "ps", "nad"]}
        for s, l in allkeys:
            f = final(data[(g, s, l)])
            for k in agg:
                agg[k] += f[k]
        fnr = agg["fns"] / agg["ps"]; fpr = agg["fps"] / agg["ns"]
        pooled[g] = dict(imlm=agg["nad"] / agg["ns"] + fpr + fnr,
                         dcf=.75 * fnr + .25 * fpr, fnr=fnr, fpr=fpr,
                         adapt=agg["nad"] / agg["ns"])

    if len(pooled) >= 2:
        gs = sorted(pooled)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        ax[0].plot(gs, [pooled[g]["imlm"] for g in gs], "o-", color="#d1495b")
        ax[0].set_ylabel("Pooled IMLM")
        ax[1].plot(gs, [pooled[g]["dcf"] for g in gs], "o-", color="#2e4057")
        ax[1].set_ylabel("Pooled DCF")
        for a in ax:
            a.set_xlabel("$\\gamma$ (discount strength)"); a.grid(alpha=.25)
            if 1.0 in pooled:
                a.axhline(pooled[1.0]["imlm"] if a is ax[0] else pooled[1.0]["dcf"],
                          color="#8d8d8d", ls="--", lw=1, label="baseline ($\\gamma$=1)")
                a.legend(fontsize=8)
        fig.suptitle("Pooled over 20 SLV target languages")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "gamma_curve.png"), dpi=120)
        plt.close(fig)

    # ---------- diversity diagnostics ----------
    diag = {}
    for g in GAMMAS:
        rows = []
        for s, l in allkeys:
            f = os.path.join(run_dir(g, s, l), "al_diversity.csv")
            if os.path.isfile(f):
                rows.append(pd.read_csv(f))
        if rows:
            a = pd.concat(rows)
            diag[g] = dict(overlap=a.get("overlap_with_runpy", a["overlap_with_baseline"]).mean(),
                           distinct=a["distinct_clusters_picked"].mean(),
                           nclust=a["n_clusters"].mean(),
                           noise=a["n_singletons"].mean() if "n_singletons" in a else float("nan"),
                           u_pick=a["mean_u_picked"].mean(),
                           u_base=a["mean_u_baseline"].mean())

    # ---------- write report ----------
    out = os.path.join(ROOT, "runs", "DQ_RESULTS.md")
    with open(out, "w") as f:
        f.write("# Cluster-quota diversity-penalized AL — results\n\n")
        f.write("`score(x) = u(x) · γ^(n_c/q_c)`, greedy, HDBSCAN clustering "
                "(min_cluster_size=2, L2+PCA-4, noise as singletons). "
                "γ=1.0 is the matched control.\n\n")

        if pooled:
            f.write("## Pooled over all 20 languages\n\n")
            f.write("| γ | IMLM | DCF | FNR | FPR | annot | ΔIMLM vs γ=1 |\n|---|---|---|---|---|---|---|\n")
            for g in sorted(pooled):
                p = pooled[g]
                dd = (f"{(p['imlm']-pooled[1.0]['imlm'])/pooled[1.0]['imlm']*100:+.1f}%"
                      if 1.0 in pooled else "—")
                f.write(f"| {g} | {p['imlm']:.4f} | {p['dcf']:.4f} | {p['fnr']:.4f} "
                        f"| {p['fpr']:.4f} | {p['adapt']:.4f} | {dd} |\n")
            f.write("\n")

        if diag:
            f.write("## Selection diagnostics (mean per session)\n\n")
            f.write("| γ | clusters | noise frac | distinct clusters in 8 picks "
                    "| overlap with baseline top-8 | mean u picked | mean u baseline |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for g in sorted(diag):
                d = diag[g]
                f.write(f"| {g} | {d['nclust']:.1f} | {d['noise']:.3f} | {d['distinct']:.2f} "
                        f"| {d['overlap']:.2f}/8 | {d['u_pick']:.4f} | {d['u_base']:.4f} |\n")
            f.write("\n")

        f.write("## Per language\n\n")
        f.write("| Corpus | Lang | metric | " + " | ".join(f"γ={g}" for g in GAMMAS) + " | best γ |\n")
        f.write("|---|---|---|" + "---|" * (len(GAMMAS) + 1) + "\n")
        wins = {g: 0 for g in GAMMAS}
        for s, (_, langs) in CORPORA.items():
            for l in langs:
                vals = {g: (final(data[(g, s, l)]) if (g, s, l) in data else None)
                        for g in GAMMAS}
                if not any(vals.values()):
                    continue
                for metric in ["imlm", "dcf"]:
                    cells = []
                    for g in GAMMAS:
                        cells.append(f"{vals[g][metric]:.4f}" if vals[g] else "—")
                    ok = {g: vals[g][metric] for g in GAMMAS if vals[g]}
                    best = min(ok, key=ok.get) if ok else None
                    if metric == "imlm" and best is not None:
                        wins[best] += 1
                    f.write(f"| {s} | {l} | {metric.upper()} | " + " | ".join(cells)
                            + f" | {best if best is not None else '—'} |\n")
        f.write(f"\n**Best-γ counts by IMLM across languages:** "
                + ", ".join(f"γ={g}: {n}" for g, n in wins.items()) + "\n")
        if 1.0 in pooled:
            beat = sum(1 for s, l in allkeys
                       if all((g, s, l) in data for g in [1.0])
                       and any((g, s, l) in data and final(data[(g, s, l)])["imlm"]
                               < final(data[(1.0, s, l)])["imlm"]
                               for g in GAMMAS if g != 1.0))
            f.write(f"\n**Languages where some γ<1 beats the baseline on IMLM: "
                    f"{beat}/{len(allkeys)}**\n")
    print("wrote", out)
    print("wrote plots to", PLOTS)


if __name__ == "__main__":
    main()
