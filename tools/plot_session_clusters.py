"""Draw the per-session clusters that the diversity extension (oalcf-main/utils/diverse_query.py) forms.

Clusters are rebuilt exactly as run_dq.py builds them: each session's 720 clips, their own ECAPA
embedding (the centre frame of the context window), agglomerative clustering with cosine distance,
average linkage and threshold 0.25 (the extension's own `cluster_session`). They depend only on the
audio, so they are identical for every target language, gamma and arm of a corpus. The rebuild is
checked session by session against the cluster counts a finished run logged (al_diversity.csv);
any mismatch stops the script.

Outputs (runs/analysis/cluster_analysis/):
  <corpus>_stream.png    every session: language make-up, clusters vs languages, largest-cluster
                         share, and how closely clusters follow languages (purity)
  <corpus>_sessions.png  four sessions drawn in 2-D (t-SNE of the embeddings), coloured by cluster
                         (top row) and by language group (bottom row)
  <corpus>_sessions.csv  the per-session numbers behind the stream figure

Usage: python tools/plot_session_clusters.py --corpus cr
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # training jobs may be running; stay light
import argparse  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/oalcf-main")
from utils.diverse_query import cluster_session  # noqa: E402  the extension's own clustering

CORPUS = {"ac": ("AfricanContinentLID", "African Continent"), "cr": ("CaucasusRegionLID", "Caucasus Region"),
          "sea": ("SoutheastAsiaLID", "Southeast Asia"), "sa": ("SouthAsiaLID", "South Asia")}
TARGETS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
           "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
# the seven filler locales plus Toki Pona (docs/other/DEVIATIONS.md)
OUT_OF_REGION = ["eo", "eu", "hu", "tr", "nl", "fa", "ar", "tok"]
SESSION = 720

# reference palette (dataviz skill), light mode
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUP = {"target": "#2a78d6", "out_of_region": "#eb6834", "regional": "#1baf7a"}  # slots 1-3
GROUP_LABEL = {"target": "target languages", "out_of_region": "out-of-region (filler + Toki Pona)",
               "regional": "other regional languages"}
CLUSTER = ["#e87ba4", "#008300", "#4a3aa7"]  # slots 5-7: the three largest clusters of a session
REST = "#d9d8d1"  # every other cluster

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})
HALO = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=list(CORPUS))
ap.add_argument("--threshold", type=float, default=0.25)
ap.add_argument("--out", default=f"{ROOT}/runs/analysis/cluster_analysis")
a = ap.parse_args()
dirname, cname = CORPUS[a.corpus]
D = f"{ROOT}/data/{dirname}"
os.makedirs(a.out, exist_ok=True)

# ---------------- rebuild every session's clusters ----------------
order = [x.strip() for x in open(f"{D}/orders/cluster_order_test") if x.strip()]
n_sess = len(order) // SESSION
whole = len(order) == n_sess * SESSION
order = order[:n_sess * SESSION]
print(f"{cname}: loading {len(order):,} embeddings ({n_sess} sessions)")
feats = np.stack([np.load(f"{D}/ecapalang/test/{n}.npy") for n in order]).astype(np.float64)
if whole:
    # src/dataset.py:317-321 wraps positions with `% (len - 1)`, so in every run the stream's very last
    # clip is represented by the embedding of the dataset's first entry: the first bootstrap
    # (train-order) clip. Replicated so the rebuilt clusters are the ones the runs used.
    first_train = open(f"{D}/orders/cluster_order_train").readline().strip()
    feats[-1] = np.load(f"{D}/ecapalang/train/{first_train}.npy")
locale = np.array([n.split("_")[2] for n in order])
group = np.where(np.isin(locale, TARGETS[a.corpus]), "target",
                 np.where(np.isin(locale, OUT_OF_REGION), "out_of_region", "regional"))

rows, labels = [], []
for s in range(n_sess):
    sl = slice(s * SESSION, (s + 1) * SESSION)
    lab, k = cluster_session(np.ascontiguousarray(feats[sl]), a.threshold)
    labels.append(lab)
    sizes = np.bincount(lab, minlength=k)
    loc, g = locale[sl], group[sl]
    majority = sum(pd.Series(loc[lab == c]).value_counts().iloc[0] for c in range(k))
    rows.append(dict(session=s, n_clusters=k, n_singletons=int((sizes == 1).sum()),
                     largest_cluster=int(sizes.max()), largest_cluster_pct=100 * sizes.max() / SESSION,
                     n_languages=len(set(loc)), purity_pct=100 * majority / SESSION,
                     n_target=int((g == "target").sum()),
                     target_pct=100 * (g == "target").mean(),
                     out_of_region_pct=100 * (g == "out_of_region").mean(),
                     regional_pct=100 * (g == "regional").mean()))
df = pd.DataFrame(rows)

ref_path = f"{ROOT}/runs/diversity_extension/oal_t0.25_g0.8/seed_0/{a.corpus}_{TARGETS[a.corpus][0]}/al_diversity.csv"
ref = pd.read_csv(ref_path)
same = (len(ref) == n_sess) and all((ref[c].values == df[c].values).all()
                                    for c in ("n_clusters", "n_singletons", "largest_cluster"))
if not same:
    sys.exit(f"STOP: the rebuilt clusters do not match {ref_path}; the figure would not show what the runs used")
print(f"rebuild matches the clusters logged during the runs in all {n_sess} sessions "
      f"(n_clusters, n_singletons, largest_cluster vs {os.path.relpath(ref_path, ROOT)})")
df.round(2).to_csv(f"{a.out}/{a.corpus}_sessions.csv", index=False)

# sessions to draw: spread over the range of cluster counts, among sessions with some target clips
cand = df[df.n_target >= 10] if (df.n_target >= 10).sum() >= 4 else df
cand = cand.sort_values(["n_clusters", "session"]).reset_index(drop=True)
picks = sorted({int(cand.session[int(round(q * (len(cand) - 1)))]) for q in (0.1, 0.4, 0.7, 0.95)})
groups = [gk for gk in ("target", "out_of_region", "regional") if df[f"{gk}_pct"].sum() > 0]


def tidy(ax, grid_y=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------- figure 1: the whole stream ----------------
fig = plt.figure(figsize=(12, 10), layout="constrained")
ax = fig.subplots(4, 1, sharex=True, height_ratios=[1.3, 1, 0.8, 0.8])
x = df.session.values
bottom = np.zeros(n_sess)
for gk in groups:
    ax[0].bar(x, df[f"{gk}_pct"], bottom=bottom, width=1.0, color=GROUP[gk], linewidth=0, label=GROUP_LABEL[gk])
    bottom += df[f"{gk}_pct"].values
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("% of session")
ax[0].set_title("Who is speaking: language make-up of each session", loc="left", fontsize=10, pad=8)
ax[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(groups), fontsize=8.5, handlelength=1.2)

ax[1].plot(x, df.n_clusters, color=INK, linewidth=1.6)
ax[1].plot(x, df.n_languages, color=MUTED, linewidth=1.4, linestyle=(0, (4, 2)))
ax[1].set_ylim(0, max(df.n_clusters.max(), df.n_languages.max()) * 1.15)
ax[1].set_ylabel("count")
ax[1].set_title(f"Clusters per session (mean {df.n_clusters.mean():.1f}) vs languages present "
                f"(mean {df.n_languages.mean():.1f})", loc="left", fontsize=10, pad=8)
ax[1].text(n_sess + 0.8, df.n_clusters.iloc[-1], "clusters", color=INK, fontsize=8.5, va="center", path_effects=HALO)
ax[1].text(n_sess + 0.8, df.n_languages.iloc[-1], "languages", color=INK2, fontsize=8.5, va="center",
           path_effects=HALO)

ax[2].bar(x, df.largest_cluster_pct, width=0.8, color=INK2, linewidth=0)
ax[2].set_ylim(0, 100)
ax[2].set_ylabel("% of session")
ax[2].set_title(f"Largest cluster's share of the session (mean {df.largest_cluster_pct.mean():.0f}%)",
                loc="left", fontsize=10, pad=8)

ax[3].plot(x, df.purity_pct, color=INK2, linewidth=1.6)
ax[3].set_ylim(0, 100)
ax[3].set_ylabel("%")
ax[3].set_title(f"Do clusters follow languages? Share of clips whose cluster is mostly their own language "
                f"(mean {df.purity_pct.mean():.0f}%)", loc="left", fontsize=10, pad=8)
ax[3].set_xlabel(f"session (720 clips each, in stream order)")
ax[3].set_xlim(-1, n_sess + 9)

for i, axi in enumerate(ax):
    tidy(axi)
    for s in picks:
        axi.axvline(s, color=INK, linewidth=0.7, linestyle=":", zorder=3)
        if i == 1:
            axi.text(s, axi.get_ylim()[1], f" S{s}", color=INK, fontsize=8, va="top", ha="left", path_effects=HALO)
fig.suptitle(f"{cname}: the extension's clusters in each of the {n_sess} sessions", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.supxlabel("Clusters rebuilt from the audio embeddings exactly as run_dq.py forms them (cosine, average "
              "linkage, threshold 0.25). They are the same for every target language and γ. "
              "Dotted lines: the sessions drawn in the second figure.", x=0.01, ha="left", fontsize=8, color=INK2)
fig.savefig(f"{a.out}/{a.corpus}_stream.png", dpi=150)
plt.close(fig)

# ---------------- figure 2: four sessions in 2-D ----------------
fig = plt.figure(figsize=(4.2 * len(picks), 9.4), layout="constrained")
top, bot = fig.subfigures(2, 1, hspace=0.04)
ax_top, ax_bot = top.subplots(1, len(picks)), bot.subplots(1, len(picks))
for j, s in enumerate(picks):
    sl = slice(s * SESSION, (s + 1) * SESSION)
    Z = feats[sl] / np.linalg.norm(feats[sl], axis=1, keepdims=True)  # cosine geometry
    P = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(Z)
    lab, loc, g, r = labels[s], locale[sl], group[sl], df.iloc[s]
    rank = np.argsort(-np.bincount(lab), kind="stable")[:3]

    axc = ax_top[j]
    rest = ~np.isin(lab, rank)
    axc.scatter(P[rest, 0], P[rest, 1], s=11, c=REST, linewidths=0.3, edgecolors=SURFACE)
    for i, c in enumerate(rank):
        m = lab == c
        axc.scatter(P[m, 0], P[m, 1], s=11, c=CLUSTER[i], linewidths=0.3, edgecolors=SURFACE)
    tg = g == "target"
    axc.scatter(P[tg, 0], P[tg, 1], s=24, facecolors="none", edgecolors=INK, linewidths=0.6)
    axc.set_title(f"Session {s}: {int(r.n_clusters)} clusters, {int(r.n_languages)} languages\n"
                  f"largest cluster {r.largest_cluster_pct:.0f}% · purity {r.purity_pct:.0f}% · "
                  f"{int(r.n_target)} target clips", loc="left", fontsize=9)

    axl = ax_bot[j]
    for gk in ("regional", "out_of_region", "target"):
        m = g == gk
        axl.scatter(P[m, 0], P[m, 1], s=11, c=GROUP[gk], linewidths=0.3, edgecolors=SURFACE)
    for code, cnt in pd.Series(loc).value_counts().head(6).items():
        if cnt < 15:
            continue
        m = loc == code
        axl.text(*np.median(P[m], axis=0), code, fontsize=10, fontweight="bold", color=INK, ha="center",
                 va="center", path_effects=HALO)
    for axi in (axc, axl):
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_color(GRID)

dot = dict(marker="o", linestyle="", markersize=6, markeredgewidth=0)
top.legend(handles=[Line2D([], [], color=CLUSTER[0], label="largest cluster", **dot),
                    Line2D([], [], color=CLUSTER[1], label="2nd largest", **dot),
                    Line2D([], [], color=CLUSTER[2], label="3rd largest", **dot),
                    Line2D([], [], color=REST, label="all other clusters", **dot),
                    Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none",
                           markeredgecolor=INK, markeredgewidth=0.8, label="target-language clip")],
           loc="outside lower center", ncol=5, fontsize=8.5)
top.suptitle("Coloured by the extension's clusters", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
bot.legend(handles=[Line2D([], [], color=GROUP[gk], label=GROUP_LABEL[gk], **dot) for gk in groups],
           loc="outside lower center", ncol=len(groups), fontsize=8.5)
bot.suptitle("Coloured by language (labels: the six largest languages in the session)", x=0.01, ha="left",
             fontsize=10.5, fontweight="bold")
fig.supxlabel("Each dot is one clip, placed by t-SNE of its ECAPA embedding; the 2-D layout is approximate and "
              "its axes mean nothing. Clusters were computed in the full 256-D space.", x=0.01, ha="left",
              fontsize=8, color=INK2)
fig.suptitle(f"{cname}: four sessions up close", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.savefig(f"{a.out}/{a.corpus}_sessions.png", dpi=150)
plt.close(fig)
print(f"wrote {a.out}/{a.corpus}_stream.png, {a.corpus}_sessions.png, {a.corpus}_sessions.csv; sessions drawn: {picks}")
