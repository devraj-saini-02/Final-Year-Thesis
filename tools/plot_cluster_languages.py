"""Show how the extension's per-session clusters relate to languages, at two thresholds.

For a few sessions of one corpus: every clip is placed by t-SNE of its own ECAPA embedding (one layout
per session, shared by both thresholds, so only the cluster outlines change), coloured by language
(target languages dark, other languages light), with an outline around each cluster of 8 or more
clips. Next to each map is the language make-up of every cluster. Clusters are the extension's own
(`cluster_session`: cosine distance, average linkage), at the threshold the runs used (0.25) and at a
tighter one (0.15). The 0.25 clusters are checked against the counts the runs logged.
Descriptive only: see docs/other/CLUSTER_OBSERVATIONS.md.

Every target gets its own dark colour, most common target first. Only the first three are fully
distinguishable in a scatter; with CR's and SA's seven targets a few pairs are close (purple and blue
most of all), which the human accepted. The legend and the bars name every target.

Output: runs/analysis/cluster_analysis/<corpus>_clusters_025_vs_015.png
Usage:  python tools/plot_cluster_languages.py --corpus ac [--sessions 31 36 49]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # training jobs may be running; stay light
import argparse  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/oalcf-main")
from utils.diverse_query import cluster_session  # noqa: E402  the extension's own clustering

CORPUS = {"ac": ("AfricanContinentLID", "African Continent"), "cr": ("CaucasusRegionLID", "Caucasus Region"),
          "sea": ("SoutheastAsiaLID", "Southeast Asia"), "sa": ("SouthAsiaLID", "South Asia")}
TARGETS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
           "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
FIRST_TARGET = {"ac": "bas", "cr": "hy-AM", "sea": "yue", "sa": "cnh"}
NAMES = {
    # African Continent
    "af": "Afrikaans", "am": "Amharic", "bas": "Basaa", "dyu": "Dyula", "ha": "Hausa", "ig": "Igbo",
    "kab": "Kabyle", "kln": "Kalenjin", "lg": "Luganda", "ln": "Lingala", "luo": "Luo", "rw": "Kinyarwanda",
    "sw": "Swahili", "ti": "Tigrinya", "tw": "Twi", "yo": "Yoruba", "zgh": "Tamazight", "zu": "Zulu",
    # Caucasus Region
    "hy-AM": "Armenian", "ckb": "Central Kurdish", "cv": "Chuvash", "kmr": "Kurmanji", "ky": "Kyrgyz",
    "sr": "Serbian", "tt": "Tatar", "mhr": "Meadow Mari", "mrj": "Hill Mari", "ba": "Bashkir", "ka": "Georgian",
    "uz": "Uzbek", "ab": "Abkhaz", "ug": "Uyghur", "sah": "Sakha", "tk": "Turkmen", "kk": "Kazakh",
    "myv": "Erzya", "mdf": "Moksha", "os": "Ossetian", "az": "Azerbaijani",
    # Southeast Asia
    "yue": "Cantonese", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "ja": "Japanese", "ko": "Korean",
    "mn": "Mongolian", "zh-CN": "Mandarin (CN)", "zh-TW": "Mandarin (TW)", "zh-HK": "Chinese (HK)",
    "nan-tw": "Taiwanese Hokkien", "lo": "Lao", "tok": "Toki Pona",
    # South Asia
    "cnh": "Hakha Chin", "dv": "Dhivehi", "hi": "Hindi", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "skr": "Saraiki", "ta": "Tamil", "bn": "Bengali", "ur": "Urdu", "as": "Assamese", "pa-IN": "Punjabi",
    "ne-NP": "Nepali", "sat": "Santali", "te": "Telugu",
    # filler
    "eo": "Esperanto", "eu": "Basque", "hu": "Hungarian", "tr": "Turkish", "nl": "Dutch", "fa": "Persian",
    "ar": "Arabic"}
SESSION, MIN_OUTLINE = 720, 8

# reference palette (dataviz skill), light mode. Targets: one dark colour each, most common target
# first. The first three are validated as a set (all pairs); with CR's and SA's seven targets the rest
# are as far apart as dark hues allow, and a few pairs are close (accepted by the human, 2026-09-15;
# the legend and the bars carry identity). Other languages: light tints; lightness carries target vs not.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DARK = ["#1c5cab", "#c0501f", "#1a8a6a", "#6b3fa0", "#a8336a", "#8a6a00", "#45443f"]
DARK_REST = "#45443f"
LIGHT = ["#d5cdf5", "#f3dc9f", "#f5c8d9", "#d3e3bf"]
LIGHT_REST = "#e4e3dc"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})
HALO = [pe.withStroke(linewidth=2.8, foreground=SURFACE)]

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="ac", choices=list(CORPUS))
ap.add_argument("--thresholds", nargs=2, type=float, default=[0.25, 0.15])
ap.add_argument("--sessions", nargs="+", type=int, default=None)
ap.add_argument("--n_sessions", type=int, default=3)
ap.add_argument("--out", default=f"{ROOT}/runs/analysis/cluster_analysis")
a = ap.parse_args()
dirname, cname = CORPUS[a.corpus]
targets = TARGETS[a.corpus]
D = f"{ROOT}/data/{dirname}"

order = [x.strip() for x in open(f"{D}/orders/cluster_order_test") if x.strip()]
n_sess = len(order) // SESSION
whole = len(order) == n_sess * SESSION
order = order[:n_sess * SESSION]
feats = np.stack([np.load(f"{D}/ecapalang/test/{n}.npy") for n in order]).astype(np.float64)
if whole:
    # src/dataset.py:317-321 wraps with `% (len - 1)`: the stream's last clip is represented by the first
    # bootstrap clip's embedding in every run (see plot_session_clusters.py)
    feats[-1] = np.load(f"{D}/ecapalang/train/{open(f'{D}/orders/cluster_order_train').readline().strip()}.npy")
locale = np.array([n.split("_")[2] for n in order])
ref = pd.read_csv(f"{ROOT}/runs/diversity_extension/oal_t0.25_g0.8/seed_0/{a.corpus}_{FIRST_TARGET[a.corpus]}/al_diversity.csv")


def sl(s):
    return slice(s * SESSION, (s + 1) * SESSION)


if a.sessions:
    picks = a.sessions
else:  # the sessions with the most languages among those with at least 30 target clips
    n_lang = [len(set(locale[sl(s)])) for s in range(n_sess)]
    n_tgt = [int(np.isin(locale[sl(s)], targets).sum()) for s in range(n_sess)]
    cand = [s for s in range(n_sess) if n_tgt[s] >= 30]
    picks = sorted(sorted(cand, key=lambda s: (-n_lang[s], -n_tgt[s], s))[:a.n_sessions])

# colours follow the language across every panel. Targets: the three most common in the drawn sessions
# get a dark hue each, any others share a dark grey. Other languages: the four most common get light
# tints, everything else one light grey.
seen_t = Counter(x for s in picks for x in locale[sl(s)] if x in targets)
dark_named = [t for t, _ in seen_t.most_common(len(DARK))]
dark_rest = [t for t in targets if t in seen_t and t not in dark_named]
others = Counter(x for s in picks for x in locale[sl(s)] if x not in targets)
fixed = [x for x, _ in others.most_common(len(LIGHT))]
COLOR = ({t: DARK[i] for i, t in enumerate(dark_named)} | {t: DARK_REST for t in dark_rest}
         | {x: LIGHT[i] for i, x in enumerate(fixed)})
SEGMENTS = dark_named + dark_rest + fixed  # bar segment order; everything else is one light-grey segment


def col(x):
    return COLOR.get(x, LIGHT_REST)


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


fig = plt.figure(figsize=(21, 5.6 * len(picks) + 1.9), layout="constrained")
rows = np.atleast_1d(fig.subfigures(len(picks), 1, hspace=0.02))
for row, s in zip(rows, picks):
    X = feats[sl(s)]
    P = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
             random_state=0).fit_transform(X / np.linalg.norm(X, axis=1, keepdims=True))
    loc = locale[sl(s)]
    is_t = np.isin(loc, targets)
    tgt = ", ".join(f"{NAMES.get(t, t)} {int((loc == t).sum())}" for t in targets if (loc == t).any())
    row.suptitle(f"Session {s}: {len(set(loc))} languages; target clips: {tgt or 'none'}", x=0.004, ha="left",
                 fontsize=11.5, fontweight="bold")
    axes = row.subplots(1, 4, width_ratios=[1.25, 0.9, 1.25, 0.9])
    for j, thr in enumerate(a.thresholds):
        lab, k = cluster_session(np.ascontiguousarray(X), thr)
        if abs(thr - 0.25) < 1e-9 and k != ref.n_clusters[s]:
            sys.exit(f"STOP: session {s} at 0.25 gives {k} clusters, the runs logged {ref.n_clusters[s]}")
        sizes = np.bincount(lab, minlength=k)
        big = [c for c in np.argsort(-sizes, kind="stable") if sizes[c] >= MIN_OUTLINE]
        letter = {c: chr(65 + i) if i < 26 else str(i + 1) for i, c in enumerate(big)}
        crumbs = [c for c in range(k) if sizes[c] < MIN_OUTLINE]

        axm, axb = axes[2 * j], axes[2 * j + 1]
        axm.scatter(P[~is_t, 0], P[~is_t, 1], s=10, c=[col(x) for x in loc[~is_t]], linewidths=0)
        axm.scatter(P[is_t, 0], P[is_t, 1], s=15, c=[col(x) for x in loc[is_t]], linewidths=0.3, edgecolors=SURFACE)
        for c in big:
            pts = P[lab == c]
            try:
                hull = ConvexHull(pts)
            except Exception:  # degenerate (collinear) cluster: no outline
                continue
            v = np.append(hull.vertices, hull.vertices[0])
            axm.plot(pts[v, 0], pts[v, 1], color=INK, linewidth=0.9)
            top = pts[hull.vertices][np.argmax(pts[hull.vertices][:, 1])]
            axm.text(top[0], top[1], letter[c], fontsize=9.5, fontweight="bold", color=INK, ha="center",
                     va="bottom", path_effects=HALO)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_color(GRID)
        used = " (used by the extension)" if abs(thr - 0.25) < 1e-9 else ""
        axm.set_title(f"Threshold {thr:g}{used}: {k} clusters, {len(big)} of them with {MIN_OUTLINE}+ clips "
                      f"(outlined)", loc="left", fontsize=9.5)

        bars = [(letter[c], loc[lab == c]) for c in big[:12]]
        if len(big) > 12:
            bars.append((f"{len(big) - 12} more ({MIN_OUTLINE}+ clips)", loc[np.isin(lab, big[12:])]))
        if crumbs:
            bars.append((f"{len(crumbs)} crumbs (<{MIN_OUTLINE} clips)", loc[np.isin(lab, crumbs)]))
        for i, (name, L) in enumerate(bars):
            vc, left = pd.Series(L).value_counts(), 0
            parts = [(x, int(vc.get(x, 0))) for x in SEGMENTS]
            parts.append((None, int(sum(v for x, v in vc.items() if x not in SEGMENTS))))
            for x, n in parts:
                if n:
                    axb.barh(i, n, left=left, height=0.72, color=col(x) if x else LIGHT_REST,
                             edgecolor=SURFACE, linewidth=0.8)
                    left += n
            axb.text(left + 8, i, f"{len(L)} clips, {len(vc)} lang.", va="center", fontsize=8, color=INK2)
        axb.set_yticks(range(len(bars)))
        axb.set_yticklabels([n for n, _ in bars], fontsize=8.5)
        axb.invert_yaxis()
        axb.set_xlim(0, SESSION * 1.3)
        axb.set_xlabel("clips")
        axb.set_title("Language make-up of each cluster", loc="left", fontsize=9.5)
        tidy(axb)

handles = [Patch(color=COLOR[t], label=f"{NAMES.get(t, t)} (target)") for t in dark_named]
if dark_rest:
    handles.append(Patch(color=DARK_REST, label="other targets: " + ", ".join(NAMES.get(t, t) for t in dark_rest)))
handles += [Patch(color=COLOR[x], label=NAMES.get(x, x)) for x in fixed]
handles += [Patch(color=LIGHT_REST, label="other languages"),
            Line2D([], [], color=INK, linewidth=0.9, label=f"cluster outline ({MIN_OUTLINE}+ clips; A = largest)")]
caption = ("Each dot is one clip (720 per session), placed by t-SNE of its ECAPA embedding; one layout per session, shared "
           "by both thresholds, so only the outlines change.\nOutlines are convex hulls in the 2-D layout and can span gaps "
           "or overlap; clusters were computed in the full 256-D space with the extension's own code. Descriptive only "
           "(uses the clips' true languages).")
leg = fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 6), fontsize=9,
                 handlelength=1.4, title=caption, title_fontsize=8)
leg.get_title().set_color(INK2)
fig.suptitle(f"{cname}: do the extension's clusters follow languages? Threshold {a.thresholds[0]:g} vs "
             f"{a.thresholds[1]:g}", x=0.004, ha="left", fontsize=14, fontweight="bold")
os.makedirs(a.out, exist_ok=True)
path = f"{a.out}/{a.corpus}_clusters_{int(round(a.thresholds[0] * 100)):03d}_vs_{int(round(a.thresholds[1] * 100)):03d}.png"
fig.savefig(path, dpi=130)
print(f"sessions drawn: {picks}; dark: {dark_named} + grey {dark_rest}; light tints: {fixed}; wrote {path}")
