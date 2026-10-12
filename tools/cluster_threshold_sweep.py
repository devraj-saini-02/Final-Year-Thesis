"""Cluster-size breakdown and threshold sweep for the diversity extension's per-session clusters.

For every session of each corpus, the 720 clips' own ECAPA embeddings are clustered exactly as
oalcf-main/utils/diverse_query.py does (cosine distance, average linkage), at the extension's
threshold 0.25 and at tighter cuts of the same tree, and three things are measured:
  (a) how the session splits into clusters by size (mini-clusters and the share of clips in them),
  (b) how well clusters match the clips' true languages: homogeneity (each cluster is one language),
      completeness (each language stays in one cluster), V-measure, purity, and the share of clips
      sitting in a cluster that is at least 90% one language,
  (c) the quota q_c = k*sqrt(|c|)/sum_j sqrt(|c_j|) the extension would hand each cluster, and the
      penalty a cluster's SECOND pick would carry, gamma**(1/q_c), at a few gammas. As clusters get
      smaller the quota shrinks, the exponent 1/q_c grows, and gamma must rise to keep the penalty
      from collapsing to "one pick per cluster".
The 0.25 cut is checked against the cluster counts the runs logged (al_diversity.csv).

Descriptive only. The true languages are test-stream labels, so nothing here may be used to choose a
threshold or a clusterer for the method itself; that needs the training split. See
docs/other/CLUSTER_OBSERVATIONS.md.

Usage: python tools/cluster_threshold_sweep.py > runs/analysis/cluster_analysis/threshold_sweep.txt
       python tools/cluster_threshold_sweep.py --corpus cr --thresholds 0.25 0.18 0.15
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from sklearn.metrics import homogeneity_completeness_v_measure  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = {"ac": "AfricanContinentLID", "cr": "CaucasusRegionLID", "sea": "SoutheastAsiaLID", "sa": "SouthAsiaLID"}
FIRST_TARGET = {"ac": "bas", "cr": "hy-AM", "sea": "yue", "sa": "cnh"}
SESSION, BUDGET, MINI = 720, 8, 8  # clips per session, AL queries per session, "mini cluster" cutoff
GAMMAS = (0.8, 0.9, 0.95)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", nargs="+", default=list(CORPUS), choices=list(CORPUS))
ap.add_argument("--thresholds", nargs="+", type=float,
                default=[0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10, 0.08])
a = ap.parse_args()
T = sorted(a.thresholds, reverse=True)


HEADER = ("  thresh | clusters  mini  single  clips in mini | biggest | homog  compl     V | purity  pure clips | "
          " q med  q clip | " + "  ".join(f"g={g}" for g in GAMMAS))


def row_line(t, rows):
    """One table row: the mean over sessions of every measurement at threshold t."""
    d = pd.DataFrame(rows).mean()
    pen = "  ".join(f"{g ** (1 / d.q_clip):5.2f}" for g in GAMMAS)
    return (f"   {t:.2f}  | {d.k:7.1f} {d.mini:6.1f} {d.singles:6.1f} {d.mini_share:11.0f}% | {d.biggest:6.0f}% | "
            f"{d.hom:5.2f}  {d.comp:5.2f} {d.v:5.2f} | {d.purity:5.0f}% {d.pure_share:10.0f}% | "
            f"{d.q_med:6.2f} {d.q_clip:6.2f} | {pen}")


def session_rows(Z, ycode, n_lang):
    """One session: cluster at every threshold, return a dict of measurements per threshold.

    ycode: language of each clip as an integer code; n_lang: how many languages in the session.
    """
    tree = linkage(Z, method="average", metric="cosine")
    out = {}
    for t in T:
        lab = fcluster(tree, t, criterion="distance")
        sizes = np.bincount(lab)[1:].astype(np.float64)
        h, c, v = homogeneity_completeness_v_measure(ycode, lab)
        # counts[cluster, language] in one pass, instead of a pandas call per cluster
        counts = np.bincount((lab - 1) * n_lang + ycode, minlength=len(sizes) * n_lang).reshape(-1, n_lang)
        top = counts.max(axis=1)
        majority = int(top.sum())
        pure_clips = int(sizes[top / sizes >= 0.9].sum())
        q = BUDGET * np.sqrt(sizes) / np.sqrt(sizes).sum()
        out[t] = dict(k=len(sizes), mini=int((sizes < MINI).sum()), singles=int((sizes == 1).sum()),
                      mini_share=100 * sizes[sizes < MINI].sum() / SESSION,
                      biggest=100 * sizes.max() / SESSION, hom=h, comp=c, v=v,
                      purity=100 * majority / SESSION, pure_share=100 * pure_clips / SESSION,
                      q_med=float(np.median(q)),
                      # quota of the cluster a typical CLIP sits in: the median over clips, not over
                      # clusters, so mini clusters do not dominate it. Queries land in the big
                      # clusters, so this is the quota the penalty actually meets.
                      q_clip=float(np.median(q[lab - 1])))
    return out


print("Per-session clusters of the diversity extension: same tree as utils/diverse_query.py (cosine, average "
      "linkage), cut at several thresholds.\nMeans over sessions. Descriptive only (uses test-stream language "
      f"labels).  mini cluster = fewer than {MINI} clips.\n")
print("  homogeneity: 1.0 = every cluster holds one language | completeness: 1.0 = every language sits in one "
      "cluster | purity: share of clips whose cluster is mostly their own language")
print(f"  pure clips: share of clips in a cluster that is >= 90% one language")
print(f"  q med: median quota over clusters | q clip: quota of the cluster a typical CLIP sits in (median over "
      f"clips), which is the quota the penalty actually meets, since queries land in the big clusters")
print(f"  g=...: what a cluster's SECOND pick is multiplied by at that gamma, gamma**(1/q clip), with budget "
      f"{BUDGET}. Near 0 means the penalty has become a hard one-pick-per-cluster rule.\n")

allrows = {t: [] for t in T}
for c in a.corpus:
    D = f"{ROOT}/data/{CORPUS[c]}"
    order = [x.strip() for x in open(f"{D}/orders/cluster_order_test") if x.strip()]
    n_sess = len(order) // SESSION
    order = order[:n_sess * SESSION]
    F = np.stack([np.load(f"{D}/ecapalang/test/{n}.npy") for n in order]).astype(np.float64)
    if len(open(f"{D}/orders/cluster_order_test").readlines()) == n_sess * SESSION:
        # src/dataset.py:317-321 wraps with `% (len - 1)`: the stream's last clip is represented by the
        # first bootstrap clip's embedding in every run (see plot_session_clusters.py)
        F[-1] = np.load(f"{D}/ecapalang/train/{open(f'{D}/orders/cluster_order_train').readline().strip()}.npy")
    loc = np.array([n.split("_")[2] for n in order])
    ref = pd.read_csv(f"{ROOT}/runs/diversity_extension/oal_t0.25_g0.8/seed_0/{c}_{FIRST_TARGET[c]}/al_diversity.csv")

    rows, match, n_lang = {t: [] for t in T}, 0, []
    for s in range(n_sess):
        sl = slice(s * SESSION, (s + 1) * SESSION)
        codes, langs = pd.factorize(loc[sl])
        n_lang.append(len(langs))
        r = session_rows(F[sl], codes, len(langs))
        for t in T:
            rows[t].append(r[t]); allrows[t].append(r[t])
        if 0.25 in r:
            match += int(r[0.25]["k"] == ref.n_clusters[s])

    check = (f"; the 0.25 cut reproduces the logged cluster count in {match}/{n_sess} sessions"
             if 0.25 in T else "")
    print(f"== {c.upper()}: {n_sess} sessions, {np.mean(n_lang):.1f} languages per session{check}")
    print(HEADER)
    for t in T:
        print(row_line(t, rows[t]))
    print()

if len(a.corpus) > 1:
    print(f"== ALL {len(a.corpus)} CORPORA POOLED ({len(allrows[T[0]])} sessions)")
    print(HEADER)
    for t in T:
        print(row_line(t, allrows[t]))
