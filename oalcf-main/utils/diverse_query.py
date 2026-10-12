"""
Cluster-quota diversity-penalized uncertainty sampling for the OAL AL channel.

EXTENSION -- not part of the published method.  Lives in its own module and is
driven by its own entry point (`run_dq.py`), so `utils/query_strategies.py` and
`run.py` stay byte-identical to upstream: the baseline query selection algorithm
is untouched and the reproduction diff stays readable.

The method
----------
    score(x) = u(x) * gamma ** (n_c / q_c)

  u(x)  least-confidence uncertainty, 1 - max_j softmax(logits)_j.  Higher means
        the classifier is less sure.  Same quantity the baseline 'smax' strategy
        ranks on (baseline sorts max-softmax ascending == u descending).
        For the binary task this is computed as min_j softmax_j, which is
        algebraically identical but numerically exact: forming 1 - max directly
        cancels catastrophically once the classifier is confident (max ~ 1), and
        was measured to manufacture 98 spurious ties in a 720-sample session.
        Those ties made gamma=1 disagree with the baseline top-8 in some
        sessions, which would have broken the control.
  c     the cluster x belongs to.
  n_c   how many samples have ALREADY been picked from cluster c this session.
  q_c   quota,  q_c = k * sqrt(|c|) / sum_j sqrt(|c_j|),  summing to exactly the
        budget k.  The square root damps size: a cluster 100x larger draws 10x
        the quota, not 100x.
  gamma in [0, 1], the discount strength.

n_c/q_c is "fair shares consumed": below 1 the cluster is inside its allocation,
above 1 it is over-drawn.  gamma=1 makes the penalty identically 1, so gamma=1
reproduces the baseline exactly and is the built-in control.  gamma=0 (a hard
quota) is deliberately not swept.

Selection is GREEDY and must be: n_c is 0 for every cluster until picks land, so
scoring the pool once and taking the top k would leave every penalty at
gamma**0 = 1 and reproduce the baseline regardless of gamma.

Clustering: agglomerative, cosine distance, average linkage, distance_threshold
--------------------------------------------------------------------------------
The quota formula only means something if cluster size reflects how much of the
batch that region genuinely occupies.  Methods that force a fixed cluster count
break exactly that property: they split a homogeneous mass into artificial
pieces and the quota then allocates budget to boundaries that do not exist.
Agglomerative clustering with `distance_threshold` set and `n_clusters=None`
yields however many clusters the data supports, is deterministic (so it adds no
variance on top of the seed noise already in play), and admits singleton
clusters naturally -- a lone point is simply its own region, so there is no
noise label and no partition to repair.  At 720 points per session the O(n^2)
cost is irrelevant (~0.2 s per session).

Cosine is the right metric: ECAPA embeddings are compared by cosine similarity,
so the unit sphere is their natural geometry, and using it directly removes any
need for normalization or dimensionality reduction first.

INFERRED (not specified in the method description; recorded in
docs/other/DEVIATIONS.md): the distance threshold.  Chosen empirically by clustering
sessions sampled across all four corpora and picking the value whose cluster
count lands in a sensible range for a 720-sample batch -- too few and the
penalty never fires, too many and every sample is its own cluster and it fires
constantly.  Measured over 40 sessions spanning the four corpora:

    thresh   k mean   k range   median quota   sessions with 5 <= k <= 30
     0.20     39.2     4-104        0.17                 38%
     0.22     21.8     3- 58        0.19                 70%
     0.24     11.9     2- 32        0.34                 88%
     0.25      9.0     2- 21        0.55                 88%
     0.26      7.1     2- 17        0.71                 70%
     0.28      4.3     1- 11        1.30                 32%

0.25 is the default: 88% of sessions land in the target window and the median
quota of 0.55 puts a second pick from a typical cluster at exponent ~1.8, which
keeps gamma's effect visible across the swept range rather than saturating it.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

from utils.query_strategies import StrategyManager

# Small clusters give q_c << 1 and therefore large exponents.  gamma**large
# underflows to 0 anyway, so clamping costs nothing and keeps things finite.
MAX_EXPONENT = 50.0

# ECAPA embeddings are 256-d; a feature vector is (2*context+1) frames
# flattened, so the sample's OWN embedding is the centre frame.
ECAPA_DIM = 256


def compute_quotas(cluster_sizes, k):
    """q_c = k * sqrt(|c|) / sum_j sqrt(|c_j|).  Sums to exactly k."""
    roots = np.sqrt(np.asarray(cluster_sizes, dtype=np.float64))
    total = roots.sum()
    if total <= 0:
        return np.full(len(cluster_sizes), float(k) / max(len(cluster_sizes), 1))
    return k * roots / total


def penalty(gamma, exponents):
    """gamma ** exponents, with 0**0 defined as 1 so a cluster's FIRST pick is
    never penalized.  (Kept for completeness; gamma=0 is not swept.)"""
    exponents = np.clip(exponents, 0.0, MAX_EXPONENT)
    if gamma == 0.0:
        return np.where(exponents == 0.0, 1.0, 0.0)
    return np.power(gamma, exponents)


def greedy_select(u, labels, quotas, gamma, k, tiebreak=None):
    """The greedy loop.

    u       (n,) uncertainty, higher = less confident
    labels  (n,) cluster id per sample, in [0, len(quotas))
    quotas  (n_clusters,)
    returns selected indices into u, in selection order

    tiebreak  (n,) optional rank array; among equal scores the LOWEST rank
              wins.  run.py ranks with torch.sort(max_softmax), and torch.sort
              is NOT stable by default (stable=False), so its order among equal
              values is implementation-defined -- "lowest index" does not
              reproduce it.  Passing torch.sort's own permutation as the rank
              makes gamma=1 select exactly run.py's queries, in run.py's order.
              This matters because max_softmax saturates to exactly 1.0f for
              many samples once the classifier is confident, so these ties are
              common rather than exotic.
    """
    u = np.asarray(u, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(u)
    k = int(min(k, n))
    rank = (np.arange(n, dtype=np.int64) if tiebreak is None
            else np.asarray(tiebreak, dtype=np.int64))
    n_c = np.zeros(len(quotas), dtype=np.float64)
    available = np.ones(n, dtype=bool)
    selected = []
    for _ in range(k):
        exponents = n_c[labels] / quotas[labels]
        score = u * penalty(gamma, exponents)
        score = np.where(available, score, -np.inf)
        best = score.max()
        cand = np.flatnonzero(available & (score == best))
        # Lowest tiebreak rank wins.  Breaking ties by cluster occupancy instead
        # would inject diversity even at gamma=1 and break the control.
        pick = int(cand[0]) if len(cand) == 1 else int(cand[np.argmin(rank[cand])])
        selected.append(pick)
        available[pick] = False
        n_c[labels[pick]] += 1.0
    return selected


def cluster_session(Z, threshold):
    """Agglomerative, cosine, average linkage, no fixed cluster count.
    Returns (labels, n_clusters).  Every point gets a cluster, so the partition
    is complete by construction and the quota denominator is never distorted."""
    n = len(Z)
    if n < 2:
        return np.zeros(n, dtype=np.int64), max(n, 1)
    # cosine distance is undefined for a zero vector; nudge any such row
    norms = np.linalg.norm(Z, axis=1)
    if np.any(norms == 0):
        Z = Z.copy()
        Z[norms == 0, 0] = 1e-12
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric='cosine', linkage='average').fit_predict(Z)
    return labels.astype(np.int64), int(labels.max()) + 1


class DiverseStrategyManager(StrategyManager):
    """Drop-in replacement for StrategyManager applying the cluster-quota
    diversity penalty to the softmax-uncertainty ranking.  Every other strategy
    in the base class is untouched; only 'smax' (the OAL baseline's default, and
    therefore run.py's `mm`) is overridden."""

    # set by run_dq.py before the run starts
    dq_gamma = 1.0
    dq_threshold = 0.25
    dq_cluster_feat = 'center'
    dq_log_path = None

    def __init__(self, params):
        super().__init__(params)
        self._session = 0
        self._log_rows = []
        if params.adapt_distr is not None:
            raise ValueError('diverse_query does not implement --adapt_distr')
        if params.combo is not None:
            raise ValueError('diverse_query does not implement --combo')

    def _session_features(self, data_module):
        """Features for the current unlabeled pool, in the SAME order as
        extract_logits() returns logits, so indices line up.

        The dataset is indexed DIRECTLY rather than through a DataLoader.  This
        is load-bearing, not a style choice: constructing a DataLoader iterator
        draws a base seed from the global torch RNG
        (`_BaseDataLoaderIter.__init__` -> `torch.empty((), dtype=torch.int64)
        .random_(generator=loader.generator)`), so an extra loader pass per
        session silently shifts the RNG state that the following trainer.fit()
        consumes.  With a loader here, gamma=1 produced a different trajectory
        from the untouched baseline despite selecting byte-identical queries.
        Indexing the dataset touches no RNG, and the loader in extract_logits()
        is the one the baseline also creates, so the RNG stream is unchanged.

        Order matches extract_logits() because its DataLoader uses
        shuffle=False, i.e. plain dataset order.
        """
        ds = (data_module.data_test if hasattr(data_module, 'data_test')
              else data_module.dataset)
        rows = []
        for ii in range(len(ds)):
            x = ds[ii][0]
            rows.append(x.numpy() if hasattr(x, 'numpy') else np.asarray(x))
        feats = np.stack(rows, axis=0) if rows else np.zeros((0, 1))
        feats = feats.reshape(feats.shape[0], -1)
        if self.dq_cluster_feat == 'center' and feats.shape[1] % ECAPA_DIM == 0:
            n_frames = feats.shape[1] // ECAPA_DIM
            if n_frames >= 3 and n_frames % 2 == 1:
                mid = n_frames // 2
                # INFERRED: cluster on the sample's OWN embedding, not the full
                # +/-3 window.  The stream is ordered by Dirichlet locale
                # clustering, so neighbouring frames correlate with locale;
                # clustering the window would let stream position leak into the
                # assignment and the penalty would partly penalize position.
                feats = feats[:, mid * ECAPA_DIM:(mid + 1) * ECAPA_DIM]
        return np.ascontiguousarray(feats, dtype=np.float64)

    def select_queries(self, data_module, method_list, module, n_queries):
        self.logits = self.extract_logits(data_module, module)
        # Everything this class does BEYOND the baseline's own work must leave
        # the RNG stream untouched, or the following trainer.fit() consumes a
        # different stream and the run diverges even when the selected queries
        # are byte-identical.  extract_logits() above is common to both paths,
        # so the snapshot is taken after it and restored before returning.
        _rng_cpu = torch.get_rng_state()
        _rng_np = np.random.get_state()
        _rng_mps = (torch.mps.get_rng_state()
                    if hasattr(torch, 'mps') and torch.backends.mps.is_available()
                    else None)
        probs = F.softmax(self.logits, dim=-1)
        smax_out = torch.max(probs, dim=-1)[0]
        # Cast to float64 BEFORE subtracting: 1.0 - float32 is exact in
        # float64.  This must stay a pure order-reversing map of smax_out, the
        # exact quantity run.py sorts on, so that gamma=1 reproduces the
        # baseline selection bit-for-bit.  (min(p) would be a strictly better
        # uncertainty estimate -- smax saturates to exactly 1.0f for many
        # samples, so 1-smax ties them while min(p) still separates them -- but
        # adopting it would silently mix a second improvement into every gamma
        # and the control would no longer isolate the diversity penalty.)
        u = (1.0 - smax_out.double()).cpu().numpy()
        n = len(u)
        if n == 0:
            return ({'smax': torch.zeros(0, dtype=torch.long)},
                    {'smax': torch.tensor(0.0)})

        Z = self._session_features(data_module)
        labels, n_clusters = cluster_session(Z, self.dq_threshold)
        sizes = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        sizes = np.maximum(sizes, 1.0)
        quotas = compute_quotas(sizes, n_queries)

        # run.py's exact ordering: torch.sort(max_softmax) ascending.  Used
        # both as the tie-break rank (so gamma=1 is bit-exact) and as the
        # reference the diagnostics compare against.
        sort_idx = torch.sort(smax_out)[1].cpu().numpy()
        rank = np.empty(n, dtype=np.int64)
        rank[sort_idx] = np.arange(n, dtype=np.int64)

        idxs = greedy_select(u, labels, quotas, self.dq_gamma, n_queries,
                             tiebreak=rank)

        ref = sort_idx[:n_queries].tolist()
        self._ref = ref
        self._log_session(labels, sizes, quotas, idxs, u, n_clusters, n_queries)
        self._session += 1

        torch.set_rng_state(_rng_cpu)
        np.random.set_state(_rng_np)
        if _rng_mps is not None:
            torch.mps.set_rng_state(_rng_mps)
        return ({'smax': torch.tensor(idxs, dtype=torch.long)},
                {'smax': torch.mean(smax_out)})

    def _log_session(self, labels, sizes, quotas, idxs, u, n_clusters, k):
        picked = labels[np.asarray(idxs, dtype=np.int64)]
        counts = np.bincount(picked, minlength=n_clusters)
        base = list(np.argsort(-u, kind='stable')[:k])
        self._log_rows.append(dict(
            session=self._session,
            n_clusters=n_clusters,
            n_singletons=int((sizes == 1).sum()),
            largest_cluster=int(sizes.max()),
            max_quota=round(float(quotas.max()), 4),
            median_quota=round(float(np.median(quotas)), 4),
            distinct_clusters_picked=len(np.unique(picked)),
            over_quota_picks=int(np.sum(np.maximum(counts - np.ceil(quotas), 0))),
            overlap_with_baseline=len(set(base) & set(idxs)),
            overlap_with_runpy=len(set(getattr(self, '_ref', base)) & set(idxs)),
            same_order_as_runpy=int(list(getattr(self, '_ref', base)) == list(idxs)),
            mean_u_picked=round(float(np.mean(u[idxs])), 6),
            mean_u_baseline=round(float(np.mean(u[base])), 6),
        ))
        if self.dq_log_path is not None:
            self.flush_log()

    def flush_log(self):
        if self.dq_log_path is None or not self._log_rows:
            return
        keys = list(self._log_rows[0].keys())
        os.makedirs(os.path.dirname(self.dq_log_path), exist_ok=True)
        with open(self.dq_log_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for row in self._log_rows:
                f.write(','.join(str(row[kk]) for kk in keys) + '\n')
