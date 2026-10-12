"""
Corrective-feedback deduplication (EXTENSION -- not part of the published method).

In OAL-CF the expert reviews every clip the classifier flags as target after each session's
fit (`--sim_type fpstps`): confirming true detections and correcting false alarms.  Many of
those clips are near-duplicates of each other (same speaker, same recording conditions).
This extension asks the expert for ONE label per group of near-duplicates and copies that
label to the rest of the group:

  1. cluster the session's CF clips on their own ECAPA embeddings (agglomerative, cosine
     distance, average linkage, `distance_threshold` -- the same clusterer as the
     diversity extension, utils/diverse_query.cluster_session);
  2. per cluster, the expert labels one representative (`--cfd_rep`):
        confident  the member the classifier is most sure is target (default)
        uncertain  the member whose p(target) is closest to 0.5
        medoid     the member with the smallest summed cosine distance to the others
  3. every other member receives the representative's label.  If that label is wrong for
     a member, the WRONG label is what enters the adaptation pool -- exactly what would
     happen with a real expert, so the cost of the shortcut is simulated, not assumed away.

The classifier is scored exactly as before (scores.csv is unchanged in meaning; its
cf_tp / cf_fp columns still count the TRUE classes of the CF clips).  What changes is the
number of clips the expert actually listened to, which is logged per session in
`cf_dedup.csv` and scored by tools/score_runs.py (IMLM_thesis, IMLM_repo_expert).

Threshold 0 makes every clip its own cluster, so nothing is copied and the run must be
bit-identical to baseline OAL-CF: the built-in control.

Nothing here may disturb the random-number streams the baseline consumes (see the RNG
notes in utils/diverse_query.py): the post-fit predictions are taken from the simulator's
own single extract_preds() call rather than recomputed, embeddings are read straight from
the cached feature files, and the whole bookkeeping block runs under an RNG snapshot.
"""
import os

import numpy as np
import torch

from utils.corrective_feedback import FeedbackSimulator
from utils.diverse_query import cluster_session
from src.dataset import ImlDataModule

REP_RULES = ('confident', 'uncertain', 'medoid')


def group_clips(Z, threshold):
    """Cluster labels for the rows of Z.  threshold <= 0 (or fewer than 2 clips) puts every
    clip in its own cluster without calling the clusterer, which is the control setting."""
    n = len(Z)
    if threshold <= 0 or n < 2:
        return np.arange(n, dtype=np.int64)
    labels, _ = cluster_session(np.asarray(Z, dtype=np.float64), threshold)
    return labels


def choose_representatives(groups, p_target, rule='confident', Z=None):
    """{cluster id: position of the member the expert labels}.  Ties go to the lowest
    position, so the choice is deterministic."""
    p_target = np.asarray(p_target, dtype=np.float64)
    reps = {}
    for c in np.unique(groups):
        members = np.flatnonzero(groups == c)
        if len(members) == 1 or rule == 'confident':
            reps[c] = int(members[np.argmax(p_target[members])])
        elif rule == 'uncertain':
            reps[c] = int(members[np.argmin(np.abs(p_target[members] - 0.5))])
        elif rule == 'medoid':
            if Z is None:
                raise ValueError('medoid rule needs the embeddings')
            X = np.asarray(Z, dtype=np.float64)[members]
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(norms > 0, norms, 1.0)
            dist = 1.0 - X @ X.T
            reps[c] = int(members[np.argmin(dist.sum(axis=1))])
        else:
            raise ValueError(f'unknown representative rule {rule!r}; use one of {REP_RULES}')
    return reps


def propagate(true_labels, groups, reps):
    """Labels the adaptation pool receives, and which positions the expert labelled."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    given = true_labels.copy()
    asked = np.zeros(len(true_labels), dtype=bool)
    for c, r in reps.items():
        members = groups == c
        given[members] = true_labels[r]
        asked[r] = True
    return given, asked


class DedupDataModule(ImlDataModule):
    """ImlDataModule that applies copied CF labels when the CF clips enter the pool.

    The flips are applied AFTER the transfer, so run.py's cf_tp / cf_fp (read from the test
    set just before the transfer) keep counting true classes, and no clip that is still
    being scored ever carries a copied label."""

    def __init__(self, params):
        super().__init__(params)
        if self.task != 'lid':
            raise ValueError('cf_dedup is implemented for the LID task only')
        self.pending_label_flips = {}

    def transfer_samples(self, idxs):
        base = [self.data_test.active_idxs[ii] for ii in idxs]
        super().transfer_samples(idxs)
        for b in base:
            if b in self.pending_label_flips:
                self.ds.labels[b] = int(self.pending_label_flips.pop(b))
        if self.pending_label_flips:
            raise RuntimeError(f'{len(self.pending_label_flips)} copied CF labels were never '
                               'applied: the CF clips and the transferred clips disagree')


class DedupFeedbackSimulator(FeedbackSimulator):
    """FeedbackSimulator that asks the expert for one label per cluster of CF clips."""

    # set by run_cfdedup.py before the run starts
    cfd_threshold = 0.10
    cfd_rep = 'confident'
    log_path = None

    def __init__(self, params):
        super().__init__(params)
        if self.sim_type is None:
            raise ValueError('cf_dedup needs corrective feedback (--sim_type, e.g. fpstps)')
        if self.sim_type == 'all':
            raise ValueError("cf_dedup does not support --sim_type all")
        if self.cfd_rep not in REP_RULES:
            raise ValueError(f'--cfd_rep must be one of {REP_RULES}')
        self._last_preds = None
        self._rows = []

    def extract_preds(self, data_module, module):
        preds, scores = super().extract_preds(data_module, module)
        self._last_preds = (preds, scores)
        return preds, scores

    def simulate(self, data_module, module):
        self._last_preds = None
        idxs = super().simulate(data_module, module)
        row = dict(session=data_module.current_batch, n_cf=len(idxs), cf_target=0,
                   cf_nontarget=0, expert_labels=0, copied=0, wrong=0,
                   nontarget_given_target=0, target_given_nontarget=0, clusters=0,
                   largest_cluster=0)
        if len(idxs) > 0:
            if self._last_preds is None:
                raise RuntimeError('corrective feedback returned clips without predictions')
            rng_cpu, rng_np = torch.get_rng_state(), np.random.get_state()
            try:
                sel = np.asarray(idxs, dtype=np.int64)
                preds = self._last_preds[0].cpu().numpy()[sel]
                scores = self._last_preds[1].cpu().numpy()[sel].astype(np.float64)
                p_target = np.where(preds == 1, scores, 1.0 - scores)
                ds = data_module.ds
                base = [data_module.data_test.active_idxs[i] for i in sel]
                true = np.array([ds.get_label(b) for b in base], dtype=np.int64)
                Z = np.stack([np.concatenate([np.asarray(ds._load_feat(ds.feat_files[b].format(nn)),
                                                         dtype=np.float64).reshape(-1)
                                              for nn in ds.feat_names]) for b in base])
                if not np.all(np.isfinite(Z)):
                    raise ValueError('non-finite embedding among the CF clips')
                groups = group_clips(Z, self.cfd_threshold)
                reps = choose_representatives(groups, p_target, self.cfd_rep, Z)
                given, asked = propagate(true, groups, reps)
                for b, g, t in zip(base, given, true):
                    if g != t:
                        data_module.pending_label_flips[b] = int(g)
                sizes = np.bincount(groups)
                row.update(cf_target=int(true.sum()), cf_nontarget=int((1 - true).sum()),
                           expert_labels=int(asked.sum()), copied=int((~asked).sum()),
                           wrong=int((given != true).sum()),
                           nontarget_given_target=int(((true == 0) & (given == 1)).sum()),
                           target_given_nontarget=int(((true == 1) & (given == 0)).sum()),
                           clusters=int(len(sizes)), largest_cluster=int(sizes.max()))
            finally:
                torch.set_rng_state(rng_cpu)
                np.random.set_state(rng_np)
        self._rows.append(row)
        self.flush_log()
        return idxs

    def flush_log(self):
        if self.log_path is None or not self._rows:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        keys = list(self._rows[0].keys())
        with open(self.log_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in self._rows:
                f.write(','.join(str(r[k]) for k in keys) + '\n')
