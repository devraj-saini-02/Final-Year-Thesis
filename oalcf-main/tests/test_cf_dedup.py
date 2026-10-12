"""Unit tests for utils/cf_dedup.py.

Runs from any working directory:  python tests/test_cf_dedup.py
"""
import os
import sys
import types

# make `utils` / `src` importable without setting PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from utils.cf_dedup import (group_clips, choose_representatives, propagate,
                            DedupFeedbackSimulator, REP_RULES)

ok = True
def check(name, cond, detail=''):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    ok = ok and bool(cond)

rng = np.random.default_rng(0)

# 1. threshold 0 (the control) never merges anything, even exact duplicates
Z = rng.normal(size=(20, 256)); Z[5] = Z[4]
g = group_clips(Z, 0.0)
check('threshold 0 -> every clip its own group', len(np.unique(g)) == 20)
check('single clip -> one group', list(group_clips(Z[:1], 0.1)) == [0])
check('no clips -> empty', len(group_clips(Z[:0], 0.1)) == 0)

# 2. near-duplicates merge, distinct clips do not
base = rng.normal(size=(3, 256))
Zd = np.vstack([base[0] + 1e-3 * rng.normal(size=(4, 256)),
                base[1] + 1e-3 * rng.normal(size=(3, 256)),
                base[2][None]])
g = group_clips(Zd, 0.10)
check('near-duplicates form 3 groups', len(np.unique(g)) == 3, f'groups={g.tolist()}')
check('group sizes 4/3/1', sorted(np.bincount(g).tolist()) == [1, 3, 4])

# 3. a zero vector does not crash the clusterer
Zz = Zd.copy(); Zz[7] = 0
check('zero embedding tolerated', len(group_clips(Zz, 0.10)) == 8)

# 4. representative rules
groups = np.array([0, 0, 0, 1, 1])
p = np.array([0.9, 0.6, 0.99, 0.51, 0.7])
check('confident picks max p(target)', choose_representatives(groups, p, 'confident') == {0: 2, 1: 4})
check('uncertain picks p closest to 0.5', choose_representatives(groups, p, 'uncertain') == {0: 1, 1: 3})
Zm = np.array([[1, 0], [1, 0.1], [1, -0.1], [0, 1], [0.1, 1]], dtype=float)
rm = choose_representatives(groups, p, 'medoid', Zm)
check('medoid picks the central member', rm[0] == 0, f'{rm}')
try:
    choose_representatives(groups, p, 'medoid'); check('medoid without Z raises', False)
except ValueError:
    check('medoid without Z raises', True)
try:
    choose_representatives(groups, p, 'bogus'); check('unknown rule raises', False)
except ValueError:
    check('unknown rule raises', True)
check('ties go to the lowest position', choose_representatives(np.array([0, 0]), np.array([0.8, 0.8])) == {0: 0})

# 5. propagation: counts, asked mask, wrong labels
true = np.array([1, 0, 1, 0, 0])
given, asked = propagate(true, groups, {0: 2, 1: 4})
check('one expert label per group', asked.sum() == 2)
check('copied label is the representative\'s', given.tolist() == [1, 1, 1, 0, 0])
check('exactly one wrong label (false alarm given target)', int((given != true).sum()) == 1)
given0, asked0 = propagate(true, np.arange(5), {i: i for i in range(5)})
check('singleton groups -> labels unchanged, all asked', (given0 == true).all() and asked0.all())

# 6. simulator guards
P = types.SimpleNamespace(sim_type=None, max_fb_samples=None, rank_cf=None)
try:
    DedupFeedbackSimulator(P); check('no --sim_type raises', False)
except ValueError:
    check('no --sim_type raises', True)
P.sim_type = 'all'
try:
    DedupFeedbackSimulator(P); check("--sim_type all raises", False)
except ValueError:
    check("--sim_type all raises", True)

# 7. the bookkeeping leaves the RNG streams untouched (fake data module, no model)
class FakeDS:
    def __init__(self, Z, labels):
        self.Z, self.labels, self.feat_names = Z, list(labels), ['ecapalang']
        self.feat_files = [f'{{}}/{i}' for i in range(len(Z))]
    def _load_feat(self, path):
        return self.Z[int(path.split('/')[-1])]
    def get_label(self, i):
        return int(self.labels[i])

class FakeTest:
    def __init__(self, n):
        self.active_idxs = list(range(n))
    def __len__(self):
        return len(self.active_idxs)

labels = np.array([1, 1, 1, 1, 0, 0, 0, 1])
dm = types.SimpleNamespace(ds=FakeDS(Zd, labels), data_test=FakeTest(8), current_batch=0,
                           pending_label_flips={}, unlabeled_len=lambda: 8,
                           get_test_labels=lambda: torch.tensor(labels))
P.sim_type = 'fpstps'
sim = DedupFeedbackSimulator(P)
sim.cfd_threshold = 0.10
all_pos = torch.ones(8, dtype=torch.long)
conf = torch.tensor([0.9, 0.8, 0.95, 0.7, 0.6, 0.99, 0.55, 0.9])
sim.extract_preds = lambda d, m: (sim.__setattr__('_last_preds', (all_pos, conf)) or (all_pos, conf))
torch.manual_seed(1); np.random.seed(1)
t0, n0 = torch.get_rng_state(), np.random.get_state()
out = sim.simulate(dm, None)
check('torch RNG untouched', torch.equal(t0, torch.get_rng_state()))
check('numpy RNG untouched', np.array_equal(n0[1], np.random.get_state()[1]))
row = sim._rows[-1]
check('all CF clips returned', len(out) == 8)
check('3 groups -> 3 expert labels', row['expert_labels'] == 3, str(row))
# group of 3 = clips 4,5,6 (true 0,0,0): rep is clip 5 (p=.99), label 0 -> no error
# group of 4 = clips 0-3 (true 1,1,1,1): no error; singleton 7: no error
check('no wrong labels on pure groups', row['wrong'] == 0 and not dm.pending_label_flips)
dm.ds.labels[6] = 1   # make group {4,5,6} impure: clip 6 is now a target, rep 5 says 0
dm.get_test_labels = lambda: torch.tensor(dm.ds.labels)
sim.simulate(dm, None); row = sim._rows[-1]
check('impure group -> one wrong label, queued as a flip',
      row['wrong'] == 1 and row['target_given_nontarget'] == 1 and dm.pending_label_flips == {6: 0}, str(row))

print('\nALL PASS' if ok else '\nSOME TESTS FAILED')
sys.exit(0 if ok else 1)
