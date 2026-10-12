"""Unit tests for utils/diverse_query.py.

Runs from any working directory:  python tests/test_diverse_query.py
"""
import os
import sys

# make `utils` importable without setting PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.diverse_query import compute_quotas, penalty, greedy_select, MAX_EXPONENT, cluster_session

ok = True
def check(name, cond, detail=''):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    ok = ok and cond

# 1. quotas sum to exactly k
q = compute_quotas([400, 289, 9], 8)
check('quotas sum to k', abs(q.sum() - 8) < 1e-12, f'sum={q.sum():.12f}')

# 2. sqrt damping: 100x the size -> 10x the quota
q2 = compute_quotas([100, 10000], 8)
check('sqrt damping 100x size -> 10x quota', abs(q2[1] / q2[0] - 10.0) < 1e-12,
      f'ratio={q2[1]/q2[0]:.6f}')

# 3. the spec's worked case: sum sqrt ~= 40
#    cluster 400 -> q = 8*20/40 = 4 ; cluster 9 -> q = 8*3/40 = 0.6
check('worked case q(400)=4', abs(q[0] - 4.0) < 1e-9, f'q={q[0]:.6f}')
check('worked case q(9)=0.6', abs(q[2] - 0.6) < 1e-9, f'q={q[2]:.6f}')
check('worked case sum sqrt = 40', abs(np.sqrt([400,289,9]).sum() - 40) < 1e-9)

# 4. discount magnitudes.  NOTE the spec's procedure ("n_c = how many have
#    ALREADY been selected"; increment AFTER the pick) makes the 2nd pick from a
#    cluster carry exponent 1/q_c, while the spec's worked paragraph quotes
#    2/q_c.  Both are checked; the implementation follows the procedure.
g = 0.4
check('procedure reading: 2nd pick from c=400 -> 0.4^0.25 = 0.795',
      abs(penalty(g, np.array([1/4.0]))[0] - 0.4**0.25) < 1e-12,
      f'{0.4**0.25:.3f}')
check('worked-para reading: 0.4^0.5 = 0.632', abs(0.4**0.5 - 0.6325) < 1e-3)
# the spec rounds the exponent to 3.3 before exponentiating; both land at ~0.05
check('worked-para reading: 0.4^3.3 rounds to 0.05', round(0.4**3.3, 2) == 0.05,
      f'0.4^3.3={0.4**3.3:.4f}, exact 0.4^(2/0.6)={0.4**(2/0.6):.4f}')

# 5. 0**0 == 1 so a cluster's first pick is never penalized
p0 = penalty(0.0, np.array([0.0, 0.001, 5.0]))
check('gamma=0: 0**0 = 1 for first pick', p0[0] == 1.0)
check('gamma=0: over-drawn -> 0', p0[1] == 0.0 and p0[2] == 0.0)

# 6. exponent clamp prevents underflow blowing up
pc = penalty(0.5, np.array([1e9]))
check('exponent clamped, finite', np.isfinite(pc[0]) and pc[0] == 0.5**MAX_EXPONENT)

# 7. gamma=1 reproduces the baseline top-k exactly
rng = np.random.default_rng(0)
u = rng.random(720)
labels = rng.integers(0, 8, 720)
quotas = compute_quotas(np.bincount(labels, minlength=8), 8)
sel1 = greedy_select(u, labels, quotas, 1.0, 8)
base = list(np.argsort(-u, kind='stable')[:8])
check('gamma=1 == baseline top-k', set(sel1) == set(base), f'{sorted(sel1)}')

# 8. the greedy loop matters: one-shot scoring would equal the baseline
sel0 = greedy_select(u, labels, quotas, 0.2, 8)
check('gamma=0.2 differs from baseline (greedy loop is doing work)',
      set(sel0) != set(base), f'overlap={len(set(sel0)&set(base))}/8')

# 9. gamma=0 with n_clusters == budget and near-equal quotas -> one per cluster
check('gamma=0 spreads across clusters',
      len(set(labels[greedy_select(u, labels, quotas, 0.0, 8)])) >= 7,
      f'distinct={len(set(labels[greedy_select(u,labels,quotas,0.0,8)]))}/8')

# 10. monotonicity: lower gamma -> more distinct clusters covered (on average)
dist = []
for gam in [1.0, 0.8, 0.5, 0.2, 0.0]:
    d = [len(set(labels[greedy_select(rng.random(720), labels, quotas, gam, 8)]))
         for _ in range(30)]
    dist.append(np.mean(d))
check('distinct clusters increases as gamma falls',
      all(dist[ii] <= dist[ii+1] + 1e-9 for ii in range(len(dist)-1)),
      ' -> '.join(f'{d:.2f}' for d in dist))

# 11. a dominant cluster cannot monopolise the budget at low gamma
big = np.zeros(720, dtype=int); big[:700] = 0; big[700:] = np.arange(20) % 7 + 1
ub = rng.random(720); ub[:700] += 1.0          # make the big cluster the most uncertain
qb = compute_quotas(np.bincount(big, minlength=8), 8)
n_big_hi = sum(1 for ii in greedy_select(ub, big, qb, 1.0, 8) if big[ii] == 0)
n_big_lo = sum(1 for ii in greedy_select(ub, big, qb, 0.2, 8) if big[ii] == 0)
check('dominant cluster capped at low gamma', n_big_hi == 8 and n_big_lo < 8,
      f'gamma=1 takes {n_big_hi}/8, gamma=0.2 takes {n_big_lo}/8 (its quota is {qb[0]:.2f})')

# 12. never returns duplicates, always returns min(k, n)
s = greedy_select(u, labels, quotas, 0.5, 8)
check('no duplicates, correct count', len(s) == len(set(s)) == 8)
s_small = greedy_select(u[:3], labels[:3], compute_quotas(np.bincount(labels[:3], minlength=8), 8), 0.5, 8)
check('k > pool size handled', len(s_small) == 3)

# --- agglomerative clustering ------------------------------------------------
# complete partition by construction: every point gets a cluster, no noise label
Zc = np.concatenate([rng.normal(0, .05, (300, 16)) + np.array([1.]+[0.]*15),
                     rng.normal(0, .05, (300, 16)) + np.array([0.,1.]+[0.]*14),
                     rng.normal(0, .05, (120, 16)) + np.array([0.,0.,1.]+[0.]*13)])
lab, kk = cluster_session(Zc, 0.25)
sz = np.bincount(lab, minlength=kk)
check('agglomerative: complete partition (no noise label)',
      lab.min() >= 0 and sz.sum() == len(Zc), f'k={kk}, sum={sz.sum()}')
check('agglomerative: recovers 3 planted groups', kk == 3, f'k={kk}')
check('agglomerative: quotas sum to k', abs(compute_quotas(sz, 8).sum() - 8) < 1e-9)
check('agglomerative: deterministic',
      np.array_equal(lab, cluster_session(Zc, 0.25)[0]))
check('agglomerative: no fixed count -- tighter threshold gives more clusters',
      cluster_session(Zc, 0.05)[1] >= kk,
      f'k(0.05)={cluster_session(Zc,0.05)[1]} >= k(0.25)={kk}')
check('agglomerative: singletons allowed (a lone point is its own region)',
      cluster_session(np.concatenate([Zc, rng.normal(0,.05,(1,16))+np.array([0.]*15+[9.])]), 0.25)[1] == kk + 1)
check('agglomerative: n<2 handled', cluster_session(Zc[:1], 0.25)[1] == 1)
check('agglomerative: zero vector does not break cosine',
      cluster_session(np.concatenate([Zc, np.zeros((1,16))]), 0.25)[0].min() >= 0)

# planted-group quotas behave: big group gets more, sqrt-damped not linear
q3 = compute_quotas(np.bincount(lab), 8)
big, small = q3[np.argmax(sz)], q3[np.argmin(sz)]
check('bigger cluster gets bigger quota, sqrt-damped',
      big > small and big / small < (sz.max() / sz.min()),
      f'q ratio {big/small:.2f} < size ratio {sz.max()/sz.min():.2f}')

print('\nALL PASS' if ok else '\nFAILURES PRESENT')
raise SystemExit(0 if ok else 1)
