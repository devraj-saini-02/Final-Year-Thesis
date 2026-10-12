"""Counterfactual probe: replays a BASELINE run (gamma = 1, so the trajectory is the archived
baseline's, checked against its scores.csv afterwards) and, every session, records

  - the unlabeled pool: base index, true label, locale, uncertainty u, pre-fit prediction
  - cluster labels at several thresholds
  - the baseline's 8 AL picks, and the picks the extension WOULD make at several settings
  - post-fit predictions for every clip of the session (incl. the ones AL just took)
  - the clips corrective feedback labels

Nothing here feeds back into the run: every extra computation is wrapped in an RNG snapshot /
restore, and features are read straight from the base dataset (no DataLoader, no ImlData pair
sampling). Usage (any cwd; writes output/ and checkpoints/ under --workdir):

  python tools/analysis/cf_probe.py --workdir SCRATCH_DIR --probe_out FILE.pkl  <run.py args...>

The 2026-10-10 batch (seed 1; ha yo ky vi skr ml; both arms) is archived in
runs/analysis/2026-10-10_oalcf_cf/probe/. Use a scratch --workdir: it receives a full copy of
the replayed run (output/, checkpoints/), which is a duplicate of the archived baseline.
"""
import os, sys, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F

MAIN = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro/oalcf-main"
sys.path.insert(0, MAIN)

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--workdir", required=True)
ap.add_argument("--probe_out", required=True)
pa, rest = ap.parse_known_args()
os.makedirs(pa.workdir, exist_ok=True)
os.chdir(pa.workdir)

import utils.device as device
device.get_device = lambda use_gpu=True: torch.device("cpu")

import run
import utils.diverse_query as dqm
from utils.diverse_query import DiverseStrategyManager, cluster_session, compute_quotas, greedy_select

SETTINGS = [(0.25, 0.8), (0.18, 0.95), (0.16, 0.93), (0.10, 0.97)]
THRESHOLDS = sorted({t for t, _ in SETTINGS})
LOG = {"sessions": []}


class RNG:
    def __enter__(self):
        self.t = torch.get_rng_state(); self.n = np.random.get_state(); return self
    def __exit__(self, *a):
        torch.set_rng_state(self.t); np.random.set_state(self.n)


def model_probs(module, ds, base_idxs):
    model = module.model
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(base_idxs), 512):
            x = torch.stack([ds[j][0] for j in base_idxs[i:i + 512]])
            out.append(F.softmax(model(x)[-1], dim=-1)[:, 1])
    if was_training:
        model.train()
    return torch.cat(out).numpy() if out else np.zeros(0)


class ProbeManager(DiverseStrategyManager):
    dq_gamma = 1.0
    dq_threshold = 0.25

    def select_queries(self, data_module, method_list, module, n_queries):
        idxs_dict, metrics = super().select_queries(data_module, method_list, module, n_queries)
        with RNG():
            probs = F.softmax(self.logits, dim=-1)
            smax = torch.max(probs, dim=-1)[0]
            u = (1.0 - smax.double()).numpy()
            p_tgt = probs[:, 1].numpy()
            sort_idx = torch.sort(smax)[1].numpy()
            rank = np.empty(len(u), dtype=np.int64); rank[sort_idx] = np.arange(len(u))
            Z = self._session_features(data_module)
            base = np.array(data_module.data_test.active_idxs)
            ds = data_module.ds
            rec = dict(session=data_module.current_batch, base=base,
                       label=np.array([ds.get_label(b) for b in base]),
                       locale=np.array([os.path.basename(ds.feat_files[b]).split("_")[2] for b in base]),
                       u=u, p_tgt_pre=p_tgt, picks_base=idxs_dict["smax"].numpy().copy(),
                       clusters={}, picks={})
            for t in THRESHOLDS:
                lab, nc = cluster_session(Z, t)
                rec["clusters"][t] = lab
            for t, g in SETTINGS:
                lab = rec["clusters"][t]
                sizes = np.maximum(np.bincount(lab).astype(float), 1.0)
                rec["picks"][(t, g)] = np.array(greedy_select(u, lab, compute_quotas(sizes, n_queries), g,
                                                              n_queries, tiebreak=rank))
            LOG["sessions"].append(rec)
        return idxs_dict, metrics


_orig_sim = run.FeedbackSimulator.simulate


def sim_probe(self, data_module, module):
    rec = LOG["sessions"][-1] if LOG["sessions"] else None
    if rec is not None and rec["session"] == data_module.current_batch:
        with RNG():
            rec["p_tgt_post"] = model_probs(module, data_module.ds, list(rec["base"]))
    idxs = _orig_sim(self, data_module, module)
    if rec is not None and rec["session"] == data_module.current_batch:
        rec["cf_base"] = np.array([data_module.data_test.active_idxs[i] for i in idxs], dtype=np.int64)
    return idxs


run.FeedbackSimulator.simulate = sim_probe
run.StrategyManager = ProbeManager

run_name = rest[rest.index("--run_name") + 1]
os.makedirs(os.path.join("output", os.path.dirname(run_name)), exist_ok=True)
os.makedirs(os.path.join("checkpoints", os.path.dirname(run_name)), exist_ok=True)
sys.argv = ["run.py"] + rest + ["--gpus", "0"]
try:
    run.main()
finally:
    pickle.dump(LOG, open(pa.probe_out, "wb"))
