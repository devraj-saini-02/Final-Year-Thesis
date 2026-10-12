"""CF deduplication: cluster each session's corrective-feedback clips (cosine, average linkage,
distance threshold, the extension's own cluster_session), ask the expert for ONE label per
cluster (the member the model is most confident is target), copy it to the rest.
Measured on the 6 OAL-CF probe runs (seed 1, baseline trajectories)."""
import os, sys, glob, pickle, types
import numpy as np, pandas as pd

ROOT = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
sys.path.insert(0, os.path.join(ROOT, "oalcf-main"))
from src.dataset import BaseLidData
from utils.diverse_query import cluster_session

DATA = os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf")
CORP = {"ac": "AfricanContinentLID", "cr": "CaucasusRegionLID", "sea": "SoutheastAsiaLID", "sa": "SouthAsiaLID"}
THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]

sess_rows, clus_rows = [], []
for f in sorted(glob.glob(os.path.join(DATA, "probe", "oalcf_*_s1.pkl"))):
    corpus, lang = os.path.basename(f)[6:-7].split("_")
    p = types.SimpleNamespace(feat_root=f"{ROOT}/data/{CORP[corpus]}/ecapalang/", env_name="test",
                              order_file="seed1", lid_target=lang, reverse_order=False,
                              bootstrap=8, samples_per_batch=720, context=3)
    ds = BaseLidData(p); ds.load_external_bootstrap()
    for r in pickle.load(open(f, "rb"))["sessions"]:
        cf = r.get("cf_base")
        if cf is None or len(cf) == 0:
            continue
        pos = {b: i for i, b in enumerate(r["base"])}
        idx = np.array([pos[b] for b in cf])
        lab = r["label"][idx]; post = r["p_tgt_post"][idx]
        assert all(ds.get_label(b) == l for b, l in zip(cf, lab))      # file mapping check
        Z = np.stack([ds._load_feat(ds.feat_files[b].format("ecapalang")) for b in cf]).astype(np.float64)
        for t in THRESHOLDS:
            cl, nc = cluster_session(Z, t)
            sizes = np.bincount(cl)
            wrong = fp_as_t = t_as_fp = impure = 0
            for c in range(nc):
                m = np.flatnonzero(cl == c)
                rep = m[np.argmax(post[m])]
                w = lab[m] != lab[rep]
                wrong += w.sum(); fp_as_t += ((lab[m] == 0) & (lab[rep] == 1)).sum()
                t_as_fp += ((lab[m] == 1) & (lab[rep] == 0)).sum()
                impure += len(np.unique(lab[m])) > 1
                clus_rows.append(dict(t=t, size=len(m), pure=len(np.unique(lab[m])) == 1, run=f"{corpus}_{lang}"))
            sess_rows.append(dict(run=f"{corpus}_{lang}", corpus=corpus, session=r["session"], t=t, n_cf=len(cf),
                                  n_tp=int(lab.sum()), n_fp=int((1 - lab).sum()), asked=nc, wrong=wrong,
                                  fp_as_t=fp_as_t, t_as_fp=t_as_fp, impure=impure, largest=sizes.max()))
S = pd.DataFrame(sess_rows); C = pd.DataFrame(clus_rows)
S.to_csv(os.path.join(DATA, "tables", "cf_dedup_sessions.csv"), index=False)
C.to_csv(os.path.join(DATA, "tables", "cf_dedup_clusters.csv"), index=False)
print("done", S.run.nunique(), "runs,", S.session.count() // len(THRESHOLDS), "sessions with CF")
