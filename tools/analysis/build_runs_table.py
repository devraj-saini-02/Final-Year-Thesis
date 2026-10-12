"""Per-run table over every finished run: scores, label composition, derived AL target hits,
and extension diagnostics. Writes runs_table.csv (and sessions.pkl, not archived) to runs/analysis/2026-10-10_oalcf_cf/tables/."""
import os, sys, pickle
import numpy as np, pandas as pd
ROOT = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from experiments import index, SESSIONS
from score_runs import score
OUT = os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf", "tables")

rows, sess = [], {}
for (algo, method, thr, gam, seed), runs in index().items():
    for (c, l), d in runs.items():
        s = pd.read_csv(os.path.join(d, "scores.csv"))
        if len(s) != SESSIONS[c]:
            continue
        r = score(d)
        tgt = np.round(s["p_target"] * s["n_samples"]).astype(int).values
        prev = np.concatenate([[4], tgt[:-1]])          # bootstrap = 4 target + 4 non-target
        al_t = np.clip(tgt - prev - s["cf_tp"].values, 0, 8)
        s = s.assign(al_tgt=al_t)
        dv = os.path.join(d, "al_diversity.csv")
        changed = ugiven = np.nan
        if os.path.exists(dv):
            q = pd.read_csv(dv)
            q = q.iloc[:len(s)]
            s = s.assign(changed=(8 - q["overlap_with_runpy"]).values,
                         u_pick=q["mean_u_picked"].values, u_base=q["mean_u_baseline"].values,
                         n_clusters=q["n_clusters"].values,
                         distinct=q["distinct_clusters_picked"].values)
            changed = s["changed"].mean()
            ugiven = 1 - s["u_pick"].sum() / max(s["u_base"].sum(), 1e-12)
        key = (algo, method, thr, gam, seed, c, l)
        sess[key] = s
        n_tgt_stream = s["pre_ps"].sum()
        rows.append(dict(algo=algo, method=method, thr=thr, gam=gam, seed=seed, corpus=c, lang=l,
                         eq1=r["IMLM_eq1"], dcf=r["DCF"], fnr=r["FNR"], fpr=r["FPR"],
                         fns=s["fns"].sum(), fps=s["fps"].sum(), pre_fns=s["pre_fns"].sum(),
                         pre_fps=s["pre_fps"].sum(), n_tgt=n_tgt_stream,
                         AL=r["AL"], cf_tp=r["CF_tp"], cf_fp=r["CF_fp"], pool=s["n_samples"].iloc[-1],
                         al_tgt=al_t.sum(), al_tgt_rate=al_t.sum() / r["AL"],
                         stream_prev=n_tgt_stream / (s["pre_ps"] + s["pre_ns"]).sum(),
                         changed=changed, ugiven=ugiven))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "runs_table.csv"), index=False)
pickle.dump(sess, open(os.path.join(OUT, "sessions.pkl"), "wb"))   # 23 MB, regenerated on demand, not archived
print(len(df), "runs")
print(df.groupby(["algo", "method", "thr", "gam"], dropna=False).size())
