"""Analyse cf_probe pickles: what the extension would swap, and what those clips are."""
import sys, pickle, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf")
import numpy as np, pandas as pd

files = sys.argv[1:] or sorted(glob.glob(os.path.join(DATA, "probe", "*.pkl")))
rows, sess_rows = [], []
for f in files:
    run = os.path.basename(f)[:-4]
    arm = run.split("_")[0]
    L = pickle.load(open(f, "rb"))
    for r in L["sessions"]:
        lab, u, loc = r["label"], r["u"], r["locale"]
        post = r.get("p_tgt_post")
        cf = set(r.get("cf_base", np.array([], dtype=int)).tolist())
        base_set = set(r["picks_base"].tolist())
        n_tgt = lab.sum()
        # how concentrated is the plain top-8 at threshold 0.16, and u gap to the best outside clip
        cl = r["clusters"][0.16]
        pb = r["picks_base"]
        top_cl = np.bincount(cl[pb]).argmax()
        outside = np.setdiff1d(np.flatnonzero(cl != top_cl), pb)
        u8 = np.sort(u[pb])[0]                        # weakest baseline pick
        u_out = u[outside].max() if len(outside) else 0
        sess_rows.append(dict(run=run, arm=arm, session=r["session"], n_tgt=n_tgt,
                              tgt_in_base_picks=lab[pb].sum(),
                              base_picks_cf_anyway=np.mean([(post[i] > 0.5) for i in pb]) if post is not None else np.nan,
                              base_picks_predpos_pre=np.mean(r["p_tgt_pre"][pb] > 0.5),
                              top8_clusters=len(np.unique(cl[pb])), top_cluster_share=np.mean(cl[pb] == top_cl),
                              u8=u8, u_out=u_out, gap=(u_out / u8) if u8 > 0 else np.nan,
                              n_cf=len(cf)))
        for (t, g), pe in r["picks"].items():
            ins = [i for i in pe if i not in base_set]
            outs = [i for i in pb if i not in set(pe.tolist())]
            for kind, idxs in (("in", ins), ("out", outs)):
                for i in idxs:
                    rows.append(dict(run=run, arm=arm, session=r["session"], setting=f"{t:g}/{g:g}", kind=kind,
                                     target=lab[i], locale=loc[i], u=u[i],
                                     pred_pos_pre=r["p_tgt_pre"][i] > 0.5,
                                     pred_pos_post=(post[i] > 0.5) if post is not None else np.nan,
                                     clsize=np.bincount(r["clusters"][t])[r["clusters"][t][i]],
                                     burst=n_tgt >= 30))
S = pd.DataFrame(sess_rows); W = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("=== per session, plain uncertainty top-8 (threshold 0.16 clusters) ===")
print(S.groupby("arm")[["tgt_in_base_picks", "base_picks_predpos_pre", "base_picks_cf_anyway",
                       "top8_clusters", "top_cluster_share", "n_cf"]].mean().round(3))
print("\nu ratio best-outside-clip / weakest-pick (penalty can only reorder if > gamma-multiplier):")
print(S.groupby("arm").gap.describe(percentiles=[.1, .25, .5, .75]).round(3))
print("\n=== swapped clips: 'out' = displaced baseline pick, 'in' = extension's replacement ===")
g = W.groupby(["arm", "setting", "kind"]).agg(n=("target", "size"), target=("target", "mean"),
        u=("u", "median"), predpos_pre=("pred_pos_pre", "mean"), predpos_post=("pred_pos_post", "mean"),
        clsize=("clsize", "median"))
print(g.round(3))
print("\nswaps in burst sessions (>=30 targets):")
print(W[W.burst].groupby(["arm", "kind"]).agg(n=("target", "size"), target=("target", "mean"),
      predpos_post=("pred_pos_post", "mean")).round(3))
S.to_csv(os.path.join(DATA, "tables", "probe_sessions.csv"), index=False)
W.to_csv(os.path.join(DATA, "tables", "probe_swaps.csv"), index=False)
