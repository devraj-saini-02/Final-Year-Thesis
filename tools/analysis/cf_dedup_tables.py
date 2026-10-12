#!/usr/bin/env python3
"""Write the CF-deduplication cluster tables (savings, errors, cluster sizes per threshold) to
runs/analysis/2026-10-10_oalcf_cf/cf_dedup_tables.txt.

Inputs: tables/cf_dedup_{sessions,clusters}.csv (offline, seed 1, made by cf_dedup_offline.py) and
the real seed-2 runs in runs/cf_dedup/oalcf_cfd0.1/seed_2/*/cf_dedup.csv.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from experiments import ALL_LANGS, run_path  # noqa: E402

DATA = os.path.join(ROOT, "runs/analysis/2026-10-10_oalcf_cf")
OUT = os.path.join(DATA, "cf_dedup_tables.txt")
SECONDS_PER_CLIP = 5.19      # AC mean clip length, Table II (82 h / 56,880 clips)


def table(df):
    return df.to_string() + "\n"


def main():
    S = pd.read_csv(os.path.join(DATA, "tables", "cf_dedup_sessions.csv"))
    C = pd.read_csv(os.path.join(DATA, "tables", "cf_dedup_clusters.csv"))
    g = S.groupby("t").sum(numeric_only=True)
    n_cf, n_tp, n_fa = int(g.n_cf.iloc[0]), int(g.n_tp.iloc[0]), int(g.n_fp.iloc[0])
    w = io.StringIO()
    p = lambda *a: print(*a, file=w)

    p("CF DEDUPLICATION: CLUSTERS, ERRORS AND SAVINGS BY DISTANCE THRESHOLD")
    p("=" * 78)
    p("Method: each session's corrective-feedback (CF) clips are clustered on their ECAPA embeddings")
    p("(cosine distance, average linkage, distance threshold t). The expert labels ONE clip per cluster")
    p("(the one the model is most confident is target) and that label is copied to the other members.")
    p("Lower t = tighter clusters = fewer copies, fewer errors.")
    p("")
    p(f"Offline data: 6 OAL-CF baseline runs, stream-order seed 1 (ac/ha ac/yo cr/ky sea/vi sa/ml sa/skr),")
    p(f"113 sessions with CF, {n_cf:,} CF clips = {n_tp:,} true detections + {n_fa:,} false alarms.")
    p("Measured on the baseline trajectories (tools/analysis/cf_dedup_offline.py).")
    p("")

    p("TABLE 1. Predicted savings and error rate per threshold")
    p("-" * 78)
    t1 = pd.DataFrame({
        "clusters (=expert labels)": g.asked.astype(int),
        "labels saved": (g.n_cf - g.asked).astype(int),
        "saved %": (100 * (1 - g.asked / g.n_cf)).round(1),
        "listening h saved": ((g.n_cf - g.asked) * SECONDS_PER_CLIP / 3600).round(1),
        "wrong copies": g.wrong.astype(int),
        "error % (of CF)": (100 * g.wrong / g.n_cf).round(2),
        "FA->target": g.fp_as_t.astype(int),
        "FA error % (of FAs)": (100 * g.fp_as_t / g.n_fp).round(1),
        "target->non-target": g.t_as_fp.astype(int),
        "mixed clusters %": (100 * g.impure / g.asked).round(1),
    })
    t1.index.name = "threshold"
    p(table(t1))
    p("FA->target: a false alarm given the label 'target' (teaches the model a wrong positive).")
    p("target->non-target: a real target given 'non-target' (the more damaging kind for misses).")
    p("Mixed cluster: a cluster holding both true targets and false alarms (the source of all errors).")
    p(f"Listening hours at {SECONDS_PER_CLIP} s per clip; total CF review = {n_cf * SECONDS_PER_CLIP / 3600:.1f} h.")
    p("")

    p("TABLE 2. How many chunks, and how chunky (share of CF clips by the size of their cluster)")
    p("-" * 78)
    rows = {}
    for t, x in C.groupby("t"):
        sz = x["size"].to_numpy(); n = sz.sum()
        share = lambda lo, hi: f"{100 * sz[(sz >= lo) & (sz <= hi)].sum() / n:.0f}%"
        rows[t] = {"clusters": len(sz), "clusters/session": round(len(sz) / S.session.count() * len(g), 1),
                   "singleton clusters %": round(100 * (sz == 1).mean()),
                   "clips alone": share(1, 1), "in 2-4": share(2, 4), "in 5-19": share(5, 19),
                   "in 20-99": share(20, 99), "in 100+": share(100, 10 ** 9),
                   "mean size": round(n / len(sz), 2),
                   "median clip's cluster": int(np.median(np.repeat(sz, sz))), "largest": int(sz.max())}
    t2 = pd.DataFrame(rows).T
    t2.index.name = "threshold"
    p(table(t2))
    p("'median clip's cluster' = size of the cluster a typical CF clip sits in.")
    p("")

    p("TABLE 3. Per language at the thresholds worth considering")
    p("-" * 78)
    for t in (0.08, 0.10, 0.12, 0.14):
        h = S[S.t == t].groupby("run").sum(numeric_only=True)
        t3 = pd.DataFrame({"CF clips": h.n_cf, "false alarms": h.n_fp, "expert labels": h.asked,
                           "saved %": (100 * (1 - h.asked / h.n_cf)).round(0).astype(int),
                           "wrong": h.wrong, "FA->target": h.fp_as_t, "target->non-target": h.t_as_fp})
        t3.index.name = f"t = {t}"
        p(table(t3))

    p("TABLE 4. Prediction vs real runs at threshold 0.10")
    p("-" * 78)
    real = []
    for c, l in ALL_LANGS:
        f = os.path.join(run_path("oalcf", "cfdedup", 0.1, None, 2, c, l), "cf_dedup.csv")
        if os.path.exists(f):
            real.append(pd.read_csv(f).sum(numeric_only=True))
    off = g.loc[0.10]
    lines = [("offline prediction (seed 1, 6 languages)", off.n_cf, off.asked, off.wrong, off.fp_as_t, off.n_fp, off.t_as_fp)]
    if real:
        r = pd.DataFrame(real).sum()
        lines.append((f"real runs (seed 2, {len(real)} languages)", r.n_cf, r.expert_labels, r.wrong,
                      r.nontarget_given_target, r.cf_nontarget, r.target_given_nontarget))
    t4 = pd.DataFrame([{"": a, "CF clips": int(n), "saved %": round(100 * (1 - e / n), 1),
                        "error % (of CF)": round(100 * wr / n, 2), "FA error %": round(100 * fa / nfa, 2),
                        "target->non-target": int(t2n)} for a, n, e, wr, fa, nfa, t2n in lines]).set_index("")
    p(table(t4))
    p("Real-run counts differ from the baseline's because a run diverges after its first wrong copy.")
    p("")
    p("Reading: 0.10 saves ~30% of CF labels at ~0.2% wrong copies; 0.12 saves ~49% at ~0.6% but")
    p("mislabels ~3% of false alarms; from 0.14 the false-alarm error climbs steeply (6% -> 33% at 0.25).")
    p(f"\nGenerated by tools/analysis/cf_dedup_tables.py")
    open(OUT, "w").write(w.getvalue())
    print(w.getvalue())
    print(f"written to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
