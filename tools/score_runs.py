#!/usr/bin/env python3
"""
Score OAL / OAL-CF run directories with BOTH IMLM formulas, from the same scores.csv
(no extra runs needed). See docs/other/DEVIATIONS.md, "Scoring".

  IMLM_repo : the repo's scoring/print_basic_results.py, unchanged upstream formula
                (N_fp + N_pool) / N_nontarget + N_fn / N_target
              N_pool = final adaptation pool = bootstrap + AL queries + every CF label
              (CF-confirmed true positives included). Matches the dissertation's Eq. 3.6.

  IMLM_eq1  : the paper's Eq. 1 (Sec. IV-A), as the authors compute it in their own plotting
              code (plotting/trend_plot.py, postquential branch):
                (N_fp + N_boot + N_AL) / N + N_fn / N_target
              N = all samples in the stream. Every false positive is counted once (in N_fp),
              including those CF corrects; CF-confirmed true positives are not charged.
              --literal: the literal reading of Eq. 1, where N_ann also includes the
              CF-corrected false positives, so each of those is counted twice.

  IMLM_thesis : the dissertation's Eq. 3.8 (= Eq. 3.6 with q = N/N_target):
                (N_adapt + N_fp) / N + N_fn / N_target
              N_adapt = samples an EXPERT labelled for adaptation: bootstrap + AL queries + CF labels.
              This is the one formula that charges every CF label, confirmations included.
  IMLM_repo_expert : the repo scorer's formula with the pool size replaced by N_adapt (expert
              labels). Equal to IMLM_repo for every run except the CF-deduplication extension.

For the CF-deduplication extension (oalcf-main/run_cfdedup.py) the expert labels only one clip
per group of near-duplicate CF clips; its cf_dedup.csv gives the per-session count, and that count
is what N_adapt uses. Every other run has CF_expert = CF_tp + CF_fp, so nothing changes for them.

DCF / FNR / FPR are the same as the repo scorer's. Compare against the paper's Fig. 12 with
IMLM_eq1 only.

Usage, from oalcf-main/:
    python ../tools/score_runs.py output/ac_oal/* output/ac_oalcf/*
    python ../tools/score_runs.py --csv results.csv output/*_oal*/*
"""
import argparse
import os

import pandas as pd


def score(run_dir, literal=False):
    s = pd.read_csv(os.path.join(run_dir, "scores.csv"))
    fps, fns = s["fps"].sum(), s["fns"].sum()
    ns, ps = s["ns"].sum(), s["ps"].sum()
    pool = s["n_samples"].iloc[-1]
    al, cf_tp, cf_fp = s["n_al"].sum(), s["cf_tp"].sum(), s["cf_fp"].sum()
    # bootstrap size = pool after session 0 minus what session 0 added
    boot = s["n_samples"].iloc[0] - s["n_al"].iloc[0] - s["cf_tp"].iloc[0] - s["cf_fp"].iloc[0]
    n_all = (s["pre_ps"] + s["pre_ns"]).sum()
    n_target = s["pre_ps"].sum()
    n_ann = boot + al + (cf_fp if literal else 0)
    # clips the expert actually labelled through CF: all of them, unless the CF-dedup extension
    # copied some labels (then cf_dedup.csv has the per-session count)
    cf_expert = cf_tp + cf_fp
    dedup = os.path.join(run_dir, "cf_dedup.csv")
    if os.path.exists(dedup):
        d = pd.read_csv(dedup)
        if int(d["n_cf"].sum()) != int(cf_tp + cf_fp):
            raise ValueError(f"{dedup}: n_cf total {int(d['n_cf'].sum())} != scores.csv CF total "
                             f"{int(cf_tp + cf_fp)}")
        cf_expert = d["expert_labels"].sum()
    n_adapt = boot + al + cf_expert
    nan = float("nan")
    div = lambda a, b: a / b if b > 0 else nan     # a corpus with no targets cannot be scored
    return dict(sessions=len(s), DCF=s["cum_dcf"].iloc[-1], FNR=div(fns, ps), FPR=div(fps, ns),
                IMLM_repo=div(fps + pool, ns) + div(fns, ps),
                IMLM_eq1=div(n_ann + fps, n_all) + div(fns, n_target),
                IMLM_thesis=div(n_adapt + fps, n_all) + div(fns, n_target),
                IMLM_repo_expert=div(fps + n_adapt, ns) + div(fns, ps),
                boot=int(boot), AL=int(al), CF_tp=int(cf_tp), CF_fp=int(cf_fp),
                CF_expert=int(cf_expert))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories containing scores.csv")
    ap.add_argument("--literal", action="store_true",
                    help="literal Eq. 1: CF-corrected false positives counted twice (in N_ann and N_fp)")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    args = ap.parse_args()

    rows = []
    for d in args.runs:
        if not os.path.exists(os.path.join(d, "scores.csv")):
            print(f"skip (no scores.csv): {d}")
            continue
        rows.append(dict(run=os.path.normpath(d), **score(d, args.literal)))
    if not rows:
        return
    df = pd.DataFrame(rows)
    fmt = {c: "{:.4f}".format for c in ["DCF", "FNR", "FPR", "IMLM_repo", "IMLM_eq1",
                                            "IMLM_thesis", "IMLM_repo_expert"]}
    print(df.to_string(index=False, formatters=fmt))
    if args.literal:
        print("(IMLM_eq1 computed with --literal: CF-corrected false positives counted twice)")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
