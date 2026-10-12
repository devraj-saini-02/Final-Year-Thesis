"""Checks for the CF-dedup smoke runs (seed 1, African, OAL-CF), as archived in
runs/cf_dedup/oalcf_cfd0.1/seed_1/ac_{ha,yo} and runs/cf_dedup/oalcf_cfd0/seed_1/ac_ha.
Run: python tools/analysis/verify_cfdedup_smoke.py"""
import os, re, sys, filecmp
import numpy as np, pandas as pd, torch
R = "/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
sys.path.insert(0, os.path.join(R, "tools"))
from score_runs import score
RUNS = {"ac_ha_t0": "runs/cf_dedup/oalcf_cfd0/seed_1/ac_ha", "ac_ha_t0.10": "runs/cf_dedup/oalcf_cfd0.1/seed_1/ac_ha",
        "ac_yo_t0.10": "runs/cf_dedup/oalcf_cfd0.1/seed_1/ac_yo"}
OFFLINE = {"ha": (360, 0), "yo": (392, 8)}      # offline prediction at 0.10: expert labels, wrong labels

for name in sorted(RUNS):
    d = os.path.join(R, RUNS[name]); lang = name.split("_")[1]; thr = float(name.split("_t")[1])
    base = os.path.join(R, f"runs/baseline/oalcf/seed_1/ac_{lang}")
    print(f"\n=== {name}")
    sc = pd.read_csv(os.path.join(d, "scores.csv")); b = pd.read_csv(os.path.join(base, "scores.csv"))
    cf = pd.read_csv(os.path.join(d, "cf_dedup.csv"))
    log = open(os.path.join(d, "train.log")).read()
    sd = torch.load(os.path.join(d, "state_dict.pt"), map_location="cpu")
    checks = {
        "full session count (83)": len(sc) == 83,
        "no NaN/inf in scores.csv": np.isfinite(sc.select_dtypes("number").values).all(),
        "final weights finite": all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()),
        "no 'nan' loss / Traceback / Warning-errors in log": not re.search(r"\bnan\b|Traceback|ZeroDivision|RuntimeWarning", log, re.I),
        "cf_dedup.csv one row per session": len(cf) == 83,
        "CF clip totals agree with scores.csv": cf.n_cf.sum() == sc.cf_tp.sum() + sc.cf_fp.sum(),
        "true CF classes agree with scores.csv": cf.cf_target.sum() == sc.cf_tp.sum() and cf.cf_nontarget.sum() == sc.cf_fp.sum(),
        "expert + copied = CF clips": (cf.expert_labels + cf.copied == cf.n_cf).all(),
        "command.txt records the threshold": f"--cfd_threshold {name.split('_t')[1]}" in open(os.path.join(d, "command.txt")).read(),
    }
    flips = cf[cf.wrong > 0]
    first_flip = int(flips.session.min()) if len(flips) else None
    same = (sc.drop(columns=[]).values == b.values).all(axis=1)
    first_diff = int(np.argmin(same)) if not same.all() else None
    if thr == 0:
        checks["control: bit-identical to baseline scores.csv"] = filecmp.cmp(os.path.join(d, "scores.csv"), os.path.join(base, "scores.csv"), shallow=False)
        checks["control: expert labels every CF clip"] = (cf.expert_labels == cf.n_cf).all() and cf.wrong.sum() == 0
    else:
        checks["identical to baseline before the first wrong label"] = (first_diff is None) or (first_flip is not None and first_diff >= first_flip)
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    r = score(d); rb = score(base)
    print(f"   CF clips {cf.n_cf.sum()}, expert labels {cf.expert_labels.sum()} ({1 - cf.expert_labels.sum() / max(cf.n_cf.sum(), 1):.0%} saved), "
          f"wrong {cf.wrong.sum()} (false alarm->target {cf.nontarget_given_target.sum()}, target->non-target {cf.target_given_nontarget.sum()})")
    if thr > 0:
        print(f"   offline prediction on the baseline trajectory: expert {OFFLINE[lang][0]}, wrong {OFFLINE[lang][1]}")
    print(f"   first wrong label at session {first_flip}; scores first differ from baseline at session {first_diff}")
    for m in ["IMLM_eq1", "IMLM_thesis", "IMLM_repo", "IMLM_repo_expert", "DCF"]:
        print(f"   {m:17s} baseline {rb[m]:.4f}  ->  {r[m]:.4f}")
