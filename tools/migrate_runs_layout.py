#!/usr/bin/env python3
"""
One-time move of runs/ into the seed-first layout (2026-09-18). Kept for the record; running it
again after the move does nothing, because there is nothing left in the old places.

    before                                          after
    runs/baseline_seed0/output/cr_oal/tt/           runs/seed_0/oal_baseline/cr_tt/
    runs/dq_oal_thresh/output/cr_oal_t16_g93/tt/    runs/seed_0/oal_t0.16_g0.93/cr_tt/
    runs/baseline_seeds/output/cr_oal_s3/tt/        runs/seed_3/oal_baseline/cr_tt/
    runs/<batch>/logs/<run>.log                     runs/seed_N/<setting>/<corpus>_<lang>/train.log
    runs/<batch>/RESULTS.md, summary*.txt           runs/reports/<batch>_<file>
    runs/<batch>/progress.log                       runs/reports/progress_logs/<batch>.log

Nothing is deleted and nothing is rewritten: every run directory is moved whole, so scores.csv,
command.txt and the weights stay exactly as they were.

    python tools/migrate_runs_layout.py --dry-run     # list every move, change nothing
    python tools/migrate_runs_layout.py               # do it
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import experiments  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
BATCHES = ["baseline_seed0", "baseline_seeds", "dq_oal", "dq_oalcf", "dq_oal_thresh",
           "dq_oalcf_thresh", "experiments"]


def setting_dir(algo, method, thr, gam):
    return f"{algo}_baseline" if method == "baseline" else f"{algo}_t{thr:g}_g{gam:g}"


def plan_runs():
    """[(source run dir, destination run dir)] from the index built on the current layout."""
    moves = []
    for (algo, method, thr, gam, seed), runs in experiments.index().items():
        for (corpus, lang), src in runs.items():
            dst = os.path.join(RUNS, f"seed_{seed}", setting_dir(algo, method, thr, gam),
                               f"{corpus}_{lang}")
            if os.path.abspath(src) != os.path.abspath(dst):
                moves.append((src, dst))
    return sorted(moves)


def plan_logs(run_moves):
    """Training logs sit in runs/<batch>/logs/<something>_<lang>.log; put each beside its run."""
    dest_of = {}
    for src, dst in run_moves:
        parts = src.split(os.sep)
        folder, lang = parts[-2], parts[-1]          # e.g. cr_oal_t16_g93 / tt
        dest_of[(folder, lang)] = dst
        if folder.endswith(f"_{lang}"):              # new layout: cr_tt (folder already has the lang)
            dest_of[(folder, "")] = dst
    moves, unmatched = [], []
    for batch in BATCHES:
        logs = os.path.join(RUNS, batch, "logs")
        if not os.path.isdir(logs):
            continue
        for name in sorted(os.listdir(logs)):
            if not name.endswith(".log"):
                continue
            stem = name[:-4]
            if batch == "baseline_seed0" and stem.startswith("baseline_"):
                stem = stem[len("baseline_"):]
            hit = None
            for (folder, lang), dst in dest_of.items():
                if lang and stem == f"{folder}_{lang}":
                    hit = dst
                    break
                if not lang and stem == folder:
                    hit = dst
                    break
            if hit:
                moves.append((os.path.join(logs, name), os.path.join(hit, "train.log")))
            else:
                unmatched.append(os.path.join(logs, name))
    return moves, unmatched


def plan_reports():
    moves = []
    for batch in BATCHES:
        d = os.path.join(RUNS, batch)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            src = os.path.join(d, name)
            if not os.path.isfile(src):
                continue
            if name == "progress.log":
                moves.append((src, os.path.join(RUNS, "reports", "progress_logs", f"{batch}.log")))
            else:
                moves.append((src, os.path.join(RUNS, "reports", f"{batch}__{name}")))
    figs = os.path.join(RUNS, "baseline_seed0", "figures")
    if os.path.isdir(figs):
        moves.append((figs, os.path.join(RUNS, "figures")))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_moves = plan_runs()
    log_moves, unmatched = plan_logs(run_moves)
    report_moves = plan_reports()

    print(f"{len(run_moves)} run directories, {len(log_moves)} logs, {len(report_moves)} reports/figures")
    if unmatched:
        print(f"{len(unmatched)} log file(s) with no matching run (left where they are):")
        for u in unmatched[:10]:
            print("   " + os.path.relpath(u, ROOT))
    for src, dst in (run_moves[:3] + report_moves[:3]):
        print(f"   {os.path.relpath(src, ROOT)}\n      -> {os.path.relpath(dst, ROOT)}")
    if args.dry_run:
        print("\ndry run: nothing moved")
        return

    for src, dst in run_moves + log_moves + report_moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"   SKIP (destination exists): {os.path.relpath(dst, ROOT)}")
            continue
        shutil.move(src, dst)

    for batch in BATCHES:                            # remove the now-empty old scaffolding
        for sub in ("output", "logs", ""):
            d = os.path.join(RUNS, batch, sub)
            if os.path.isdir(d) and not any(os.scandir(d)):
                os.rmdir(d)
    for batch in BATCHES:
        d = os.path.join(RUNS, batch)
        if os.path.isdir(d):
            leftovers = [p for p in os.listdir(d)]
            if leftovers:
                print(f"   left in runs/{batch}: {leftovers[:5]}")
    print("\ndone. Check with: python tools/experiments.py")


if __name__ == "__main__":
    main()
