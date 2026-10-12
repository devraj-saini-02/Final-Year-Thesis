#!/usr/bin/env python3
"""
One-time move of runs/ from the seed-first layout to a method-first layout (2026-10-10).

    before                                         after
    runs/seed_N/oal_baseline/<c>_<l>/              runs/baseline/oal/seed_N/<c>_<l>/
    runs/seed_N/oal_t0.16_g0.93/<c>_<l>/           runs/diversity_extension/oal_t0.16_g0.93/seed_N/<c>_<l>/
    runs/seed_N/oalcf_cfd0.1/<c>_<l>/              runs/cf_dedup/oalcf_cfd0.1/seed_N/<c>_<l>/
    runs/cluster_analysis/                         runs/analysis/cluster_analysis/
    runs/comparisons/<name>/                       runs/comparisons/<category of the tested setting>/<name>/

Whole folders are renamed, never edited or copied: progress.log and failed/ travel with their batch.
Every scores.csv is checksummed before and verified after; the script refuses to overwrite anything.

    python tools/migrate_runs_by_method.py            # dry run: print what would move
    python tools/migrate_runs_by_method.py --apply    # do it, verify, write the log
Log: runs/reports/layout_migration_2026-10-10.txt
"""
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import ROOT, CATEGORY, parse_setting, seed_dir, index  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
LOG = os.path.join(RUNS, "reports", "layout_migration_2026-10-10.txt")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def comparison_category(name):
    """runs/comparisons/<A>_vs_<B>_seeds..._<metric>: category of B, the setting being tested."""
    m = re.fullmatch(r".+?_vs_(.+?)_seeds[\d-]+_.+", name)
    if not m:
        return None
    b = m.group(1)
    if "_cfd" in b:
        return "cf_dedup"
    if re.search(r"_t[\d.]+_g[\d.]+", b):
        return "diversity_extension"
    return "baseline"


def plan():
    moves = []
    for sdir in sorted(os.listdir(RUNS)):
        m = re.fullmatch(r"seed_(\d+)", sdir)
        if not m:
            continue
        for setting in sorted(os.listdir(os.path.join(RUNS, sdir))):
            src = os.path.join(RUNS, sdir, setting)
            parsed = parse_setting(setting)
            if not os.path.isdir(src) or not parsed:
                raise SystemExit(f"unexpected entry, refusing to guess: {src}")
            moves.append((src, seed_dir(*parsed, int(m.group(1)))))
    ca = os.path.join(RUNS, "cluster_analysis")
    if os.path.isdir(ca):
        moves.append((ca, os.path.join(RUNS, "analysis", "cluster_analysis")))
    comp = os.path.join(RUNS, "comparisons")
    for name in sorted(os.listdir(comp)) if os.path.isdir(comp) else []:
        if name in set(CATEGORY.values()):
            continue
        cat = comparison_category(name)
        if cat is None:
            raise SystemExit(f"cannot classify comparison folder: {name}")
        moves.append((os.path.join(comp, name), os.path.join(comp, cat, name)))
    return moves


def main():
    apply = "--apply" in sys.argv
    moves = plan()
    clash = [d for _, d in moves if os.path.exists(d)]
    if clash:
        raise SystemExit(f"destination already exists, nothing moved: {clash[:3]}")
    before = {}
    for key, runs in index().items():
        for (c, l), d in runs.items():
            before[key + (c, l)] = md5(os.path.join(d, "scores.csv"))
    rel = lambda p: os.path.relpath(p, ROOT)
    for s, d in moves:
        print(f"{'move' if apply else 'would move'}  {rel(s)}  ->  {rel(d)}")
    print(f"\n{len(moves)} folders, {len(before)} runs")
    if not apply:
        print("dry run - nothing changed. Re-run with --apply.")
        return
    for s, d in moves:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        os.rename(s, d)
    for sdir in os.listdir(RUNS):
        if re.fullmatch(r"seed_\d+", sdir):
            os.rmdir(os.path.join(RUNS, sdir))          # fails loudly if anything was left behind
    after = {}
    for key, runs in index().items():
        for (c, l), d in runs.items():
            after[key + (c, l)] = (md5(os.path.join(d, "scores.csv")), d)
    missing = [k for k in before if k not in after]
    changed = [k for k in before if k in after and after[k][0] != before[k]]
    misplaced = [k for k, (_, d) in after.items() if not d.startswith(os.path.join(RUNS, CATEGORY[k[1]]))]
    ok = not (missing or changed or misplaced) and len(after) == len(before)
    with open(LOG, "w") as f:
        f.write("runs/ layout migration, 2026-10-10 (tools/migrate_runs_by_method.py)\n")
        f.write("seed-first  runs/seed_N/<setting>/<corpus>_<lang>  ->  "
                "method-first  runs/<category>/<setting>/seed_N/<corpus>_<lang>\n\n")
        for s, d in moves:
            f.write(f"{rel(s)}  ->  {rel(d)}\n")
        f.write(f"\n{len(before)} runs before, {len(after)} after; missing {len(missing)}, "
                f"scores.csv changed {len(changed)}, outside their category {len(misplaced)}: "
                f"{'VERIFIED' if ok else 'PROBLEM'}\n")
    print(f"\nverified: {len(after)}/{len(before)} runs found, scores.csv byte-identical: "
          f"{not changed}, none missing: {not missing}, all in the right category: {not misplaced}")
    print(f"log: {rel(LOG)}")
    if not ok:
        raise SystemExit("VERIFICATION FAILED - see the log")


if __name__ == "__main__":
    main()
