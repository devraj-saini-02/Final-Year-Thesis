#!/usr/bin/env python3
"""
Compare two settings language by language, with the significance test the paper uses.

    python tools/compare_runs.py oal oalcf                     # does CF help? (seed 0)
    python tools/compare_runs.py oal oal@0.16/0.93             # does the extension help OAL?
    python tools/compare_runs.py oal oalcf --seeds 0 1 2 3 4   # average over 5 stream orders
    python tools/compare_runs.py oalcf oalcf@0.25/0.8 --metric DCF
    python tools/compare_runs.py --list                        # what has been run

Every comparison is saved in its own folder under runs/comparisons/<category of B>/ (baseline,
diversity_extension or cf_dedup), holding report.txt (the whole
report) and table.csv (the per-language table). The first run of a comparison keeps the plain folder
name; later runs of the same comparison get <name>_1, <name>_2, ... so nothing is overwritten.
Pass --no-save to skip writing, or --csv PATH to also write the table somewhere specific.

Specs: `oal`, `oalcf`, or `<algo>@<threshold>/<gamma>` for the extension (tools/experiments.py).
B is the one being tested; A is what it is compared against. Lower is better for every metric,
so a "win" means B scored lower than A on that language.

With several seeds, each language is first averaged over the seeds it has in both settings, and
the test runs on those 20 averages (never on 20 x n_seeds rows, which would fake independence).

Metrics (tools/score_runs.py): IMLM_eq1 is the paper's Eq. 1 and the default - use it for anything
compared against the paper. IMLM_repo is the released scorer's variant, DCF/FNR/FPR are error rates.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (ALL_LANGS, CATEGORY, CORPUS_NAME, DUPLICATES, LANGS, ROOT,  # noqa: E402
                         SESSIONS, index, parse_spec, spec_name)
from score_runs import score  # noqa: E402


def collect(key4, seeds, metric, idx):
    """{(corpus, lang): {seed: value}} for one setting."""
    out = {}
    for seed in seeds:
        for (c, l), d in idx.get(key4 + (seed,), {}).items():
            r = score(d)
            if r["sessions"] != SESSIONS[c]:
                print(f"   WARNING: {d} has {r['sessions']}/{SESSIONS[c]} sessions, skipped")
                continue
            out.setdefault((c, l), {})[seed] = r[metric]
    return out


def save_name(na, nb, seeds, metric):
    """File stem for a comparison: oal_vs_oal_t0.16_g0.93_seeds0-1_IMLM_eq1."""
    def safe(spec):
        return spec.replace("@", "_t").replace("/", "_g").replace("+", "_")
    return f"{safe(na)}_vs_{safe(nb)}_seeds{'-'.join(str(s) for s in seeds)}_{metric}"


def main():
    ap = argparse.ArgumentParser(description="Compare two settings across the 20 SLV languages.")
    ap.add_argument("a", nargs="?", help="baseline spec, e.g. oal")
    ap.add_argument("b", nargs="?", help="spec being tested, e.g. oal@0.16/0.93")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="stream-order seeds (default 0)")
    ap.add_argument("--metric", default="IMLM_eq1",
                    choices=["IMLM_eq1", "IMLM_repo", "IMLM_thesis", "IMLM_repo_expert",
                             "DCF", "FNR", "FPR"])
    ap.add_argument("--list", action="store_true", help="list everything that has been run, then exit")
    ap.add_argument("--csv", help="also write the per-language table to this exact path")
    ap.add_argument("--no-save", action="store_true",
                    help="do not save to runs/comparisons/ (printing only)")
    args = ap.parse_args()

    if args.list or not (args.a and args.b):
        import experiments
        experiments.main()
        return

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    idx = index()
    try:
        ka, kb = parse_spec(args.a), parse_spec(args.b)
    except ValueError as e:
        print(e)
        return
    if ka == kb:
        print(f"'{args.a}' and '{args.b}' are the same setting - nothing to compare.")
        return
    A, B = collect(ka, args.seeds, args.metric, idx), collect(kb, args.seeds, args.metric, idx)
    na, nb = spec_name(*ka), spec_name(*kb)

    rows = []
    for c, l in ALL_LANGS:
        shared = sorted(set(A.get((c, l), {})) & set(B.get((c, l), {})))
        if shared:
            rows.append((c, l, np.mean([A[(c, l)][s] for s in shared]),
                         np.mean([B[(c, l)][s] for s in shared]), shared))
    if not rows:
        missing_a = [s for s in args.seeds if not idx.get(ka + (s,))]
        missing_b = [s for s in args.seeds if not idx.get(kb + (s,))]
        print(f"Nothing to compare: '{na}' missing seeds {missing_a}, '{nb}' missing seeds {missing_b}.")
        print("Run them first (see GUIDE.md), or use --list to see what exists.")
        return

    if DUPLICATES:
        print(f"   NOTE: {len(DUPLICATES)} run(s) exist in two folders; the first is used. "
              f"Details: python tools/experiments.py")

    seeds_used = sorted({s for *_, ss in rows for s in ss})
    partial = [f"{l}({len(ss)})" for _, l, _, _, ss in rows if len(ss) < len(seeds_used)]
    out(f"\n{args.metric}: {nb}  vs  {na}   (lower is better)")
    unusable = [x for x in args.seeds if x not in seeds_used]
    if unusable:
        miss_a = [x for x in unusable if not idx.get(ka + (x,))]
        miss_b = [x for x in unusable if not idx.get(kb + (x,))]
        out(f"   NOTE: you asked for seeds {args.seeds}; only {seeds_used} exist for both settings.")
        out(f"         not run yet: {miss_a or 'none'} for {na}, {miss_b or 'none'} for {nb}")
    out(f"seeds {seeds_used}, {len(rows)}/20 languages" +
          (f"; fewer seeds for: {' '.join(partial)}" if partial else ""))
    if len(rows) < 20:
        out("   INCOMPLETE - the 20-language test below is not the paper's test until all 20 are in.")

    out(f"\n   {'':4s}{'lang':8s}{na[:15]:>15s}  {nb[:15]:>15s}{'change':>10s}   {'better':<15s}")
    for c, l, a, b, ss in rows:
        out(f"   {c:4s}{l:8s}{a:15.4f}  {b:15.4f}{100 * (b / a - 1):+9.0f}%   {(nb if b < a else na):<15s}")

    a = np.array([r[2] for r in rows]); b = np.array([r[3] for r in rows])
    wins = int((b < a).sum())
    p = wilcoxon(b, a, alternative="less").pvalue if np.any(b != a) else float("nan")
    out(f"\n   {nb} better in {wins}/{len(rows)} languages")
    out(f"   mean {a.mean():.4f} -> {b.mean():.4f} ({100 * (b.mean() / a.mean() - 1):+.0f}%)")
    out(f"   one-sided Wilcoxon (is {nb} lower?): p = {p:.4f}"
          f"{'   SIGNIFICANT at 0.05' if p < 0.05 else ''}")
    out(f"   median per-language change {100 * np.median(b / a - 1):+.0f}%, "
          f"worst {100 * (b / a - 1).max():+.0f}% ({rows[int(np.argmax(b / a - 1))][1]}), "
          f"best {100 * (b / a - 1).min():+.0f}% ({rows[int(np.argmin(b / a - 1))][1]})")

    out(f"\n   per corpus:")
    for corpus in LANGS:
        sub = [r for r in rows if r[0] == corpus]
        if not sub:
            continue
        sa = np.array([r[2] for r in sub]); sb = np.array([r[3] for r in sub])
        out(f"      {CORPUS_NAME[corpus]:18s} {int((sb < sa).sum())}/{len(sub)} better | "
              f"mean {sa.mean():.4f} -> {sb.mean():.4f} ({100 * (sb.mean() / sa.mean() - 1):+.0f}%)")

    if len(seeds_used) > 1:
        out(f"\n   per seed ({nb} better in ... of the languages that seed has):")
        for s in seeds_used:
            pairs = [(A[(c, l)][s], B[(c, l)][s]) for c, l in [(r[0], r[1]) for r in rows]
                     if s in A.get((c, l), {}) and s in B.get((c, l), {})]
            if pairs:
                x = np.array([q[0] for q in pairs]); y = np.array([q[1] for q in pairs])
                out(f"      seed {s}: {int((y < x).sum())}/{len(pairs)} | "
                      f"mean {x.mean():.4f} -> {y.mean():.4f} ({100 * (y.mean() / x.mean() - 1):+.0f}%)")
        out("   (per-seed rows are for inspection; the test above is the one to report)")

    def write_csv(path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "lang", na, nb, "pct_change", "seeds"])
            for c, l, x, y, ss in rows:
                w.writerow([c, l, f"{x:.6f}", f"{y:.6f}", f"{100 * (y / x - 1):.1f}",
                            " ".join(map(str, ss))])

    if args.csv:
        write_csv(args.csv)
        print(f"\n   table written to {args.csv}")

    if not args.no_save:
        stem = save_name(na, nb, seeds_used, args.metric)
        folder = os.path.join(ROOT, "runs", "comparisons", CATEGORY[kb[1]], stem)
        if os.path.exists(folder):                 # keep the first one; number the later ones
            n = 1
            while os.path.exists(f"{folder}_{n}"):
                n += 1
            folder = f"{folder}_{n}"
        os.makedirs(folder)
        with open(os.path.join(folder, "report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        write_csv(os.path.join(folder, "table.csv"))
        print(f"\n   saved to {os.path.relpath(folder, ROOT)}/ "
              f"(report.txt, table.csv)")
    print()


if __name__ == "__main__":
    main()
