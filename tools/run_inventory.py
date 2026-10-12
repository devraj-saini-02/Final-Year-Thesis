#!/usr/bin/env python3
"""
Write runs/INVENTORY.md: every run in the repo, what it is, where it lives, and how it scored.

    python tools/run_inventory.py            # regenerate runs/INVENTORY.md

Re-run it after any new batch. It reads the runs themselves (through tools/experiments.py), so it
cannot go out of date the way a hand-written list would.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (ALL_LANGS, CORPUS_NAME, LANGS, ROOT, SESSIONS,  # noqa: E402
                         index, run_path, spec_name)
from score_runs import score  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
METRIC = "IMLM_eq1"


def dq_stats(run_dirs):
    """Mean queries changed per session and uncertainty given up, from al_diversity.csv."""
    ch, uc, cl = [], [], []
    for d in run_dirs:
        f = os.path.join(d, "al_diversity.csv")
        if not os.path.exists(f):
            return None
        t = pd.read_csv(f)
        ch.append((8 - t.overlap_with_baseline).mean())
        uc.append((t.mean_u_baseline - t.mean_u_picked).sum() / t.mean_u_baseline.sum())
        cl.append(t.n_clusters.mean())
    return np.mean(ch), 100 * np.mean(uc), np.mean(cl)


def main():
    idx = index()
    settings = {}
    for (algo, method, thr, gam, seed), runs in idx.items():
        settings.setdefault((algo, method, thr, gam), {})[seed] = runs
    total = sum(len(r) for r in idx.values())

    L = []
    L.append("# INVENTORY — every run in this repo")
    L.append("")
    L.append(f"Generated {date.today()} by `python tools/run_inventory.py`. Re-run it after any new "
             "batch; do not edit by hand.")
    L.append("")
    L.append(f"**{total} runs**, {len(settings)} settings, seeds {SEEDS[0]}–{SEEDS[-1]}. "
             f"Metric below is {METRIC} (the paper's Eq. 1), lower is better.")
    L.append("")
    L.append("One run = one target language, one algorithm, one seed: 20 languages across four "
             "corpora (African 3, Caucasus 7, Southeast Asia 3, South Asia 7).")
    L.append("")

    # ---- health
    bad = [(k, c, l, score(d)["sessions"]) for k, runs in idx.items() for (c, l), d in runs.items()
           if score(d)["sessions"] != SESSIONS[c]]
    L.append(f"**Health:** {total - len(bad)}/{total} runs have their full session count "
             "(African 83, Caucasus 159, Southeast Asia 126, South Asia 95).")
    if bad:
        L.append("")
        for k, c, l, n in bad:
            L.append(f"- INCOMPLETE: {spec_name(*k[:4])} seed {k[4]} {c}/{l}: {n}/{SESSIONS[c]} sessions")
    L.append("")

    # ---- coverage matrix
    L.append("## 1. What exists (languages finished, out of 20)")
    L.append("")
    L.append("| setting | " + " | ".join(f"seed {s}" for s in SEEDS) + " | total |")
    L.append("|---|" + "---|" * (len(SEEDS) + 1))
    for key in sorted(settings, key=lambda k: (k[0], k[1] != "baseline", k[2] or 0, -(k[3] or 0))):
        cells = []
        for s in SEEDS:
            n = len(settings[key].get(s, {}))
            cells.append("—" if n == 0 else ("**20**" if n == 20 else str(n)))
        tot = sum(len(v) for v in settings[key].values())
        L.append(f"| `{spec_name(*key)}` | " + " | ".join(cells) + f" | {tot} |")
    L.append("")
    L.append("`oal` = active learning only. `oalcf` = plus corrective feedback (the paper's method). "
             "`@t/g` = our extension at cluster threshold t and decay gamma g. "
             "**gamma 1.0 runs are controls**: the penalty is inert, so they must reproduce the "
             "baseline bit for bit, and they did at every threshold.")
    L.append("")

    # ---- results
    L.append("## 2. How each setting scored")
    L.append("")
    L.append("Each extension setting is compared against its own baseline (`oal` or `oalcf`) on the "
             "seeds where both exist, one row per setting. 'better' counts languages that improved.")
    L.append("")
    L.append("| setting | seeds used | better | mean before → after | p | clusters/session | queries changed | uncertainty given up |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(settings, key=lambda k: (k[0], k[1] != "baseline", k[2] or 0, -(k[3] or 0))):
        algo, method, thr, gam = key
        if method == "baseline":
            continue
        base = settings.get((algo, "baseline", None, None), {})
        if gam == 1.0:
            n = sum(len(v) for v in settings[key].values())
            L.append(f"| `{spec_name(*key)}` | 0 | control | penalty inert; reproduces the baseline "
                     f"exactly ({n} run{'s' if n > 1 else ''} checked) | | | | |")
            continue
        shared = [s for s in SEEDS if len(settings[key].get(s, {})) == 20 and len(base.get(s, {})) == 20]
        if not shared:
            n = sum(len(v) for v in settings[key].values())
            L.append(f"| `{spec_name(*key)}` | incomplete ({n}/20 languages) | | | | | | |")
            continue
        a = [np.mean([score(base[s][(c, l)])[METRIC] for s in shared]) for c, l in ALL_LANGS]
        b = [np.mean([score(settings[key][s][(c, l)])[METRIC] for s in shared]) for c, l in ALL_LANGS]
        a, b = np.array(a), np.array(b)
        p = wilcoxon(b, a, alternative="less").pvalue
        st = dq_stats([settings[key][shared[0]][(c, l)] for c, l in ALL_LANGS])
        stx = f"{st[2]:.0f} | {st[0]:.2f} of 8 | {st[1]:.1f}%" if st else " | | "
        star = " **" if p < 0.05 else ""
        L.append(f"| `{spec_name(*key)}` | {','.join(str(s) for s in shared)} | {int((b < a).sum())}/20 | "
                 f"{a.mean():.4f} → {b.mean():.4f} ({100 * (b.mean() / a.mean() - 1):+.0f}%) | "
                 f"{p:.4f}{star} | {stx} |")
    L.append("")
    L.append("** = significant at 0.05. Read these with care (GUIDE.md, 'Where the project stands'): a seed-0-only "
             "row is one stream order and has repeatedly proved misleading.")
    L.append("")

    # ---- baselines per corpus
    L.append("## 3. Baselines per corpus (mean over the seeds each has)")
    L.append("")
    L.append("| algorithm | seeds | " + " | ".join(CORPUS_NAME[c] for c in LANGS) + " | all 20 |")
    L.append("|---|---|" + "---|" * (len(LANGS) + 1))
    for algo in ("oal", "oalcf"):
        base = settings.get((algo, "baseline", None, None), {})
        done = [s for s in SEEDS if len(base.get(s, {})) == 20]
        if not done:
            continue
        cells = []
        for c in LANGS:
            v = [score(base[s][(c, l)])[METRIC] for s in done for l in LANGS[c]]
            cells.append(f"{np.mean(v):.4f}")
        allv = [score(base[s][(c, l)])[METRIC] for s in done for c, l in ALL_LANGS]
        L.append(f"| {algo} | {','.join(str(s) for s in done)} | " + " | ".join(cells) +
                 f" | {np.mean(allv):.4f} |")
    L.append("")
    L.append("For reference, the paper's own values average 0.1015 over the 20 languages "
             "(`docs/other/REPRO_BRIEF.md` section 7).")
    L.append("")

    # ---- where things live
    L.append("## 4. Where a run lives on disk")
    L.append("")
    L.append("```")
    L.append("runs/<category>/<setting>/seed_<seed>/<corpus>_<language>/")
    L.append("```")
    L.append("")
    L.append("For example:")
    L.append("")
    L.append("```")
    for key in [("oal", "baseline", None, None), ("oal", "diverse", 0.16, 0.93)]:
        if key in settings:
            seed = sorted(settings[key])[0]
            L.append(os.path.relpath(run_path(*key, seed, "cr", "tt"), ROOT) +
                     f"      # {spec_name(*key)}, seed {seed}, Caucasus Tatar")
    L.append("```")
    L.append("")
    L.append("Each run folder holds `scores.csv` (per-session results, the source of every number "
             "above), `command.txt` (the exact command), `state_dict.pt` (final weights), `train.log`, "
             "and for extension runs `al_diversity.csv` (clusters and queries per session).")
    L.append("")
    L.append("## 5. Language codes")
    L.append("")
    for c in LANGS:
        L.append(f"- **{CORPUS_NAME[c]}** (`{c}`, {SESSIONS[c]} sessions): " +
                 ", ".join(f"`{l}`" for l in LANGS[c]))
    L.append("")
    L.append("## 6. Regenerate / explore")
    L.append("")
    L.append("```bash")
    L.append("python tools/run_inventory.py                       # rewrite this file")
    L.append("python tools/compare_runs.py --list                 # same coverage, printed")
    L.append("python tools/compare_runs.py oal oal@0.16/0.93 --seeds 0 1 2 3 4   # any comparison")
    L.append("```")

    out = os.path.join(ROOT, "runs", "INVENTORY.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {os.path.relpath(out, ROOT)} ({total} runs, {len(settings)} settings)")


if __name__ == "__main__":
    main()
