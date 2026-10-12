#!/usr/bin/env python3
"""
One index of every finished run in this repo, old layout or new.

A run is identified by five things:
    algo      'oal' or 'oalcf'                 (oalcf = OAL plus corrective feedback)
    method    'baseline', 'diverse' or 'cfdedup'
              (diverse = the cluster-quota AL extension, run_dq.py;
               cfdedup = the corrective-feedback deduplication extension, run_cfdedup.py)
    threshold cluster distance threshold       (None for baseline)
    gamma     penalty strength                 (None for baseline; 1.0 = control, same as baseline)
    seed      stream-order seed, 0-4

Other modules import `index()` and `parse_spec()`; `python tools/experiments.py` prints
everything that exists. Nothing here runs a model.

Spec strings, as typed on the command line by tools/compare_runs.py:
    oal                 baseline OAL
    oalcf               baseline OAL-CF
    oal@0.16/0.93       OAL + extension, threshold 0.16, gamma 0.93
    oalcf@0.25/0.8      OAL-CF + extension, threshold 0.25, gamma 0.8
    oalcf+cfd0.1        OAL-CF + CF deduplication (run_cfdedup.py), grouping threshold 0.1
                        (method 'cfdedup', gamma None)

Where a run lives (since 2026-10-10):  runs/<category>/<setting>/seed_<N>/<corpus>_<lang>/
    category  baseline/            the paper's methods as published      setting: oal, oalcf
              diversity_extension/ our cluster-quota AL extension        setting: <algo>_t<T>_g<G>
              cf_dedup/            our CF deduplication extension        setting: <algo>_cfd<T>
`run_path()` / `seed_dir()` are the one place that knows this; everything else asks them.
`python tools/experiments.py --dir SPEC SEED` prints a seed folder (used by run_experiment.sh).
"""
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANGS = {"ac": ["bas", "ha", "yo"],
         "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
         "sea": ["yue", "id", "vi"],
         "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESSIONS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
ALL_LANGS = [(c, l) for c in LANGS for l in LANGS[c]]
CORPUS_NAME = {"ac": "African Continent", "cr": "Caucasus Region",
               "sea": "Southeast Asia", "sa": "South Asia"}

# Where finished runs live.
#   current layout: runs/<category>/<setting>/seed_<N>/<corpus>_<lang>   <- everything lives here
#                   (see the module docstring; tools/migrate_runs_by_method.py, 2026-10-10)
#   older layouts:  runs/seed_<N>/<setting>/<corpus>_<lang>   setting oal_baseline, <algo>_t<T>_g<G>, ...
#                   runs/experiments/<exp_id>/output/<corpus>_<lang>
#                   runs/<batch>/output/<corpus>_<arm><suffix>/<lang>
#   The old ones are still read so that an archived or restored folder keeps working
#   (tools/migrate_runs_layout.py moved everything to the current layout on 2026-09-18).
_OLD_ROOTS = ["runs/baseline_seed0/output", "runs/baseline_seeds/output", "runs/dq_oal/output",
              "runs/dq_oalcf/output", "runs/dq_oal_thresh/output", "runs/dq_oalcf_thresh/output"]


def _gamma(tok):
    """Old folder names spell gamma without a dot: g93 -> 0.93, g975 -> 0.975, g100 -> 1.0."""
    if tok == "100":
        return 1.0
    return round(float(tok) / (100 if len(tok) == 2 else 1000), 3)


def _thresh(tok):
    """t16 -> 0.16, t18 -> 0.18."""
    return round(float(tok) / 100, 2)


def _parse_old(folder):
    """Read an old-layout folder name. Returns (algo, method, threshold, gamma, seed) or None."""
    m = re.fullmatch(r"(ac|cr|sea|sa)_(oal|oalcf)(_s(\d+)|_dq(\d+)|_t(\d+)_g(\d+))?", folder)
    if not m:
        return None
    corpus, algo, _, seed, dq, thr, gam = m.groups()
    if dq:                      # threshold 0.25 batch, seed 0
        return corpus, algo, "diverse", 0.25, round(float(dq) / 100, 2), 0   # dq080 -> 0.80
    if thr:                     # tighter-threshold batch, seed 0
        return corpus, algo, "diverse", _thresh(thr), _gamma(gam), 0
    return corpus, algo, "baseline", None, None, int(seed or 0)


CATEGORY = {"baseline": "baseline", "diverse": "diversity_extension", "cfdedup": "cf_dedup"}


def setting_name(algo, method, threshold, gamma):
    """Folder name of a setting inside its category: oal | oal_t0.16_g0.93 | oalcf_cfd0.1."""
    if method == "baseline":
        return algo
    if method == "cfdedup":
        return f"{algo}_cfd{threshold:g}"
    return f"{algo}_t{threshold:g}_g{gamma:g}"


def seed_dir(algo, method, threshold, gamma, seed):
    """runs/<category>/<setting>/seed_<N>: one batch's folder (holds progress.log and the runs)."""
    return os.path.join(ROOT, "runs", CATEGORY[method], setting_name(algo, method, threshold, gamma),
                        f"seed_{seed}")


def parse_category_setting(category, name):
    """Inverse of setting_name within a category folder; None for anything else."""
    if category == "baseline" and name in ("oal", "oalcf"):
        return name, "baseline", None, None
    parsed = parse_setting(name)
    if parsed and CATEGORY[parsed[1]] == category:
        return parsed
    return None


DUPLICATES = []   # (setting key, corpus, lang, kept dir, ignored dir); filled by index()


def parse_setting(name):
    """'oal_baseline' -> (oal, baseline, None, None); 'oal_t0.16_g0.93' -> (oal, diverse, .16, .93)."""
    m = re.fullmatch(r"(oal|oalcf)_baseline", name)
    if m:
        return m.group(1), "baseline", None, None
    m = re.fullmatch(r"(oal|oalcf)_t([\d.]+)_g([\d.]+)", name)
    if m:
        return m.group(1), "diverse", float(m.group(2)), float(m.group(3))
    m = re.fullmatch(r"(oalcf)_cfd([\d.]+)", name)
    if m:
        return m.group(1), "cfdedup", float(m.group(2)), None
    return None


def index():
    """{(algo, method, threshold, gamma, seed): {(corpus, lang): run_dir}} for finished runs.

    If the same setting/seed/language exists in two places (e.g. an old batch folder and a new
    runs/experiments one), the first found wins and the clash is recorded in DUPLICATES.
    """
    DUPLICATES.clear()
    out = defaultdict(dict)

    def add(key, corpus, lang, d):
        if (corpus, lang) in out[key]:
            DUPLICATES.append((key, corpus, lang, out[key][(corpus, lang)], d))
        else:
            out[key][(corpus, lang)] = d

    RUNS = os.path.join(ROOT, "runs")
    for category in sorted(set(CATEGORY.values())):            # current layout
        cdir = os.path.join(RUNS, category)
        for name in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
            parsed = parse_category_setting(category, name)
            if not parsed:
                continue
            for sd in sorted(os.listdir(os.path.join(cdir, name))):
                m = re.fullmatch(r"seed_(\d+)", sd)
                if not m:
                    continue
                for corpus, lang in ALL_LANGS:
                    d = os.path.join(cdir, name, sd, f"{corpus}_{lang}")
                    if os.path.exists(os.path.join(d, "scores.csv")):
                        add(parsed + (int(m.group(1)),), corpus, lang, d)
    for sdir in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:   # seed-first layout (before 2026-10-10)
        m = re.fullmatch(r"seed_(\d+)", sdir)
        if not m:
            continue
        seed = int(m.group(1))
        base = os.path.join(RUNS, sdir)
        for setting in sorted(os.listdir(base)):
            parsed = parse_setting(setting)
            if not parsed:
                continue
            for corpus, lang in ALL_LANGS:
                d = os.path.join(base, setting, f"{corpus}_{lang}")
                if os.path.exists(os.path.join(d, "scores.csv")):
                    add(parsed + (seed,), corpus, lang, d)
    for rel in _OLD_ROOTS:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        for folder in sorted(os.listdir(base)):
            parsed = _parse_old(folder)
            if not parsed:
                continue
            corpus, algo, method, thr, gam, seed = parsed
            for lang in LANGS[corpus]:
                d = os.path.join(base, folder, lang)
                if os.path.exists(os.path.join(d, "scores.csv")):
                    add((algo, method, thr, gam, seed), corpus, lang, d)
    exp = os.path.join(ROOT, "runs/experiments")
    if os.path.isdir(exp):
        for exp_id in sorted(os.listdir(exp)):
            key = parse_exp_id(exp_id)
            if not key:
                continue
            base = os.path.join(exp, exp_id, "output")
            for corpus, lang in ALL_LANGS:
                d = os.path.join(base, f"{corpus}_{lang}")
                if os.path.exists(os.path.join(d, "scores.csv")):
                    add(key, corpus, lang, d)
    return dict(out)


def run_path(algo, method, threshold, gamma, seed, corpus, lang):
    """Where a run lives: runs/<category>/<setting>/seed_<N>/<corpus>_<lang>."""
    return os.path.join(seed_dir(algo, method, threshold, gamma, seed), f"{corpus}_{lang}")


def parse_exp_id(name):
    m = re.fullmatch(r"(oal|oalcf)_baseline_s(\d+)", name)
    if m:
        return m.group(1), "baseline", None, None, int(m.group(2))
    m = re.fullmatch(r"(oal|oalcf)_t([\d.]+)_g([\d.]+)_s(\d+)", name)
    if m:
        return m.group(1), "diverse", float(m.group(2)), float(m.group(3)), int(m.group(4))
    return None


def parse_spec(spec):
    """'oal' / 'oalcf@0.16/0.93' -> (algo, method, threshold, gamma). Raises ValueError."""
    spec = spec.strip()
    m = re.fullmatch(r"(oalcf)\+cfd([\d.]+)", spec)
    if m:
        return m.group(1), "cfdedup", float(m.group(2)), None
    m = re.fullmatch(r"(oal|oalcf)(?:@([\d.]+)/([\d.]+))?", spec)
    if not m:
        raise ValueError(f"bad spec '{spec}'. Use: oal | oalcf | oal@0.16/0.93 | oalcf@0.25/0.8 | oalcf+cfd0.1")
    algo, thr, gam = m.groups()
    if thr is None:
        return algo, "baseline", None, None
    return algo, "diverse", float(thr), float(gam)


def spec_name(algo, method, threshold, gamma):
    if method == "baseline":
        return algo
    if method == "cfdedup":
        return f"{algo}+cfd{threshold:g}"
    return f"{algo}@{threshold:g}/{gamma:g}"


def find(algo, method, threshold, gamma, seed, idx=None):
    """Runs for one setting and seed, or {} if that combination was never run."""
    idx = index() if idx is None else idx
    return idx.get((algo, method, threshold, gamma, seed), {})


def main():
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--dir":
        # used by tools/run_experiment.sh: the folder a setting's batch for one seed writes to
        print(seed_dir(*parse_spec(sys.argv[2]), int(sys.argv[3])))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--have":
        # used by tools/run_experiment.sh: which languages already exist for this setting and seed
        algo, method, thr, gam = parse_spec(sys.argv[2])
        for corpus, lang in sorted(find(algo, method, thr, gam, int(sys.argv[3]))):
            print(f"{corpus} {lang}")
        return
    idx = index()
    groups = defaultdict(list)
    for (algo, method, thr, gam, seed), runs in idx.items():
        groups[(algo, method, thr, gam)].append((seed, len(runs)))
    print(f"{'spec':22s}{'seeds (languages done)':40s}")
    print("-" * 62)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0)):
        seeds = ", ".join(f"s{s}: {n}/20" for s, n in sorted(groups[key]))
        print(f"{spec_name(*key):22s}{seeds}")
    print(f"\n{len(idx)} setting/seed combinations, "
          f"{sum(len(v) for v in idx.values())} finished runs.")
    if DUPLICATES:
        print(f"\nWARNING: {len(DUPLICATES)} run(s) exist in two places; the first is used:")
        for key, c, l, kept, ignored in DUPLICATES[:10]:
            print(f"   {spec_name(*key[:4])} seed {key[4]} {c}/{l}\n      using  {kept}\n      ignoring {ignored}")
    print("Compare any two:  python tools/compare_runs.py <spec A> <spec B> --seeds 0")


if __name__ == "__main__":
    main()
