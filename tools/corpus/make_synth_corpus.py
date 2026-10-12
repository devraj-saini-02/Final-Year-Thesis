#!/usr/bin/env python3
"""
Build a synthetic, AC-SLV-shaped corpus so the OAL / OAL-CF pipeline can be
exercised end to end before any Common Voice data exists.

This verifies PLUMBING, NOT PERFORMANCE. The features are Gaussian blobs, not
ECAPA embeddings. Numbers produced from this corpus say nothing about whether
the paper reproduces. See the "What this does and does not prove" section at
the bottom.

Layout produced (matches src/dataset.py:278-288 and utils/org_cluster.py:12-21):

    <root>/                                  # name must contain "LID", no underscores
      ecapalang/train/common_voice_<loc>_<id>.npy
      ecapalang/dev/  common_voice_<loc>_<id>.npy
      ecapalang/test/ common_voice_<loc>_<id>.npy
      orders/                                # left empty; org_cluster.py fills it

Usage:
    python make_synth_corpus.py --root ./synthLID
    cd synthLID && python <repo>/utils/org_cluster.py     # after the ecapalang repoint
"""

import argparse
import os
import numpy as np

# The three AC targets, at their Table II test-split prevalences.
TARGETS = {"bas": 0.0095, "ha": 0.0116, "yo": 0.0175}

# Filler locales. Real CV locale codes from the African set, used here only so
# the file names and the Dirichlet chunk count look realistic. No claim is made
# that these are the paper's AC locale list -- that is still unknown.
FILLERS = ["sw", "rw", "lg", "wo", "ig", "am", "ff", "nso", "zu"]

FEAT_DIM = 256  # ECAPA lang-id embedding dimension (paper V-B)


def build_split(rng, n_sessions, samples_per_session, target_prev, centers, noise, out_dir):
    n = n_sessions * samples_per_session
    locales = []
    for loc, p in target_prev.items():
        locales += [loc] * int(round(p * n))
    n_filler = n - len(locales)
    if n_filler <= 0:
        raise ValueError(f"target prevalences exceed split size for {out_dir}")
    locales += list(rng.choice(FILLERS, size=n_filler))
    rng.shuffle(locales)

    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for i, loc in enumerate(locales):
        counts[loc] = counts.get(loc, 0) + 1
        feat = centers[loc] + rng.normal(0.0, noise, size=FEAT_DIM)
        np.save(os.path.join(out_dir, f"common_voice_{loc}_{i:07d}.npy"),
                feat.astype(np.float32))
    return n, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./synthLID",
                    help='corpus root; must contain "LID" and no underscores')
    ap.add_argument("--test-sessions", type=int, default=10,
                    help="sessions in the test split (paper AC has 79)")
    ap.add_argument("--train-sessions", type=int, default=6,
                    help="sessions in the train split; only the head is used, "
                         "for the bootstrap corpus")
    ap.add_argument("--dev-sessions", type=int, default=1)
    ap.add_argument("--samples-per-session", type=int, default=720)
    ap.add_argument("--train-target-prev", type=float, default=0.02,
                    help="target prevalence in train; higher than test so the "
                         "bootstrap scan terminates early. Raise if "
                         "load_external_bootstrap prepends too many sessions.")
    ap.add_argument("--separation", type=float, default=1.2,
                    help="distance between class centres. Lower = harder task. "
                         "Set well below 1.0 to check the pipeline still runs "
                         "when the classifier cannot learn.")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if "LID" not in os.path.basename(root):
        raise SystemExit('root basename must contain "LID" -- src/dataset.py:20 '
                         "routes the SLV loader on that substring")
    if "_" in os.path.basename(root):
        raise SystemExit("root basename must not contain '_' -- the locale is "
                         "parsed as split('_')[2] of the path")

    rng = np.random.default_rng(args.seed)

    # One centre per locale, so the task is learnable but not trivial.
    all_locales = list(TARGETS) + FILLERS
    centers = {loc: rng.normal(0.0, 1.0, size=FEAT_DIM) * args.separation
               for loc in all_locales}

    feat_root = os.path.join(root, "ecapalang")
    os.makedirs(os.path.join(root, "orders"), exist_ok=True)

    train_prev = {k: args.train_target_prev for k in TARGETS}
    plan = [
        ("train", args.train_sessions, train_prev),
        ("dev",   args.dev_sessions,   TARGETS),
        ("test",  args.test_sessions,  TARGETS),
    ]

    for split, n_sessions, prev in plan:
        n, counts = build_split(rng, n_sessions, args.samples_per_session, prev,
                                centers, args.noise,
                                os.path.join(feat_root, split))
        line = ", ".join(f"{loc} {counts.get(loc, 0)} "
                         f"({100 * counts.get(loc, 0) / n:.2f}%)"
                         for loc in TARGETS)
        print(f"{split:5s} {n:6d} clips  {n_sessions:3d} sessions   {line}")

    print(f"\nwrote {root}")
    print("next:")
    print(f"  cd {root} && python <repo>/utils/org_cluster.py")
    print("  (org_cluster.py must be repointed from wavlm/ to ecapalang/ first)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# What this does and does not prove
#
# Does prove, if a run completes:
#   - the MPS device path works end to end (fit, test, AL scoring, CF, ddm)
#   - the corpus layout, order-file format and locale parse are correct
#   - bootstrap-from-train and the n_boot_batches session offset work
#   - IMLM / DCF / FNR / FPR are computed and written to scores.csv
#   - a full 79-session run's wall-clock cost can be extrapolated
#
# Does NOT prove:
#   - anything about IMLM values, OAL vs OAL-CF ordering, or the budget sweep
#   - that the AC corpus is built correctly
#   - that ECAPA features behave like these blobs
#
# Gaussian blobs with well-separated centres are a far easier problem than real
# spoken language verification. Expect error rates far below the paper's and
# treat any comparison between the two as meaningless.
# ---------------------------------------------------------------------------
