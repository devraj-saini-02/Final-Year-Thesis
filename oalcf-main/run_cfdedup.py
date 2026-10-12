"""
Entry point for the corrective-feedback deduplication experiment (utils/cf_dedup.py).

Takes every argument run.py takes (it needs --sim_type, e.g. fpstps), plus:
  --cfd_threshold   cosine distance threshold for grouping the CF clips (default 0.10);
                    0 puts every clip in its own group == baseline OAL-CF (the control)
  --cfd_rep         which group member the expert labels: confident (default), uncertain, medoid

run.py, utils/corrective_feedback.py and src/dataset.py are NOT modified.  The simulator and
the data module are swapped in by rebinding the names run.py imported, before run.main().
Per-session expert-label counts go to output/<run_name>/cf_dedup.csv.
"""
import sys
import os
import argparse


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--cfd_threshold', type=float, default=0.10)
    ap.add_argument('--cfd_rep', type=str, default='confident',
                    choices=['confident', 'uncertain', 'medoid'])
    cfd_args, rest = ap.parse_known_args()

    # hide our arguments from run.py's own parser, which uses parse_args()
    sys.argv = [sys.argv[0]] + rest

    if not (0.0 <= cfd_args.cfd_threshold <= 2.0):
        raise SystemExit('--cfd_threshold must be in [0, 2] (cosine distance)')
    if '--sim_type' not in rest:
        raise SystemExit('run_cfdedup.py needs corrective feedback: pass --sim_type fpstps')

    import run
    from utils.cf_dedup import DedupFeedbackSimulator, DedupDataModule

    DedupFeedbackSimulator.cfd_threshold = cfd_args.cfd_threshold
    DedupFeedbackSimulator.cfd_rep = cfd_args.cfd_rep
    if '--run_name' in sys.argv:
        run_name = sys.argv[sys.argv.index('--run_name') + 1]
        DedupFeedbackSimulator.log_path = os.path.join('output', run_name, 'cf_dedup.csv')

    print(f'[run_cfdedup] threshold={cfd_args.cfd_threshold} rep={cfd_args.cfd_rep}')

    run.FeedbackSimulator = DedupFeedbackSimulator
    run.ImlDataModule = DedupDataModule
    run.main()


if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    print(time.time() - t0)
