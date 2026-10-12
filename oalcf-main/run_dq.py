"""
Entry point for the cluster-quota diversity-penalized AL experiment.

Takes every argument run.py takes, plus:
  --dq_gamma         discount strength in [0,1].  1.0 == baseline OAL.
  --dq_threshold     agglomerative cosine distance_threshold (default 0.25).
                     Sets how many clusters the data supports; no fixed count.
  --dq_cluster_feat  'center' (the sample's own ECAPA embedding) or 'full'
                     (the whole +/-3 context window)

run.py and utils/query_strategies.py are NOT modified.  The strategy object is
swapped in by rebinding the name run.py imported, before run.main() is called.
"""
import sys
import os
import argparse


def main():
    dq = argparse.ArgumentParser(add_help=False)
    dq.add_argument('--dq_gamma', type=float, default=1.0)
    dq.add_argument('--dq_threshold', type=float, default=0.25)
    dq.add_argument('--dq_cluster_feat', type=str, default='center',
                    choices=['center', 'full'])
    dq_args, rest = dq.parse_known_args()

    # hide our arguments from run.py's own parser, which uses parse_args()
    sys.argv = [sys.argv[0]] + rest

    if not (0.0 <= dq_args.dq_gamma <= 1.0):
        raise SystemExit('--dq_gamma must be in [0, 1]')

    import run
    from utils.diverse_query import DiverseStrategyManager

    DiverseStrategyManager.dq_gamma = dq_args.dq_gamma
    DiverseStrategyManager.dq_threshold = dq_args.dq_threshold
    DiverseStrategyManager.dq_cluster_feat = dq_args.dq_cluster_feat

    run_name = None
    for ii, tok in enumerate(sys.argv):
        if tok == '--run_name' and ii + 1 < len(sys.argv):
            run_name = sys.argv[ii + 1]
    if run_name is not None:
        DiverseStrategyManager.dq_log_path = os.path.join(
            'output', run_name, 'al_diversity.csv')

    print(f'[run_dq] gamma={dq_args.dq_gamma} threshold={dq_args.dq_threshold} '
          f'cluster_feat={dq_args.dq_cluster_feat}')

    run.StrategyManager = DiverseStrategyManager
    run.main()


if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    print(time.time() - t0)
