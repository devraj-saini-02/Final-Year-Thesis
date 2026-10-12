#!/bin/bash
# 10 reshuffled test-stream-order seeds for AC OAL (bas/ha/yo), investigating
# whether the bas/yo FNR coincidence (0.3880 vs 0.3992) is a seed-0 artifact
# of where the Dirichlet ordering happens to place each language's burst.
#
# Only the TEST order is reshuffled per seed. The bootstrap (train-derived,
# seed-0) is left untouched and copied as-is for every seed -- bootstrap
# composition is a separate initialization step independent of test stream
# order, and varying it would require downloading new train-split CV shards
# (confirmed: the seed-0 bootstrap's shared nontarget anchor landed in `kab`,
# a locale requiring ~1GB shards; a new seed's anchor could land in `rw`,
# whose shards are several GB each). Keeping train order fixed at seed 0
# means this sweep needs zero new downloads.
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
CORPUS="$ROOT/data/AfricanContinentLID"
MAIN="$ROOT/oalcf-main"
LANGS=(bas ha yo)

for SEED in $(seq 1 10); do
  echo "=== seed $SEED: building test order ==="
  cd "$CORPUS"
  python3 "$MAIN/utils/org_cluster.py" --seed "$SEED" --splits test
  cp orders/cluster_order_test "orders/seed${SEED}_order_test"
  cp orders/cluster_order_train "orders/seed${SEED}_order_train"
  # org_cluster.py writes the default cluster_order_test in place, so the last
  # sweep seed would otherwise be left masquerading as the seed-0 baseline.
  # Restore seed 0 before exiting this iteration.
  python3 "$MAIN/utils/org_cluster.py" --seed 0 --splits test

  for LANG in "${LANGS[@]}"; do
    echo "=== seed $SEED / $LANG : OAL run ==="
    cd "$MAIN"
    mkdir -p "output/ac_oal_seedsweep/seed${SEED}_${LANG}" "checkpoints/ac_oal_seedsweep/seed${SEED}_${LANG}"
    python3 run.py \
      --run_name "ac_oal_seedsweep/seed${SEED}_${LANG}" \
      --env_name test --lid_target "$LANG" \
      --feat_root "$CORPUS/ecapalang/" \
      --order_file "seed${SEED}" \
      --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
      > "$ROOT/runs/ac_oal_seedsweep_seed${SEED}_${LANG}.log" 2>&1
    echo "    done: $(python3 scoring/print_basic_results.py output/ac_oal_seedsweep/seed${SEED}_${LANG} 2>/dev/null | tail -1)"
  done
done

echo "=== seed sweep complete ==="
