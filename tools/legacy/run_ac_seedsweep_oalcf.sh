#!/bin/bash
# OAL-CF counterpart to tools/run_ac_seed_sweep.sh -- same 10 pre-built
# test-stream-order seeds (seed{1..10}_order_test/train, bootstrap fixed at
# seed 0, orders already on disk from the OAL sweep), same 3 languages, just
# --sim_type fpstps. Answers: does the CF win/loss pattern seen at seed 0
# hold across seeds, or was it also seed-0-specific like the OAL absolute
# numbers turned out to be?
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
CORPUS="$ROOT/data/AfricanContinentLID"
MAIN="$ROOT/oalcf-main"
LANGS=(bas ha yo)

for SEED in $(seq 1 10); do
  for LANG in "${LANGS[@]}"; do
    echo "=== seed $SEED / $LANG : OAL-CF run ==="
    cd "$MAIN"
    mkdir -p "output/ac_oalcf_seedsweep/seed${SEED}_${LANG}" "checkpoints/ac_oalcf_seedsweep/seed${SEED}_${LANG}"
    python3 run.py \
      --run_name "ac_oalcf_seedsweep/seed${SEED}_${LANG}" \
      --env_name test --lid_target "$LANG" \
      --feat_root "$CORPUS/ecapalang/" \
      --order_file "seed${SEED}" \
      --sim_type fpstps \
      --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
      > "$ROOT/runs/ac_oalcf_seedsweep_seed${SEED}_${LANG}.log" 2>&1
    echo "    done: $(python3 scoring/print_basic_results.py output/ac_oalcf_seedsweep/seed${SEED}_${LANG} 2>/dev/null | tail -1)"
  done
done

echo "=== oalcf seed sweep complete ==="
