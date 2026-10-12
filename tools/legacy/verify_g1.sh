#!/bin/bash
# Full-length gamma=1.0 verification: run_dq.py must reproduce the untouched
# baseline OAL runs bit-for-bit on languages whose stream order is intact.
set -uo pipefail
ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"; PY="$ROOT/venv/bin/python"; cd "$MAIN"
mkdir -p output/dqverify checkpoints/dqverify
PAIRS=("sa:SouthAsiaLID:cnh:sa_oal_precap" "sea:SoutheastAsiaLID:id:sea_oal_precap" "cr:CaucasusRegionLID:hy-AM:cr_oal_precap")
fail=0
for entry in "${PAIRS[@]}"; do
  IFS=':' read -r short corpus lang arch <<< "$entry"
  echo "=== $(date '+%H:%M:%S') gamma=1.0 full run: ${short}/${lang} ==="
  "$PY" run_dq.py --dq_gamma 1.0 --run_name "dqverify/${short}_${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/dqverify_${short}_${lang}.log" 2>&1
  if diff "output/dqverify/${short}_${lang}/scores.csv" "output/${arch}/${lang}/scores.csv" > /dev/null; then
    echo "    BIT-IDENTICAL to output/${arch}/${lang}"
  else
    echo "    *** DIFFERS from output/${arch}/${lang} ***"; fail=1
  fi
done
echo "=== verification $([ $fail -eq 0 ] && echo PASSED || echo FAILED) ==="
