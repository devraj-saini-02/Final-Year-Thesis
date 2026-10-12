#!/bin/bash
# Test whether --max_fb_samples 16 (the dissertation's actual baseline CF cap,
# section 4.4 -- 8 TP + 8 FP balanced per session) resolves the classifier-
# collapse pattern found in the uncapped CR/SEA/SA OAL-CF runs. One
# representative collapsed language per corpus, all otherwise identical to
# the original baseline command.
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
cd "$MAIN"

run_one () {
  local short="$1" corpus="$2" lang="$3"
  echo "=== ${short}/${lang} : OAL-CF, max_fb_samples=16 ==="
  mkdir -p "output/cftest_${short}/${lang}" "checkpoints/cftest_${short}/${lang}"
  python3 run.py \
    --run_name "cftest_${short}/${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --sim_type fpstps --max_fb_samples 16 \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/cftest_${short}_${lang}.log" 2>&1
  echo "    done: $(python3 scoring/print_basic_results.py "output/cftest_${short}/${lang}" 2>/dev/null | tail -1)"
}

run_one "cr" "CaucasusRegionLID" "hy-AM"
run_one "sa" "SouthAsiaLID" "dv"
run_one "sea" "SoutheastAsiaLID" "id"

echo "=== CF cap test complete ==="
