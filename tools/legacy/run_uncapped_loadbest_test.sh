#!/bin/bash
# PAPER-FAITHFUL test: CF left UNCAPPED (paper III-B: "all positive predictions
# in the session get labels" -- the paper states no limit), on the two languages
# with the smallest uncapped OAL-CF losses.
#
# The single variable is --load_best, which the paper never specifies: with it,
# each session carries forward the best checkpoint instead of the last epoch's
# weights. This is the most plausible paper-compatible brake on the runaway
# post-CF fit that drives the collapse.
#
# Controls (already on disk, uncapped, no load_best):
#   sa/cnh  OAL 0.0172 -> CF 0.0286 (+66%)
#   cr/cv   OAL 0.2157 -> CF 0.4968 (+130%)
set -euo pipefail
ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
cd "$MAIN"

run_one () {
  local short="$1" corpus="$2" lang="$3"
  echo "=== ${short}/${lang} : OAL-CF UNCAPPED + --load_best ==="
  mkdir -p "output/uncap_lb_${short}/${lang}" "checkpoints/uncap_lb_${short}/${lang}"
  python3 run.py \
    --run_name "uncap_lb_${short}/${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --sim_type fpstps --load_best \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/uncap_lb_${short}_${lang}.log" 2>&1
  echo "    done: $(python3 scoring/print_basic_results.py "output/uncap_lb_${short}/${lang}" 2>/dev/null | tail -1)"
}

run_one "sa" "SouthAsiaLID" "cnh"
run_one "cr" "CaucasusRegionLID" "cv"
echo "=== uncapped + load_best test complete ==="
