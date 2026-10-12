#!/bin/bash
# Diagnostic: does the dissertation's "Informative CF" (§5.4.1) flip the two
# languages that still show CF losing after the L=16 cap was applied?
# Informative CF = most-certain false positives + least-certain true positives,
# enabled by --rank_cf. The paper never mentions this method, so this tests a
# hypothesis about what was actually run -- it is not a fidelity fix.
set -euo pipefail
ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
cd "$MAIN"

run_one () {
  local short="$1" corpus="$2" lang="$3"
  echo "=== ${short}/${lang} : OAL-CF, cap=16 + informative CF ==="
  mkdir -p "output/infocf_${short}/${lang}" "checkpoints/infocf_${short}/${lang}"
  python3 run.py \
    --run_name "infocf_${short}/${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --sim_type fpstps --max_fb_samples 16 --rank_cf uncertainty \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/infocf_${short}_${lang}.log" 2>&1
  echo "    done: $(python3 scoring/print_basic_results.py "output/infocf_${short}/${lang}" 2>/dev/null | tail -1)"
}

run_one "cr" "CaucasusRegionLID" "hy-AM"
run_one "sa" "SouthAsiaLID" "dv"
echo "=== informative-CF test complete ==="
