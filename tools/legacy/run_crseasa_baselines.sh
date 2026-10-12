#!/bin/bash
# Phase 5 baselines for CR/SEA/SA, mirroring AC's Phase 5 exactly: one OAL run
# and one OAL-CF run per target language, at the default budget (--n_queries 8,
# unset). Corpus and order files were built in an earlier session (Phase 4) --
# this script only runs the model, it does not touch data/.
#
# Target locales are Table II's, cross-checked against tools/find_corpus.py's
# REGIONS dict (the same checksums used to identify each corpus's release).
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"

# short_name : corpus_dir : space-separated target locales
CORPORA=(
  "cr:CaucasusRegionLID:hy-AM ckb cv kmr ky sr tt"
  "sea:SoutheastAsiaLID:yue id vi"
  "sa:SouthAsiaLID:cnh dv hi ml mr or skr"
)

run_one () {
  local short="$1" corpus="$2" lang="$3" arm="$4" simflag="$5"
  local outdir="output/${short}_${arm}/${lang}"
  local ckptdir="checkpoints/${short}_${arm}/${lang}"
  cd "$MAIN"
  mkdir -p "$outdir" "$ckptdir"
  echo "=== ${short} / ${lang} : ${arm} run ==="
  python3 run.py \
    --run_name "${short}_${arm}/${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    $simflag \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/crseasa_baseline_${short}_${arm}_${lang}.log" 2>&1
  echo "    done: $(python3 scoring/print_basic_results.py "$outdir" 2>/dev/null | tail -1)"
}

for entry in "${CORPORA[@]}"; do
  IFS=':' read -r short corpus langs <<< "$entry"
  for lang in $langs; do
    run_one "$short" "$corpus" "$lang" "oal" ""
    run_one "$short" "$corpus" "$lang" "oalcf" "--sim_type fpstps"
  done
  echo "=== ${short} baselines complete ==="
done

echo "=== all CR/SEA/SA baselines complete ==="
