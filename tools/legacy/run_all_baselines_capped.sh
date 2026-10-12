#!/bin/bash
# Full reproduction rerun, all four SLV corpora, both arms, seed 0 -- now with
# the corrective-feedback fix applied: --max_fb_samples 16 (8 TP + 8 FP
# balanced per session), matching the dissertation's actual baseline config
# (Lindsey 2024, sections 4.4 / 5.4), which the original runs omitted.
#
# AC is included even though OAL-CF was already run once (uncapped, archived
# to output/ac_oalcf_precap) -- it used the same uncapped CF as CR/SEA/SA, so
# its own numbers may change under the fix too, and this makes all four
# corpora's capped comparison come from one coherent batch.
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
cd "$MAIN"

run_one () {
  local short="$1" corpus="$2" lang="$3" arm="$4" simflag="$5"
  local outdir="output/${short}_${arm}/${lang}"
  local ckptdir="checkpoints/${short}_${arm}/${lang}"
  mkdir -p "$outdir" "$ckptdir"
  echo "=== ${short} / ${lang} : ${arm} run ==="
  python3 run.py \
    --run_name "${short}_${arm}/${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    $simflag \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/capped_${short}_${arm}_${lang}.log" 2>&1
  echo "    done: $(python3 scoring/print_basic_results.py "$outdir" 2>/dev/null | tail -1)"
}

# short : corpus_dir : space-separated target locales
CORPORA=(
  "ac:AfricanContinentLID:bas ha yo"
  "cr:CaucasusRegionLID:hy-AM ckb cv kmr ky sr tt"
  "sea:SoutheastAsiaLID:yue id vi"
  "sa:SouthAsiaLID:cnh dv hi ml mr or skr"
)

for entry in "${CORPORA[@]}"; do
  IFS=':' read -r short corpus langs <<< "$entry"
  for lang in $langs; do
    run_one "$short" "$corpus" "$lang" "oal" ""
    run_one "$short" "$corpus" "$lang" "oalcf" "--sim_type fpstps --max_fb_samples 16"
  done
  echo "=== ${short} baselines complete ==="
done

echo "=== all corpora baselines complete (capped) ==="

echo "=== generating convergence plots and summary ==="
python3 "$ROOT/tools/plot_convergence.py"

echo "=== full capped reproduction run complete ==="
