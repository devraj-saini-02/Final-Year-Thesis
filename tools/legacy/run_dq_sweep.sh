#!/bin/bash
# Cluster-quota diversity-penalized AL sweep, OAL arm, all 20 target languages.
#
#   score(x) = u(x) * gamma ** (n_c / q_c)      greedy, one pick at a time
#
# Clustering: agglomerative, cosine, average linkage, n_clusters=None,
# distance_threshold=0.25 (chosen empirically; see utils/diverse_query.py).
#
# PHASE 1 -- AC baselines (gamma=1.0).  AC's orders/cluster_order_test was
# overwritten on Sep 11, so the archived ac_oal_precap runs are on a stream that
# no longer exists and cannot serve as AC's baseline.  gamma=1.0 is verified
# bit-identical to untouched run.py, so these ARE baseline OAL runs on the
# current stream.  CR/SEA/SA order files are intact (Sep 7) and their archived
# *_oal_precap runs are used as baselines directly -- no reruns needed.
#
# PHASE 2 -- gamma = 0.9 and 0.7 across all 20 languages.
set -uo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
PY="$ROOT/venv/bin/python"
cd "$MAIN"

GAMMAS=("$@"); if [ ${#GAMMAS[@]} -eq 0 ]; then GAMMAS=(0.9 0.7); fi

CORPORA=(
  "ac:AfricanContinentLID:bas ha yo"
  "cr:CaucasusRegionLID:hy-AM ckb cv kmr ky sr tt"
  "sea:SoutheastAsiaLID:yue id vi"
  "sa:SouthAsiaLID:cnh dv hi ml mr or skr"
)

run_one () {  # gamma short corpus lang
  local g="$1" short="$2" corpus="$3" lang="$4"
  local t; t="g$(printf '%03d' "$(echo "$g * 100" | bc | cut -d. -f1)")"
  local outdir="output/dq_${t}/${short}_${lang}"
  if [ -f "$outdir/scores.csv" ] && [ -f "$outdir/state_dict.pt" ]; then
    echo "    skip (complete): ${t} ${short}/${lang}"; return 0
  fi
  mkdir -p "output/dq_${t}" "checkpoints/dq_${t}"
  echo "    $(date '+%H:%M:%S')  gamma=$g  ${short}/${lang}"
  "$PY" run_dq.py --dq_gamma "$g" \
    --run_name "dq_${t}/${short}_${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 \
    > "$ROOT/runs/dq_${t}_${short}_${lang}.log" 2>&1 \
    && echo "        done: $("$PY" scoring/print_basic_results.py "$outdir" 2>/dev/null | tail -1)" \
    || echo "        FAILED -- see runs/dq_${t}_${short}_${lang}.log"
}

echo "############ PHASE 1: AC baselines (gamma=1.0, current stream) ############"
date
for lang in bas ha yo; do run_one 1.0 ac AfricanContinentLID "$lang"; done

echo
echo "############ PHASE 2: gamma = ${GAMMAS[*]} across all 20 languages ############"
date
for entry in "${CORPORA[@]}"; do
  IFS=':' read -r short corpus langs <<< "$entry"
  for g in "${GAMMAS[@]}"; do
    echo "  --- ${short} corpus, gamma=${g} ---"
    for lang in $langs; do run_one "$g" "$short" "$corpus" "$lang"; done
  done
  echo "  === ${short} complete ==="
done

echo
echo "############ generating tables and convergence graphs ############"
"$PY" "$ROOT/tools/plot_dq.py"
echo "############ sweep complete ############"
date
