#!/bin/bash
# Resume after the boundary-bleed fix to the CR/SEA/SA bootstrap (see
# docs/other/DEVIATIONS.md and tools/assemble_crseasa_bootstrap.py). Re-fetches
# (only the 4 new locales actually transfer, the other 31 are already on
# disk), re-assembles with the corrected logic, re-extracts features for the
# newly-added clips only, then a --debug sanity pass per corpus (traverses
# every session boundary cheaply, exercises the exact bug that crashed the
# first attempt) before committing to the real 34-run baseline batch again.
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
cd "$ROOT"

echo "=== [1/5] fetching newly-identified bootstrap locales (ab, ar, eu, ta -- 6.00 GB) ==="
python3 tools/fetch_crseasa_bootstrap.py --go

echo "=== [2/5] re-assembling bootstrap clips (corrected boundary logic) ==="
python3 tools/assemble_crseasa_bootstrap.py

echo "=== [3/5] ECAPA feature extraction for newly-added clips ==="
for corpus in CaucasusRegionLID SoutheastAsiaLID SouthAsiaLID; do
  echo "--- $corpus ---"
  cd "$ROOT/data/$corpus"
  python3 "$MAIN/utils/extract_ecapa.py"
  cd "$ROOT"
done

echo "=== [4/5] debug sanity pass: one language per corpus, full session sweep ==="
# macOS ships bash 3.2 (no associative arrays, added in bash 4) -- parallel
# indexed arrays instead.
SANITY_CORPUS=(CaucasusRegionLID SoutheastAsiaLID SouthAsiaLID)
SANITY_LANG=(hy-AM id or)
SANITY_SHORT=(cr sea sa)
for i in 0 1 2; do
  corpus="${SANITY_CORPUS[$i]}"
  lang="${SANITY_LANG[$i]}"
  short="${SANITY_SHORT[$i]}"
  echo "--- sanity: $corpus / $lang (OAL, --debug) ---"
  cd "$MAIN"
  mkdir -p "output/_sanity2/${short}_${lang}" "checkpoints/_sanity2/${short}_${lang}"
  python3 run.py \
    --run_name "_sanity2/${short}_${lang}" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 --debug
  cd "$ROOT"
  echo "--- sanity: $corpus / $lang (OAL-CF, --debug) ---"
  cd "$MAIN"
  mkdir -p "output/_sanity2/${short}_${lang}_cf" "checkpoints/_sanity2/${short}_${lang}_cf"
  python3 run.py \
    --run_name "_sanity2/${short}_${lang}_cf" \
    --env_name test --lid_target "$lang" \
    --feat_root "$ROOT/data/${corpus}/ecapalang/" \
    --sim_type fpstps \
    --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 --debug
  cd "$ROOT"
done
echo "=== sanity pass clean on all 3 corpora, both arms ==="

echo "=== [5/5] Phase 5 baselines: 17 languages x 2 arms = 34 runs ==="
bash tools/run_crseasa_baselines.sh

echo "=== CR/SEA/SA pipeline complete ==="
