#!/bin/bash
# Full CR/SEA/SA pipeline: bootstrap fetch -> bootstrap assemble -> bootstrap
# feature extraction -> Phase 5 baselines (34 runs: 17 target languages x 2
# arms). Each step logged separately under runs/.
set -euo pipefail

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
cd "$ROOT"

echo "=== [1/4] fetching bootstrap train shards (13.39 GB, 31 locales) ==="
python3 tools/fetch_crseasa_bootstrap.py --go

echo "=== [2/4] extracting needed bootstrap clips into each corpus's train/ ==="
python3 tools/assemble_crseasa_bootstrap.py

echo "=== [3/4] ECAPA feature extraction, train split only (test/dev already done, skipped) ==="
for corpus in CaucasusRegionLID SoutheastAsiaLID SouthAsiaLID; do
  echo "--- $corpus ---"
  cd "$ROOT/data/$corpus"
  python3 "$ROOT/oalcf-main/utils/extract_ecapa.py"
  cd "$ROOT"
done

echo "=== [4/4] Phase 5 baselines: 17 languages x 2 arms = 34 runs ==="
bash tools/run_crseasa_baselines.sh

echo "=== CR/SEA/SA pipeline complete ==="
