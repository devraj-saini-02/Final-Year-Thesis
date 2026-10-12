#!/bin/bash
set -euo pipefail
ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
for CORPUS in CaucasusRegionLID SoutheastAsiaLID SouthAsiaLID; do
  echo "=== extracting $CORPUS ==="
  cd "$ROOT/data/$CORPUS"
  python3 "$ROOT/oalcf-main/utils/extract_ecapa.py" 2>&1 | tee "$ROOT/data/$CORPUS/extract.log"
  echo "=== $CORPUS done ==="
done
echo "=== all extraction complete ==="
