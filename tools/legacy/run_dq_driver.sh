#!/bin/bash
# Caffeinated driver for the diversity sweep. caffeinate wraps the sweep, so it
# exits exactly when the sweep does.
ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
exec caffeinate -dimsu "$ROOT/tools/run_dq_sweep.sh" "$@"
