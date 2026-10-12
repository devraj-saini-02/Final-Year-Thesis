#!/bin/bash
# Cluster-quota diversity-penalized AL (oalcf-main/run_dq.py) with OAL-CF, Caucasus Region, on CPU.
#
# One queue, two jobs at a time:
#   1. gamma = 1.0 on sr (the fastest CR OAL-CF baseline). When it finishes, scores.csv,
#      al_samples.txt and every weight of the final model must equal the archived baseline OAL-CF
#      run. If not, the queue is emptied (a run already in progress finishes; nothing new starts)
#      and no summary is written.
#   2. gamma in {0.8, 0.5, 0.2} x 7 CR languages, OAL-CF (uncapped CF, --sim_type fpstps).
#      Within each gamma the longest baseline runs go first, so the two workers finish together.
# Then health check, scoring, and comparison against baseline OAL-CF, baseline OAL, and the same
# gamma with OAL (runs/dq_oal).
#
# Resumable: finished runs (output/<run>/state_dict.pt) are skipped. Training checkpoints of a
# finished run are deleted (unused; the final model is output/<run>/state_dict.pt).
# Outputs: oalcf-main/output/cr_oalcf_dq<100*gamma>/<lang>, archived to runs/dq_oalcf/output.
# Progress: runs/dq_oalcf/progress.log     Summary: runs/dq_oalcf/summary.txt
#
# Usage: bash tools/run_dq_oalcf_cr_cpu.sh      (bash, not zsh)

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/dq_oalcf"
BASE="$ROOT/runs/baseline_seed0/output"
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"
PROG="$OUT/progress.log"
QUEUE="$OUT/.queue"
LOCK="$OUT/.queue.lock"
FAILED="$OUT/.gamma1_failed"
FEAT="$ROOT/data/CaucasusRegionLID/ecapalang/"
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 --sim_type fpstps)

gtag() { case $1 in 1.0) echo dq100 ;; 0.8) echo dq080 ;; 0.5) echo dq050 ;; 0.2) echo dq020 ;; esac; }

check_gamma1() {  # check_gamma1 <lang>: gamma = 1.0 OAL-CF must equal the archived baseline OAL-CF
  python3 - "$MAIN/output/cr_oalcf_dq100/$1" "$BASE/cr_oalcf/$1" >> "$PROG" 2>&1 <<'EOF'
import filecmp, sys, torch
new, base = sys.argv[1], sys.argv[2]
same_scores = filecmp.cmp(f"{new}/scores.csv", f"{base}/scores.csv", shallow=False)
same_samples = filecmp.cmp(f"{new}/al_samples.txt", f"{base}/al_samples.txt", shallow=False)
a, b = torch.load(f"{new}/state_dict.pt", map_location="cpu"), torch.load(f"{base}/state_dict.pt", map_location="cpu")
same_model = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
ok = same_scores and same_samples and same_model
print(f"  gamma=1.0 OAL-CF {new.split('/')[-1]}: scores.csv identical={same_scores}  "
      f"al_samples.txt identical={same_samples}  final weights identical={same_model}")
print("GAMMA=1 CHECK PASSED" if ok else "GAMMA=1 CHECK FAILED")
sys.exit(0 if ok else 1)
EOF
  if [ $? -ne 0 ]; then
    until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
    : > "$QUEUE"; rmdir "$LOCK"
    touch "$FAILED"
    echo "$(date '+%m-%d %H:%M') === stopping: gamma=1.0 does not reproduce baseline OAL-CF; queue emptied" >> "$PROG"
  fi
}

run_job() {  # run_job <corpus> <lang> <gamma>
  local c=$1 l=$2 g=$3
  local run="${c}_oalcf_$(gtag "$g")/${l}"
  if [ -f "output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"
  else
    mkdir -p "output/${c}_oalcf_$(gtag "$g")" "checkpoints/${c}_oalcf_$(gtag "$g")"
    echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
    local s=$(date +%s)
    python3 ../tools/run_cpu.py --script run_dq.py --dq_gamma "$g" --run_name "$run" --lid_target "$l" \
      --feat_root "$FEAT" "${COMMON[@]}" > "$OUT/logs/${c}_oalcf_$(gtag "$g")_${l}.log" 2>&1
    local rc=$?
    rm -rf "checkpoints/$run"
    echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
  fi
  [ "$g" = "1.0" ] && check_gamma1 "$l"
}

rm -rf "$LOCK" "$FAILED"; : > "$QUEUE"
echo "cr sr 1.0" >> "$QUEUE"
for g in 0.8 0.5 0.2; do for l in tt ckb kmr hy-AM ky cv sr; do echo "cr $l $g" >> "$QUEUE"; done; done

pop() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}
worker() { while :; do job=$(pop); [ -z "$job" ] && break; run_job $job; done; }

echo "$(date '+%m-%d %H:%M') === $(wc -l < "$QUEUE" | tr -d ' ') runs (gamma=1.0 check on sr, then gamma 0.8/0.5/0.2 x 7 CR languages, OAL-CF), 2 at a time" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"
if [ -f "$FAILED" ]; then exit 1; fi

mkdir -p "$OUT/output"
for d in output/cr_oalcf_dq*; do cp -R "$d" "$OUT/output/"; done
echo "$(date '+%m-%d %H:%M') === all runs finished and archived; summarizing" >> "$PROG"

python3 "$ROOT/tools/summarize_dq_oalcf.py" --corpus cr --gammas 0.8 0.5 0.2 > "$OUT/summary.txt" 2>&1
echo "$(date '+%m-%d %H:%M') === complete: see runs/dq_oalcf/summary.txt" >> "$PROG"
