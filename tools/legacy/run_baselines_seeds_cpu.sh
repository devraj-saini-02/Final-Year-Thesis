#!/bin/bash
# Paper reproduction over stream-order seeds: baseline OAL and OAL-CF, 20 SLV languages, per seed
# given as arguments (default 1 2 3 4; seed 0 is runs/baseline_seed0). CPU via tools/run_cpu.py,
# two jobs at a time, caffeinate for the life of the script.
#
# A seed reshuffles the TEST stream only (`--order_file seed{N}`, data/*/orders/seed{N}_order_test);
# seed{N}_order_train is a byte copy of seed 0's train order, so the bootstrap and the model seed
# (params.py default) are unchanged. See docs/other/DEVIATIONS.md.
#
# Jobs are queued seed by seed (all 40 of seed 1, then seed 2, ...), so stopping early still leaves
# whole seeds. Resumable: a run whose state_dict.pt exists is skipped; just run the script again.
#
# Progress: runs/baseline_seeds/progress.log   Logs: runs/baseline_seeds/logs/
# At the end: runs/baseline_seeds/output/ (archived runs), summary.txt (tools/summarize_seeds.py)
#
# Usage: bash tools/run_baselines_seeds_cpu.sh [seeds...] [--arms oal|oalcf|both]   (bash, not zsh)
#   --arms oal    ~2.5 h per seed      --arms oalcf  ~10 h per seed      both (default) ~12.5 h

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/baseline_seeds"
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs" "$OUT/output"

ARMS=(oalcf oal); SEEDS=()
while [ $# -gt 0 ]; do
  case $1 in
    --arms) case $2 in oal) ARMS=(oal);; oalcf) ARMS=(oalcf);; both) ARMS=(oalcf oal);;
                       *) echo "--arms must be oal, oalcf or both"; exit 1;; esac; shift 2 ;;
    *) SEEDS+=("$1"); shift ;;
  esac
done
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4)
PROG="$OUT/progress.log"
QUEUE="$OUT/.queue"
LOCK="$OUT/.queue.lock"
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0)

corpus_dir() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID" ;;
  esac
}

for seed in "${SEEDS[@]}"; do
  for c in ac cr sea sa; do
    for split in test train; do
      f="$(corpus_dir $c)/orders/seed${seed}_order_${split}"
      [ -f "$f" ] || { echo "missing order file: $f" >> "$PROG"; exit 1; }
    done
  done
done

# Within a seed, longest jobs first so the two slots finish close together.
rm -rf "$LOCK"; : > "$QUEUE"
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for lang in hy-AM ckb cv kmr ky sr tt; do echo "$seed cr $lang $arm"  >> "$QUEUE"; done
    for lang in yue id vi;                 do echo "$seed sea $lang $arm" >> "$QUEUE"; done
    for lang in cnh dv hi ml mr or skr;    do echo "$seed sa $lang $arm"  >> "$QUEUE"; done
    for lang in bas ha yo;                 do echo "$seed ac $lang $arm"  >> "$QUEUE"; done
  done
done

pop() {  # atomically take the first job off the queue (mkdir is the lock)
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}

run_job() {  # run_job <seed> <corpus> <lang> <arm>
  local seed=$1 short=$2 lang=$3 arm=$4
  local run="${short}_${arm}_s${seed}/${lang}"
  if [ -f "output/$run/state_dict.pt" ] || [ -f "$OUT/output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"; return
  fi
  mkdir -p "output/${short}_${arm}_s${seed}" "checkpoints/${short}_${arm}_s${seed}"
  local extra=()
  [ "$arm" = oalcf ] && extra=(--sim_type fpstps)
  echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
  local s=$(date +%s)
  python3 ../tools/run_cpu.py --run_name "$run" --lid_target "$lang" \
    --feat_root "$(corpus_dir "$short")/ecapalang/" --order_file "seed${seed}" \
    "${COMMON[@]}" "${extra[@]}" > "$OUT/logs/${short}_${arm}_s${seed}_${lang}.log" 2>&1
  local rc=$?
  echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
  if [ $rc -eq 0 ] && [ -f "output/$run/state_dict.pt" ]; then
    mkdir -p "$OUT/output/${short}_${arm}_s${seed}"
    mv "output/$run" "$OUT/output/$run"
    rm -rf "checkpoints/$run"
  fi
}

worker() {
  while :; do
    job=$(pop)
    [ -z "$job" ] && break
    run_job $job
  done
}

echo "$(date '+%m-%d %H:%M') === seeds ${SEEDS[*]}, arms ${ARMS[*]}: $(wc -l < "$QUEUE" | tr -d ' ') jobs, 2 at a time, CPU" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"
echo "$(date '+%m-%d %H:%M') === all jobs finished; summarizing" >> "$PROG"
python3 ../tools/summarize_seeds.py --seeds 0 "${SEEDS[@]}" > "$OUT/summary.txt" 2>&1
echo "$(date '+%m-%d %H:%M') === complete: see runs/baseline_seeds/summary.txt" >> "$PROG"
