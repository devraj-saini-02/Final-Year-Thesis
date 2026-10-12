#!/bin/bash
# Cluster-quota diversity-penalized AL (oalcf-main/run_dq.py) with OAL-CF, on CPU, for the corpora
# given as arguments: gamma in {0.8, 0.5, 0.2} x every target language, two jobs at a time.
#
# No gamma = 1.0 check here: it already passed bit-for-bit against baseline OAL-CF on CR/sr
# (tools/run_dq_oalcf_cr_cpu.sh) and against baseline OAL on AC/yo, CR/hy-AM and SA/dv.
# Queue order: corpora in the order given; within a corpus, gamma 0.8 then 0.5 then 0.2, and the
# longest baseline runs first. Pass the slowest corpus first (e.g. `sea ac`) so the short runs fill
# the tail.
#
# Resumable: finished runs (output/<run>/state_dict.pt) are skipped. Training checkpoints of a
# finished run are deleted (unused; the final model is output/<run>/state_dict.pt).
# Outputs: oalcf-main/output/<corpus>_oalcf_dq<100*gamma>/<lang>, archived to runs/dq_oalcf/output.
# Progress: runs/dq_oalcf/progress.log
# Summaries: runs/dq_oalcf/summary_<corpus>.txt for each corpus given, and
#            runs/dq_oalcf/summary_<c1>_<c2>...txt pooled over every corpus whose runs are complete.
#
# Usage: bash tools/run_dq_oalcf_cpu.sh sea ac      (bash, not zsh)

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/dq_oalcf"
[ $# -eq 0 ] && { echo "usage: bash tools/run_dq_oalcf_cpu.sh <corpus> [<corpus> ...]   (ac cr sea sa)"; exit 1; }
for c in "$@"; do case $c in ac|cr|sea|sa) ;; *) echo "unknown corpus: $c"; exit 1 ;; esac; done
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"
PROG="$OUT/progress.log"
QUEUE="$OUT/.queue"
LOCK="$OUT/.queue.lock"
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 --sim_type fpstps)

feat_root() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID/ecapalang/" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID/ecapalang/" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID/ecapalang/" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID/ecapalang/" ;;
  esac
}
# target languages, longest baseline OAL-CF run first
langs() {
  case $1 in
    ac)  echo "bas yo ha" ;;
    cr)  echo "tt ckb kmr hy-AM ky cv sr" ;;
    sea) echo "id yue vi" ;;
    sa)  echo "hi dv skr cnh or mr ml" ;;
  esac
}
gtag() { case $1 in 0.8) echo dq080 ;; 0.5) echo dq050 ;; 0.2) echo dq020 ;; esac; }

run_job() {  # run_job <corpus> <lang> <gamma>
  local c=$1 l=$2 g=$3
  local run="${c}_oalcf_$(gtag "$g")/${l}"
  if [ -f "output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"; return
  fi
  mkdir -p "output/${c}_oalcf_$(gtag "$g")" "checkpoints/${c}_oalcf_$(gtag "$g")"
  echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
  local s=$(date +%s)
  python3 ../tools/run_cpu.py --script run_dq.py --dq_gamma "$g" --run_name "$run" --lid_target "$l" \
    --feat_root "$(feat_root "$c")" "${COMMON[@]}" > "$OUT/logs/${c}_oalcf_$(gtag "$g")_${l}.log" 2>&1
  local rc=$?
  rm -rf "checkpoints/$run"
  echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
}

rm -rf "$LOCK"; : > "$QUEUE"
for c in "$@"; do for g in 0.8 0.5 0.2; do for l in $(langs "$c"); do echo "$c $l $g" >> "$QUEUE"; done; done; done

pop() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}
worker() { while :; do job=$(pop); [ -z "$job" ] && break; run_job $job; done; }

echo "$(date '+%m-%d %H:%M') === $(wc -l < "$QUEUE" | tr -d ' ') runs (gamma 0.8/0.5/0.2 x $*, OAL-CF), 2 at a time" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"

mkdir -p "$OUT/output"
for c in "$@"; do for d in output/${c}_oalcf_dq*; do cp -R "$d" "$OUT/output/"; done; done
echo "$(date '+%m-%d %H:%M') === all runs finished and archived; summarizing" >> "$PROG"

for c in "$@"; do
  python3 "$ROOT/tools/summarize_dq_oalcf.py" --corpus "$c" --gammas 0.8 0.5 0.2 > "$OUT/summary_$c.txt" 2>&1
done
complete=()
for c in ac cr sea sa; do
  ok=1
  for t in dq080 dq050 dq020; do for l in $(langs "$c"); do
    [ -f "$OUT/output/${c}_oalcf_$t/$l/state_dict.pt" ] || ok=0
  done; done
  [ $ok = 1 ] && complete+=("$c")
done
if [ ${#complete[@]} -gt 1 ]; then
  python3 "$ROOT/tools/summarize_dq_oalcf.py" --corpus "${complete[@]}" --gammas 0.8 0.5 0.2 \
    > "$OUT/summary_$(IFS=_; echo "${complete[*]}").txt" 2>&1
fi
echo "$(date '+%m-%d %H:%M') === complete: summaries in runs/dq_oalcf/summary_*.txt" >> "$PROG"
