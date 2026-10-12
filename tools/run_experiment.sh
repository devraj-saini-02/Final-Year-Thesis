#!/bin/bash
# Run one experiment: an algorithm, on some stream-order seeds, with or without the diversity
# extension. This is the only script you need to start runs. See GUIDE.md.
#
# Usage (bash, not zsh; run from the repo root):
#   bash tools/run_experiment.sh --algo oal   --seeds 1 2
#   bash tools/run_experiment.sh --algo oalcf --seeds 0   --diverse --threshold 0.16 --gamma 0.93
#   bash tools/run_experiment.sh --algo oal   --seeds 0   --corpora cr --diverse --threshold 0.18 --gamma 0.95
#   bash tools/run_experiment.sh --algo oalcf --seeds 1   --cfdedup 0.10
#
#   --algo oal|oalcf      oal = active learning only; oalcf = active learning + corrective feedback
#   --seeds N [N ...]     stream-order seeds, 0-4 (seed files must exist in data/*/orders/)
#   --diverse             use the cluster-quota extension (oalcf-main/run_dq.py) instead of the baseline
#   --threshold T         cluster distance threshold, only with --diverse (e.g. 0.25, 0.18, 0.16)
#   --gamma G             penalty strength 0-1, only with --diverse (1.0 = no penalty = baseline)
#   --cfdedup T           corrective-feedback deduplication (oalcf-main/run_cfdedup.py): the expert
#                         labels one clip per group of near-duplicate CF clips, grouped at cosine
#                         distance threshold T (e.g. 0.10; 0 = no grouping = baseline). oalcf only.
#   --corpora "ac cr"     default: all four (ac cr sea sa)
#   --jobs N              parallel runs, default 2 (2 is the tested setting on this machine)
#   --force               re-run and OVERWRITE runs that already exist (default: keep them and skip)
#
# Output:   runs/<category>/<setting>/seed_<N>/<corpus>_<lang>/
#             baseline/oal, baseline/oalcf, diversity_extension/oal_t0.16_g0.93, cf_dedup/oalcf_cfd0.1
#           each run folder holds scores.csv, command.txt, state_dict.pt and its own train.log
#           progress log: runs/<category>/<setting>/seed_<N>/progress.log
#           at the end, comparisons against the matching baseline are saved to runs/comparisons/
# Time:     OAL ~2.5 h per seed (20 languages), OAL-CF ~10 h per seed, on CPU, 2 at a time.
# Safe to re-run: finished runs are skipped, so an interrupted batch resumes where it stopped.
# When the batch ends it refreshes runs/INVENTORY.md, so that file always reflects what exists.
# The screen may sleep; the machine will not (caffeinate runs for the life of this script).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN="$ROOT/oalcf-main"

ALGO=""; SEEDS=(); DIVERSE=0; THRESHOLD=""; GAMMA=""; CFDEDUP=""; CORPORA=(ac cr sea sa); JOBS=2; FORCE=0
die() { echo "ERROR: $*" >&2; echo "Run 'bash tools/run_experiment.sh' with no arguments for usage." >&2; exit 1; }

while [ $# -gt 0 ]; do
  case $1 in
    --algo)      ALGO=${2:-}; shift 2 ;;
    --seeds)     shift; while [ $# -gt 0 ] && [[ $1 != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --corpora)   shift; CORPORA=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do CORPORA+=("$1"); shift; done ;;
    --diverse)   DIVERSE=1; shift ;;
    --threshold) THRESHOLD=${2:-}; shift 2 ;;
    --gamma)     GAMMA=${2:-}; shift 2 ;;
    --cfdedup)   CFDEDUP=${2:-}; [ -z "$CFDEDUP" ] && die "--cfdedup needs a threshold (e.g. 0.10)"; shift 2 ;;
    --jobs)      JOBS=${2:-}; shift 2 ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   sed -n '2,29p' "$0"; exit 0 ;;
    *)           die "unknown argument '$1'" ;;
  esac
done

[ -z "$ALGO" ] && { sed -n '2,29p' "$0"; exit 0; }
case $ALGO in oal|oalcf) ;; *) die "--algo must be oal or oalcf" ;; esac
[ ${#SEEDS[@]} -eq 0 ] && die "--seeds is required, e.g. --seeds 0 1"
if [ $DIVERSE -eq 1 ]; then
  [ -z "$THRESHOLD" ] && die "--diverse needs --threshold (e.g. 0.16)"
  [ -z "$GAMMA" ] && die "--diverse needs --gamma (e.g. 0.93)"
else
  { [ -n "$THRESHOLD" ] || [ -n "$GAMMA" ]; } && die "--threshold/--gamma only apply with --diverse"
fi

if [ -n "$CFDEDUP" ]; then
  [ "$ALGO" = oalcf ] || die "--cfdedup only applies to --algo oalcf (it changes corrective feedback)"
  [ $DIVERSE -eq 1 ] && die "--cfdedup and --diverse cannot be combined"
  awk -v t="$CFDEDUP" 'BEGIN{exit !(t+0>=0 && t+0<=2)}' || die "--cfdedup threshold must be between 0 and 2"
fi

case $JOBS in ''|*[!0-9]*) die "--jobs must be a whole number" ;; esac
[ "$JOBS" -lt 1 ] && die "--jobs must be at least 1"
for s in "${SEEDS[@]}"; do
  case $s in ''|*[!0-9]*) die "seeds must be whole numbers, got '$s'" ;; esac
done
if [ $DIVERSE -eq 1 ]; then
  awk -v t="$THRESHOLD" 'BEGIN{exit !(t+0>0 && t+0<=1)}' || die "--threshold must be between 0 and 1"
  awk -v g="$GAMMA" 'BEGIN{exit !(g+0>0 && g+0<=1)}' || die "--gamma must be between 0 and 1 (1 = no penalty)"
fi
if pgrep -f "run_experiment.sh" | grep -v -e "^$$\$" -e "^$PPID\$" | grep -q .; then
  echo "note: another run_experiment.sh is already running. Two batches on different settings are"
  echo "      fine; two batches on the SAME setting and seed would collide. Check with: pgrep -fl run_experiment.sh"
fi

corpus_dir() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID" ;;
    *)   echo "" ;;
  esac
}
langs_of() {
  case $1 in
    ac)  echo "bas ha yo" ;;
    cr)  echo "hy-AM ckb cv kmr ky sr tt" ;;
    sea) echo "yue id vi" ;;
    sa)  echo "cnh dv hi ml mr or skr" ;;
  esac
}

for c in "${CORPORA[@]}"; do
  [ -z "$(corpus_dir "$c")" ] && die "unknown corpus '$c' (use ac, cr, sea or sa)"
  for s in "${SEEDS[@]}"; do
    if [ "$s" = 0 ]; then f="$(corpus_dir "$c")/orders/cluster_order_test"
    else f="$(corpus_dir "$c")/orders/seed${s}_order_test"; fi
    [ -f "$f" ] || die "no order file for seed $s in $c: $f (see GUIDE.md, 'Adding a new seed')"
  done
done

if [ $DIVERSE -eq 1 ]; then
  EXP_BASE="${ALGO}_t$(printf '%g' "$THRESHOLD")_g$(printf '%g' "$GAMMA")"
  ENTRY=run_dq.py
elif [ -n "$CFDEDUP" ]; then
  EXP_BASE="${ALGO}_cfd$(printf '%g' "$CFDEDUP")"
  ENTRY=run_cfdedup.py
else
  EXP_BASE="${ALGO}_baseline"
  ENTRY=run.py
fi
source "$ROOT/venv/bin/activate" || die "no venv at $ROOT/venv (see GUIDE.md, 'Setup')"
caffeinate -ims -w $$ &
cd "$MAIN" || die "no $MAIN"

COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0)
[ "$ALGO" = oalcf ] && COMMON+=(--sim_type fpstps)
[ $DIVERSE -eq 1 ] && COMMON+=(--dq_threshold "$THRESHOLD" --dq_gamma "$GAMMA")
[ -n "$CFDEDUP" ] && COMMON+=(--cfd_threshold "$CFDEDUP")

QUEUE=$(mktemp); LOCK="$(mktemp -d)/lock"
SPEC="$ALGO"; [ $DIVERSE -eq 1 ] && SPEC="$ALGO@$(printf '%g' "$THRESHOLD")/$(printf '%g' "$GAMMA")"
[ -n "$CFDEDUP" ] && SPEC="$ALGO+cfd$(printf '%g' "$CFDEDUP")"
seed_dir() {  # runs/<category>/<setting>/seed_<N>: tools/experiments.py is the one place that knows the layout
  python3 "$ROOT/tools/experiments.py" --dir "$SPEC" "$1"
}
SKIPPED=0
for seed in "${SEEDS[@]}"; do
  # languages already finished for this setting and seed, anywhere in runs/ (old layouts included)
  HAVE=""
  [ $FORCE -eq 0 ] && HAVE=$(python3 "$ROOT/tools/experiments.py" --have "$SPEC" "$seed" 2>/dev/null)
  for c in "${CORPORA[@]}"; do
    for lang in $(langs_of "$c"); do
      if echo "$HAVE" | grep -qx "$c $lang"; then SKIPPED=$((SKIPPED + 1)); continue; fi
      echo "$seed $c $lang" >> "$QUEUE"
    done
  done
done
touch "$QUEUE"
TOTAL=$(wc -l < "$QUEUE" | tr -d ' ')
if [ "$SKIPPED" -gt 0 ]; then
  echo "note: $SKIPPED run(s) already exist for $SPEC on seed(s) ${SEEDS[*]} and are skipped."
  echo "      see them with: python tools/compare_runs.py --list"
  echo "      to re-run and overwrite them instead, add --force"
fi
[ $FORCE -eq 1 ] && echo "--force: existing runs for $SPEC will be REPLACED."
if [ "$TOTAL" -eq 0 ]; then
  echo "Nothing left to run - every requested run already exists."
  exit 0
fi

pop() {  # take the first job off the queue; mkdir is the lock, so two workers never take the same one
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}

run_job() {  # run_job <seed> <corpus> <lang>
  local seed=$1 c=$2 lang=$3
  local out; out=$(seed_dir "$seed")                   # runs/<category>/<setting>/seed_<N>
  local dest="$out/${c}_${lang}"
  local run="${out#$ROOT/runs/}/${c}_${lang}"
  local work="${EXP_BASE}_s${seed}"                      # temporary, under oalcf-main/output/
  mkdir -p "$out"
  if [ -f "$dest/state_dict.pt" ]; then
    if [ $FORCE -eq 0 ]; then
      echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$out/progress.log"; return
    fi
    echo "$(date '+%m-%d %H:%M') --force: replacing existing run: $run" >> "$out/progress.log"
    rm -rf "$dest"
  fi
  mkdir -p "output/$work" "checkpoints/$work"
  local order=()
  [ "$seed" != 0 ] && order+=(--order_file "seed${seed}")   # seed 0 uses the default order file
  echo "$(date '+%m-%d %H:%M') start: $run" >> "$out/progress.log"
  local t0=$(date +%s)
  python3 "$ROOT/tools/run_cpu.py" --script "$ENTRY" \
    --run_name "$work/${c}_${lang}" --lid_target "$lang" \
    --feat_root "$(corpus_dir "$c")/ecapalang/" ${order[@]+"${order[@]}"} "${COMMON[@]}" \
    > "/tmp/${work}_${c}_${lang}.log" 2>&1
  local rc=$?
  echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - t0) / 60 )) min" \
    >> "$out/progress.log"
  if [ $rc -eq 0 ] && [ -f "output/$work/${c}_${lang}/state_dict.pt" ]; then
    mv "output/$work/${c}_${lang}" "$dest"
    mv "/tmp/${work}_${c}_${lang}.log" "$dest/train.log"     # the log lives with its run
    rm -rf "checkpoints/$work/${c}_${lang}"
    rmdir "output/$work" "checkpoints/$work" 2>/dev/null
  else
    mkdir -p "$out/failed"
    mv "/tmp/${work}_${c}_${lang}.log" "$out/failed/${c}_${lang}.log" 2>/dev/null
    echo "   FAILED - see ${out#$ROOT/}/failed/${c}_${lang}.log" >> "$out/progress.log"
  fi
}

worker() { while :; do job=$(pop); [ -z "$job" ] && break; run_job $job; done; }

echo "experiment: $EXP_BASE, seeds ${SEEDS[*]}, $TOTAL runs, $JOBS at a time"
first_dir=$(seed_dir "${SEEDS[0]}")
echo "progress:   tail -f ${first_dir#$ROOT/}/progress.log"
for seed in "${SEEDS[@]}"; do
  d=$(seed_dir "$seed"); mkdir -p "$d"
  echo "$(date '+%m-%d %H:%M') === $EXP_BASE seed $seed: queued in a batch of $TOTAL runs, $JOBS at a time" \
    >> "$d/progress.log"
done

PIDS=()
for ((i = 0; i < JOBS; i++)); do worker & PIDS+=($!); sleep 5; done
wait "${PIDS[@]}"     # only the workers; a bare `wait` would also wait on caffeinate and hang
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"

for seed in "${SEEDS[@]}"; do
  echo "$(date '+%m-%d %H:%M') === batch finished" >> "$(seed_dir "$seed")/progress.log"
done

python3 "$ROOT/tools/run_inventory.py"      # keep runs/INVENTORY.md current

# Save the comparisons against the matching baseline (runs/comparisons/<category>/), on the seeds
# of this batch and on every seed both settings have. Partial results are labelled INCOMPLETE.
case $SPEC in
  oal)                      PAIRS=("oal oalcf");          METRICS=(IMLM_eq1 DCF) ;;   # needs both baselines
  oalcf)                    PAIRS=("oal oalcf");          METRICS=(IMLM_eq1 DCF) ;;
  *+cfd*)                   PAIRS=("oalcf $SPEC");        METRICS=(IMLM_eq1 IMLM_thesis IMLM_repo_expert DCF) ;;
  *)                        PAIRS=("$ALGO $SPEC");        METRICS=(IMLM_eq1 DCF) ;;
esac
echo
echo "saving comparisons to runs/comparisons/ ..."
SEEDSETS=("${SEEDS[*]}"); [ "${SEEDS[*]}" != "0 1 2 3 4" ] && SEEDSETS+=("0 1 2 3 4")
for pair in "${PAIRS[@]}"; do
  for m in "${METRICS[@]}"; do
    for seedset in "${SEEDSETS[@]}"; do
      python3 "$ROOT/tools/compare_runs.py" $pair --seeds $seedset --metric "$m" 2>&1 \
        | grep -E "saved to|better in|Nothing to compare" | sed 's/^/   /'
    done
  done
done
echo "done."
