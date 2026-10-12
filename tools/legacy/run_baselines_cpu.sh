#!/bin/bash
# All 40 SLV baselines (20 target languages x OAL / OAL-CF) on CPU via tools/run_cpu.py,
# two jobs at a time, kept awake with caffeinate for the life of this script.
#
# Resumable: a job whose output/<run>/state_dict.pt exists is skipped (run.py writes that
# file only at the very end of a run, and wipes a partial output dir itself on restart).
# Just run this script again after an interruption.
#
# Progress:  runs/baseline_progress.log      Per-run logs:  runs/baseline_<corpus>_<arm>_<lang>.log
# At the end: runs/baseline_scores.{txt,csv} (both scorers), runs/baseline_health.txt
#
# Usage: bash tools/run_baselines_cpu.sh      (bash, not zsh)

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1

PROG="$ROOT/runs/baseline_progress.log"
QUEUE="$ROOT/runs/.baseline_queue"
LOCK="$ROOT/runs/.baseline_queue.lock"
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0)

feat_root() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID/ecapalang/" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID/ecapalang/" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID/ecapalang/" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID/ecapalang/" ;;
  esac
}

# Longest jobs first (OAL-CF, most sessions first) so the two slots finish close together.
rm -rf "$LOCK"; : > "$QUEUE"
for arm in oalcf oal; do
  for lang in hy-AM ckb cv kmr ky sr tt; do echo "cr $lang $arm"  >> "$QUEUE"; done
  for lang in yue id vi;                 do echo "sea $lang $arm" >> "$QUEUE"; done
  for lang in cnh dv hi ml mr or skr;    do echo "sa $lang $arm"  >> "$QUEUE"; done
  for lang in bas ha yo;                 do echo "ac $lang $arm"  >> "$QUEUE"; done
done

pop() {  # atomically take the first job off the queue (mkdir is the lock)
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}

run_job() {  # run_job <corpus> <lang> <arm>
  local short=$1 lang=$2 arm=$3
  local run="${short}_${arm}/${lang}"
  if [ -f "output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"; return
  fi
  mkdir -p "output/${short}_${arm}" "checkpoints/${short}_${arm}"
  local extra=()
  [ "$arm" = oalcf ] && extra=(--sim_type fpstps)
  echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
  local s=$(date +%s)
  python3 ../tools/run_cpu.py --run_name "$run" --lid_target "$lang" --feat_root "$(feat_root "$short")" \
    "${COMMON[@]}" "${extra[@]}" > "$ROOT/runs/baseline_${short}_${arm}_${lang}.log" 2>&1
  local rc=$?
  echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
}

worker() {
  while :; do
    job=$(pop)
    [ -z "$job" ] && break
    run_job $job
  done
}

echo "$(date '+%m-%d %H:%M') === baseline batch started: $(wc -l < "$QUEUE" | tr -d ' ') jobs, 2 at a time, CPU" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"
echo "$(date '+%m-%d %H:%M') === all jobs finished; scoring" >> "$PROG"

python3 ../tools/score_runs.py --csv "$ROOT/runs/baseline_scores.csv" \
  output/ac_oal/* output/ac_oalcf/* output/cr_oal/* output/cr_oalcf/* \
  output/sea_oal/* output/sea_oalcf/* output/sa_oal/* output/sa_oalcf/* > "$ROOT/runs/baseline_scores.txt" 2>&1

python3 - "$ROOT" > "$ROOT/runs/baseline_health.txt" 2>&1 <<'EOF'
import os, sys, torch
ROOT = sys.argv[1]; M = f"{ROOT}/oalcf-main"
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
LANGS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
         "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
bad = 0
print(f"{'run':20s}{'sessions':>10}{'NaN sess':>10}{'weights':>9}{'device':>8}  status")
for c, langs in LANGS.items():
    for arm in ("oal", "oalcf"):
        for l in langs:
            d = f"{M}/output/{c}_{arm}/{l}"; log = f"{ROOT}/runs/baseline_{c}_{arm}_{l}.log"
            rows = sum(1 for _ in open(f"{d}/scores.csv")) - 1 if os.path.exists(f"{d}/scores.csv") else 0
            txt = open(log, errors="ignore").read().replace("\r", "\n") if os.path.exists(log) else ""
            nan = sum(1 for ln in txt.splitlines() if "test/loss" in ln and "nan" in ln.lower()) // 2
            dev = "cpu" if "used: False" in txt else ("MPS!" if "used: True" in txt else "?")
            fin = None
            if os.path.exists(f"{d}/state_dict.pt"):
                sd = torch.load(f"{d}/state_dict.pt", map_location="cpu")
                fin = all(torch.isfinite(v).all().item() for v in sd.values() if v.is_floating_point())
            ok = rows == SESS[c] and nan == 0 and fin is True and dev == "cpu"
            bad += not ok
            print(f"{c+'_'+arm+'/'+l:20s}{f'{rows}/{SESS[c]}':>10}{nan:>10}{str(fin):>9}{dev:>8}  {'OK' if ok else 'CHECK'}")
print(f"\n{40 - bad}/40 runs OK" + ("" if bad == 0 else f"; {bad} need checking"))
EOF
echo "$(date '+%m-%d %H:%M') === batch complete: see runs/baseline_scores.txt and runs/baseline_health.txt" >> "$PROG"
