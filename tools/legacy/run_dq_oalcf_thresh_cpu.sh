#!/bin/bash
# Diversity extension (oalcf-main/run_dq.py) with OAL-CF at a chosen cluster threshold, on CPU.
#
# Usage: bash tools/run_dq_oalcf_thresh_cpu.sh 0.16:0.93        (bash, not zsh)
#   Each argument is <dq_threshold>:<dq_gamma>, run over all 20 target languages with corrective
#   feedback (--sim_type fpstps, uncapped). Run names carry both, e.g. cr_oalcf_t16_g93/tt, so
#   nothing collides with the threshold-0.25 runs in runs/dq_oalcf/ or the OAL runs in
#   runs/dq_oal_thresh/.
#
# Why 0.16 / gamma 0.93: chosen from the OAL screen (runs/dq_oal_thresh/RESULTS.md) as the best of
# seven settings - Caucasus 7/7, -37%, p = 0.0078; pooled 15/20, -32%, and still -18% (p = 0.0198)
# with the burst-prone African corpus excluded. This batch is the CONFIRMATORY test on the arm that
# matters, against the pre-declared prediction recorded in RESULTS.md before launch.
#
# Note: corrective feedback makes the model more confident, so the same gamma moves fewer queries
# than in the OAL arm (measured at threshold 0.25: 0.69 changed queries per session with CF vs 1.31
# without). Read the realised disturbance off al_diversity.csv from the first finished run.
#
# A gamma = 1.0 control is queued first (ac/ha): gamma = 1 makes the penalty identically 1 whatever
# the clustering, so it must reproduce baseline OAL-CF bit-for-bit. If it fails, the queue is emptied.
#
# Resumable: finished runs (output/<run>/state_dict.pt) are skipped; checkpoints are deleted.
# Outputs: oalcf-main/output/<corpus>_oalcf_t<NN>_g<NN>/<lang>, archived to runs/dq_oalcf_thresh/output.
# Progress: runs/dq_oalcf_thresh/progress.log   Summary: runs/dq_oalcf_thresh/summary.txt

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/dq_oalcf_thresh"
BASE="$ROOT/runs/baseline_seed0/output"
[ $# -eq 0 ] && { echo "usage: bash tools/run_dq_oalcf_thresh_cpu.sh <threshold>:<gamma> [...]"; exit 1; }
for spec in "$@"; do
  case $spec in *:*) ;; *) echo "bad argument '$spec', expected <threshold>:<gamma>"; exit 1 ;; esac
done
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"
PROG="$OUT/progress.log"
QUEUE="$OUT/.queue"
LOCK="$OUT/.queue.lock"
FAILED="$OUT/.control_failed"
# OAL-CF: corrective feedback on every positive prediction, uncapped (never pass --max_fb_samples)
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0 --sim_type fpstps)

feat_root() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID/ecapalang/" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID/ecapalang/" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID/ecapalang/" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID/ecapalang/" ;;
  esac
}
# longest baseline OAL-CF run first within each corpus, so the two workers finish together
langs() {
  case $1 in
    ac)  echo "bas yo ha" ;;
    cr)  echo "tt ckb kmr hy-AM ky cv sr" ;;
    sea) echo "id yue vi" ;;
    sa)  echo "hi dv skr cnh or mr ml" ;;
  esac
}
tag_t() { echo "t$(echo "$1" | sed 's/^0\.//')"; }
tag_g() { echo "g$(echo "$1" | sed 's/^1\.0*$/100/; s/^0\.//')"; }
run_name() { echo "${1}_oalcf_$(tag_t "$3")_$(tag_g "$4")/${2}"; }

check_control() {  # check_control <corpus> <lang> <thresh>: gamma=1.0 must equal baseline OAL-CF
  local run; run=$(run_name "$1" "$2" "$3" 1.0)
  python3 - "$MAIN/output/$run" "$BASE/${1}_oalcf/${2}" "$3" >> "$PROG" 2>&1 <<'EOF'
import filecmp, sys, torch
new, base, thr = sys.argv[1], sys.argv[2], sys.argv[3]
same_scores = filecmp.cmp(f"{new}/scores.csv", f"{base}/scores.csv", shallow=False)
same_samples = filecmp.cmp(f"{new}/al_samples.txt", f"{base}/al_samples.txt", shallow=False)
a, b = torch.load(f"{new}/state_dict.pt", map_location="cpu"), torch.load(f"{base}/state_dict.pt", map_location="cpu")
same_model = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
ok = same_scores and same_samples and same_model
print(f"  gamma=1.0 OAL-CF control at threshold {thr}: scores.csv identical={same_scores}  "
      f"al_samples identical={same_samples}  final weights identical={same_model}")
print(f"CONTROL PASSED (threshold {thr})" if ok else f"CONTROL FAILED (threshold {thr})")
sys.exit(0 if ok else 1)
EOF
  if [ $? -ne 0 ]; then
    until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
    : > "$QUEUE"; rmdir "$LOCK"
    touch "$FAILED"
    echo "$(date '+%m-%d %H:%M') === stopping: gamma=1.0 does not reproduce baseline OAL-CF at threshold $3" >> "$PROG"
  fi
}

run_job() {  # run_job <corpus> <lang> <thresh> <gamma>
  local c=$1 l=$2 thr=$3 g=$4 run dir
  run=$(run_name "$c" "$l" "$thr" "$g"); dir=$(dirname "$run")
  if [ -f "output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"
  else
    mkdir -p "output/$dir" "checkpoints/$dir"
    echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
    local s; s=$(date +%s)
    python3 ../tools/run_cpu.py --script run_dq.py --dq_gamma "$g" --dq_threshold "$thr" \
      --run_name "$run" --lid_target "$l" --feat_root "$(feat_root "$c")" "${COMMON[@]}" \
      > "$OUT/logs/$(echo "$run" | tr '/' '_').log" 2>&1
    local rc=$?
    rm -rf "checkpoints/$run"
    echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
  fi
  [ "$g" = "1.0" ] && check_control "$c" "$l" "$thr"
}

rm -rf "$LOCK" "$FAILED"; : > "$QUEUE"
for thr in $(for spec in "$@"; do echo "${spec%%:*}"; done | sort -u); do
  echo "ac ha $thr 1.0" >> "$QUEUE"      # control first, on the fastest corpus
done
for spec in "$@"; do
  thr="${spec%%:*}"; g="${spec##*:}"
  for c in cr sea sa ac; do for l in $(langs "$c"); do echo "$c $l $thr $g" >> "$QUEUE"; done; done
done

pop() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}
worker() { while :; do job=$(pop); [ -z "$job" ] && break; run_job $job; done; }

echo "$(date '+%m-%d %H:%M') === $(wc -l < "$QUEUE" | tr -d ' ') runs: OAL-CF, $* (threshold:gamma), 2 at a time" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"
if [ -f "$FAILED" ]; then exit 1; fi

mkdir -p "$OUT/output"
for d in output/*_oalcf_t*_g*; do cp -R "$d" "$OUT/output/"; done
echo "$(date '+%m-%d %H:%M') === all runs finished and archived; summarizing" >> "$PROG"

python3 - "$ROOT" "$@" > "$OUT/summary.txt" 2>&1 <<'EOF'
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT, specs = sys.argv[1], sys.argv[2:]
sys.path.insert(0, f"{ROOT}/tools")
from score_runs import score
A, B = f"{ROOT}/runs/dq_oalcf_thresh/output", f"{ROOT}/runs/baseline_seed0/output"
L = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
     "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
pairs = [(c, l) for c in L for l in L[c]]
tag_t = lambda t: "t" + t.replace("0.", "")
tag_g = lambda g: "g" + ("100" if float(g) == 1.0 else g.replace("0.", ""))
print("Diversity extension with OAL-CF at a tighter cluster threshold, seed 0, CPU\n")
for spec in specs:
    thr, g = spec.split(":")
    d = lambda c, l: f"{A}/{c}_oalcf_{tag_t(thr)}_{tag_g(g)}/{l}"
    bad = [f"{c}/{l}" for c, l in pairs if not os.path.exists(f"{d(c, l)}/state_dict.pt")
           or sum(1 for _ in open(f"{d(c, l)}/scores.csv")) - 1 != SESS[c]]
    print(f"== threshold {thr}, gamma {g}   ({len(pairs) - len(bad)}/{len(pairs)} runs OK"
          + (f"; CHECK {bad}" if bad else "") + ")")
    if bad:
        print(); continue
    for label, ref in (("vs baseline OAL-CF", "oalcf"), ("vs baseline OAL", "oal")):
        print(f"  {label}")
        for metric, key in (("IMLM Eq.1", "IMLM_eq1"), ("IMLM repo", "IMLM_repo"), ("DCF", "DCF")):
            x = np.array([score(d(c, l))[key] for c, l in pairs])
            y = np.array([score(f"{B}/{c}_{ref}/{l}")[key] for c, l in pairs])
            p = wilcoxon(x, y, alternative="less").pvalue if np.any(x != y) else 1.0
            print(f"    {metric:12s} better in {int((x < y).sum()):2d}/20 | mean {y.mean():.4f} -> {x.mean():.4f} | p = {p:.4f}")
    print("  per corpus (IMLM Eq.1, vs baseline OAL-CF)")
    for c in L:
        x = np.array([score(d(c, l))["IMLM_eq1"] for l in L[c]])
        y = np.array([score(f"{B}/{c}_oalcf/{l}")["IMLM_eq1"] for l in L[c]])
        p = wilcoxon(x, y, alternative="less").pvalue if np.any(x != y) else 1.0
        print(f"    {c:4s} {int((x < y).sum())}/{len(x)} | mean {y.mean():.4f} -> {x.mean():.4f} "
              f"({100 * (x.mean() / y.mean() - 1):+.0f}%) | p = {p:.4f}")
    df = pd.concat([pd.read_csv(f"{d(c, l)}/al_diversity.csv") for c, l in pairs])
    give = 100 * ((df.mean_u_baseline - df.mean_u_picked) / df.mean_u_baseline.replace(0, np.nan)).mean()
    print(f"  selection: {8 - df.overlap_with_runpy.mean():.2f} of 8 queries changed per session | "
          f"distinct clusters {df.distinct_clusters_picked.mean():.2f} | clusters/session {df.n_clusters.mean():.1f} | "
          f"uncertainty given up {give:.1f}%")
    print("\n  per language, IMLM Eq.1: baseline OAL-CF -> this setting")
    for c, l in pairs:
        b = score(f"{B}/{c}_oalcf/{l}")["IMLM_eq1"]; n = score(d(c, l))["IMLM_eq1"]
        print(f"    {c:4s}{l:6s} {b:.4f} -> {n:.4f} ({100 * (n / b - 1):+.0f}%)")
EOF
echo "$(date '+%m-%d %H:%M') === complete: see runs/dq_oalcf_thresh/summary.txt" >> "$PROG"
