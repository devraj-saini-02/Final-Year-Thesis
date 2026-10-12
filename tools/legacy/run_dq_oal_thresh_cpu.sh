#!/bin/bash
# Diversity extension (oalcf-main/run_dq.py) with OAL at TIGHTER cluster thresholds, on CPU.
#
# Usage: bash tools/run_dq_oal_thresh_cpu.sh 0.18:0.95 0.16:0.975     (bash, not zsh)
#   Each argument is <dq_threshold>:<dq_gamma> and is run over all 20 target languages.
#   Run names carry both, e.g. cr_oal_t18_g95/tt, so nothing collides with the threshold-0.25 runs
#   in runs/dq_oal/.
#
# Why these gammas: the penalty multiplies a cluster's SECOND pick by gamma**(1/q), where q is the
# quota of the cluster a typical clip sits in. q falls with the threshold (4.30 at 0.25, 0.95 at 0.18,
# 0.47 at 0.16), so the same gamma bites much harder on smaller clusters. gamma 0.95 at 0.18 and
# 0.975 at 0.16 both give m = gamma**(1/q) ~ 0.95, the effective strength of gamma 0.8 at threshold
# 0.25 - the only setting that ever beat a baseline. Holding m fixed isolates the change in cluster
# granularity. See docs/other/CLUSTER_OBSERVATIONS.md sections 3a-3c.
#
# A gamma = 1.0 control at each distinct threshold is queued first (ac/ha): gamma = 1 makes the
# penalty identically 1 whatever the clustering, so it must reproduce baseline OAL bit-for-bit. It
# checks the new --dq_threshold plumbing and nothing else. If a control fails, the queue is emptied.
#
# Resumable: finished runs (output/<run>/state_dict.pt) are skipped; checkpoints are deleted.
# Outputs: oalcf-main/output/<corpus>_oal_t<NN>_g<NN>/<lang>, archived to runs/dq_oal_thresh/output.
# Progress: runs/dq_oal_thresh/progress.log   Summary: runs/dq_oal_thresh/summary.txt

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/dq_oal_thresh"
BASE="$ROOT/runs/baseline_seed0/output"
[ $# -eq 0 ] && { echo "usage: bash tools/run_dq_oal_thresh_cpu.sh <threshold>:<gamma> [...]"; exit 1; }
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
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0)   # OAL: no --sim_type

feat_root() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID/ecapalang/" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID/ecapalang/" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID/ecapalang/" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID/ecapalang/" ;;
  esac
}
langs() {
  case $1 in
    ac)  echo "bas yo ha" ;;
    cr)  echo "tt ckb kmr hy-AM ky cv sr" ;;
    sea) echo "id yue vi" ;;
    sa)  echo "hi dv skr cnh or mr ml" ;;
  esac
}
tag_t() { echo "t$(echo "$1" | sed 's/^0\.//')"; }              # 0.18 -> t18
tag_g() { echo "g$(echo "$1" | sed 's/^1\.0*$/100/; s/^0\.//')"; }  # 0.95 -> g95, 0.975 -> g975, 1.0 -> g100

run_name() { echo "${1}_oal_$(tag_t "$3")_$(tag_g "$4")/${2}"; }  # corpus lang thresh gamma

check_control() {  # check_control <corpus> <lang> <thresh>: gamma=1.0 must equal baseline OAL
  local run; run=$(run_name "$1" "$2" "$3" 1.0)
  python3 - "$MAIN/output/$run" "$BASE/${1}_oal/${2}" "$3" >> "$PROG" 2>&1 <<'EOF'
import filecmp, sys, torch
new, base, thr = sys.argv[1], sys.argv[2], sys.argv[3]
same_scores = filecmp.cmp(f"{new}/scores.csv", f"{base}/scores.csv", shallow=False)
same_samples = filecmp.cmp(f"{new}/al_samples.txt", f"{base}/al_samples.txt", shallow=False)
a, b = torch.load(f"{new}/state_dict.pt", map_location="cpu"), torch.load(f"{base}/state_dict.pt", map_location="cpu")
same_model = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
ok = same_scores and same_samples and same_model
print(f"  gamma=1.0 control at threshold {thr}: scores.csv identical={same_scores}  "
      f"al_samples identical={same_samples}  final weights identical={same_model}")
print(f"CONTROL PASSED (threshold {thr})" if ok else f"CONTROL FAILED (threshold {thr})")
sys.exit(0 if ok else 1)
EOF
  if [ $? -ne 0 ]; then
    until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
    : > "$QUEUE"; rmdir "$LOCK"
    touch "$FAILED"
    echo "$(date '+%m-%d %H:%M') === stopping: gamma=1.0 does not reproduce baseline OAL at threshold $3" >> "$PROG"
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

echo "$(date '+%m-%d %H:%M') === $(wc -l < "$QUEUE" | tr -d ' ') runs: OAL, $* (threshold:gamma), 2 at a time" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"
if [ -f "$FAILED" ]; then exit 1; fi

mkdir -p "$OUT/output"
for d in output/*_oal_t*_g*; do cp -R "$d" "$OUT/output/"; done
echo "$(date '+%m-%d %H:%M') === all runs finished and archived; summarizing" >> "$PROG"

python3 - "$ROOT" "$@" > "$OUT/summary.txt" 2>&1 <<'EOF'
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT, specs = sys.argv[1], sys.argv[2:]
sys.path.insert(0, f"{ROOT}/tools")
from score_runs import score
A, B = f"{ROOT}/runs/dq_oal_thresh/output", f"{ROOT}/runs/baseline_seed0/output"
L = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
     "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
pairs = [(c, l) for c in L for l in L[c]]
tag_t = lambda t: "t" + t.replace("0.", "")
tag_g = lambda g: "g" + ("100" if float(g) == 1.0 else g.replace("0.", ""))
print("Diversity extension with OAL at tighter cluster thresholds, seed 0, CPU\n")
print("A cluster's 2nd pick is multiplied by gamma**(1/q); q = quota of a typical clip's cluster")
print("  threshold 0.25: q 4.30 | 0.18: q 0.95 | 0.16: q 0.47   (docs/other/CLUSTER_OBSERVATIONS.md 3a)\n")
for spec in specs:
    thr, g = spec.split(":")
    d = lambda c, l: f"{A}/{c}_oal_{tag_t(thr)}_{tag_g(g)}/{l}"
    bad = [f"{c}/{l}" for c, l in pairs
           if not os.path.exists(f"{d(c, l)}/state_dict.pt")
           or sum(1 for _ in open(f"{d(c, l)}/scores.csv")) - 1 != SESS[c]]
    print(f"== threshold {thr}, gamma {g}   ({len(pairs) - len(bad)}/{len(pairs)} runs OK"
          + (f"; CHECK {bad}" if bad else "") + ")")
    if bad:
        print()
        continue
    for metric, key in (("IMLM Eq.1 (authors)", "IMLM_eq1"), ("IMLM repo", "IMLM_repo"), ("DCF", "DCF")):
        x = np.array([score(d(c, l))[key] for c, l in pairs])
        y = np.array([score(f"{B}/{c}_oal/{l}")[key] for c, l in pairs])
        p = wilcoxon(x, y, alternative="less").pvalue if np.any(x != y) else 1.0
        print(f"   {metric:20s} better in {int((x < y).sum()):2d}/20 | mean {y.mean():.4f} -> {x.mean():.4f} | p = {p:.4f}")
    df = pd.concat([pd.read_csv(f"{d(c, l)}/al_diversity.csv") for c, l in pairs])
    changed = 100 * (df.overlap_with_runpy < 8).mean()
    give = 100 * ((df.mean_u_baseline - df.mean_u_picked) / df.mean_u_baseline.replace(0, np.nan)).mean()
    print(f"   selection: {8 - df.overlap_with_runpy.mean():.2f} of 8 queries changed per session | "
          f"picks land in {df.distinct_clusters_picked.mean():.2f} distinct clusters | "
          f"clusters/session {df.n_clusters.mean():.1f} (singletons {df.n_singletons.mean():.1f})")
    print(f"   cost: a pick moved in {changed:.0f}% of sessions | uncertainty given up {give:.1f}%\n")
print("Per language, IMLM Eq.1: baseline OAL -> each setting")
head = "  ".join(f"{s:>15s}" for s in specs)
print(f"  {'':4s}{'lang':7s} {'baseline':>9s}  {head}")
for c, l in pairs:
    b = score(f"{B}/{c}_oal/{l}")["IMLM_eq1"]
    cells = "  ".join(f"{score(f'{A}/{c}_oal_{tag_t(s.split(chr(58))[0])}_{tag_g(s.split(chr(58))[1])}/{l}')['IMLM_eq1']:15.4f}"
                      for s in specs)
    print(f"  {c:4s}{l:7s} {b:9.4f}  {cells}")
EOF
echo "$(date '+%m-%d %H:%M') === complete: see runs/dq_oal_thresh/summary.txt" >> "$PROG"
