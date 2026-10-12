#!/bin/bash
# Cluster-quota diversity-penalized AL (oalcf-main/run_dq.py) with OAL, on CPU.
#
# Stage 1: gamma = 1.0 must reproduce the archived baseline OAL bit-for-bit (scores.csv and
#          every weight of the final model) for hy-AM (CR), yo (AC) and dv (SA).
#          If any differs, stop: the extension would be changing more than the query choice.
# Stage 2 (only if stage 1 passes): gamma in {0.8, 0.5, 0.2} x 20 languages, OAL, two jobs at
#          a time, then health check, scoring and comparison against baseline OAL.
#
# Resumable: finished runs (output/<run>/state_dict.pt) are skipped. Training checkpoints of a
# finished run are deleted (unused; the final model is output/<run>/state_dict.pt).
# Outputs: oalcf-main/output/<corpus>_oal_dq<100*gamma>/<lang>, archived to runs/dq_oal/output.
# Progress: runs/dq_oal/progress.log     Summary: runs/dq_oal/summary.txt
#
# Usage: bash tools/run_dq_oal_cpu.sh      (bash, not zsh)

ROOT="/Users/devrajsaini/Documents/BTP-1/Active Learning/oalcf-repro"
MAIN="$ROOT/oalcf-main"
OUT="$ROOT/runs/dq_oal"
BASE="$ROOT/runs/baseline_seed0/output"
source "$ROOT/venv/bin/activate"
caffeinate -ims -w $$ &
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"
PROG="$OUT/progress.log"
QUEUE="$OUT/.queue"
LOCK="$OUT/.queue.lock"
COMMON=(--env_name test --auto_weight --auto_mult 10 --min_delta 1e-3 --nworkers 0)

feat_root() {
  case $1 in
    ac)  echo "$ROOT/data/AfricanContinentLID/ecapalang/" ;;
    cr)  echo "$ROOT/data/CaucasusRegionLID/ecapalang/" ;;
    sea) echo "$ROOT/data/SoutheastAsiaLID/ecapalang/" ;;
    sa)  echo "$ROOT/data/SouthAsiaLID/ecapalang/" ;;
  esac
}
gtag() { case $1 in 1.0) echo dq100 ;; 0.8) echo dq080 ;; 0.5) echo dq050 ;; 0.2) echo dq020 ;; esac; }

run_job() {  # run_job <corpus> <lang> <gamma>
  local c=$1 l=$2 g=$3
  local run="${c}_oal_$(gtag "$g")/${l}"
  if [ -f "output/$run/state_dict.pt" ]; then
    echo "$(date '+%m-%d %H:%M') skip (already done): $run" >> "$PROG"; return
  fi
  mkdir -p "output/${c}_oal_$(gtag "$g")" "checkpoints/${c}_oal_$(gtag "$g")"
  echo "$(date '+%m-%d %H:%M') start: $run" >> "$PROG"
  local s=$(date +%s)
  python3 ../tools/run_cpu.py --script run_dq.py --dq_gamma "$g" --run_name "$run" --lid_target "$l" \
    --feat_root "$(feat_root "$c")" "${COMMON[@]}" > "$OUT/logs/${c}_oal_$(gtag "$g")_${l}.log" 2>&1
  local rc=$?
  rm -rf "checkpoints/$run"
  echo "$(date '+%m-%d %H:%M') done:  $run  exit $rc  $(( ($(date +%s) - s) / 60 )) min" >> "$PROG"
}

# ---------------- Stage 1: gamma = 1.0 must equal the baseline ----------------
echo "$(date '+%m-%d %H:%M') === stage 1: gamma=1.0 vs archived baseline OAL (hy-AM, yo, dv)" >> "$PROG"
( run_job cr hy-AM 1.0 ) & P1=$!
( run_job ac yo 1.0; run_job sa dv 1.0 ) & P2=$!
wait $P1 $P2
python3 - "$MAIN" "$BASE" >> "$PROG" 2>&1 <<'EOF'
import filecmp, sys, torch
M, B = sys.argv[1], sys.argv[2]
ok = True
for c, l in (("cr", "hy-AM"), ("ac", "yo"), ("sa", "dv")):
    new, base = f"{M}/output/{c}_oal_dq100/{l}", f"{B}/{c}_oal/{l}"
    same_scores = filecmp.cmp(f"{new}/scores.csv", f"{base}/scores.csv", shallow=False)
    a, b = torch.load(f"{new}/state_dict.pt", map_location="cpu"), torch.load(f"{base}/state_dict.pt", map_location="cpu")
    same_model = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ok &= same_scores and same_model
    print(f"  gamma=1.0 {c}/{l}: scores.csv identical={same_scores}  final weights identical={same_model}")
print("STAGE 1 PASSED" if ok else "STAGE 1 FAILED")
sys.exit(0 if ok else 1)
EOF
if [ $? -ne 0 ]; then
  echo "$(date '+%m-%d %H:%M') === stopping: gamma=1.0 does not reproduce the baseline; stage 2 NOT started" >> "$PROG"
  exit 1
fi

# ---------------- Stage 2: gamma in {0.8, 0.5, 0.2}, all 20 languages ----------------
rm -rf "$LOCK"; : > "$QUEUE"
for g in 0.8 0.5 0.2; do for l in hy-AM ckb cv kmr ky sr tt; do echo "cr $l $g" >> "$QUEUE"; done; done
for g in 0.8 0.5 0.2; do for l in yue id vi; do echo "sea $l $g" >> "$QUEUE"; done; done
for g in 0.8 0.5 0.2; do for l in cnh dv hi ml mr or skr; do echo "sa $l $g" >> "$QUEUE"; done; done
for g in 0.8 0.5 0.2; do for l in bas ha yo; do echo "ac $l $g" >> "$QUEUE"; done; done

pop() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  head -n 1 "$QUEUE"
  tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
  rmdir "$LOCK"
}
worker() { while :; do job=$(pop); [ -z "$job" ] && break; run_job $job; done; }

echo "$(date '+%m-%d %H:%M') === stage 2: $(wc -l < "$QUEUE" | tr -d ' ') runs (gamma 0.8/0.5/0.2 x 20 languages, OAL), 2 at a time" >> "$PROG"
worker & W1=$!
sleep 5
worker & W2=$!
wait $W1 $W2      # only the workers -- a bare `wait` would also wait on caffeinate (deadlock)
rm -f "$QUEUE" "$QUEUE.tmp"; rm -rf "$LOCK"

mkdir -p "$OUT/output"
for d in output/*_oal_dq*; do cp -R "$d" "$OUT/output/"; done
echo "$(date '+%m-%d %H:%M') === all runs finished and archived; summarizing" >> "$PROG"

python3 - "$ROOT" > "$OUT/summary.txt" 2>&1 <<'EOF'
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = sys.argv[1]; sys.path.insert(0, f"{ROOT}/tools")
from score_runs import score
A, B = f"{ROOT}/runs/dq_oal/output", f"{ROOT}/runs/baseline_seed0/output"
LANGS = {"ac": ["bas", "ha", "yo"], "cr": ["hy-AM", "ckb", "cv", "kmr", "ky", "sr", "tt"],
         "sea": ["yue", "id", "vi"], "sa": ["cnh", "dv", "hi", "ml", "mr", "or", "skr"]}
SESS = {"ac": 83, "cr": 159, "sea": 126, "sa": 95}
TAGS = {"0.8": "dq080", "0.5": "dq050", "0.2": "dq020"}
print("Diversity-penalized AL with OAL vs baseline OAL, seed 0, CPU\n")
print("== health")
bad = 0
for g, t in TAGS.items():
    for c, langs in LANGS.items():
        for l in langs:
            d = f"{A}/{c}_oal_{t}/{l}"; log = f"{ROOT}/runs/dq_oal/logs/{c}_oal_{t}_{l}.log"
            rows = sum(1 for _ in open(f"{d}/scores.csv")) - 1 if os.path.exists(f"{d}/scores.csv") else 0
            txt = open(log, errors="ignore").read().replace("\r", "\n") if os.path.exists(log) else ""
            nan = sum(1 for x in txt.splitlines() if "test/loss" in x and "nan" in x.lower()) // 2
            ok = rows == SESS[c] and nan == 0 and "used: False" in txt and os.path.exists(f"{d}/state_dict.pt")
            if not ok:
                bad += 1; print(f"  CHECK gamma={g} {c}/{l}: sessions {rows}/{SESS[c]}, NaN {nan}")
print(f"  {60 - bad}/60 runs OK\n")
print("== each gamma vs baseline OAL over 20 languages (one-sided Wilcoxon: gamma lower than baseline;"
      " with 3 gammas, significant only if p < 0.0167)")
rows = []
for g, t in TAGS.items():
    for metric, key in (("IMLM Eq.1 (authors)", "IMLM_eq1"), ("IMLM repo", "IMLM_repo"), ("DCF", "DCF")):
        base, new = [], []
        for c, langs in LANGS.items():
            for l in langs:
                base.append(score(f"{B}/{c}_oal/{l}")[key]); new.append(score(f"{A}/{c}_oal_{t}/{l}")[key])
        base, new = np.array(base), np.array(new)
        p = wilcoxon(new, base, alternative="less").pvalue if np.any(new != base) else 1.0
        print(f"  gamma={g}  {metric:20s} better in {int((new < base).sum()):2d}/20, worse in {int((new > base).sum()):2d}/20 | "
              f"mean {base.mean():.4f} -> {new.mean():.4f} | p = {p:.4f}{'  *' if p < 0.0167 else ''}")
print("\n== how much each gamma changed the selection (al_diversity.csv, mean per session)")
for g, t in TAGS.items():
    df = pd.concat([pd.read_csv(f"{A}/{c}_oal_{t}/{l}/al_diversity.csv") for c in LANGS for l in LANGS[c]])
    print(f"  gamma={g}: queries shared with plain uncertainty top-8 {df.overlap_with_runpy.mean():.2f}/8 | "
          f"distinct clusters among the 8 {df.distinct_clusters_picked.mean():.2f} | clusters per session {df.n_clusters.mean():.1f}")
print("\n== per language, IMLM Eq.1 (authors' reading): baseline OAL -> gamma 0.8 / 0.5 / 0.2")
for c, langs in LANGS.items():
    for l in langs:
        b = score(f"{B}/{c}_oal/{l}")["IMLM_eq1"]
        v = [score(f"{A}/{c}_oal_{t}/{l}")["IMLM_eq1"] for t in TAGS.values()]
        print(f"  {c:4s}{l:6s} {b:.4f} -> " + " / ".join(f"{x:.4f}" for x in v))
EOF
echo "$(date '+%m-%d %H:%M') === complete: see runs/dq_oal/summary.txt" >> "$PROG"
