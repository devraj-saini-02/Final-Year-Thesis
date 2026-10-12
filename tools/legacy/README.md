# tools/legacy/ — historical batch scripts. Do not use for new work.

Each script here ran one specific batch and is kept only because a published result refers to it by
name (`runs/reports/*RESULTS.md`). **Use `bash tools/run_experiment.sh` instead**; it does the
same things with one set of options, and it skips runs that already exist.

They also write to (and read from) the pre-2026-09-18 folder layout, which no longer exists:
results now live in `runs/seed_<N>/<setting>/<corpus>_<lang>/`. So they would need their paths fixed
before they could run at all. `summarize_dq_oalcf.py` here has the same problem.

Two things to know if you ever run one of these anyway:

1. **They are all hard-wired to seed 0** (except `run_baselines_seeds_cpu.sh`). They never pass
   `--order_file`, so they re-run the stream order that is already finished and archived.
2. **They write to the old folders** (`runs/dq_oal/`, `runs/dq_oalcf_thresh/`, ...), not to
   `runs/experiments/`. A run that lands in both places is reported as a duplicate by
   `python tools/experiments.py`, and the comparison tool uses whichever it finds first.

Worth knowing about, independent of the batches:

- `repro_mps_batchnorm_grad.py` — the minimal demonstration that Apple's GPU backend computes wrong
  gradients for this model. The reason everything runs on CPU.
- `verify_g1.sh` — checks that the extension with gamma = 1.0 reproduces the baseline bit for bit.
- `run_baselines_seeds_cpu.sh` — produced `runs/baseline_seeds/` (seeds 1-4, OAL).
