#!/usr/bin/env python3
"""
Run oalcf-main/run.py with EVERY device request resolved to CPU.

Why: on PyTorch 2.3.0 the MPS backend computes wrong weight/bias gradients for a
BatchNorm1d whose input does not require grad. The model's first layer (bn_layer, on
the raw ECAPA features) is exactly that case, so every MPS training step feeds that
layer a garbage gradient, and in OAL-CF it eventually overflows to NaN. The CPU kernel
computes the gradient correctly. See docs/other/DEVIATIONS.md.

`--gpus 0` alone is not enough: it moves only the Lightning Trainer to CPU, while the
query-strategy, corrective-feedback and drift helpers call utils.device.get_device()
with no argument and would still pick MPS. This wrapper replaces get_device before
run.py (and anything it imports) binds it, then appends `--gpus 0` so command.txt
records the CPU run. No repo code is modified.

Usage, from oalcf-main/ (run.py writes output/ and checkpoints/ relative to cwd):
    python ../tools/run_cpu.py <run.py args...>
    python ../tools/run_cpu.py --stop_after N <run.py args...>   # short tests only
    python ../tools/run_cpu.py --script run_dq.py --dq_gamma 0.5 <run.py args...>

--script runs another entry point that wraps run.py (the diversity-penalized AL extension,
run_dq.py, or the CF deduplication extension, run_cfdedup.py). Those strip their own --dq_* /
--cfd_* options before run.py writes command.txt, so they are appended to command.txt after the run.
"""
import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
import utils.device as device  # noqa: E402

device.get_device = lambda use_gpu=True: torch.device("cpu")

args = sys.argv[1:]
script = "run.py"
if "--script" in args:
    i = args.index("--script")
    script = args[i + 1]
    del args[i:i + 2]
stop_after = None
if "--stop_after" in args:
    i = args.index("--stop_after")
    stop_after = int(args[i + 1])
    del args[i:i + 2]

if stop_after is not None:
    import src.module as module  # noqa: E402

    def on_train_end(self):
        if int(self.trainer.datamodule.current_batch) > stop_after:
            os._exit(0)

    module.VtdModule.on_train_end = on_train_end

sys.argv = [script] + args + ["--gpus", "0"]
runpy.run_path(script, run_name="__main__")

if script != "run.py" and "--run_name" in args:
    prefixes = ("--dq_", "--cfd_")
    wrapper_opts = [a for j, a in enumerate(args)
                    if a.startswith(prefixes) or (j > 0 and args[j - 1].startswith(prefixes))]
    cmd_file = os.path.join("output", args[args.index("--run_name") + 1], "command.txt")
    with open(cmd_file, "a") as f:
        f.write(f"\n# via {script}: {' '.join(wrapper_opts)}\n")
