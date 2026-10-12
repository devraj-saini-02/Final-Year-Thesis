#!/usr/bin/env python3
"""
Equivalence harness: run a given oalcf source tree's run.py under an IDENTICAL
environment, so the only variable between two runs is the source code itself.

Used to check that our modified tree implements the same OAL / OAL-CF algorithm
as the author's unmodified upstream tree.

Shims applied identically to every tree (none of them edit a source file):
  - Tensor.cuda / Module.cuda -> no-op   this machine has no CUDA, and upstream
                                         utils/ddm.py calls .cuda()
                                         unconditionally every session
  - MPS hidden                           forces every get_device() to CPU, so
                                         both trees run on the same backend
  - Lightning Trainer forced to cpu, 1   upstream passes devices=[0]
  - torch.set_flush_denormal(True)       our tree sets it (paper Sec. V-D);
                                         upstream does not.  Applied to both so
                                         it is not a variable in the comparison.

Usage: eqrun.py <tree_dir> <run.py args...>
"""
import os
import sys
import runpy

tree = os.path.abspath(sys.argv[1])
args = sys.argv[2:]

import torch
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.backends.mps.is_available = lambda: False
torch.set_flush_denormal(True)

import pytorch_lightning as pl
_Base = pl.Trainer


class CpuTrainer(_Base):
    def __init__(self, *a, **k):
        k['accelerator'] = 'cpu'
        k['devices'] = 1
        super().__init__(*a, **k)


pl.Trainer = CpuTrainer

os.chdir(tree)
sys.path.insert(0, tree)
sys.argv = [os.path.join(tree, 'run.py')] + args
runpy.run_path(os.path.join(tree, 'run.py'), run_name='__main__')
