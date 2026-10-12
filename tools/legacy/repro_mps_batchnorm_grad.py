#!/usr/bin/env python3
"""
Minimal repro for the reason all training runs on CPU (docs/other/DEVIATIONS.md, "Training device").

On PyTorch 2.3.0 with the MPS backend, BatchNorm1d's weight and bias gradients are wrong
when the BatchNorm's input does not require grad. That is exactly the model's first layer
(`bn_layer`, applied to the raw ECAPA features). CPU computes them correctly, and setting
requires_grad on the input makes MPS correct too.

The network below is random data through BN -> Linear -> BN -> LeakyReLU -> Linear; nothing
from the repo is used. The true first-layer bias gradient is ~1e-10 (the next BatchNorm
cancels any constant shift of its input).

Usage: python tools/repro_mps_batchnorm_grad.py
"""
import copy

import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS not available -- nothing to reproduce on this machine")

torch.manual_seed(0)
net0 = torch.nn.Sequential(
    torch.nn.BatchNorm1d(512), torch.nn.Linear(512, 128),
    torch.nn.BatchNorm1d(128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 16))


def first_bn_grads(x, dev, input_requires_grad):
    net = copy.deepcopy(net0).to(dev).train()
    xd = x.detach().clone().to(dev).requires_grad_(input_requires_grad)
    (net(xd) ** 2).mean().backward()
    return net[0].bias.grad.cpu(), net[0].weight.grad.cpu()


print("torch", torch.__version__)
for req in (False, True):
    wrong = 0
    for i in range(30):
        x = torch.randn(256, 512, generator=torch.Generator().manual_seed(100 + i)) * 4 + 2
        cb, cw = first_bn_grads(x, "cpu", req)
        mb, mw = first_bn_grads(x, "mps", req)
        ok = torch.allclose(mb, cb, atol=1e-6) and torch.allclose(mw, cw, rtol=1e-3, atol=1e-6)
        wrong += not ok
    print(f"input.requires_grad={req}: MPS first-BN gradients wrong in {wrong}/30 calls "
          f"(last call: |bias grad| CPU {cb.abs().max():.1e}, MPS {mb.abs().max():.1e})")
