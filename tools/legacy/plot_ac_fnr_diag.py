#!/usr/bin/env python3
"""Diagnostic: cumulative FNR by session for AC OAL runs (bas/ha/yo).

Investigates why bas OAL FNR (0.3880) and yo OAL FNR (0.3992) land within 3%
of each other -- front-loaded cold-start vs. session-burst hypothesis.
Reads archived runs/ac_oal/{lang}/scores.csv, no new runs required.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LANGS = ["bas", "ha", "yo"]
COLORS = {"bas": "#4C72B0", "ha": "#DD8452", "yo": "#55A868"}

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for lang in LANGS:
    df = pd.read_csv(f"runs/ac_oal/{lang}/scores.csv")
    cum_fns = df["fns"].cumsum()
    cum_ps = df["ps"].cumsum().replace(0, np.nan)
    cum_fnr = cum_fns / cum_ps
    axes[0].plot(df["pass"], cum_fnr, label=f"{lang} OAL", color=COLORS[lang], linewidth=2)
    axes[1].plot(df["pass"], df["ps"], label=f"{lang} target count/session", color=COLORS[lang], linewidth=1.5)

axes[0].set_ylabel("Cumulative postquential FNR")
axes[0].set_title("AC OAL: cumulative FNR by session (bas/yo flatline to 0 error after their burst session)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_ylabel("Target (ps) samples in session")
axes[1].set_xlabel("Session (pass)")
axes[1].set_title("Target-class samples per session -- shows the Dirichlet burst")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = "runs/ac_fnr_burst_diagnostic.png"
plt.savefig(out, dpi=140)
print("saved", out)
