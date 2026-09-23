# -*- coding: utf-8 -*-
"""
fig4_resolution.py  ->  figures/fig4_resolution.pdf   (paper Fig. 4)

Probability of resolution versus angular separation (1..10 deg), for SNR 0 dB
(left) and 5 dB (right).  The first source is fixed at -48.1 deg.  A trial is
counted as resolved if two peaks straddle the true midpoint and the summed angle
error is below the separation.  All methods use the SAME data.

The whole pipeline runs every time:
  for each (SNR, separation): generate data -> estimate -> test resolution.

Run:  CUDA_VISIBLE_DEVICES="" python code/figures/fig4_resolution.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (COLOR, MARKER, OUTDIR, N_MC, gen_cov, est_dl, est_music, resolved)

FIRST = -48.1
SEPS = list(range(1, 11))
SNRS = [0, 5]
METHODS = ["Sigmoid", "Gaussian", "0-1 hard", "BCE", "MUSIC"]

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, snr in zip(axes, SNRS):
    curves = {m: [] for m in METHODS}
    for sep in SEPS:
        doas = np.asarray([FIRST, FIRST + sep], float)
        cov = gen_cov(doas, snr, n_mc=N_MC)
        for m in METHODS:
            est = est_music(cov) if m == "MUSIC" else est_dl(m, cov)
            ok = sum(resolved(est[i], doas) for i in range(est.shape[0]))
            curves[m].append(ok / est.shape[0])
    for m in METHODS:
        ax.plot(SEPS, curves[m], color=COLOR[m], marker=MARKER[m], lw=2.4, ms=8, label=m)
    ax.set_xlabel("Angular separation (deg)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_xticks(SEPS)
    ax.set_ylim(-0.05, 1.09); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, prop={"size": 11, "weight": "bold"})
    ax.set_title(f"SNR = {snr} dB", fontsize=13)
for ax in axes:
    ax.set_ylabel("Probability of DOA resolution", fontsize=15, fontweight="bold")
for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.text(0.5, -0.22, tag, transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(OUTDIR / "fig4_resolution.pdf")
fig.savefig(OUTDIR / "fig4_resolution.png", dpi=300)
print("saved", OUTDIR / "fig4_resolution.pdf")
