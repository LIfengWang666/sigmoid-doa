# -*- coding: utf-8 -*-
"""
fig3_rmse.py  ->  figures/fig3_rmse.pdf   (paper Fig. 3)

RMSE versus SNR for a small (2 deg, left) and a large (20 deg, right) separation,
comparing Sigmoid / Gaussian / 0-1 hard / BCE / MUSIC.

The whole pipeline runs every time:
  for each (angle-pair, SNR): generate data -> estimate with each method -> RMSE.
All methods use the SAME data.

Run:  CUDA_VISIBLE_DEVICES="" python code/figures/fig3_rmse.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (COLOR, MARKER, LS, OUTDIR, N_MC,
                    gen_cov, est_dl, est_music, rmse)

SNRS = [-20, -15, -10, -5, 0, 5]
PAIRS = [([-48.1, -46.1], "(a)  2 deg"), ([-48.1, -28.1], "(b)  20 deg")]
METHODS = ["Sigmoid", "Gaussian", "0-1 hard", "BCE", "MUSIC"]

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (doas, tag) in zip(axes, PAIRS):
    doas = np.asarray(doas, float)
    curves = {m: [] for m in METHODS}
    for snr in SNRS:                         # generate + estimate per SNR
        cov = gen_cov(doas, snr, n_mc=N_MC)
        for m in METHODS:
            est = est_music(cov) if m == "MUSIC" else est_dl(m, cov)
            curves[m].append(rmse(est, doas))
    for m in METHODS:
        ax.semilogy(SNRS, curves[m], color=COLOR[m], ls=LS[m],
                    marker=MARKER[m], lw=2.2, ms=7, label=m)
    ax.set_xlabel("SNR (dB)", fontsize=15, fontweight="bold")
    ax.set_ylabel("RMSE of DOA estimates (deg)", fontsize=15, fontweight="bold")
    ax.set_xticks(SNRS); ax.set_ylim(0.03, 100)
    ax.grid(True, which="both", ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, prop={"size": 11, "weight": "bold"})
    ax.set_title(tag, fontsize=13)

fig.tight_layout()
fig.savefig(OUTDIR / "fig3_rmse.pdf")
fig.savefig(OUTDIR / "fig3_rmse.png", dpi=300)
print("saved", OUTDIR / "fig3_rmse.pdf")
