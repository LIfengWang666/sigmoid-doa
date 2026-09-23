# -*- coding: utf-8 -*-
"""
fig1_label.py  ->  figures/fig1_label.pdf   (paper Fig. 1)

The Sigmoid label (a, before derivative) and its first derivative (b, after
derivative) for two sources separated by 2 deg.  Pure synthetic -- no model.
Run:  python code/figures/fig1_label.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).resolve().parents[2] / "figures"
OUTDIR.mkdir(exist_ok=True)

W = 0.2                          # sigmoid transition width (deg)
DOAS = [-43.7, -41.7]            # two sources, 2 deg apart

th = np.arange(-60, 60, 0.005)
y = np.zeros_like(th)
for d in DOAS:
    y += 1.0 / (1.0 + np.exp(-(th - d) / W))
y /= len(DOAS)
dy = np.gradient(y, th)

# normalize each curve by its maximum inside the shown window -> peaks reach 1
XLO, XHI = -47.0, -38.0
win = (th >= XLO) & (th <= XHI)
y = y / y[win].max()
dy = dy / dy[win].max()

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

axes[0].plot(th, y, color="#2E86C1", lw=2.4)
axes[0].set_xlabel("Angle (deg)", fontsize=15, fontweight="bold")
axes[0].set_ylabel("Label value", fontsize=15, fontweight="bold")

axes[1].plot(th, dy, color="#C0392B", lw=2.4)
axes[1].set_xlabel("Angle (deg)", fontsize=15, fontweight="bold")
axes[1].set_ylabel("Derivative", fontsize=15, fontweight="bold")

for ax in axes:
    for d in DOAS:
        ax.axvline(d, color="k", ls=":", alpha=0.6, lw=1.1)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.text(0.5, -0.24, tag, transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(OUTDIR / "fig1_label.pdf")
fig.savefig(OUTDIR / "fig1_label.png", dpi=300)
print("saved", OUTDIR / "fig1_label.pdf")
