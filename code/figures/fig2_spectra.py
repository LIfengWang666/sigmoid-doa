# -*- coding: utf-8 -*-
"""
fig2_spectra.py  ->  figures/fig2_spectra.pdf   (paper Fig. 2)

Output spectra at a 2 deg separation (two sources at -48.1 and -46.1 deg, SNR 5 dB):
(a) Gaussian spectrum vs Sigmoid derivative;  (b) 0-1 hard, BCE, MUSIC.

The whole pipeline is run every time (so it is transparent to the reader):
  1. generate the test data            (common.gen_cov)
  2. forward each trained model        (common.model_raw)
  3. Sigmoid read-out: monotonicity + first derivative
  4. classical MUSIC for reference
  5. plot and save.

Run:  CUDA_VISIBLE_DEVICES="" python code/figures/fig2_spectra.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from common import (COLOR, LS, OUTDIR, D, LAMBDA, ARRAY_NUM,
                    gen_cov, model_raw)

DOAS = [-48.1, -46.1]
SNR = 5

# 1. data (a single trial is enough to draw the spectra)
cov = gen_cov(DOAS, SNR, n_mc=1)

# 2. forward the four trained models
raw = {n: model_raw(n, cov)[0] for n in ["Gaussian", "Sigmoid", "0-1 hard", "BCE"]}

# 3. Sigmoid read-out: monotonize then take the first derivative
p = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(np.arange(480), raw["Sigmoid"])
sigmoid_deriv = np.clip(np.gradient(p, 0.25), 0, None)

# 4. MUSIC spectrum
th = -60 + np.arange(480) * 0.25
A = np.exp(-1j * 2 * np.pi / LAMBDA * np.outer(D, np.sin(np.deg2rad(th))))
w, v = np.linalg.eigh(cov[0]); En = v[:, :ARRAY_NUM - 2]
music = 1.0 / (np.einsum("gn,gh,hn->n", A.conj(), En @ En.conj().T, A).real + 1e-12)

th480 = -60 + np.arange(480) * 0.25
th121 = -60 + np.arange(121) * 1.0


def norm(x):
    x = np.clip(x, 0, None)
    return (x - x.min()) / (np.ptp(x) + 1e-12)


def norm_max(x):
    x = np.clip(x, 0, None)
    return x / (x.max() + 1e-12)


def draw(ax, items, normf=norm):
    xlo, xhi = min(DOAS) - 8, max(DOAS) + 8
    for label, thx, spec, key in items:
        spec = np.clip(np.asarray(spec, float), 0, None)
        win = (thx >= xlo) & (thx <= xhi)          # normalise by the max inside the shown window
        y = normf(spec) if win.all() else spec / (spec[win].max() + 1e-12)
        ax.plot(thx, y, color=COLOR[key], ls=LS[key], lw=2.0, label=label)
    for t in DOAS:
        ax.axvline(t, color="k", ls=":", alpha=0.5, lw=1.1)
    ax.set_xlabel("Angular separation (deg)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Normalized Amplitude", fontsize=15, fontweight="bold")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, prop={"size": 11, "weight": "bold"})


# 5. plot
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw(axes[0], [("Gaussian", th480, raw["Gaussian"], "Gaussian"),
               ("Sigmoid (deriv.)", th480, sigmoid_deriv, "Sigmoid")], normf=norm_max)
draw(axes[1], [("0-1 hard", th121, raw["0-1 hard"], "0-1 hard"),
               ("BCE", th121, raw["BCE"], "BCE"),
               ("MUSIC", th480, music, "MUSIC")], normf=norm_max)
for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.text(0.5, -0.22, tag, transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTDIR / "fig2_spectra.pdf")
fig.savefig(OUTDIR / "fig2_spectra.png", dpi=300)
print("saved", OUTDIR / "fig2_spectra.pdf")
