# -*- coding: utf-8 -*-
"""
common.py -- shared utilities for reproducing the paper figures.

Everything is CPU-only by default (set CUDA_VISIBLE_DEVICES="" for CPU).
All data are generated on the fly; the only external files are the four
pre-trained checkpoints under ../weights/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks as scipy_fp
from sklearn.isotonic import IsotonicRegression

# make `code/` importable (partricnntran.py, toolbox.py live there)
ROOT = Path(__file__).resolve().parents[2]        # repo root
CODE = Path(__file__).resolve().parents[1]        # code/
sys.path.insert(0, str(CODE))
from partricnntran import partricnntran           # noqa: E402
from toolbox import batch_extract_upper_triangle  # noqa: E402

WEIGHTS = ROOT / "weights"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True)

# ---------------- fixed simulation settings ----------------
ARRAY_NUM = 16
LAMBDA = 0.3
D = np.arange(ARRAY_NUM) * LAMBDA / 2             # half-wavelength ULA
N_SNAP = 1000                                     # snapshots
N_MC = 1000                                       # Monte-Carlo trials

# method name -> (case in partricnntran, checkpoint folder, grid step, grid size)
METHODS = {
    "Sigmoid":  ("StairCase", "exp-staircase_w02_grid025_480_snap1000_2src", 0.25, 480),
    "Gaussian": ("Gauss",     "exp-gaussian_grid025_480_snap1000_2src",      0.25, 480),
    "0-1 hard": ("Hard",      "exp-hardlabel_mse_nosig_snap1000_2src",       1.0,  121),
    "BCE":      ("BCE",       "exp-hardlabel_snap1000_2src",                 1.0,  121),
}
COLOR = {"Sigmoid": "#C0392B", "Gaussian": "#2E86C1",
         "0-1 hard": "#28B463", "BCE": "#E67E22", "MUSIC": "#7D3C98"}
MARKER = {"Sigmoid": "o", "Gaussian": "s", "0-1 hard": "^",
          "BCE": "D", "MUSIC": "*"}
LS = {"Sigmoid": "-", "Gaussian": "--", "0-1 hard": "-.", "BCE": ":", "MUSIC": "-"}


# ---------------- data generation ----------------
def gen_cov(doas, snr_db, n_mc=N_MC):
    """Generate n_mc sample covariance matrices for sources at `doas` (deg).

    Mirrors the original data generator (val_data_gen.py): both random-number
    generators are created ONCE, OUTSIDE the Monte-Carlo loop --
      * the source matrix is drawn with a fixed seed (10),
      * the noise generator is seeded with `sig_num*N_SNAP + 10`,
    and the loop advances the same generator (it is not re-seeded per trial).
    The function is called inside the SNR loop of the figure scripts.
    """
    doas = np.asarray(doas, dtype=float)
    sig_num = len(doas)
    rad = np.deg2rad(doas)
    A = np.exp(-1j * 2 * np.pi / LAMBDA * D.reshape(-1, 1) * np.sin(rad))

    rng = np.random.default_rng(10)                       # source (fixed seed, once)
    S = rng.standard_normal((sig_num, N_SNAP)) + 1j * rng.standard_normal((sig_num, N_SNAP))
    S *= (np.sqrt(10 ** (snr_db / 10)) / np.linalg.norm(S, axis=1, keepdims=True))

    rng = np.random.default_rng(sig_num * N_SNAP + 10)    # noise (seeded once, outside the loop)
    cov = np.zeros((n_mc, ARRAY_NUM, ARRAY_NUM), complex)
    for i in range(n_mc):
        rp = rng.standard_normal((ARRAY_NUM, N_SNAP)); rp /= np.linalg.norm(rp, axis=1, keepdims=True) * np.sqrt(2)
        ip = rng.standard_normal((ARRAY_NUM, N_SNAP)); ip /= np.linalg.norm(ip, axis=1, keepdims=True) * np.sqrt(2)
        X = A @ S + (rp + 1j * ip)
        cov[i] = X @ X.T.conj() / N_SNAP
    return cov


# ---------------- model loading / inference ----------------
_MODELS = {}


def load_model(name):
    if name not in _MODELS:
        case, folder, _, _ = METHODS[name]
        m = partricnntran(case1=case)
        m.load_state_dict(torch.load(WEIGHTS / folder / "best_val.pth", map_location="cpu"))
        m.eval()
        _MODELS[name] = m
    return _MODELS[name]


def _to_input(cov):
    R = np.stack([np.real(cov), np.imag(cov), np.angle(cov)], -1)
    t = torch.tensor(R, dtype=torch.float32).permute(0, 3, 1, 2)
    return batch_extract_upper_triangle(t).view(cov.shape[0], 3, 1, -1)


def model_raw(name, cov):
    with torch.no_grad():
        return load_model(name)(_to_input(cov)).numpy()


# ---------------- read-outs ----------------
def _top2(peaks, values):
    order = peaks[np.argsort(-values[peaks])][:2]
    return np.sort(-60 + order)  # returns grid indices here; caller maps to angles


def est_from_spec(name, spec):
    """spec: spectrum over the method's grid -> two sorted angles (deg)."""
    _, _, step, ng = METHODS[name]
    dmin = max(1, int(round(1.0 / step)))
    if name == "Sigmoid":
        pk = scipy_fp(spec, height=0.05, distance=dmin)[0]   # derivative, 0..1
    else:
        pk = scipy_fp(spec, distance=dmin)[0]                # raw model output
    order = pk[np.argsort(-spec[pk])][:2]
    a = np.sort(-60 + order * step)
    if len(a) >= 2:
        return np.array([a[0], a[1]])
    if len(a) == 1:
        return np.array([a[0], 100.0])
    return np.array([100.0, 100.0])


def est_dl(name, cov):
    """Estimate DOAs with a trained model (+ method-specific read-out)."""
    out = model_raw(name, cov)
    est = np.zeros((out.shape[0], 2))
    for i, row in enumerate(out):
        if name == "Sigmoid":
            p = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(np.arange(len(row)), row)
            spec = np.clip(np.gradient(p, METHODS[name][2]), 0, None)
        else:
            spec = row
        est[i] = est_from_spec(name, spec)
    return est


def est_music(cov):
    step = 0.25
    th = -60 + np.arange(480) * step
    A = np.exp(-1j * 2 * np.pi / LAMBDA * np.outer(D, np.sin(np.deg2rad(th))))
    est = np.zeros((cov.shape[0], 2))
    for i in range(cov.shape[0]):
        w, v = np.linalg.eigh(cov[i])
        En = v[:, :ARRAY_NUM - 2]
        P = 1.0 / (np.einsum("gn,gh,hn->n", A.conj(), En @ En.conj().T, A).real + 1e-12)
        est[i] = est_from_spec("Gaussian", P)   # 0.25 grid, local-maxima read-out
    return est


def rmse(est, doas):
    return float(np.sqrt(np.mean((est - np.asarray(doas)) ** 2)))


def resolved(est_row, doas):
    """Resolution criterion: two peaks straddling the midpoint and err_sum < sep."""
    doas = np.asarray(doas)
    if not np.all(est_row < 90):           # a failed/missing peak
        return False
    mid = doas.mean()
    sep = abs(doas[1] - doas[0])
    return (est_row[0] < mid < est_row[1]) and (abs(est_row[0] - doas[0]) + abs(est_row[1] - doas[1]) < sep)
