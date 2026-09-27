# CNN-Transformer with Sigmoid Label for High-Resolution Sub-Degree DOA Estimation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LIfengWang666/sigmoid-doa/HEAD?filepath=code/figures/figures.ipynb)

**▶ Run online (no setup):** click the **Binder** badge above, then
**Run ▸ Run All Cells** in the notebook `code/figures/figures.ipynb`.
Binder builds the environment automatically and shows all four figures.

> To activate your own Binder link, replace `USERNAME/REPO` in the badge URL
> above (and in the line below) with your GitHub user and repository name.
> This repository: `https://mybinder.org/v2/gh/LIfengWang666/sigmoid-doa/HEAD?filepath=code/figures/figures.ipynb`

Code to reproduce the figures of the paper. Every figure is produced by running
the **full pipeline** (data generation → trained models → read-out → plot), so
the reader can follow the logic end to end.

## 1. Get the code

Clone the repository (or click **Code ▸ Download ZIP** on the GitHub page):

```
git clone https://github.com/LIfengWang666/sigmoid-doa.git
cd sigmoid-doa
```

## 2. Install the dependencies

```
pip install -r requirements.txt
```

## 3. Reproduce each figure (one command each)

Run from the **`code/figures/`** directory:

| Paper figure | Command | Output |
|---|---|---|
| **Fig. 1** – Sigmoid label & derivative | `python fig1_label.py` | `figures/fig1_label.pdf` |
| **Fig. 2** – Spatial spectra (2°) | `python fig2_spectra.py` | `figures/fig2_spectra.pdf` |
| **Fig. 3** – RMSE vs SNR (2° / 20°) | `python fig3_rmse.py` | `figures/fig3_rmse.pdf` |
| **Fig. 4** – Resolution (0 / 5 dB) | `python fig4_resolution.py` | `figures/fig4_resolution.pdf` |

Each script prints the path of the saved PDF (a `.png` is saved too).
No arguments, no configuration. To force CPU add the prefix
`CUDA_VISIBLE_DEVICES=""`. Fig. 3 and Fig. 4 run 1000 Monte-Carlo trials per
point, so they take a few minutes on CPU.

## 4. Online run on Binder (no installation)

Click the Binder badge at the top. It opens `code/figures/figures.ipynb`; choose
**Run All Cells**. The notebook runs the four scripts and displays the resulting
figures inline.

## 5. The pipeline in each script (so nothing is hidden)

Every `figN_*.py` is short and self-contained; it does:

1. **generate** the test data (`common.gen_cov`),
2. **forward** the trained model(s) (`common.model_raw`),
3. **read out** the DOAs (`common.est_dl`: for the Sigmoid label, monotonicity +
   first derivative + top-2 peaks; `common.est_music` for the MUSIC reference),
4. **plot** and save.

`fig1_label.py` is pure synthetic (it shows the label and its derivative; no model).

## Repository layout

```
.
├── README.md
├── requirements.txt              # numpy, scipy, scikit-learn, matplotlib, torch, einops, timm
├── runtime.txt                   # python-3.10 (used by Binder)
├── weights/                      # 4 pre-trained checkpoints (best_val.pth)
│   ├── exp-staircase_w02_grid025_480_snap1000_2src/
│   ├── exp-gaussian_grid025_480_snap1000_2src/
│   ├── exp-hardlabel_mse_nosig_snap1000_2src/
│   └── exp-hardlabel_snap1000_2src/
├── figures/                      # output (created when you run the scripts)
└── code/
    ├── partricnntran.py          # CNN-Transformer backbone (HMC-ViT)
    ├── toolbox.py                # upper-triangle extraction / helpers
    ├── val_data_gen.py           # reference data generator
    ├── clean_scratch.py          # one-off: remove exploratory scripts
    ├── train_sigmoid/            # training code, Sigmoid label (MSE loss)
    ├── train_gaussian/           # training code, Gaussian label (MSE loss)
    ├── train_nosig/              # training code, 0-1 hard label (MSE loss)
    └── figures/
        ├── common.py             # shared: data gen + model loading + read-outs
        ├── figures.ipynb         # Binder notebook (Run All)
        ├── fig1_label.py
        ├── fig2_spectra.py
        ├── fig3_rmse.py
        └── fig4_resolution.py
```

## Simulation settings (fixed inside `common.py`)

| Item | Value |
|---|---|
| Array | ULA, M = 16, half-wavelength |
| Wavelength | λ = 0.3 m |
| Snapshots | N = 1000 |
| Monte-Carlo trials | 1000 (Fig. 3, 4) |
| Sigmoid label width | w = 0.2° |
| Grid (Sigmoid / Gaussian) | 0.25° (480 points) |
| Grid (0-1 hard / BCE) | 1° (121 points) |
| Backbone | CNN-Transformer (HMC-ViT), 8 class tokens |
