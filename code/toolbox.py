## Contains utility functions
## For tasks such as calculating errors, finding peaks, etc.
## Xie Yuxuan 2024-08-322
import numpy as np
import torch
import torch.nn as nn

## Calculate the error for a single estimation
## This error calculation is partial; further processing is needed to obtain RMSE
## estimate_doa: estimated DOA; real_doa: actual DOA
## Returns RMSE based on peak calculation
## Xie Yuxuan 2024-03-30
def get_MSE(estimate_doa, real_doa):

    inner_real_doa = np.sort(real_doa)
    inner_estimate_doa = np.sort(estimate_doa)
    squared_diff = [(x - y) ** 2 for x, y in zip(inner_estimate_doa, inner_real_doa)]
    MSE = sum(squared_diff) / len(real_doa)  # Divide by the number of real DOA

    return MSE


## Search for a specified number of strong peaks in a given spectrum
## spectrum - input spectrum; sig_num - number of peaks to search for
## sig_num defaults to None; in this case, all spectrum peak indices are returned
## Returns - indices of the strongest peaks
def find_peaks(spectrum, sig_num=None):
    peaks = []
    peaks_value = []

    # Handle cases where the spectrum peaks are at the edges
    if spectrum[0] > spectrum[1]: 
        peaks.append(0)
        peaks_value.append(spectrum[0])
    if spectrum[len(spectrum)-1] > spectrum[len(spectrum) - 2]: 
        peaks.append(len(spectrum)-1)
        peaks_value.append(spectrum[len(spectrum)-1])

    # Handle cases where spectrum peaks are in the middle
    # Traverse the spatial spectrum and find elements larger than both of their neighbors, considering them as peaks
    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
            peaks.append(i)
            peaks_value.append(spectrum[i])

    peaks = np.array(peaks)[np.argsort(peaks_value)[::-1]]
    #plt.figure(figsize=(8, 8))
    #plt.plot(spectrum)
    #plt.savefig("result.png", dpi=300, bbox_inches="tight")
    #Determine whether to return all peaks
    if sig_num is not None:
        top_n_peaks = peaks[: sig_num]  # Take the top n peaks
        return top_n_peaks
    
    return peaks


## Template-fit read-out for cumulative-staircase labels (PRIMARY read-out).
## Least-squares fit of the ideal two-step staircase to the monotonized prediction:
## coarse O(G^2) search over all grid angle pairs via the Gram identity, then a joint
## 2-D fine search in a +/-fine_range deg window at fine_step resolution. Returns TWO
## sorted DOA ANGLES (degrees), NOT grid indices. Level-crossing is biased inward at
## small source separation, so template-fit is used instead for the go/no-go numbers.

# module-level cache for the template basis (depends only on w and the grid, not pred)
_TEMPLATE_CACHE = {}

def _template_basis(w, num_grid, doa_min, doa_max):
    key = (round(float(w), 6), int(num_grid), float(doa_min), float(doa_max))
    if key not in _TEMPLATE_CACHE:
        theta_g = (doa_min + np.arange(num_grid)).astype(np.float64)          # grid angles
        L = 1.0 / (1.0 + np.exp(-(theta_g[:, None] - theta_g[None, :]) / w))  # (G,G)
        Gram = L.T @ L
        _TEMPLATE_CACHE[key] = (theta_g, L, Gram, np.diag(Gram).copy())
    return _TEMPLATE_CACHE[key]

def template_fit(spectrum, sig_num=None, w=0.5, doa_min=-60.0, doa_max=60.0,
                 num_grid=121, fine_range=1.5, fine_step=0.02):
    p = np.maximum.accumulate(np.asarray(spectrum, dtype=np.float64))  # enforce monotone
    theta_g, L, Gram, diagG = _template_basis(w, num_grid, doa_min, doa_max)

    # exact LS objective for template (a,b): ||p||^2 + d_a + d_b + 0.5*Gram_ab
    c = p @ L                                       # (G,)
    d = -c + 0.25 * diagG                           # (G,)
    M = d[:, None] + d[None, :] + 0.5 * Gram        # (G, G)
    iu = np.triu_indices(num_grid)                  # a <= b
    flat_best = int(np.argmin(M[iu]))
    t1 = float(theta_g[iu[0][flat_best]])
    t2 = float(theta_g[iu[1][flat_best]])

    # joint 2-D fine refinement: independent windows around each coarse theta
    lo, hi = float(doa_min), float(doa_max)
    cand1 = np.arange(t1 - fine_range, t1 + fine_range + 1e-9, fine_step)
    cand2 = np.arange(t2 - fine_range, t2 + fine_range + 1e-9, fine_step)
    cand1 = cand1[(cand1 >= lo) & (cand1 <= hi)]
    cand2 = cand2[(cand2 >= lo) & (cand2 <= hi)]
    L1 = 1.0 / (1.0 + np.exp(-(theta_g[:, None] - cand1[None, :]) / w))   # (G, n1)
    L2 = 1.0 / (1.0 + np.exp(-(theta_g[:, None] - cand2[None, :]) / w))   # (G, n2)
    c1 = p @ L1
    c2 = p @ L2
    n1 = np.einsum('gi,gi->i', L1, L1)
    n2 = np.einsum('gj,gj->j', L2, L2)
    cross = L1.T @ L2
    Mf = -(c1[:, None] + c2[None, :]) + 0.25 * (n1[:, None] + 2.0 * cross + n2[None, :])
    fi, fj = np.unravel_index(int(np.argmin(Mf)), Mf.shape)
    return np.sort(np.array([cand1[fi], cand2[fj]], dtype=float))


from scipy.signal import find_peaks as scipy_find_peaks, savgol_filter
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

def template_fit2(pred) -> np.ndarray:
    """Least-squares fit of the ideal two-step staircase to pred_mono.

    Monotonizes pred, does an exact O(G^2) coarse search over all grid pairs
    (theta1 <= theta2) using the Gram-matrix identity, then a JOINT 2-D fine search
    over independent sub-grid windows around each coarse theta (full cross-product,
    evaluated with the same Gram identity).  A joint search is used rather than
    alternating coordinate descent because at small separation the objective has a
    narrow diagonal valley in which coordinate descent stalls.  Always returns two
    sorted angles; there is no peak-merging step, so a close second source can never
    be suppressed.
    """
    # p = monotonize(pred)

    # # 保序回归，increasing=True：强制非递减
    x = np.arange(len(pred))  # 索引坐标
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    p = ir.fit_transform(x, pred)

    # p = savgol_filter(pred, window_length=11, polyorder=2)
    # 线性插值
    x_old = np.arange(-60, 61, 1)
    _ng = len(pred)
    _st = 0.1 if _ng == 1200 else (0.25 if _ng == 480 else 1.0)
    x_new = -60 + np.arange(_ng) * _st
    # p_new = np.interp(x_new, x_old, p)
    p_new = p

    # 2. 数值求导（不要用 y*(1-y)）
    grid = _st
    dy = np.gradient(p_new, grid)

    # plt.figure(figsize=(8, 8))
    # plt.plot(dy)
    # plt.savefig("result.png", dpi=300, bbox_inches="tight")
    # 3. 找峰值（每个峰对应一个 Sigmoid 跳变中心）
    peaks, properties = scipy_find_peaks(dy, height=0.05, distance=1)  # distance 根据 w 设定，避免同一个源被检测多次
    heights = properties["peak_heights"]
    # 按峰高降序重排
    idx_sort = np.argsort(-heights)
    peaks_sorted_by_height = peaks[idx_sort]
    heights_sorted = heights[idx_sort]


    peaks = peaks[idx_sort]

    peaks = peaks[0:2] # take the first K peaks
    peaks = sorted(peaks) # sort the index in a ascending order
    # ---- Generate data ----

    # 4. 得到源个数和位置
    K_est = len(peaks)
    if len(peaks)<2:
        theta_est = np.ones(2)*100
        theta_est[0] = x_new[peaks]
    else:
        theta_est = x_new[peaks]

    return np.sort(np.array([theta_est[0], theta_est[1]], dtype=float))

# Extract the upper triangular elements (excluding the diagonal) of each channel in a tensor
# Used for extracting upper triangular elements from batch input data
# Adapted for batch processing, allowing handling of a batch at once
def batch_extract_upper_triangle(tensor):
    batch_size, channels, height, width = tensor.shape
    assert height == width, "Input matrices must be square"
    # Get indices for the upper triangle (excluding the diagonal)
    indices = torch.triu_indices(height, width, offset=1)  # (2, num_elements)
    # Extract upper triangular elements
    upper_triangle_elements = tensor[:, :, indices[0], indices[1]]  # (batch_size, channels, num_elements)
    return upper_triangle_elements