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

    # Determine whether to return all peaks
    if sig_num is not None:
        top_n_peaks = peaks[: sig_num]  # Take the top n peaks
        return top_n_peaks
    
    return peaks


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