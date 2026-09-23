## Generate test data
## Used for rough comparison of the performance of trained models and other methods
## A large amount of data is generated with a wide range of conditions. If performance is good here, it will perform even better on other data.
## Xie Yuxuan 2024-08-22 UTF-8
import numpy as np
import random
import os
from multiprocessing import Pool, cpu_count

array_num = 16  # Number of array elements
lamda = 0.3  # Wavelength
d = np.arange(array_num) * lamda / 2  # Array position vector
doa = [[-48.1, -48.1 + sep] for sep in range(1, 11)]
mc_num = 1000
snapshots = [1000]
snrs = [-10]
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'val_data_res_uniform')  # Save path (relative to script)

# Create save path (auto-create, no interactive prompt)
os.makedirs(save_dir, exist_ok=True)
print(f"Test data save directory '{save_dir}' is ready.")

## Generate data based on given conditions
## data_num - Data identifier; sig_num - Number of signals; snap_num - Number of snapshots; doa - Angles; snr - Signal-to-noise ratio; mc_num - Number of Monte Carlo experiments
## Save data in npz format, including covariance matrix arrays for all Monte Carlo experiments and experimental conditions
def generate_data(data_num, sig_num, snap_num, doa, snr, doa_gap, mc_num):

    rng = np.random.default_rng(10)  # Set random seed to 10
    doa_rad = np.deg2rad(doa)
    data_rec_list = []
    data_cov_list = []  # Store covariance data for all Monte Carlo experiments under these conditions

    # Generate signal data matrix following a 0-1 standard normal distribution
    real_part = rng.standard_normal((sig_num, snap_num))
    imaginary_part = rng.standard_normal((sig_num, snap_num))
    S = real_part + 1j * imaginary_part
    norms = np.linalg.norm(S, axis=1, keepdims=True)  # Normalize the signal matrix to make its energy 1, then scale it using SNR
    snr_scale_factor = np.sqrt(10 ** (snr / 10))  # Calculate scaling factor using SNR
    S *= (snr_scale_factor / norms)  # Scale the signal matrix using SNR

    rng = np.random.default_rng(sig_num * snap_num + 10)  # Set random seed based on iteration
    ## Perform Monte Carlo experiments
    for i in range(mc_num):

        # Generate noise data matrix following a 0-1 standard normal distribution and normalize
        real_part = rng.standard_normal((array_num, snap_num))
        norms = np.linalg.norm(real_part, axis=1, keepdims=True)
        real_part = real_part / norms
        real_part = real_part * 1 / np.sqrt(2)

        imaginary_part = rng.standard_normal((array_num, snap_num))
        norms = np.linalg.norm(imaginary_part, axis=1, keepdims=True)
        imaginary_part = imaginary_part / norms
        imaginary_part = imaginary_part * 1 / np.sqrt(2)
        N = real_part + 1j * imaginary_part  # Generate noise data matrix

        # Generate array manifold matrix; use sin since angles are in the range of -90 to 90
        phase_shifts = -1j * 2 * np.pi / lamda * d.reshape(-1, 1) * np.sin(doa_rad)
        A = np.exp(phase_shifts)
        X = np.dot(A, S) + N  # Generate data matrix
        R = np.dot(X, X.T.conj()) / X.shape[0]  # Compute covariance matrix

        # data_rec_list.append(X)  # Add the received data from this Monte Carlo experiment to the list
        data_cov_list.append(R)  # Add the covariance matrix from this Monte Carlo experiment to the list

    # Save data
    np.savez(os.path.join(save_dir, f"data_num-{data_num}&sig_num-{sig_num}&snr-{snr}&snap_num-{snap_num}&doa-{doa}"), \
                            data_cov=data_cov_list, snr=snr, sig_num=sig_num, doa=doa, snap_num=snap_num)
    print(f"Data {data_num} saved successfully!")

## Generate a list of conditions and use these as parameters to call the generate_data function for multi-process data generation
## sig_nums - List of signal counts; snapshots - List of snapshot counts; doa_min - Minimum angle; doa_max - Maximum angle; snrs - List of SNRs; doa_num - Number of angle combinations; mc_num - Number of Monte Carlo experiments
def get_data_conditions(snapshots, snrs, mc_num, doa):

    data_num = 1  # Used to assign unique identifiers to generated test data
    tasks = []  # Store all data generation conditions

    for angle in doa:
        for snap_num in snapshots:  # Iterate over snapshot counts
                for snr in snrs:  # Iterate over SNRs
                    if len(angle) == 1: doa_gap = 0
                    else: doa_gap = round(angle[1] - angle[0])
                    tasks.append((data_num, len(angle), snap_num, angle, snr, doa_gap, mc_num))  # Add data generation conditions to the task list
                    data_num += 1

    # Use multi-core processing to handle tasks
    with Pool(20) as pool:
        pool.starmap(generate_data, tasks)

## Main function
if __name__ == '__main__':
    get_data_conditions(snapshots, snrs, mc_num, doa)