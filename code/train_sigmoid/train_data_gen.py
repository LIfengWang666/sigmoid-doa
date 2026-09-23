## Generate training data
## Training data conditions are set in global parameters in the file
## 2024-08-01 Xie Yuxuan UTF-8
import h5py
import numpy as np
import itertools
import random
import os
from concurrent.futures import ThreadPoolExecutor

########## Global Parameter Settings ##########
array_num = 16  # Number of array elements
lamda = 0.3  # Wavelength
d = np.arange(array_num) * lamda / 2  # Array position vector
doa_min = -60  # Direction of arrival range - minimum angle
doa_max = 60  # Direction of arrival range - maximum angle
sig_nums = [2]  # Number of signals — only two-source training
snapshot_num = 1000  # Number of snapshots
snrs = [-15, -10, -5, 0, 5]  # Signal-to-noise ratios of training data
save_dir = '/home/yxxie/Documents/cnn_transformer/training_data'
num_threads = 25  # Number of threads to use

#################### Functions ####################

## Generate data conditions based on global variables, other functions read these conditions to generate data
## Generate training data following the dual_class_token_transformer paper
## sig_nums - number of signal sources; doa_min/max - range of source angles; snrs - SNRs to generate
## 2024-08-12 Xie Yuxuan
def get_data_conditions(sig_nums, doa_min, doa_max, snrs):
    # Used to store data conditions; each element is a dictionary containing sig_num, doa, snr
    data_list_numeq_1 = []
    data_list_numeq_2 = []
    data_list_numeq_3 = []

    doa_range = list(range(doa_min, doa_max + 1))

    # Generate data condition lists based on the number of signal sources
    for num in sig_nums:
        
        ## Handle the case of one target
        if num == 1:
            combinations = list(itertools.combinations(doa_range, num))
            for snr in snrs:
                for combo in combinations:
                    data_condition = {
                        'sig_num': num,
                        'doa': list(combo),
                        'snr': snr
                    }
                    data_list_numeq_1.append(data_condition)
            data_list_numeq_1 = data_list_numeq_1 * 2380  # Each combination is repeated 2380 times

        ## Handle the case of two targets
        if num == 2:
            combinations = list(itertools.combinations(doa_range, num))
            for snr in snrs:
                for combo in combinations:
                    data_condition = {
                        'sig_num': num,
                        'doa': list(combo),
                        'snr': snr
                    }
                    data_list_numeq_2.append(data_condition)
            data_list_numeq_2 = data_list_numeq_2 * 10  # Each combination is repeated 10 times

        ## Handle the case of three targets
        if num == 3:
            combinations = list(itertools.combinations(doa_range, num))
            for snr in snrs:
                for combo in combinations:
                    data_condition = {
                        'sig_num': num,
                        'doa': list(combo),
                        'snr': snr
                    }
                    data_list_numeq_3.append(data_condition)

    data_list = []
    data_list = data_list_numeq_1 + data_list_numeq_2 + data_list_numeq_3

    return data_list

## Generate training labels with cumulative-staircase (CDF) smoothing
## Input DOA, generate the normalized cumulative-staircase target:
##   y[g] = (1/K) * sum_k sigmoid((theta_g - theta_k) / w)
## Each source contributes a logistic step (0 -> 1); divided by K the curve rises
## monotonically from 0 to 1, with the k-th step mid-point at y=(2k-1)/(2K) sitting on
## theta_k.  y lies in [0,1] by construction (no clip needed) and matches the model's
## sigmoid output head.  Read-out uses level-crossing, NOT peak finding.
## doa - signal angles, a list
def get_label(doa):
    w = 0.2                                       # gaussian std (deg)
    grid = 0.1
    num_grid = int((doa_max - doa_min) / grid)                  # 121 grid bins (-60..60 deg)
    theta_g = (doa_min + np.arange(num_grid)*grid).astype(np.float64)  # grid angles -60..60
    K = len(doa)
    label = np.zeros(num_grid)
    for d in doa:
        label += 1.0 / (1.0 + np.exp(-(theta_g - d) / w))  # logistic step
    return (label / K).astype(np.float32)      # clip to [0,1]

## Generate data by reading data conditions
## Generate one piece of data at a time for multi-threaded calls
## data_condition - data condition; seed - random seed to ensure different threads use different random numbers
## 2024-08-13 Xie Yuxuan
def generate_data(data_condition, seed):
    # Create an independent random number generator
    rng = np.random.default_rng(seed)

    # Extract data conditions
    sig_num = data_condition['sig_num']
    doa = data_condition['doa']
    snr = data_condition['snr']
    doa_rad = np.deg2rad(doa)  # Convert angles to radians

    # Generate signal data matrix, follows standard normal distribution
    real_part = rng.standard_normal((sig_num, snapshot_num))
    imaginary_part = rng.standard_normal((sig_num, snapshot_num))
    S = real_part + 1j * imaginary_part

    norms = np.linalg.norm(S, axis=1, keepdims=True)  # Normalize signal matrix
    snr_scale_factor = np.sqrt(10 ** (snr / 10))  # Calculate scaling factor using SNR
    S *= (snr_scale_factor / norms)  # Scale signal matrix using SNR

    # Generate noise data matrix, follows standard normal distribution and normalize
    real_part = rng.standard_normal((array_num, snapshot_num))
    norms = np.linalg.norm(real_part, axis=1, keepdims=True)
    real_part = real_part / norms
    real_part = real_part * 1 / np.sqrt(2)

    imaginary_part = rng.standard_normal((array_num, snapshot_num))
    norms = np.linalg.norm(imaginary_part, axis=1, keepdims=True)
    imaginary_part = imaginary_part / norms
    imaginary_part = imaginary_part * 1 / np.sqrt(2)

    N = real_part + 1j * imaginary_part  # Generate noise matrix

    # Generate array manifold matrix, use sine for angles from -90 to 90
    phase_shifts = -1j * 2 * np.pi / lamda * d.reshape(-1, 1) * np.sin(doa_rad)
    A = np.exp(phase_shifts)
    X = np.dot(A, S) + N  # Generate data matrix
    R = np.dot(X, X.T.conj()) / X.shape[0]  # Compute covariance matrix

    # Assemble training data with three channels: real part, imaginary part, and phase
    Re = np.real(R)
    Im = np.imag(R)
    Phase = np.angle(R)
    data = np.stack((Re, Im, Phase), axis=-1)  # Data with dimensions (array elements * array elements * 3)
    label = get_label(doa)  # Label, size doa_max - doa_min + 1

    return data, label, sig_num, snr, doa

## Call the generate_data function with multiple threads to generate data
## data_condition_list - list of data conditions; save_dir - directory to save data; num_threads - number of threads to use
## 2024-08-13 Xie Yuxuan
def get_data(data_condition_list, save_dir, num_threads):
    np.random.seed(42)  # Set random seed for the main thread

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        # Submit all tasks to multi-threading
        for idx, data_condition in enumerate(data_condition_list):
            seed = np.random.SeedSequence().entropy + idx  # Create unique random seed
            futures.append(executor.submit(generate_data, data_condition, seed))  
        
        # Used to store results
        data_list = []
        label_list = []
        sig_num_list = []
        snr_list = []
        doa_list = []

        # Process results of completed tasks
        for future in futures:
            data, label, sig_num, snr, doa = future.result()  # Return values from generate_data
            data_list.append(data)
            label_list.append(label)
            sig_num_list.append(sig_num)
            snr_list.append(snr)
            doa_list.append(doa)

    # Convert to numpy arrays and save
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    sig_num_list = np.array(sig_num_list)
    snr_list = np.array(snr_list)

    np.savez(os.path.join(save_dir, 'data.npz'), data=data_list, label=label_list, signum=sig_num_list, snr=snr_list)

## Main function
## First call get_data_conditions to generate the list of conditions for training data
## Then call get_data to generate data with multi-threading based on the conditions
if __name__ == '__main__':
    data_condition_list = get_data_conditions(sig_nums, doa_min, doa_max, snrs)
    get_data(data_condition_list, save_dir, num_threads)