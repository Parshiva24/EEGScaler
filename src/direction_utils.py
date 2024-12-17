import scipy.io
import numpy as np
import os
import glob
import torch
from scipy import signal 
from sklearn.model_selection import LeaveOneOut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import constants


SEED=constants.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) 
np.random.seed(SEED)
random.seed(SEED)

 # Euclidean Alignment (EA)
def euclidean_alignment_raw(Xtarget, Xsource):
    """
    Parameters:
    Xtarget : ndarray
        Target dataset with shape (n_trials, n_channels, n_samples)
    Xsource : ndarray
        Source dataset with shape (n_trials, n_channels, n_samples)

    Returns:
    Xadapted : ndarray
        Adapted source dataset after Euclidean Alignment.
    """

    if len(Xtarget)>1:
        Xtarget = np.concatenate(Xtarget, axis=0)

    # Check if Xtarget is empty
    if len(Xtarget) == 0 and len(Xsource) > 0:
        Xtarget = Xsource  # Assign X2 to X1 if X1 is empty
    else:
        Xsource = np.array(Xsource).squeeze()
        Xtarget = np.array(Xtarget).squeeze()

    n_trials, n_channels, n_samples = Xsource.shape

            
    # Step 1: Compute the average covariance matrix for each frequency band
    cov_target = np.zeros((n_channels, n_channels))  # Separate R for each band
    for trial in range(Xtarget.shape[0]):
        Xt_trial = Xtarget[trial, :, :]  # Shape: (n_channels, n_samples, n_bands)
        cov_target += np.cov(Xt_trial)
    cov_target /= n_trials

    
    # Step 2: Compute the inverse square root of the average covariance matrix for each band
    eigvals, eigvecs = np.linalg.eigh(cov_target)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


    # Step 3: Apply Euclidean Alignment to the source data for each band
    Xadapted = np.zeros_like(Xsource)
    
    for trial in range(n_trials):
        Xadapted[trial] = R_inv_sqrt @ Xsource[trial]
        
    
    return Xadapted

def load_mat_file(filepath):
    """ Load .mat file and return Xtr, Ytr always; Xte, Yte only if they are not NaN """
    mat_data = scipy.io.loadmat(filepath)
    
    Xtr = mat_data['Xtrain']
    Ytr = mat_data['Ytrain']
    Xte = mat_data['Xtest']
    Yte = mat_data['Ytest']
    Yte_fb = mat_data['Ytest_fb']
    
    # Check if Xte and Yte are NaN or entirely NaN arrays
    if np.isnan(Xte).all() or np.isnan(Yte).all() or np.isnan(Yte_fb).all():
        # If all values in Xte or Yte are NaN, return only Xtr and Ytr
        return Xtr, Ytr
    else:
        # Otherwise, return Xtr, Ytr, Xte, Yte
        return Xtr, Ytr, Xte, Yte, Yte_fb

# def create_dataset(current_subject, base_path=None):
#     """"
#     Create training and test dataset for the current subjects.
    
#     Parameters:
#     current_subject (int): Subject number (1 to 20).
#     base_path (str): The directory where .mat files are stored.

#     Returns:
#     X_train, Y_train, X_test, Y_test
#     """
#     Xtr_all = []
#     Ytr_all = []
#     for sub in range(1, current_subject+1):
#         rel_path = f'data/S{sub:02d}_mitrials.mat'
#         filepath = os.path.join(base_path, rel_path)

#         mat_vars = load_mat_file(filepath)
#         if len(mat_vars)==2:
#             Xtr, Ytr = mat_vars
#             Xtr_all.append(Xtr)
#             Ytr_all.append(Ytr)

#         else:
#             Xtr, Ytr, Xte, Yte, Yte_fb = mat_vars
#             Xtr_all.append(Xtr)
#             Ytr_all.append(Ytr)
#             if (sub == current_subject):
#                 continue
#             else:
#                 # Xtr_all.append(Xte)
#                 # Ytr_all.append(Yte)
#                 indx = np.where(Yte==Yte_fb)[0]
#                 Xtr_all.append(Xte[indx, :, :])
#                 Ytr_all.append(Yte[indx])
                
#     X_train = np.concatenate(Xtr_all, axis=0)
#     Y_train = np.concatenate(Ytr_all, axis=0)

#     X_test = Xte
#     Y_test = Yte
        
#     return X_train, Y_train, X_test, Y_test

# Baseline Correction 
def baseline_correction(X, baseline_samples=500):
    n_trails, n_samples, n_channels = X.shape
    Xbc = np.zeros_like(X)
    for t in range(n_trails):
        Xbase = X[t, :baseline_samples-1,:]
        Xeeg = X[t, baseline_samples:, :]
        Xbc[t, baseline_samples:, :] = Xeeg - np.mean(Xbase, axis=0) #baseline correction
    
    Xnew = Xbc[:, baseline_samples:, :]
    return Xnew

# Preprocessing: Surface Laplacian, Bandpass Filter
def bandpass_filtering(X, fs=500, fcut=[0.1, 90], filt_order=5):
    n_trials, n_samples, n_channels = X.shape
    X1 = np.zeros_like(X)
    notch_freq = 50
    q_factor = 20
    b_notch, a_notch = signal.iirnotch(notch_freq, q_factor, fs)
    b,a = signal.butter(filt_order, fcut, fs=fs, btype = 'band', output='ba') 

    for t in range(n_trials):
        for c in range(n_channels):
            #Error here.
            raw_signal = X[t, :, c]
            notch_signal = signal.filtfilt(b_notch, a_notch, raw_signal)
            filt_signal = signal.filtfilt(b, a, notch_signal)
            X1[t, :, c] = filt_signal
            # Xfilt[t, :, c] = signal.filtfilt(b, a, X[t, :, c])
    return X1

# Multi-band Filter Function without averaging over trials
def filter_eeg_multi_band(X, fs=500):
    # Define the frequency bands
    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), 
             (24, 28), (28, 32), (32, 36), (36, 40)]
    
    n_trials, n_samples, n_channels = X.shape
    n_bands = len(bands)
    
    # Initialize output array: n_trials x samples x channels x n_bands
    Xout = np.zeros((n_trials, n_samples, n_channels, n_bands))
    
    # Apply bandpass filtering for each band
    for i, (low_cut, high_cut) in enumerate(bands):
        # print(f"Filtering band {low_cut}-{high_cut} Hz")
        X_filt = bandpass_filtering(X, fs=fs, fcut=[low_cut, high_cut], filt_order=5)
        
        # Store the filtered data without averaging across trials
        Xout[:, :, :, i] = X_filt
    
    Xout = np.transpose(Xout, (0, 2, 1, 3))  # (trials, electrodes, samples, n_bands)
    return Xout

def calib_sess_dataset(base_path=None):
    X = []
    Y = []
    for sub in range(1, 8):  #S01 to S07 has calibration session only
        rel_path = f'data/S{sub:02d}_mitrials.mat'
        filepath = os.path.join(base_path, rel_path)

        mat_vars = load_mat_file(filepath)
        if len(mat_vars)==2:
            Xtr, Ytr = mat_vars
            X.append(Xtr)
            Y.append(Ytr)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    return X, Y

def online_sess_dataset(subnum, base_path=None):
    rel_path = f'data/S{subnum:02d}_mitrials.mat'
    filepath = os.path.join(base_path, rel_path)

    mat_vars = load_mat_file(filepath)
    Xtr, Ytr, Xte, Yte, Yte_fb = mat_vars

    return np.array(Xtr), np.array(Ytr), np.array(Xte), np.array(Yte) 


def data_augmentation_timeshift(real_data, real_label, shift_range=(0.1, 0.5)):
    """
    Apply time shifting to EEG data using np.roll.
    Args:
        data: EEG data, shape (Nt, Nc, Ns).
        shift_range: Tuple (min_shift, max_shift) in seconds.
    Returns:
        Augmented data with time-shifting applied.
    """
    Nt, Nc, Ns = real_data.shape
    num_augmented_trials = 3*Nt
    fs = constants.fs
    augmented_data = []
    augmented_label = []
    for _ in range(num_augmented_trials):
        # Choose a random trial to augment 
        trial_idx = np.random.randint(0, Nt)

        # Random shift witinh the specified range
        shift_seconds = np.random.uniform(*shift_range)
        shift_samples = int(shift_seconds*fs)

        #Applt time shift to all the channels of the selected trial
        shifted_trial = np.array([np.roll(channel, shift_samples) for channel in real_data[trial_idx]])

        # Append time shift to all channels of the selected trial 
        augmented_data.append(shifted_trial)
        augmented_label.append(real_label[trial_idx])
    
    augmented_data = np.array(augmented_data)
    augmented_label = np.array(augmented_label)

    # Combine original and augmented data
    combined_data = np.concatenate((real_data, augmented_data), axis=0)
    combined_label = np.concatenate((real_label, augmented_label), axis=0)

    return combined_data, combined_label


def speed_dataset(train_subnum, test_subnum, base_path=None):

    Xtr = []
    Ytr = []
    for sub in train_subnum:
        rel_path = f'data/S{sub:02d}_midata.mat'
        filepath = os.path.join(base_path, rel_path)
        
        mat_data = scipy.io.loadmat(filepath)
        Xtr.append(mat_data['Xtrain'])
        Ytr.append(mat_data['Ytrain'])
    Xtr = np.concatenate(Xtr, axis=0)
    Ytr = np.concatenate(Ytr, axis=0)
    
    # print(type(test_subnum), test_subnum, test_subnum[0])
    if len(test_subnum)==1:
        rel_path = f'data/S{test_subnum[0]:02d}_midata.mat'
        filepath = os.path.join(base_path, rel_path)
        
        mat_data = scipy.io.loadmat(filepath)

        Xte = mat_data['Xtrain']
        Yte = mat_data['Ytrain']
    else:
        print(f'There are more than 1 test subjects')

    return np.array(Xtr), np.array(Ytr), np.array(Xte), np.array(Yte)


if __name__=='__main__':
    # Generate random EEG data for demonstration (replace with your data)
    
    Nt, Nc, Ns = 60, 33, 2000  # Original EEG data dimensions
    eeg_data = np.random.rand(Nt, Nc, Ns)  # Shape: (60, 33, 2000)
    X = data_augmentation_timeshift(eeg_data, shift_range=(0.1, 0.5), num_augmented_trials=150)
    # Final shape of augmented data
    print(f"Original data shape: {eeg_data.shape}")
    print(f"Augmented data shape: {X.shape}")


    


