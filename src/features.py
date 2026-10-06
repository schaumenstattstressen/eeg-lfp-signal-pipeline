# Welch's method is a technique for estimating the power spectral density (PSD) of a signal. 
# It involves dividing the signal into overlapping segments, applying a window function to each segment, 
# computing the periodogram for each segment, and then averaging the periodograms to obtain a smoother estimate of the PSD. 
# This method reduces the variance of the PSD estimate compared to using a single segment.

import numpy as np
from scipy.signal import welch
import mne
import pandas as pd
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from src.preprocessing import fit_ica, apply_ica_cleaning

def compute_psd(data, fs=160.0, nperseg=None):
    # nperseg=None -> int(2*fs) = 320 samples (2 seconds of data at 160 Hz) a clean frequency resolution for frequency bands to fall on exacct integer bins
    # when noverlap is not set, python dynamically adjusts whenever nperseg scale
    """
    Computes Power Spectral Density (PSD) using Welch's method across all channels.
    
    Parameters:
        data (ndarray): Signal array of shape (channels, time_points).
        fs (float): Sampling frequency in Hz (default 160.0).
        nperseg (int): Length of each segment for FFT (defaults to 2 * fs).
        
    Returns:
        freqs (ndarray): Array of sample frequencies (Hz).
        psd (ndarray): Power spectral density values (V^2 / Hz or uV^2 / Hz).
    """
    if nperseg is None:
        nperseg = int(2 * fs)  # Default to 2 seconds of data for 0.5 Hz resolution

    # Compute Welch's PSD
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=-1)  # axis=-1 to compute along the last dimension (time_points)

    return freqs, psd

def extract_band_power(freqs, psd, band_limits=(8.0, 12.0)):
    """
    Calculates average band power within a specific frequency band (e.g., Alpha: 8-12 Hz).
    
    Parameters:
        freqs (ndarray): 1D frequency array from Welch calculation.
        psd (ndarray): PSD matrix (channels x frequencies).
        band_limits (tuple): Frequency window bounds (low, high) in Hz.
        
    Returns:
        band_power (ndarray): Mean power per channel within specified frequency range.
    """
    low, high = band_limits
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # Integrate/average PSD across frequency indices in target band
    band_power = np.mean(psd[:, idx_band], axis=-1)
    return band_power

def create_motor_epochs(raw, tmin=-0.5, tmax=3.0, baseline=(-0.5, 0)):
    """
    Extracts motor imagery events and epoch slices from continuous raw EEG.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed raw instance.
    tmin, tmax : float
        Start and end times of the epoch relative to event onset (seconds).
    baseline : tuple or None
        Time interval for baseline correction.
        
    Returns:
    --------
    epochs : mne.Epochs
        Epoched dataset grouped by event IDs.
    """
    # 1. Extract event annotations from raw dataset
    events, event_dict = mne.events_from_annotations(raw)
    
    # 2. Filter for task target events (T1: left fist, T2: right fist)
    target_event_dict = {k: v for k, v in event_dict.items() if k in ['T1', 'T2']}
    
    # 3. Epoch the signal around target events
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=target_event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False
    )
    
    return epochs

def extract_band_powers(epochs, fmin=8.0, fmax=30.0):
    """
    Computes Power Spectral Density (PSD) per channel across specified frequency band.
    
    Returns:
    --------
    df_psd : pd.DataFrame
        DataFrame containing mean band power per epoch and channel.
    """
    # Compute spectral power density using Multitaper/Welch method
    spectrum = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax)
    psd_data = spectrum.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
    
    # Average power across frequency bins
    mean_power = psd_data.mean(axis=-1)
    
    # Convert to clean Pandas DataFrame for easy analysis
    df = pd.DataFrame(mean_power, columns=epochs.ch_names)
    df['condition'] = epochs.events[:, -1]
    
    return df

def evaluate_motor_imagery_classifier(epochs, n_components=4, cv_folds=5):
    """
    Trains and evaluates a CSP and Logistic Regression pipeline using cross-validation.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched motor imagery data.
    n_components : int
        Number of spatial patterns to extract with CSP.
    cv_folds : int
        Number of stratified cross-validation splits.
    
    Returns:
    --------
    scores : numpy.darray
        Classification accuracy scores across cross-validation folds.
    pipeline : sklearn.pipeline.Pipeline
        Fitted CSP + Logistic Regression pipeline.
    """
    # 1. extract raw data matrix X and labels y
    X = epochs.get_data(copy=True)  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    # 2. build CSP + Logistic Regression pipeline
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = make_pipeline(csp, LogisticRegression(solver='liblinear', random_state=42))

    # 3. stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    # 4. fit final model on all epochs
    clf.fit(X, y)
    return scores, clf

def load_multi_subject_data(subject_ids, runs=[4, 8, 12]):
    """Loads, cleans, and epochs data for multiple PhysioNet subjects silently."""
    X_list, y_list, group_list = [], [], []

    # MNE verbosity level setting to keep notebook output clean
    mne.set_log_level("WARNING")

    for sub_id in subject_ids:
        try:
            raw_fnames = mne.datasets.eegbci.load_data(
                subjects=sub_id, runs=runs, verbose=False
            )
            raws = [
                mne.io.read_raw_edf(f, preload=True, verbose=False)
                for f in raw_fnames
            ]
            raw = mne.concatenate_raws(raws, verbose=False)

            raw.resample(sfreq=100, verbose=False)

            mne.datasets.eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("colin27_1020")
            raw.set_montage(montage, on_missing="ignore", verbose=False)

            ica = fit_ica(
                raw, n_components=10, max_iter=500, random_state=42
            )
            raw_clean = apply_ica_cleaning(raw, ica, exclude_components=[0])

            events, event_id = mne.events_from_annotations(
                raw_clean, verbose=False
            )
            target_ids = {
                k: v for k, v in event_id.items() if k in ["T1", "T2"]
            }

            if not target_ids:
                continue

            epochs = mne.Epochs(
                raw_clean,
                events,
                event_id=target_ids,
                tmin=-0.5,
                tmax=1.0,
                baseline=(-0.5, 0.0),
                preload=True,
                verbose=False,
            )

            X_sub = epochs.get_data(copy=True)
            y_sub = epochs.events[:, -1]
            groups_sub = np.full(len(y_sub), fill_value=sub_id)

            X_list.append(X_sub)
            y_list.append(y_sub)
            group_list.append(groups_sub)

        except Exception as e:
            print(f"Skipping Subject {sub_id} due to error: {e}")
            continue

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(group_list, axis=0)

    return X, y, groups

def evaluate_loso_cross_validation(X, y, groups, n_components=4):
    """
    Performs Leave-One-Subject-Out (LOSO) cross-validation for motor imagery classification.

    Returns:
    --------
    subject_scores : dict
        Dictionary mapping subject IDs to their accuracy scores.
    """
    loso = LeaveOneGroupOut()
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = make_pipeline(csp, LogisticRegression(solver='liblinear', random_state=42))

    subject_scores = {}

    for train_idx, test_idx in loso.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = groups[test_idx][0]

        # Build and fit the pipeline
        clf.fit(X_train, y_train)

        # Evaluate on the left-out subject
        score = clf.score(X_test, y_test)
        subject_scores[test_subject] = score

    return subject_scores