# holds modular signal processing logic

import numpy as np
from scipy.signal import butter, filtfilt
import mne
from mne.preprocessing import ICA

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
    - data: The input signal (1D array).
    - lowcut: The lower frequency cutoff (in Hz).
    - highcut: The upper frequency cutoff (in Hz).
    - fs: The sampling frequency of the signal (in Hz).
    - order: The order of the filter (default is 4).

    Returns:
    - filtered_data: The bandpass filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Butterworth bandpass filter design
    b, a = butter(order, [low, high], btype='band')

    # apply zero-phase digital filter to prevents phase distortion
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def fit_ica(raw_inst, n_components=15, l_freq=1.0, max_iter=800, random_state=42):
    """Fits FastICA on high-pass filtered raw EEG data to prevent warnings."""
    # 1. High-pass filter data to remove low-frequency drift before ICA
    raw_filtered = raw_inst.copy().filter(l_freq=l_freq, h_freq=None, verbose=False)

    # 2. Initialize and fit ICA model with higher iteration limit to ensure convergence
    ica = mne.preprocessing.ICA(
        n_components=n_components, max_iter=max_iter, random_state=random_state
    )
    ica.fit(raw_filtered)
    return ica


def apply_ica_cleaning(raw, ica, exclude_components=None):
    """Applies fitted ICA decomposition to raw data excluding specified artifact components."""
    raw_cleaned = raw.copy()
    if exclude_components is not None:
        ica.exclude = exclude_components
    ica.apply(raw_cleaned)
    return raw_cleaned