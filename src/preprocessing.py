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

def fit_ica(raw_inst, n_components=15, random_state=42):
    """
    Fits Independent Component Analysis (ICA) to decompose multi-channel EEG
    signals into statistically independent sources (e.g., blinks, muscle).
    
    Parameters:
        raw_inst (mne.io.Raw): MNE Raw object containing filtered EEG data.
        n_components (int): Number of principal components to extract (default 15).
        random_state (int): Seed for reproducible ICA decomposition.
        
    Returns:
        ica (mne.preprocessing.ICA): Fitted ICA object.
    """
    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter='auto'
    )
    
    # Fit ICA on the MNE Raw object
    ica.fit(raw_inst)
    return ica


def apply_ica_cleaning(raw_inst, ica, exclude_components):
    """
    Removes selected artifact components and reconstructs cleaned EEG signals.
    
    Parameters:
        raw_inst (mne.io.Raw): Original MNE Raw instance.
        ica (mne.preprocessing.ICA): Fitted ICA object.
        exclude_components (list): List of component indices to drop (e.g., [0, 2]).
        
    Returns:
        cleaned_raw (mne.io.Raw): Copy of raw data with excluded components zeroed out.
    """
    raw_cleaned = raw_inst.copy()
    ica.exclude = exclude_components
    ica.apply(raw_cleaned)
    return raw_cleaned