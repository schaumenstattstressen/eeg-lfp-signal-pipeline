# Welch's method is a technique for estimating the power spectral density (PSD) of a signal. 
# It involves dividing the signal into overlapping segments, applying a window function to each segment, 
# computing the periodogram for each segment, and then averaging the periodograms to obtain a smoother estimate of the PSD. 
# This method reduces the variance of the PSD estimate compared to using a single segment.

import numpy as np
from scipy.signal import welch
import mne
import pandas as pd

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