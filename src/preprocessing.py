# holds modular signal processing logic

import numpy as np
from scipy.signal import butter, filtfilt

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

# --- STANDALONE TEST BLOCK ---
if __name__ == '__main__':
    # Generate 10 seconds of synthetic 64-channel EEG data at 160 Hz
    fs = 160.0
    duration = 10.0
    n_channels = 64
    n_samples = int(fs * duration)
    
    # Create dummy random signal matrix: (64 channels x 1600 samples)
    dummy_data = np.random.randn(n_channels, n_samples)
    
    # Test the Butterworth filter function
    filtered_output = butter_bandpass_filter(
        data=dummy_data,
        lowcut=8.0,
        highcut=30.0,
        fs=fs,
        order=4
    )
    
    print("--- Preprocessing Module Test ---")
    print(f"Input Shape:    {dummy_data.shape}")
    print(f"Filtered Shape: {filtered_output.shape}")
    print("Module test passed successfully!")