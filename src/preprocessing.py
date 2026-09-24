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