import matplotlib.pyplot as plt
import numpy as np

def plot_eeg_signal(time, raw_signal, filtered_signal, channel_name="Channel", duration=5.0, fs=160.0):
    """
    Plot a segment of the EEG signal.

    Parameters:
    - time: 1D array representing the time axis (in seconds).
    - raw_signal: 1D array of un-filtered voltage data for a single electrode channel.
    - filtered_signal: 1D array of filtered voltage data for the same channel (electrode).
    - channel_name: The electrode label string (default is "Channel").
    - duration: how many seconds of signals to display on screen (default is 5.0).
    - fs: Sampling frequency of the signal (in Hz, default is 160.0).
    """
    # Calculate the number of samples to display based on duration and sampling frequency
    num_samples = int(duration * fs) #5.0 seconds * 160 Hz = 800 samples
    
    # slice the time array to keep only the first 800 time steps
    t_window = time[:num_samples]
    raw_window = raw_signal[:num_samples] * 1e6  # Convert to microvolts for better visualization
    filtered_window = filtered_signal[:num_samples] * 1e6  # Convert to microvolts

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True) # creates a figure layout with 2rows and 1column (stacked plots)
    # axes[0] == top plot, axes[1] == bottom plot
    # sharex=True: links the x-axis of both plots, so zooming/panning in one plot affects the other
    axes[0].plot(t_window, raw_window, color='blue', linewidth=1.2, label='Raw Signal')
    axes[0].set_title(f"Raw Signal from {channel_name} (Raw)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitude (µV)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6) # alpha for semi-transparency, subtle grid background
    axes[0].legend(loc='upper right') # places to top-right corner of the plot

    axes[1].plot(t_window, filtered_window, color='green', label='Filtered Signal')
    axes[1].set_title(f"Bandpass Filtered (Alpha/Beta: 8-30 Hz) from {channel_name} (Filtered)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Amplitude (µV)", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    return fig, axes

def plot_psd_comparison(freqs, raw_psd, filtered_psd, channel_idx=0, channel_name="Channel", max_freq=50.0):
    """
    Plots Welch Power Spectral Density (PSD) comparison between raw and filtered signals.
    """
    # Select target channel PSD and convert to uV^2 / Hz
    raw_psd_ch = raw_psd[channel_idx] * 1e12
    filt_psd_ch = filtered_psd[channel_idx] * 1e12
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(freqs, raw_psd_ch, color='#1f77b4', linewidth=1.5, label='Raw Signal')
    ax.semilogy(freqs, filt_psd_ch, color='#2ca02c', linewidth=1.5, label='Filtered Signal (8–30 Hz)')
    
    ax.set_xlim(0, max_freq)
    ax.set_title(f"Power Spectral Density (PSD) — {channel_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Power ($\mu\text{V}^2$ / Hz)")
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig, ax