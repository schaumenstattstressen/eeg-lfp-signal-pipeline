# create synthetic EEG data for testing

import pytest
import numpy as np
import mne
from src.preprocessing import fit_ica, apply_ica_cleaning

@pytest.fixture
def dummy_raw():
    """Generate a small synthetic 64-channel MNE raw object for fast testing"""
    ch_names = [f"EEG {i:03d}" for i in range(1, 65)] # format integer to 3-digit zero-padded string
    info = mne.create_info(ch_names=ch_names, sfreq=160, ch_types='eeg')
    data = np.random.randn(64, 160*5) # 5 secs of random signal
    raw = mne.io.RawArray(data, info)
    return raw

def test_fit_ica_components(dummy_raw):
    """Test that ICA fitting returns the correct number of components"""
    n_components = 10
    ica = fit_ica(dummy_raw, n_components=n_components, random_state=42)
    assert ica.n_components_ == n_components, f"Expected {n_components} components, got {ica.n_components_}"

def test_apply_ica_cleaning_shape(dummy_raw):
    """Ensures that the cleaned data has the same shape as the original after ICA cleaning
    """
    ica = fit_ica(dummy_raw, n_components=10, random_state=42)

    exclude_components = [0]  # Exclude first component

    cleaned_raw = apply_ica_cleaning(dummy_raw, ica, exclude_components)
    
    assert cleaned_raw.get_data().shape == dummy_raw.get_data().shape, "Cleaned data shape does not match original"
    assert (cleaned_raw.info["sfreq"] == dummy_raw.info["sfreq"]), "Sampling frequency changed after ICA cleaning!"