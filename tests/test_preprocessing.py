# create synthetic EEG data for testing

import pytest
import numpy as np
import mne
from src.preprocessing import fit_ica, apply_ica_cleaning
from src.features import create_motor_epochs, extract_band_powers

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

@pytest.fixture
def dummy_annotated_raw():
    """Creates synthetic Raw EEG data containing dummy event annotations."""
    ch_names = [f"EEG {i:03d}" for i in range(1, 65)]
    info = mne.create_info(ch_names=ch_names, sfreq=160, ch_types="eeg")
    data = np.random.randn(64, 160 * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)
    
    # Add annotations mimicking PhysioNet motor events
    onset = [1.0, 4.0, 7.0]
    duration = [2.0, 2.0, 2.0]
    description = ['T1', 'T2', 'T1']
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)
    return raw

def test_create_motor_epochs(dummy_annotated_raw):
    """Verifies epoch creation returns the expected number of trials."""
    epochs = create_motor_epochs(dummy_annotated_raw, tmin=-0.5, tmax=1.0)
    assert len(epochs) == 3, f"Expected 3 epochs, got {len(epochs)}"

def test_extract_band_powers_dataframe(dummy_annotated_raw):
    """Verifies PSD extraction returns correct DataFrame dimensions."""
    epochs = create_motor_epochs(dummy_annotated_raw, tmin=-0.5, tmax=1.0)
    df_psd = extract_band_powers(epochs, fmin=8.0, fmax=30.0)
    
    # Check that rows match epoch count and columns match channels + condition
    assert len(df_psd) == 3, "DataFrame row count does not match epoch count!"
    assert 'condition' in df_psd.columns, "'condition' column missing from output!"