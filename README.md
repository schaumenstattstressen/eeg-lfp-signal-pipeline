# EEG/LFP Neural Signal Processing & Feature Extraction

**Primary Goal:** Build an automated, modular pipeline that transforms continuous, unprocessed raw voltage signals into clean, structured spectral and spatial-geometric features for cross-subject Brain-Computer Interface (BCI) motor imagery decoding.

## Key Features
- **Bandpass Filtering & Resampling:** 4th-order zero-phase Butterworth filtering ($8\text{--}30\text{ Hz}$ sensorimotor $\mu/\beta$ isolation, plus Delta, Theta, Alpha, Beta, and Gamma band support) and $100\text{ Hz}$ signal resampling for >90% runtime acceleration.
- **Artifact Rejection:** Automated FastICA decomposition and component inspection to remove ocular and muscular noise.
- **Spectral & Spatial-Geometric Feature Extraction:** Welch Power Spectral Density (PSD), Short-Time Fourier Transform (STFT), Common Spatial Patterns (CSP), and Ledoit-Wolf regularized Covariance estimation with **Riemannian Tangent Space projection** (`pyriemann`).
- **Dataset Integration & Cross-Subject Validation:** Automated ingestion of the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB) via MNE-Python, evaluated under Leave-One-Subject-Out (LOSO) cross-validation.
- **Automated Testing:** Unit test suite (`pytest`) verifying array dimensions, channel standardization, and MNE/Scikit-Learn axis transpositions.

## Cross-Subject Benchmark (10 Subjects, LOSO)

Models are evaluated on zero-shot cross-subject generalization (Left vs. Right Hand motor imagery, Run 4, 150 total epochs across Subjects 1–10) without subject-specific calibration:

| Decoding Pipeline | Validation Scheme | Cohort Size | Mean Accuracy |
| :--- | :--- | :--- | :--- |
| Chance Level (Binary) | — | — | 50.0% |
| Regularized CSP (`ledoit_wolf`) + Logistic Regression | LOSO | 10 Subjects | 49.3% |
| **Ledoit-Wolf Covariance + Riemannian Tangent Space + Logistic Regression** | **LOSO** | **10 Subjects** | **54.7%** |

## Project Structure
```text
eeg-lfp-signal-pipeline/
├── data/                   # Raw & processed data (gitignored)
├── docs/                   # Figures and exported visual reports
├── src/                    # Core modular code
│   ├── preprocessing.py    # Standardization, Butterworth filtering, FastICA
│   ├── features.py         # Multi-subject ingestion & Riemannian LOSO pipeline
│   └── visualization.py    # Signal traces, ICA topomaps, and LOSO benchmark plots
├── tests/                  # Automated unit tests
│   └── test_preprocessing.py
├── pipeline_demo.ipynb     # End-to-end walkthrough & benchmark results
└── requirements.txt        # Dependencies

## Quick Start
conda activate eeg_env
pip install -r requirements.txt
pytest
jupyter notebook pipeline_demo.ipynb