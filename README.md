# EEG/LFP neural signal processing & feature extraction

Primary Goal: to build an automated, modular pipeline that takes continuous, unprocesses raw voltage signals and turn them into clean, structured spectral features (such as Power Spectral Density across Alpha, Beta, and Gamma bands).

## features
- **Bandpass Filtering:** zero-phase butterworth filter for isolating Delta, Theta, Alpha, Beta, and Gamma bands
- **Spectral Feature Extraction:** welch power spectral density (PSD) and short-time fourier transform (STFT)
- **Dataset Integration:** PhysioNet EEG motor movement/imagery dataset via MNE

## Project Structure
eeg-lfp-signal-pipeline/
│── data/               # Raw & processed data (gitignored)
│── docs/               # Figures and exported visual reports
│── notebooks/          # Exploratory analysis & pipeline validation
│── src/                # Core modular code
│   └── preprocessing.py # Filtering logic
└── requirements.txt    # Dependencies

## Quick Start
```bash
conda activate eeg_env
jupyter notebook notebooks/pipeline_demo.ipynb 