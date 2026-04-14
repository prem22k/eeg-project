# Windows Environment Setup Script for inner_speech
# Run in PowerShell as Administrator if possible

$ErrorActionPreference = 'Stop'

Write-Host '================================' -ForegroundColor Cyan
Write-Host 'Inner Speech Conda Setup' -ForegroundColor Cyan
Write-Host '================================' -ForegroundColor Cyan

Write-Host ''
Write-Host '[1/4] Cleaning up any failed environments...' -ForegroundColor Yellow
conda env remove -n inner_speech -y 2>$null | Out-Null
Write-Host 'Done.' -ForegroundColor Green

Write-Host ''
Write-Host '[2/4] Creating lean base environment (fast path)...' -ForegroundColor Yellow
conda create -n inner_speech --solver libmamba -c conda-forge python=3.10 pip -y

Write-Host '[2/4] Installing required packages with pip...' -ForegroundColor Yellow
conda run -n inner_speech python -m pip install --upgrade pip
conda run -n inner_speech python -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.9.2 tensorflow==2.18.0 mne==1.8.0 tensorflow-datasets==4.9.6

Write-Host ''
Write-Host '[3/4] Verifying environment...' -ForegroundColor Yellow
$envExists = conda info --envs | Select-String 'inner_speech'
if (-not $envExists) {
    Write-Host 'Environment not found. Check disk space and RAM.' -ForegroundColor Red
    exit 1
}
Write-Host 'Environment found.' -ForegroundColor Green

Write-Host ''
Write-Host '[4/4] Testing imports...' -ForegroundColor Yellow
conda run -n inner_speech python -c "import tensorflow as tf; print('TensorFlow ' + tf.__version__ + ' OK')"
conda run -n inner_speech python -c "import mne; print('MNE ' + mne.__version__ + ' OK')"
conda run -n inner_speech python -c "import numpy as np; print('NumPy ' + np.__version__ + ' OK')"

Write-Host ''
Write-Host '================================' -ForegroundColor Green
Write-Host 'Setup Complete!' -ForegroundColor Green
Write-Host '================================' -ForegroundColor Green
Write-Host 'To use: conda activate inner_speech' -ForegroundColor Cyan
