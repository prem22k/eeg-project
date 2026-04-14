# Windows Environment Setup Script for inner_speech
# Run in PowerShell as Administrator if possible

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Inner Speech Conda Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Step 1: Clean up any failed environment
Write-Host "`n[1/4] Cleaning up any failed environments..." -ForegroundColor Yellow
conda env remove -n inner_speech -y 2>$null
Write-Host "Done." -ForegroundColor Green

# Step 2: Try environment.yml method (safest)
Write-Host "`n[2/4] Attempting to create from environment.yml..." -ForegroundColor Yellow
conda env create -f environment.yml -y

# Check if it succeeded
$envExists = conda info --envs | Select-String "inner_speech"
if ($envExists) {
    Write-Host "Success! Environment created from environment.yml" -ForegroundColor Green
} else {
    Write-Host "environment.yml method failed. Trying direct package install..." -ForegroundColor Red
    Write-Host "`n[2/4] (Retry) Installing with direct command..." -ForegroundColor Yellow
    conda create -n inner_speech -c conda-forge `
        python=3.10 `
        tensorflow=2.18 `
        mne=1.8 `
        numpy=1.26 `
        scipy=1.13 `
        scikit-learn=1.6 `
        pandas=2.2 `
        matplotlib=3.9 `
        tensorflow-datasets=4.9 `
        -y
}

# Step 3: Verify environment
Write-Host "`n[3/4] Verifying environment..." -ForegroundColor Yellow
$envExists = conda info --envs | Select-String "inner_speech"
if ($envExists) {
    Write-Host "✓ inner_speech environment found" -ForegroundColor Green
} else {
    Write-Host "✗ Environment still not found. Check disk space and RAM." -ForegroundColor Red
    exit 1
}

# Step 4: Activate and test
Write-Host "`n[4/4] Testing imports..." -ForegroundColor Yellow
conda activate inner_speech
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__}')" 2>&1
python -c "import mne; print(f'✓ MNE {mne.__version__}')" 2>&1
python -c "import numpy as np; print(f'✓ NumPy {np.__version__}')" 2>&1

Write-Host "`n================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "To use: conda activate inner_speech" -ForegroundColor Cyan
