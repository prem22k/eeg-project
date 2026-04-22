# Windows Conda Environment Diagnostic Script
# Collects deterministic machine + conda details for troubleshooting.

$ErrorActionPreference = 'Continue'
$Report = Join-Path $PWD 'windows_env_diagnostics.txt'

Start-Transcript -Path $Report -Force | Out-Null

Write-Host "=== System Info ===" -ForegroundColor Cyan
systeminfo | Select-Object -First 25

Write-Host "`n=== Conda Version and Config ===" -ForegroundColor Cyan
conda --version
conda info
conda config --show-sources

Write-Host "`n=== Active Environments ===" -ForegroundColor Cyan
conda info --envs

Write-Host "`n=== inner_speech Package Snapshot ===" -ForegroundColor Cyan
conda list -n inner_speech

Write-Host "`n=== Key Package Filter ===" -ForegroundColor Cyan
conda list -n inner_speech | Select-String 'python|tensorflow|tensorflow-datasets|numpy|scipy|pandas|scikit-learn|matplotlib|mne'

Write-Host "`n=== Python Import Check (inner_speech) ===" -ForegroundColor Cyan
conda run -n inner_speech python -c "import sys; print(sys.version)"
conda run -n inner_speech python -c "import tensorflow as tf; print('TF', tf.__version__)"
conda run -n inner_speech python -c "import tensorflow_datasets as tfds; print('TFDS', tfds.__version__)"
conda run -n inner_speech python -c "import numpy, scipy, pandas, sklearn, matplotlib, mne; print('Core imports OK')"

Write-Host "`n=== Channel Availability Check ===" -ForegroundColor Cyan
conda search -c defaults "tensorflow-cpu=2.18.1"
conda search -c defaults "tensorflow-datasets=4.9.9"
conda search -c conda-forge "mne=1.8"

Stop-Transcript | Out-Null

Write-Host "`n=== Diagnosis Complete ===" -ForegroundColor Green
Write-Host "Report file created: $Report" -ForegroundColor Yellow
