# Windows Conda Environment Diagnostic Script
# Run this in PowerShell after conda finishes

Write-Host "=== Conda Environment Check ===" -ForegroundColor Cyan
conda info --envs

Write-Host "`n=== Attempting to Activate Environment ===" -ForegroundColor Cyan
conda activate inner_speech

Write-Host "`n=== Python Version ===" -ForegroundColor Cyan
python --version

Write-Host "`n=== TensorFlow Import Test ===" -ForegroundColor Cyan
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__} OK')" 2>&1

Write-Host "`n=== MNE Import Test ===" -ForegroundColor Cyan
python -c "import mne; print(f'MNE {mne.__version__} OK')" 2>&1

Write-Host "`n=== NumPy Import Test ===" -ForegroundColor Cyan
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')" 2>&1

Write-Host "`n=== Conda List (installed packages) ===" -ForegroundColor Cyan
conda list | grep -E "tensorflow|mne|numpy|scipy"

Write-Host "`n=== Diagnosis Complete ===" -ForegroundColor Green
Write-Host "Copy the entire output above and send to debugging." -ForegroundColor Yellow
