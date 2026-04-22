# Windows Environment Setup Script for inner_speech
# Robust conda-only setup for win-64 with retry and verification.
# Run in PowerShell, preferably as Administrator.

$ErrorActionPreference = 'Stop'
$EnvName = 'inner_speech'

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Script,
        [int]$Retries = 2
    )

    for ($i = 1; $i -le $Retries; $i++) {
        try {
            Write-Host "[$Name] Attempt $i/$Retries" -ForegroundColor Yellow
            & $Script
            Write-Host "[$Name] Success" -ForegroundColor Green
            return
        } catch {
            Write-Host "[$Name] Failed: $($_.Exception.Message)" -ForegroundColor Red
            if ($i -eq $Retries) {
                throw
            }
            Start-Sleep -Seconds 2
        }
    }
}

Write-Host '================================' -ForegroundColor Cyan
Write-Host 'Inner Speech Conda Setup (Windows)' -ForegroundColor Cyan
Write-Host '================================' -ForegroundColor Cyan

Invoke-Step -Name 'Cleanup old environment' -Retries 1 -Script {
    conda deactivate 2>$null
    conda deactivate 2>$null
    conda env remove -n $EnvName -y 2>$null | Out-Null
}

Invoke-Step -Name 'Conda cache cleanup' -Retries 1 -Script {
    conda clean --all -y
}

Invoke-Step -Name 'Set libmamba solver' -Retries 1 -Script {
    conda config --set solver libmamba
}

Invoke-Step -Name 'Create TensorFlow base (defaults, py310)' -Retries 3 -Script {
    conda create -n $EnvName --override-channels -c defaults --strict-channel-priority -y `
        python=3.10 tensorflow-cpu=2.18.1 tensorflow-datasets=4.9.9
}

Invoke-Step -Name 'Install scientific stack (conda-forge, frozen TF stack)' -Retries 3 -Script {
    conda run -n $EnvName conda install -y -c conda-forge --strict-channel-priority --freeze-installed `
        numpy=1.26 scipy=1.13 pandas=2.2 scikit-learn=1.6 matplotlib=3.9 mne=1.8
}

Write-Host ''
Write-Host '[Verification] Import checks...' -ForegroundColor Cyan
conda run -n $EnvName python -c "import sys; print(sys.version)"
conda run -n $EnvName python -c "import tensorflow as tf; print('TensorFlow ' + tf.__version__ + ' OK')"
conda run -n $EnvName python -c "import tensorflow_datasets as tfds; print('TFDS ' + tfds.__version__ + ' OK')"
conda run -n $EnvName python -c "import numpy, scipy, pandas, sklearn, matplotlib, mne; print('Core scientific imports OK')"

Write-Host ''
Write-Host '================================' -ForegroundColor Green
Write-Host 'Setup Complete!' -ForegroundColor Green
Write-Host '================================' -ForegroundColor Green
Write-Host 'Use with: conda activate inner_speech' -ForegroundColor Cyan
