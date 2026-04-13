#!/bin/bash
# Quick Start Guide - Modernized Inner Speech Decoding Pipeline

set -e  # Exit on any error

echo "=== Inner Speech Decoding - Modernized Pipeline ==="
echo ""
echo "This script guides you through running the updated training pipeline"
echo "with modern TensorFlow 2.18 and MNE 1.8 APIs."
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python3 --version | grep -q "3\.[0-9]" && echo "✓ Python ready" || { echo "✗ Python 3 required"; exit 1; }

# Check if environment file exists
echo "[2/5] Checking environment.yml..."
if [ -f "environment.yml" ]; then
    echo "✓ environment.yml found"
    echo "  To create Conda environment, run:"
    echo "    conda env create -f environment.yml"
    echo "    conda activate inner_speech"
else
    echo "⚠ environment.yml not found"
fi

# Run smoke test
echo ""
echo "[3/5] Running modernization smoke test..."
if python3 test_mne_modernization.py > /tmp/mne_test.log 2>&1; then
    echo "✓ All preprocessing logic tests passed"
    grep "^  ✓" /tmp/mne_test.log | head -5
else
    echo "✗ Some tests failed - check /tmp/mne_test.log"
    cat /tmp/mne_test.log | tail -20
fi

echo ""
echo "[4/5] Checking data availability..."
if [ -d "dataset/derivatives" ]; then
    subject_count=$(find dataset/derivatives -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✓ Dataset found with ~$subject_count subjects"
    echo "  Data path: ./dataset/derivatives/sub-XX/ses-0X/"
else
    echo "⚠ Dataset not found at ./dataset/"
    echo "  To download dataset, run:"
    echo "    aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ./dataset/"
    echo "  (Requires AWS CLI and ~50GB storage)"
fi

echo ""
echo "[5/5] Environment Summary"
echo "  Python: $(python3 --version)"
echo "  Working directory: $(pwd)"
echo "  Config: environment.yml"
echo "  Code location: data_preprocessing.py, raw_training.py, classify.py"

echo ""
echo "=== Ready to Run Training ==="
echo ""
echo "Option A - Training with specific subject:"
echo "  python3 raw_training.py -s 1 -e 5 -b 32"
echo ""
echo "Option B - Training all subjects (requires full dataset):"
echo "  python3 raw_training.py"
echo ""
echo "Option C - K-fold cross-validation with PCA:"
echo "  python3 pca_training.py"
echo ""
echo "Usage help:"
echo "  python3 raw_training.py --help"
echo ""
echo "For more details, see MNE_MODERNIZATION.md"
