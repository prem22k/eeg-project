#!/usr/bin/env python3
"""
Demo training run with synthetic data.
Shows what the full pipeline output looks like.
"""
import numpy as np
import sys

print("\n" + "="*80)
print("INNER SPEECH DECODING - DEMO TRAINING (Synthetic Data)")
print("="*80)
print()
print("This demo shows what the training pipeline output looks like.")
print("It uses synthetic EEG data instead of downloading the full dataset.")
print()

# Mock TensorFlow if needed
try:
    import tensorflow as tf
    print(f"✓ Using TensorFlow {tf.__version__}")
except ImportError:
    print("⚠ TensorFlow not available - using mock for demo")
    class MockTF:
        __version__ = "2.18 (mocked)"
        class keras:
            class backend:
                @staticmethod
                def clear_session():
                    pass
            class models:
                class Model:
                    def fit(self, *args, **kwargs):
                        return type('History', (), {'history': {'loss': [0.5], 'accuracy': [0.6]}})()
                    def evaluate(self, *args, **kwargs):
                        return [0.4, 0.75]
                @staticmethod
                def clone_model(model):
                    return MockTF.keras.models.Model()
                @staticmethod
                def load_model(path):
                    return MockTF.keras.models.Model()
            class optimizers:
                class Adam:
                    def __init__(self, *args, **kwargs):
                        pass
            class losses:
                class CategoricalCrossentropy:
                    pass
    sys.modules['tensorflow'] = MockTF()
    tf = MockTF()

import data_preprocessing as dp

print("\n" + "-"*80)
print("PIPELINE INITIALIZATION")
print("-"*80)

# Hyperparameters (from raw_training.py)
EPOCHS = 5
SUBJECT_S = [1]
DROPOUT = 0.4
KERNEL_LENGTH = 64
N_CHECKS = 2
BATCH_SIZE = 8
PRETRAIN_EPOCHS = 3

print(f"EPOCHS: {EPOCHS}")
print(f"SUBJECTS: {SUBJECT_S}")
print(f"DROPOUT: {DROPOUT}")
print(f"KERNEL_LENGTH: {KERNEL_LENGTH}")
print(f"N_CHECKS: {N_CHECKS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"PRETRAIN_EPOCHS: {PRETRAIN_EPOCHS}")

print("\n" + "-"*80)
print("DATA LOADING")
print("-"*80)

# Generate synthetic data
print("Creating synthetic EEG data for demonstration...")
num_subjects = len(SUBJECT_S)
num_trials_per_subject = 40
num_channels = 128
num_timepoints = 640  # ~2.5 seconds at 256 Hz

# Create synthetic data for testing
synthetic_data = np.random.randn(num_trials_per_subject, num_channels, num_timepoints).astype(np.float32)
synthetic_events = np.zeros((num_trials_per_subject, 4), dtype=int)
for i in range(num_trials_per_subject):
    synthetic_events[i, 0] = i * 1024  # sample index
    synthetic_events[i, 1] = i % 4  # direction (0-3)
    synthetic_events[i, 2] = i % 3  # condition (0-2)
    synthetic_events[i, 3] = (i % 3) + 1  # event id (1-3)

print(f"✓ Synthetic data shape: {synthetic_data.shape}")
print(f"✓ Synthetic events shape: {synthetic_events.shape}")
print(f"  - {num_trials_per_subject} trials")
print(f"  - {num_channels} EEG channels")
print(f"  - {num_timepoints} timepoints per trial (~2.5s at 256Hz)")

# Validate events
print("\nValidating event table...")
try:
    dp._validate_event_table(synthetic_events)
    print("✓ Event table validation passed")
    print(f"  - Conditions found: {np.unique(synthetic_events[:, 2]).tolist()}")
    print(f"  - Directions found: {np.unique(synthetic_events[:, 1]).tolist()}")
except ValueError as e:
    print(f"✗ Event validation failed: {e}")
    sys.exit(1)

# Filter inner speech condition
print("\nFiltering to inner speech condition (condition=1)...")
keep_pos = synthetic_events[:, 2] == 1
filtered_data = synthetic_data[keep_pos]
filtered_events = synthetic_events[keep_pos]
print(f"✓ Filtered to {filtered_data.shape[0]} trials")

print("\n" + "-"*80)
print("TRAINING LOOP")
print("-"*80)

for subject in SUBJECT_S:
    print(f"\n{'='*80}")
    print(f"SUBJECT {subject}")
    print(f"{'='*80}")
    
    # Simulate k-fold training
    k_fold = 4
    val_accuracies = []
    
    for fold in range(k_fold):
        print(f"\n[FOLD {fold+1}/{k_fold}]")
        print(f"{'─'*80}")
        
        # Simulate data split
        fold_size = len(filtered_data) // k_fold
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        train_data = np.concatenate([filtered_data[:test_start], filtered_data[test_end:]])
        test_data = filtered_data[test_start:test_end]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Simulate training
        print(f"\nTraining EEGNet model...")
        
        # Show epoch progress
        for epoch in range(1, EPOCHS + 1):
            # Simulate training metrics
            loss = 1.2 - (epoch * 0.15) + np.random.randn() * 0.05
            acc = 0.3 + (epoch * 0.12) + np.random.randn() * 0.03
            val_loss = 1.1 - (epoch * 0.14) + np.random.randn() * 0.06
            val_acc = 0.35 + (epoch * 0.11) + np.random.randn() * 0.04
            
            # Clamp to reasonable ranges
            loss = max(0.2, loss)
            acc = min(0.95, max(0.2, acc))
            val_loss = max(0.2, val_loss)
            val_acc = min(0.95, max(0.2, val_acc))
            
            bar_length = int(acc * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  Epoch {epoch}/{EPOCHS} "
                  f"loss={loss:.4f} accuracy={acc:.4f} "
                  f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f} "
                  f"|{bar}|")
        
        # Simulate evaluation
        test_loss = 0.45
        test_acc = 0.68 + np.random.randn() * 0.05
        test_acc = min(0.95, max(0.3, test_acc))
        
        print(f"\nFold Evaluation: loss={test_loss:.4f}, accuracy={test_acc:.4f}")
        val_accuracies.append(test_acc)
    
    # Summary for subject
    mean_acc = np.mean(val_accuracies)
    std_acc = np.std(val_accuracies)
    print(f"\n{'='*80}")
    print(f"SUBJECT {subject} SUMMARY")
    print(f"{'='*80}")
    print(f"Average K-Fold Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in val_accuracies]}")

print("\n" + "="*80)
print("DEMO TRAINING COMPLETE")
print("="*80)
print("\nNote: This is a demonstration with synthetic data.")
print("To run with real data:")
print("  1. Download dataset: aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ./dataset/")
print("  2. Install dependencies: conda env create -f environment.yml")
print("  3. Run training: python3 raw_training.py -s 1 -e 10")
print("="*80 + "\n")
