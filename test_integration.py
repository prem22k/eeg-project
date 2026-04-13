#!/usr/bin/env python3
"""
Integration test for modernized preprocessing pipeline.
Does NOT require TensorFlow or a dataset - tests pure logic.
"""
import sys
import numpy as np

print("\n" + "="*70)
print("INNER SPEECH DECODING - MODERNIZED PIPELINE INTEGRATION TEST")
print("="*70 + "\n")

# Mock TensorFlow if not available
try:
    import tensorflow as tf
except ImportError:
    print("[SETUP] TensorFlow not available, mocking for testing...\n")
    class MockTF:
        class keras:
            class backend:
                @staticmethod
                def clear_session():
                    pass
            class models:
                @staticmethod
                def load_model(path):
                    return None
                @staticmethod
                def save(obj, path):
                    pass
            class optimizers:
                class Adam:
                    def __init__(self, *args, **kwargs):
                        pass
            class losses:
                class CategoricalCrossentropy:
                    pass
            class layers:
                class Layer:
                    pass
    sys.modules['tensorflow'] = MockTF()

# Test 1: Import all preprocessing modules
print("[TEST 1] Importing preprocessing module...")
try:
    import data_preprocessing as dp
    print("✓ PASS: data_preprocessing imported successfully\n")
except ImportError as e:
    print(f"✗ FAIL: Could not import data_preprocessing: {e}\n")
    sys.exit(1)

# Test 2: Event label parsing
print("[TEST 2] Testing event label parsing...")
test_cases = [
    ('pronounced_speech_up', (0, 0)),
    ('inner_speech_down', (1, 1)),
    ('visualized_condition_left', (2, 2)),
    ('direction_right', (-1, 3)),
]

passed = 0
failed = 0
for label, (expected_cond, expected_dir) in test_cases:
    cond, direction = dp._extract_condition_and_direction(label)
    if cond == expected_cond and direction == expected_dir:
        print(f"  ✓ '{label}' parsed as condition={cond}, direction={direction}")
        passed += 1
    else:
        print(f"  ✗ '{label}' expected ({expected_cond},{expected_dir}) "
              f"but got ({cond},{direction})")
        failed += 1

if failed == 0:
    print(f"✓ PASS: {passed}/{len(test_cases)} labels parsed correctly\n")
else:
    print(f"✗ FAIL: {failed} labels failed\n")
    sys.exit(1)

# Test 3: Event table validation
print("[TEST 3] Testing event table validation...")

# Valid table
valid_events = np.array([
    [0, 0, 0, 1],
    [256, 1, 1, 2],
    [512, 2, 2, 3],
    [768, 3, 1, 4],
])
try:
    dp._validate_event_table(valid_events)
    print("  ✓ Valid event table [samples=4, columns=4] accepted")
except ValueError as e:
    print(f"  ✗ Valid table rejected: {e}")
    sys.exit(1)

# Invalid condition
invalid_cond = np.array([
    [0, 0, 0, 1],
    [256, 1, 5, 2],  # condition=5 invalid
])
try:
    dp._validate_event_table(invalid_cond)
    print("  ✗ Invalid condition table was NOT rejected")
    sys.exit(1)
except ValueError:
    print("  ✓ Invalid condition (5) correctly rejected")

# Invalid direction
invalid_dir = np.array([
    [0, 5, 0, 1],  # direction=5 invalid
])
try:
    dp._validate_event_table(invalid_dir)
    print("  ✗ Invalid direction table was NOT rejected")
    sys.exit(1)
except ValueError:
    print("  ✓ Invalid direction (5) correctly rejected")

print("✓ PASS: Event validation working correctly\n")

# Test 4: Data-event alignment
print("[TEST 4] Testing data-event alignment...")

# Create synthetic data
data = np.random.rand(10, 128, 1024)  # 10 trials, 128 channels, 1024 timepoints
samples = np.arange(0, 10*1024, 1024)  # sample indices for each trial
events = np.array([
    [0, 0, 0, 1],
    [1024, 1, 1, 2],
    [2048, 2, 2, 3],
    [3072, 3, 0, 4],
    [4096, 0, 1, 5],
    [5120, 1, 2, 6],
    [6144, 2, 0, 7],
    [7168, 3, 1, 8],
    [8192, 0, 2, 9],
    [9216, 1, 0, 10],
])

try:
    aligned_data, aligned_events = dp._align_data_and_events(data, events, samples)
    print(f"  ✓ Alignment successful: {aligned_data.shape} data with {aligned_events.shape} events")
except Exception as e:
    print(f"  ✗ Alignment failed: {e}")
    sys.exit(1)

# Test misalignment detection
print("  Testing misalignment detection...")
misaligned_events = np.array([
    [0, 0, 0, 1],
    [1024, 1, 1, 2],
    [9999, 2, 2, 3],  # Sample 9999 doesn't exist
])
try:
    dp._align_data_and_events(data, misaligned_events, samples)
    print("  ✗ Misalignment was NOT detected")
    sys.exit(1)
except ValueError as e:
    print(f"  ✓ Misalignment correctly detected: '{str(e)[:50]}...'")

print("✓ PASS: Data-event alignment working correctly\n")

# Test 5: Mapping integrity
print("[TEST 5] Testing condition/direction mappings...")
print(f"  Conditions: {dp._CONDITION_MAP}")
print(f"  Directions: {dp._DIRECTION_MAP}")

if len(dp._CONDITION_MAP) == 3 and len(dp._DIRECTION_MAP) == 4:
    print("  ✓ Correct number of conditions (3) and directions (4)")
    print("✓ PASS: Mappings are complete\n")
else:
    print("✗ FAIL: Incorrect mapping sizes")
    sys.exit(1)

# Test 6: MNE API availability
print("[TEST 6] Checking modern MNE-Python APIs...")
try:
    import mne
    assert hasattr(mne, 'io') and hasattr(mne.io, 'read_raw_fif')
    assert hasattr(mne, 'events_from_annotations')
    assert hasattr(mne, 'Epochs')
    print("  ✓ mne.io.read_raw_fif available")
    print("  ✓ mne.events_from_annotations available")
    print("  ✓ mne.Epochs available")
    print("✓ PASS: Modern MNE-Python APIs ready\n")
except (AssertionError, AttributeError) as e:
    print(f"✗ FAIL: MNE API missing: {e}\n")
    sys.exit(1)

# Summary
print("="*70)
print("INTEGRATION TEST RESULTS: ALL TESTS PASSED ✓")
print("="*70)
print("\nPipeline is ready for training!")
print("\nNext steps:")
print("1. Install dependencies: pip install tensorflow mne scikit-learn...")
print("2. Download dataset: aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ./dataset/")
print("3. Run training: python3 raw_training.py -s 1 -e 10 -b 32")
print("\nFor full Conda environment setup:")
print("  conda env create -f environment.yml")
print("  conda activate inner_speech")
print("="*70 + "\n")
