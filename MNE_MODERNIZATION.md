# MNE-Python Modernization & Testing Report

## Summary of Changes

All preprocessing has been migrated to **modern MNE-Python APIs** (MNE 1.8+) while maintaining backward compatibility with existing CNN training pipeline.

### Key Updates

#### 1. Modern Raw File I/O
- **Old**: Direct `mne.read_epochs()` with legacy parameters
- **New**: 
  - `mne.io.read_raw_fif()` with preload
  - `mne.events_from_annotations()` for annotation extraction
  - `mne.Epochs()` for trial construction

#### 2. Event Parsing & Conversion
- **New Functions**:
  - `_normalize_label()` - Robust label normalization
  - `_extract_condition_and_direction()` - Semantic parsing of event descriptions
  - `_event_table_from_epochs()` - Legacy-compatible [sample, direction, condition, id] format
  - `_load_events()` - Unified .dat and annotation-derived event loading

#### 3. Data-Event Alignment
- **New**: `_align_data_and_events()` - Index-based alignment (no silent truncation)
- **Validation**: `_validate_event_table()` - Strict label range checking

#### 4. Error Handling
- Session load failures now logged instead of silently ignored
- Misaligned data raises hard errors with diagnostics
- Invalid labels caught immediately with clear messages

## Test Results

✓ **All Core Tests Passed**:
- ✓ Event extraction logic (pronounced_speech, inner_speech directions)
- ✓ Valid event tables accepted
- ✓ Invalid events (e.g., direction=4) correctly rejected
- ✓ Data-event alignment succeeds with matching samples
- ✓ Misaligned data correctly detected and error raised
- ✓ All modern MNE APIs available
- ✓ Condition/direction mappings validated (3 conditions, 4 directions)

## Next Steps to Run Full Pipeline

### Option 1: Using Conda (Recommended)
```bash
# Create environment with modernized dependencies
conda env create -f environment.yml

# Activate environment
conda activate inner_speech

# Download dataset
aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ./dataset/

# Run training
python raw_training.py -s 1  # Run for subject 1 only
```

### Option 2: Using pip
```bash
# Install dependencies
pip install tensorflow==2.18 mne==1.8 scikit-learn==1.6 numpy==1.26 \
            scipy==1.13 pandas==2.2 matplotlib==3.9 tensorflow-datasets==4.9

# Download dataset
aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ./dataset/

# Run training
python raw_training.py
```

## Modernization Details by File

### data_preprocessing.py
- **Lines 38-54**: `_event_table_from_epochs()` - Parse epochs with strict label validation
- **Lines 55-74**: `_load_events()` - Load from .dat or fallback to annotations
- **Lines 75-103**: `_align_data_and_events()` - Alignment with misalignment detection
- **Lines 105-121**: `_validate_event_table()` - Event label range validation
- **Lines 124-160**: `_read_session_epochs()` - Main MNE loader (handles both raw and epochs)
- **Lines 191-196**: Improved error handling for session loading

### raw_training.py
- **Line 6**: Modernized keras import (`from tensorflow.keras.utils import to_categorical`)
- **Line 58, 80, 130**: Direct `to_categorical()` calls (no np_utils alias)
- **Line 137**: Updated `MirroredStrategy()` auto-device selection

### classify.py
- **Line 72, 103**: Updated `MirroredStrategy()` 
- **Line 125**: Modern shape accessor (`v.shape` instead of `v.get_shape()`)

### data_preprocessing.py (part 2)
- **Line 271**: Modern dataset save (`dataset.save()` instead of `tf.data.experimental.save()`)

## Known Limitations & Future Improvements

1. **Label Parsing Edge Cases**: The current label normalization works for standard annotation formats. If your dataset uses unusual label strings, you may need to extend `_CONDITION_MAP` and `_DIRECTION_MAP` in data_preprocessing.py

2. **Event Count Validation**: The strict alignment requires event files to have exactly matching samples. If there's legitimate variance, you may need to adjust `_align_data_and_events()` tolerance.

3. **Python 3.10 Compatibility**: Environment.yml specifies Python 3.11, but tested against 3.10. Both should work with TensorFlow 2.18+

## Validation Checklist

After environment creation, verify:
- [ ] `python test_mne_modernization.py` passes all tests
- [ ] `python -c "import data_preprocessing; print('OK')"` succeeds
- [ ] Dataset successfully loads from `./dataset/` 
- [ ] At least one subject's data loads without error
- [ ] Training converges (no NaN losses)

## Rollback Instructions

If issues occur with modernized code:
1. Original code backed up in git (if available)
2. Key changes are isolated to data_preprocessing.py and small sections of raw_training.py/classify.py
3. To revert: restore old `mne.read_epochs()` call and remove new helper functions

## Support

For issues with:
- **MNE APIs**: Refer to https://mne.tools/ (v1.8 docs)
- **TensorFlow compatibility**: https://www.tensorflow.org/api/tf/distribute/MirroredStrategy
- **Event alignment**: Check dataset format in `./dataset/derivatives/sub-XX/ses-0X/`
