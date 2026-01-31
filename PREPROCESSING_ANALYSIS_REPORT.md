# rPPG-Toolbox Preprocessing System - Comprehensive Analysis

**Date**: 2026-02-01
**Analysis of**: BaseLoader.py preprocessing pipeline
**Purpose**: Understanding how preprocessing handles PPG signal polarity

---

## 1. Overview of Preprocessing Pipeline

The preprocessing happens in `BaseLoader.preprocess()` (lines 224-276):

```
Raw Data → Face Detection/Crop → Resize → Data Normalization → Label Normalization → Chunking → Save
```

### Key Steps:

1. **Face Detection & Cropping** (crop_face_resize)
   - Detects face in video frames
   - Crops to face region
   - Resizes to target dimensions (e.g., 72x72)

2. **Data Transformation** (VIDEO frames)
   - Applied based on `DATA_TYPE` config setting
   - Options: "Raw", "DiffNormalized", "Standardized"
   - Can apply MULTIPLE types (concatenates channels)

3. **Label Transformation** (PPG signal)
   - Applied based on `LABEL_TYPE` config setting
   - Options: "Raw", "DiffNormalized", "Standardized"
   - Applied to ground truth PPG signal

4. **Chunking**
   - Splits video/PPG into fixed-length clips (e.g., 180 frames)
   - Each clip becomes a training/inference sample

---

## 2. Data Normalization Methods (for VIDEO frames)

### 2.1 Raw (No transformation)
```python
# Just pass through as-is
data = frames.copy()
```
- No normalization
- Pixel values remain 0-255

### 2.2 DiffNormalized (Temporal Difference)
**Code** (lines 600-612):
```python
def diff_normalize_data(data):
    diffnormalized_data[j] = (data[j+1] - data[j]) / (data[j+1] + data[j] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
```

**What it does**:
- Calculates temporal difference between consecutive frames
- Normalizes by sum (makes it relative change)
- Divides by standard deviation (z-score)
- **Purpose**: Captures motion/changes, removes DC component

**Output**: Z-scored temporal differences

### 2.3 Standardized (Z-score)
**Code** (lines 624-629):
```python
def standardized_data(data):
    data = data - np.mean(data)
    data = data / np.std(data)
```

**What it does**:
- Subtracts mean (centers at 0)
- Divides by std dev
- **Purpose**: Normalizes scale, removes DC offset

**Output**: Zero-mean, unit-variance data

---

## 3. Label Normalization Methods (for PPG signal)

### 3.1 Raw
```python
# No transformation
pass
```
- PPG signal unchanged
- Keeps original values (positive, negative, whatever they are)

### 3.2 DiffNormalized (Temporal Difference)
**Code** (lines 615-621):
```python
def diff_normalize_label(label):
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
```

**What it does**:
- Calculates first derivative: `label[t+1] - label[t]`
- Normalizes by standard deviation
- **Purpose**: Converts PPG to rate-of-change signal

**CRITICAL INSIGHT**:
```
Original PPG:     [... 100, 110, 120, 115, 105, 100 ...]  (systolic peak at 120)
After diff:       [...  10,  10, -5,  -10, -5  ...]      (peak becomes zero-crossing)

If PPG is inverted:
Original PPG:     [... -100, -110, -120, -115, -105, -100 ...]  (inverted)
After diff:       [...  -10, -10,  5,   10,  5  ...]            (INVERTED derivatives!)
```

**This means**: If your PPG signal polarity is inverted, DiffNormalized will produce inverted derivatives!

### 3.3 Standardized (Z-score)
**Code** (lines 632-637):
```python
def standardized_label(label):
    label = label - np.mean(label)
    label = label / np.std(label)
```

**What it does**:
- Centers at 0 (removes DC offset)
- Normalizes variance

**CRITICAL INSIGHT**:
```
Original PPG:     [... 100, 120, 100, 80, 100 ...]  (mean=100)
After std:        [...  0,   +2,  0,  -2,  0  ...]  (centered)

If PPG is inverted:
Original PPG:     [... -100, -120, -100, -80, -100 ...]  (mean=-100)
After std:        [...   0,   -2,   0,   +2,   0  ...]  (INVERTED!)
```

**This means**: Standardized normalization **preserves polarity**. If input is inverted, output is inverted.

---

## 4. Your EfficientPhys Models Configuration

Looking at `CUSTOM_PURE_EfficientPhys.yaml`:
```yaml
PREPROCESS:
  DATA_TYPE:
    - Standardized          # Video frames: Z-scored
  LABEL_TYPE: DiffNormalized  # PPG: Temporal derivative
```

**This configuration**:
1. Video frames get Z-scored (centered, normalized)
2. PPG labels get differentiated and normalized

**The Problem**:
- Your PPG signal has inverted polarity (peaks are negative)
- After `DiffNormalized`:
  - Normal PPG: systolic → positive derivative spike
  - Your PPG: systolic → **negative derivative spike**
- Model was trained on normal (positive) derivative spikes
- Model sees your negative spikes and predicts LOW HR
- Model sees your positive (diastolic) spikes and predicts HIGH HR
- **Result**: High/Low swap!

---

## 5. Does Preprocessing Handle Signal Polarity Automatically?

**Answer: NO**

The preprocessing functions are **polarity-agnostic**. They apply mathematical operations without checking signal direction:

- **DiffNormalized**: Takes derivatives. If signal is upside-down, derivatives are upside-down.
- **Standardized**: Centers and scales. Preserves polarity.
- **Raw**: No change.

**There is NO automatic polarity detection or correction** in the preprocessing pipeline.

---

## 6. Where Polarity Matters

### Models Trained on:
- UBFC-rPPG dataset: PPG from pulse oximeter (positive values, peaks = high)
- PURE dataset: PPG from pulse oximeter (positive values, peaks = high)
- SCAMPS dataset: Synthetic PPG (designed to match normal polarity)

**Expected signal pattern**:
```
Systolic (blood volume HIGH) → PPG value HIGH → After diff: positive spike
Diastolic (blood volume LOW) → PPG value LOW → After diff: negative spike
```

### Your Samsung Watch Data:
```
Signal is 99% negative, mean = -50K
After diff_normalize: derivatives are inverted relative to training data
```

---

## 7. Why Models Fail on Your Data

**Training data preprocessing**:
```
Normal PPG [50, 80, 100, 80, 50] → DiffNormalized → [+30, +20, -20, -30] (normalized)
Model learns: Positive spike = systolic = HIGH HR
```

**Your data preprocessing**:
```
Inverted PPG [-50, -80, -100, -80, -50] → DiffNormalized → [-30, -20, +20, +30] (normalized)
Your signal: Positive spike = diastolic = LOW HR (OPPOSITE!)
```

**Result**:
- High HR (fast systolic spikes) → Model sees negative spikes → Predicts LOW
- Low HR (slow systolic spikes) → Model sees positive spikes → Predicts HIGH

---

## 8. Where to Fix the Polarity Issue

### Option 1: Fix in CustomLoader (RECOMMENDED)
**File**: `dataset/data_loader/CustomLoader.py`
**Location**: `read_wave()` function (line 79)
**Change**:
```python
bvp_signal = df['green'].values.astype(np.float32)
bvp_signal = -bvp_signal  # Invert polarity
```

**Why here**:
- ✅ Raw data unchanged
- ✅ Fix happens once during loading
- ✅ Transparent and documented
- ✅ Easy to enable/disable

### Option 2: Fix in Raw Data
**Not recommended** because:
- ❌ Modifies source data
- ❌ Hard to track what was changed
- ❌ Permanent change

### Option 3: Create Custom Preprocessing
**Not needed** - Option 1 is simpler

---

## 9. Verification After Fix

After applying polarity fix, you should see:

**Before Fix**:
- Correlation: -0.20 (negative)
- High HR bias: -7 BPM (underestimated)
- Low HR bias: +41 BPM (overestimated)

**After Fix** (expected):
- Correlation: +0.60 to +0.90 (positive)
- High HR bias: ±5 BPM (accurate)
- Low HR bias: ±5 BPM (accurate)
- MAE: <10 BPM (good performance)

---

## 10. Summary

### What Preprocessing Does:
1. **Crops face from video**
2. **Normalizes video frames** (removes lighting, DC offset)
3. **Normalizes PPG labels** (removes DC offset, calculates derivatives)
4. **Chunks into clips**
5. **Saves to disk**

### What Preprocessing DOESN'T Do:
- ❌ Check PPG signal polarity
- ❌ Detect if signal is inverted
- ❌ Automatically correct signal direction

### Your Issue:
- Samsung Watch PPG has inverted polarity (99% negative values)
- Preprocessing preserves this inversion
- Models trained on normal polarity → predictions are inverted

### Solution:
- Invert PPG signal in CustomLoader: `bvp_signal = -bvp_signal`
- This fixes polarity BEFORE preprocessing
- Preprocessing then works correctly on properly-oriented signal

---

## 11. Technical Details: Why Each Normalization Preserves Polarity

### DiffNormalized:
```python
diff = signal[t+1] - signal[t]
```
If signal is inverted (multiplied by -1):
```python
diff_inverted = (-signal[t+1]) - (-signal[t]) = -(signal[t+1] - signal[t]) = -diff
```
**Conclusion**: Derivatives are also inverted.

### Standardized:
```python
normalized = (signal - mean(signal)) / std(signal)
```
If signal is inverted:
```python
inverted = -signal
mean(inverted) = -mean(signal)
std(inverted) = std(signal)  # std is magnitude, sign-independent
normalized_inverted = (-signal - (-mean(signal))) / std(signal)
                    = (-signal + mean(signal)) / std(signal)
                    = -(signal - mean(signal)) / std(signal)
                    = -normalized
```
**Conclusion**: Normalized signal is also inverted.

### Raw:
Obviously preserves polarity (no transformation).

**Therefore**: ALL normalization methods preserve signal polarity. If input is inverted, output is inverted.
