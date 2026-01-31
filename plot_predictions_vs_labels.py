#!/usr/bin/env python3
"""
Plot predictions vs ground truth to visualize the high/low swap issue
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import glob

def extract_hr_fft(signal, fs=30):
    """Extract HR using FFT"""
    N = len(signal)
    signal = signal.flatten() if len(signal.shape) > 1 else signal

    # FFT
    yf = fft(signal - signal.mean())
    xf = fftfreq(N, 1/fs)

    # Only positive frequencies
    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])

    # HR range: 40-180 BPM = 0.67-3 Hz
    hr_mask = (xf >= 0.67) & (xf <= 3.0)
    hr_freqs = xf[hr_mask]
    hr_power = yf[hr_mask]

    if len(hr_power) > 0:
        peak_idx = np.argmax(hr_power)
        hr_hz = hr_freqs[peak_idx]
        hr_bpm = hr_hz * 60
        return hr_bpm
    return None

# Find result files
result_files = glob.glob("runs/inference_*/*/saved_test_outputs/*.pickle")

if not result_files:
    print("No result files found!")
    exit(1)

print(f"Found {len(result_files)} result files")
print(f"Using: {result_files[0]}")
print()

# Load results
with open(result_files[0], 'rb') as f:
    data = pickle.load(f)

predictions = data['predictions']
labels = data['labels']

# Extract HRs for all subjects
pred_hrs = []
label_hrs = []
subject_names = []

for subject_id in predictions.keys():
    pred = predictions[subject_id]
    label = labels[subject_id]

    # Handle nested dict structure
    if isinstance(pred, dict):
        continue  # Skip if it's another dict level
    if isinstance(label, dict):
        continue

    # Convert torch tensors to numpy if needed
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(label, 'cpu'):
        label = label.cpu().numpy()

    # Handle different array shapes
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    pred_hr = extract_hr_fft(pred, fs=30)
    label_hr = extract_hr_fft(label, fs=30)

    if pred_hr and label_hr and 40 < pred_hr < 180 and 40 < label_hr < 180:
        pred_hrs.append(pred_hr)
        label_hrs.append(label_hr)
        subject_names.append(subject_id)

pred_hrs = np.array(pred_hrs)
label_hrs = np.array(label_hrs)

print(f"Analyzed {len(pred_hrs)} subjects")
print()

# Statistics
correlation = np.corrcoef(pred_hrs, label_hrs)[0, 1]
mae = np.mean(np.abs(pred_hrs - label_hrs))
bias = np.mean(pred_hrs - label_hrs)

print(f"Overall Statistics:")
print(f"  Correlation: {correlation:.3f}")
print(f"  MAE: {mae:.2f} BPM")
print(f"  Bias: {bias:.2f} BPM")
print()

# Check high vs low separately
high_mask = label_hrs > 85
low_mask = label_hrs < 65
mid_mask = (label_hrs >= 65) & (label_hrs <= 85)

if high_mask.sum() > 0:
    high_bias = np.mean(pred_hrs[high_mask] - label_hrs[high_mask])
    print(f"High HR (>85 BPM): {high_mask.sum()} samples, bias = {high_bias:.2f} BPM")

if mid_mask.sum() > 0:
    mid_bias = np.mean(pred_hrs[mid_mask] - label_hrs[mid_mask])
    print(f"Mid HR (65-85 BPM): {mid_mask.sum()} samples, bias = {mid_bias:.2f} BPM")

if low_mask.sum() > 0:
    low_bias = np.mean(pred_hrs[low_mask] - label_hrs[low_mask])
    print(f"Low HR (<65 BPM): {low_mask.sum()} samples, bias = {low_bias:.2f} BPM")

print()

# Inversion check
if correlation < 0:
    print("⚠️  NEGATIVE CORRELATION - Signal is INVERTED!")
elif high_mask.sum() > 0 and low_mask.sum() > 0:
    if high_bias < -10 and low_bias > 10:
        print("⚠️  HIGH→LOW and LOW→HIGH pattern detected - Signal likely INVERTED!")
    elif abs(bias) > 10:
        print(f"⚠️  Systematic bias of {bias:.1f} BPM - May need calibration")
    else:
        print("✓ No obvious inversion pattern")

# Create plot
plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(label_hrs, pred_hrs, alpha=0.5, s=50)
plt.plot([40, 180], [40, 180], 'r--', label='Perfect prediction')
plt.xlabel('Ground Truth HR (BPM)')
plt.ylabel('Predicted HR (BPM)')
plt.title(f'Predictions vs Ground Truth\n(Correlation: {correlation:.3f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.xlim(40, 180)
plt.ylim(40, 180)

# Bland-Altman plot
plt.subplot(1, 2, 2)
diff = pred_hrs - label_hrs
mean_hr = (pred_hrs + label_hrs) / 2
plt.scatter(mean_hr, diff, alpha=0.5, s=50)
plt.axhline(y=0, color='r', linestyle='--', label='Zero bias')
plt.axhline(y=np.mean(diff), color='b', linestyle='-', label=f'Mean bias: {np.mean(diff):.1f}')
plt.xlabel('Average HR (BPM)')
plt.ylabel('Difference (Pred - Truth) BPM')
plt.title('Bland-Altman Plot')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: prediction_analysis.png")
print(f"  Open this image to visualize the prediction pattern")
