#!/usr/bin/env python3
"""
Simple analysis of predictions vs ground truth
"""

import pickle
import numpy as np
import glob
from scipy.fft import fft, fftfreq

def extract_hr_fft(signal, fs=30):
    """Extract HR using FFT"""
    if hasattr(signal, 'cpu'):
        signal = signal.cpu().numpy()
    signal = np.array(signal).flatten()

    N = len(signal)
    yf = fft(signal - signal.mean())
    xf = fftfreq(N, 1/fs)

    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])

    hr_mask = (xf >= 0.67) & (xf <= 3.0)
    hr_freqs = xf[hr_mask]
    hr_power = yf[hr_mask]

    if len(hr_power) > 0:
        peak_idx = np.argmax(hr_power)
        hr_bpm = hr_freqs[peak_idx] * 60
        return hr_bpm
    return None

# Load data
files = glob.glob('runs/inference_*/*/saved_test_outputs/*.pickle')
print(f"Found {len(files)} result files\n")

with open(files[0], 'rb') as f:
    data = pickle.load(f)

predictions = data['predictions']
labels = data['labels']

# Analyze clip-by-clip
pred_hrs = []
label_hrs = []

print("Analyzing clips...")
for subject_id in list(predictions.keys())[:10]:  # First 10 subjects
    pred_clips = predictions[subject_id]
    label_clips = labels[subject_id]

    for clip_id in pred_clips.keys():
        pred = pred_clips[clip_id]
        label = label_clips[clip_id]

        pred_hr = extract_hr_fft(pred, fs=30)
        label_hr = extract_hr_fft(label, fs=30)

        if pred_hr and label_hr and 40 < pred_hr < 180 and 40 < label_hr < 180:
            pred_hrs.append(pred_hr)
            label_hrs.append(label_hr)

            # Print first few for inspection
            if len(pred_hrs) <= 10:
                diff = pred_hr - label_hr
                pattern = ""
                if label_hr > 85 and pred_hr < label_hr - 10:
                    pattern = " [HIGH→LOW]"
                elif label_hr < 65 and pred_hr > label_hr + 10:
                    pattern = " [LOW→HIGH]"
                print(f"  Clip: Pred={pred_hr:.1f}, Label={label_hr:.1f}, Diff={diff:+.1f}{pattern}")

pred_hrs = np.array(pred_hrs)
label_hrs = np.array(label_hrs)

print(f"\n{'='*70}")
print(f"ANALYSIS ({len(pred_hrs)} clips)")
print('='*70)

# Overall stats
corr = np.corrcoef(pred_hrs, label_hrs)[0, 1]
mae = np.mean(np.abs(pred_hrs - label_hrs))
bias = np.mean(pred_hrs - label_hrs)

print(f"\nOverall:")
print(f"  Correlation: {corr:.3f}")
print(f"  MAE: {mae:.2f} BPM")
print(f"  Bias: {bias:.2f} BPM")

# By HR range
high_mask = label_hrs > 85
mid_mask = (label_hrs >= 65) & (label_hrs <= 85)
low_mask = label_hrs < 65

print(f"\nBy HR Range:")
if high_mask.sum() > 0:
    high_bias = np.mean(pred_hrs[high_mask] - label_hrs[high_mask])
    print(f"  High (>85):  {high_mask.sum():3d} clips, bias = {high_bias:+.2f} BPM")

if mid_mask.sum() > 0:
    mid_bias = np.mean(pred_hrs[mid_mask] - label_hrs[mid_mask])
    print(f"  Mid (65-85): {mid_mask.sum():3d} clips, bias = {mid_bias:+.2f} BPM")

if low_mask.sum() > 0:
    low_bias = np.mean(pred_hrs[low_mask] - label_hrs[low_mask])
    print(f"  Low (<65):   {low_mask.sum():3d} clips, bias = {low_bias:+.2f} BPM")

# Diagnosis
print(f"\n{'='*70}")
print("DIAGNOSIS")
print('='*70)

if corr < -0.3:
    print("⚠️  STRONG NEGATIVE CORRELATION - Signal is INVERTED!")
    print("   → Need to flip PPG signal polarity")
elif corr < 0:
    print("⚠️  NEGATIVE CORRELATION - Signal likely inverted")
    print("   → Need to flip PPG signal polarity")
elif high_mask.sum() > 0 and low_mask.sum() > 0:
    if high_bias < -15 and low_bias > 15:
        print("⚠️  Clear HIGH→LOW and LOW→HIGH swap pattern!")
        print(f"   → High HR underestimated by {abs(high_bias):.1f} BPM")
        print(f"   → Low HR overestimated by {low_bias:.1f} BPM")
        print("   → Signal polarity is INVERTED")
    elif abs(bias) > 15:
        print(f"⚠️  Systematic bias of {bias:.1f} BPM")
        print("   → May need calibration or different preprocessing")
    else:
        print("✓ No obvious inversion - predictions look reasonable")
else:
    if corr > 0.7:
        print("✓ Good correlation - model working correctly")
    else:
        print("⚠️  Low correlation - check data quality")
