#!/usr/bin/env python3
"""
Diagnose if PPG signals are inverted during preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import os

print("="*70)
print("SIGNAL INVERSION DIAGNOSTIC")
print("="*70)

# 1. Check raw ground truth PPG
print("\n1. Checking raw ground truth PPG signal...")
raw_data_path = "raw_data"
subjects = [d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]

if subjects:
    subject = subjects[0]
    ppg_file = os.path.join(raw_data_path, subject, "ppg.csv")

    if os.path.exists(ppg_file):
        df = pd.read_csv(ppg_file)
        print(f"   Subject: {subject}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Shape: {df.shape}")

        if 'green' in df.columns:
            ppg = df['green'].values[:300]  # First 300 samples
            print(f"   PPG range: [{ppg.min():.2f}, {ppg.max():.2f}]")
            print(f"   PPG mean: {ppg.mean():.2f}")
            print(f"   PPG std: {ppg.std():.2f}")

            # Check if signal increases or decreases at peaks
            # For reflection PPG: peaks should have HIGH values
            # For absorption PPG: peaks should have LOW values
            diff = np.diff(ppg)
            print(f"   Mean derivative: {diff.mean():.4f}")
            print(f"   Signal trend: {'Increasing' if diff.mean() > 0 else 'Decreasing'}")
        else:
            print(f"   ERROR: No 'green' column found!")
    else:
        print(f"   ERROR: No ppg.csv found for {subject}")
else:
    print("   ERROR: No subjects found in raw_data/")

# 2. Check model predictions vs ground truth
print("\n2. Checking model predictions vs ground truth...")
result_files = glob.glob("runs/inference_*/*/saved_test_outputs/*.pickle")

if result_files:
    # Use first available result
    result_file = result_files[0]
    model_name = result_file.split('/')[-4].replace('inference_', '')

    print(f"   Model: {model_name}")
    print(f"   File: {result_file}")

    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    # Handle dict structure
    if isinstance(data, dict):
        if 'predictions' in data:
            predictions = data['predictions']
            if isinstance(predictions, dict):
                # Get first key's data
                first_key = list(predictions.keys())[0]
                predictions = predictions[first_key]
        labels = data.get('labels', data.get('label', None))
        if isinstance(labels, dict):
            first_key = list(labels.keys())[0]
            labels = labels[first_key]
    else:
        predictions = data
        labels = None

    if predictions is not None:
        print(f"   Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
    if labels is not None:
        print(f"   Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")

    # Extract HR from first few samples
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq

    def extract_hr_fft(signal, fs=30):
        """Extract HR using FFT"""
        # FFT
        N = len(signal)
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

        # Peak frequency
        if len(hr_power) > 0:
            peak_idx = np.argmax(hr_power)
            hr_hz = hr_freqs[peak_idx]
            hr_bpm = hr_hz * 60
            return hr_bpm
        return None

    if predictions is None or labels is None:
        print("   ERROR: Could not extract predictions or labels from pickle file")
        print(f"   Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
    elif len(predictions) == 0 or len(labels) == 0:
        print("   ERROR: Empty predictions or labels")
    else:
        print("\n   Sample HR comparison:")
        print("   " + "-"*60)
        print("   Sample | Pred HR | Label HR | Diff | Pattern")
        print("   " + "-"*60)

        for i in range(min(5, len(predictions))):
            pred_hr = extract_hr_fft(predictions[i], fs=30)
            label_hr = extract_hr_fft(labels[i], fs=30)

            if pred_hr and label_hr:
                diff = pred_hr - label_hr

                # Check for inversion pattern
                if label_hr > 85:  # High HR
                    pattern = "HIGH→LOW" if pred_hr < label_hr - 10 else "OK"
                elif label_hr < 65:  # Low HR
                    pattern = "LOW→HIGH" if pred_hr > label_hr + 10 else "OK"
                else:
                    pattern = "NORMAL"

                print(f"   {i:6d} | {pred_hr:7.1f} | {label_hr:8.1f} | {diff:+5.1f} | {pattern}")

# 3. Check correlation pattern
print("\n3. Checking correlation pattern...")
if result_files and 'predictions' in locals() and predictions is not None and labels is not None:
    # Extract HRs for all samples
    pred_hrs = []
    label_hrs = []

    for i in range(len(predictions)):
        pred_hr = extract_hr_fft(predictions[i], fs=30)
        label_hr = extract_hr_fft(labels[i], fs=30)
        if pred_hr and label_hr:
            pred_hrs.append(pred_hr)
            label_hrs.append(label_hr)

    pred_hrs = np.array(pred_hrs)
    label_hrs = np.array(label_hrs)

    # Check correlation
    correlation = np.corrcoef(pred_hrs, label_hrs)[0, 1]
    print(f"   Correlation: {correlation:.3f}")

    # Check if inverse correlation suggests inversion
    if correlation < 0:
        print("   WARNING: NEGATIVE CORRELATION - Signal is likely INVERTED!")
    elif correlation < 0.5:
        print("   WARNING: LOW CORRELATION - Check preprocessing")
    else:
        print("   Correlation looks OK")

    # Check systematic bias
    bias = (pred_hrs - label_hrs).mean()
    print(f"   Mean bias: {bias:.2f} BPM")

    # Check high vs low separately
    high_mask = label_hrs > 85
    low_mask = label_hrs < 65

    if high_mask.sum() > 0:
        high_bias = (pred_hrs[high_mask] - label_hrs[high_mask]).mean()
        print(f"   High HR bias (>85 BPM): {high_bias:.2f} BPM")

    if low_mask.sum() > 0:
        low_bias = (pred_hrs[low_mask] - label_hrs[low_mask]).mean()
        print(f"   Low HR bias (<65 BPM): {low_bias:.2f} BPM")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nNext steps:")
print("- If correlation is NEGATIVE: PPG signal is inverted")
print("- If high HR has negative bias and low HR has positive bias: systematic inversion")
print("- Solution: Multiply PPG signal by -1 in preprocessing")
