#!/usr/bin/env python3
"""
Simple check for PPG signal inversion
"""

import numpy as np
import pandas as pd
import os

print("="*70)
print("PPG SIGNAL POLARITY CHECK")
print("="*70)

# Check first few subjects
raw_data_path = "raw_data"
subjects = sorted([d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))])[:5]

print(f"\nChecking {len(subjects)} subjects...")
print()

for subject in subjects:
    ppg_file = os.path.join(raw_data_path, subject, "ppg.csv")

    if not os.path.exists(ppg_file):
        print(f"{subject}: NO PPG FILE")
        continue

    df = pd.read_csv(ppg_file)

    if 'green' not in df.columns:
        print(f"{subject}: NO 'green' COLUMN")
        continue

    ppg = df['green'].values

    # Statistics
    ppg_min = ppg.min()
    ppg_max = ppg.max()
    ppg_mean = ppg.mean()
    ppg_std = ppg.std()

    # Check trend
    trend = np.mean(np.diff(ppg))

    # Print summary
    print(f"{subject}:")
    print(f"  Range: [{ppg_min:.0f}, {ppg_max:.0f}]")
    print(f"  Mean: {ppg_mean:.0f}")
    print(f"  Trend: {'↓ Decreasing' if trend < -100 else ('↑ Increasing' if trend > 100 else '→ Stable')}")

    # Check for issues
    issues = []
    if ppg_mean < 0:
        issues.append("NEGATIVE MEAN")
    if ppg_min < -10000:
        issues.append("VERY NEGATIVE VALUES")
    if trend < -1000:
        issues.append("STRONG DOWNWARD TREND")

    if issues:
        print(f"  ⚠️  WARNING: {', '.join(issues)}")
    print()

print("="*70)
print("ANALYSIS")
print("="*70)
print("""
Normal PPG Signal:
  - Range: Usually 0 to ~300,000 (raw sensor values)
  - Mean: Should be positive
  - Trend: Should be relatively stable (small drift OK)

Your PPG Signal:
  - Has NEGATIVE values
  - Has DECREASING trend
  - This suggests the signal might be INVERTED or has wrong polarity

Possible Solutions:
  1. Check if sensor recorded in absorption mode (inverted)
  2. Multiply PPG values by -1 before feeding to model
  3. Add offset to make all values positive
  4. Check with data source about signal polarity
""")
