#!/usr/bin/env python3
"""
Fix PPG signal polarity by inverting and adding offset
"""

import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime

print("="*70)
print("PPG POLARITY FIX")
print("="*70)

# Backup and fix settings
BACKUP_DIR = "raw_data_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_DATA_PATH = "raw_data"
DRY_RUN = True  # Set to False to actually modify files

print(f"\nSettings:")
print(f"  Backup directory: {BACKUP_DIR}")
print(f"  Raw data path: {RAW_DATA_PATH}")
print(f"  Dry run: {DRY_RUN}")
print()

if DRY_RUN:
    print("⚠️  DRY RUN MODE - No files will be modified")
    print("   Set DRY_RUN = False in the script to apply changes")
    print()

# Get all subjects
subjects = sorted([d for d in os.listdir(RAW_DATA_PATH)
                   if os.path.isdir(os.path.join(RAW_DATA_PATH, d))])

print(f"Found {len(subjects)} subjects\n")

# Process each subject
stats = {'processed': 0, 'skipped': 0, 'errors': 0}

for subject in subjects:
    ppg_file = os.path.join(RAW_DATA_PATH, subject, "ppg.csv")

    if not os.path.exists(ppg_file):
        print(f"{subject}: No PPG file - SKIP")
        stats['skipped'] += 1
        continue

    try:
        # Read PPG data
        df = pd.read_csv(ppg_file)

        if 'green' not in df.columns:
            print(f"{subject}: No 'green' column - SKIP")
            stats['skipped'] += 1
            continue

        # Get original statistics
        original = df['green'].values
        orig_min = original.min()
        orig_max = original.max()
        orig_mean = original.mean()

        # Method 1: Invert signal (multiply by -1)
        # This flips the signal so peaks become valleys and vice versa
        inverted = -original

        # Method 2: Add offset to make all values positive
        # Shifts the signal up so minimum value is 0
        offset = abs(inverted.min())
        corrected = inverted + offset

        # New statistics
        new_min = corrected.min()
        new_max = corrected.max()
        new_mean = corrected.mean()

        print(f"{subject}:")
        print(f"  Original: [{orig_min:.0f}, {orig_max:.0f}], mean={orig_mean:.0f}")
        print(f"  Corrected: [{new_min:.0f}, {new_max:.0f}], mean={new_mean:.0f}")

        if not DRY_RUN:
            # Backup original file (first time only)
            if stats['processed'] == 0:
                os.makedirs(BACKUP_DIR, exist_ok=True)

            backup_subject_dir = os.path.join(BACKUP_DIR, subject)
            os.makedirs(backup_subject_dir, exist_ok=True)
            backup_file = os.path.join(backup_subject_dir, "ppg.csv")
            shutil.copy2(ppg_file, backup_file)

            # Update dataframe with corrected values
            df['green'] = corrected

            # Save corrected file
            df.to_csv(ppg_file, index=False)
            print(f"  ✓ Fixed and saved")
        else:
            print(f"  ✓ Would be fixed (dry run)")

        stats['processed'] += 1

    except Exception as e:
        print(f"{subject}: ERROR - {e}")
        stats['errors'] += 1

    print()

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Processed: {stats['processed']}")
print(f"Skipped: {stats['skipped']}")
print(f"Errors: {stats['errors']}")

if DRY_RUN:
    print(f"\n⚠️  This was a DRY RUN - no files were modified")
    print(f"Review the changes above, then set DRY_RUN = False to apply")
else:
    print(f"\n✓ PPG files have been corrected")
    print(f"✓ Originals backed up to: {BACKUP_DIR}")
    print(f"\nNext steps:")
    print(f"1. Delete preprocessed_data/Custom_* directories")
    print(f"2. Re-run models with DO_PREPROCESS: true")
