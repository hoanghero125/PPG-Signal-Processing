#!/usr/bin/env python3
import pickle
import numpy as np

pickle_path = "runs/inference_TSCAN_PURE/Custom_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_TSCAN_Custom_outputs.pickle"

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("\nPredictions type:", type(data['predictions']))
print("Number of subjects:", len(data['predictions']))

# Check first subject
first_subject = list(data['predictions'].keys())[0]
print(f"\n=== First subject: {first_subject} ===")
print(f"Predictions type: {type(data['predictions'][first_subject])}")

if isinstance(data['predictions'][first_subject], dict):
    print(f"Predictions is a dict with keys: {list(data['predictions'][first_subject].keys())}")
    # Check what's inside
    for key, value in data['predictions'][first_subject].items():
        print(f"  {key}: type={type(value)}, ", end="")
        if isinstance(value, np.ndarray):
            print(f"shape={value.shape}, values={value[:3]}")
        else:
            print(f"value={value}")
else:
    print(f"Predictions shape/len: {data['predictions'][first_subject].shape if hasattr(data['predictions'][first_subject], 'shape') else len(data['predictions'][first_subject])}")

print(f"\nLabels type: {type(data['labels'][first_subject])}")
if isinstance(data['labels'][first_subject], dict):
    print(f"Labels is a dict with keys: {list(data['labels'][first_subject].keys())}")
    for key, value in data['labels'][first_subject].items():
        print(f"  {key}: type={type(value)}, ", end="")
        if isinstance(value, np.ndarray):
            print(f"shape={value.shape}, values={value[:3]}")
        else:
            print(f"value={value}")
else:
    print(f"Labels shape/len: {data['labels'][first_subject].shape if hasattr(data['labels'][first_subject], 'shape') else len(data['labels'][first_subject])}")
