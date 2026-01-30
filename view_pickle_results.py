#!/usr/bin/env python3
"""
Export pickle output files from model inference to CSV
"""
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def export_pickle_to_csv(pickle_path, output_path=None):
    """Load pickle file and export to CSV"""
    print(f"Loading: {pickle_path}")

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Auto-generate output path if not provided
        if output_path is None:
            pickle_file = Path(pickle_path)
            output_path = pickle_file.parent / f"{pickle_file.stem}.csv"

        # Export based on data structure
        if isinstance(data, dict):
            print(f"\nData type: Dictionary with keys: {list(data.keys())}")

            # Check if predictions and labels are nested dicts (per-subject results)
            if 'predictions' in data and isinstance(data['predictions'], dict):
                print(f"  Found predictions dict with {len(data['predictions'])} subjects")

                # Extract per-subject-chunk data
                rows = []
                for subject_id in data['predictions'].keys():
                    pred_dict = data['predictions'][subject_id]

                    # Check if it's a nested dict of chunks
                    if isinstance(pred_dict, dict):
                        for chunk_id, pred_tensor in pred_dict.items():
                            row = {
                                'subject_id': subject_id,
                                'chunk_id': chunk_id
                            }

                            # Convert torch tensor to numpy if needed
                            if hasattr(pred_tensor, 'numpy'):
                                pred_values = pred_tensor.cpu().numpy().flatten()
                            elif isinstance(pred_tensor, np.ndarray):
                                pred_values = pred_tensor.flatten()
                            else:
                                pred_values = np.array(pred_tensor).flatten()

                            row['prediction_mean'] = pred_values.mean()
                            row['prediction_std'] = pred_values.std()
                            row['prediction_min'] = pred_values.min()
                            row['prediction_max'] = pred_values.max()
                            row['prediction_length'] = len(pred_values)

                            # Get labels for same subject/chunk
                            if 'labels' in data and subject_id in data['labels']:
                                label_dict = data['labels'][subject_id]
                                if isinstance(label_dict, dict) and chunk_id in label_dict:
                                    label_tensor = label_dict[chunk_id]

                                    if hasattr(label_tensor, 'numpy'):
                                        label_values = label_tensor.cpu().numpy().flatten()
                                    elif isinstance(label_tensor, np.ndarray):
                                        label_values = label_tensor.flatten()
                                    else:
                                        label_values = np.array(label_tensor).flatten()

                                    row['label_mean'] = label_values.mean()
                                    row['label_std'] = label_values.std()
                                    row['label_min'] = label_values.min()
                                    row['label_max'] = label_values.max()
                                    row['label_length'] = len(label_values)

                            rows.append(row)
                    else:
                        # Old format: direct arrays per subject
                        row = {'subject_id': subject_id}
                        if isinstance(pred_dict, np.ndarray):
                            row['prediction_mean'] = pred_dict.mean()
                            row['prediction_std'] = pred_dict.std()
                            row['prediction_min'] = pred_dict.min()
                            row['prediction_max'] = pred_dict.max()
                        rows.append(row)

                df = pd.DataFrame(rows)

                # Add metadata
                if 'label_type' in data:
                    print(f"  Label type: {data['label_type']}")
                if 'fs' in data:
                    print(f"  Sampling frequency: {data['fs']}")

            else:
                # Try to create a DataFrame from the dictionary
                df_data = {}

                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        # Flatten if needed
                        if isinstance(value, np.ndarray) and value.ndim > 1:
                            print(f"  {key}: shape {value.shape} - flattening for CSV")
                            df_data[key] = value.flatten()
                        else:
                            df_data[key] = value
                    elif isinstance(value, (int, float, str, bool)):
                        # Single values - replicate to match length of arrays
                        df_data[key] = value
                    else:
                        print(f"  {key}: {type(value)} - skipping (complex type)")

                # Create DataFrame with index for scalar values
                if df_data:
                    df = pd.DataFrame(df_data, index=[0])
                else:
                    print("  No exportable data found")
                    return None

            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"\n✅ Exported to: {output_path}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")

        elif isinstance(data, (list, np.ndarray)):
            print(f"\nData type: {'List' if isinstance(data, list) else 'NumPy array'}")

            # Convert to DataFrame
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    df = pd.DataFrame({'values': data})
                else:
                    df = pd.DataFrame(data)
            else:
                df = pd.DataFrame({'values': data})

            df.to_csv(output_path, index=False)
            print(f"\n✅ Exported to: {output_path}")
            print(f"   Shape: {df.shape}")

        else:
            print(f"\nData type: {type(data)} - cannot export to CSV")
            return None

        return output_path

    except Exception as e:
        print(f"❌ Error processing pickle file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) > 1:
        pickle_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default to TSCAN results if no argument provided
        pickle_path = "runs/inference_TSCAN_PURE/Custom_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_TSCAN_Custom_outputs.pickle"
        output_path = None

    if not Path(pickle_path).exists():
        print(f"❌ File not found: {pickle_path}")
        print("\nUsage: python view_pickle_results.py [path_to_pickle_file] [optional_output_csv_path]")
        return

    export_pickle_to_csv(pickle_path, output_path)

if __name__ == "__main__":
    main()
