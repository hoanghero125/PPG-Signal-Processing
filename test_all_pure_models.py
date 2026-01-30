#!/usr/bin/env python3
"""
Batch testing script for all PURE-trained pretrained models
"""
import subprocess
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# List of 9 config files to test
CONFIGS = [
    "configs/infer_configs/CUSTOM_TSCAN_PURE.yaml",
    "configs/infer_configs/CUSTOM_DeepPhys_PURE.yaml",
    "configs/infer_configs/CUSTOM_EfficientPhys_PURE.yaml",
    "configs/infer_configs/CUSTOM_PhysNet_PURE.yaml",
    "configs/infer_configs/CUSTOM_PhysFormer_PURE.yaml",
    "configs/infer_configs/CUSTOM_PhysMamba_PURE.yaml",
    "configs/infer_configs/CUSTOM_RhythmFormer_PURE.yaml",
    "configs/infer_configs/CUSTOM_FactorizePhys_PURE.yaml",
    "configs/infer_configs/CUSTOM_iBVPNet_PURE.yaml",
]

def convert_pickle_to_csv(pickle_path):
    """Convert pickle file to CSV (copied from view_pickle_results.py)"""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Auto-generate output path
        pickle_file = Path(pickle_path)
        output_path = pickle_file.parent / f"{pickle_file.stem}.csv"

        # Check if predictions and labels are nested dicts (per-subject results)
        if isinstance(data, dict) and 'predictions' in data and isinstance(data['predictions'], dict):
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

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            return str(output_path)

    except Exception as e:
        print(f"  ⚠️  Error converting pickle to CSV: {e}")
        return None

def run_inference(config_path):
    """Run inference for a single model"""
    model_name = Path(config_path).stem

    print(f"\n{'='*70}")
    print(f"🚀 Testing: {model_name}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            ["python", "main.py", "--config_file", config_path],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            print(f"✅ {model_name} completed in {duration:.1f}s")
            status = "success"

            # Parse metrics from stdout
            import re
            metrics = {}
            mae_match = re.search(r'FFT MAE[^:]*:\s*([0-9.]+)', result.stdout)
            rmse_match = re.search(r'FFT RMSE[^:]*:\s*([0-9.]+)', result.stdout)
            mape_match = re.search(r'FFT MAPE[^:]*:\s*([0-9.]+)', result.stdout)
            pearson_match = re.search(r'FFT Pearson[^:]*:\s*([0-9.]+)', result.stdout)
            snr_match = re.search(r'FFT SNR[^:]*:\s*([-.0-9]+)', result.stdout)

            if mae_match:
                metrics['MAE'] = float(mae_match.group(1))
            if rmse_match:
                metrics['RMSE'] = float(rmse_match.group(1))
            if mape_match:
                metrics['MAPE'] = float(mape_match.group(1))
            if pearson_match:
                metrics['Pearson'] = float(pearson_match.group(1))
            if snr_match:
                metrics['SNR'] = float(snr_match.group(1))
        else:
            print(f"❌ {model_name} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            status = "failed"
            metrics = {}

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"⏰ {model_name} timed out after {duration:.1f}s")
        status = "timeout"
        result = None
        metrics = {}

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"💥 {model_name} crashed: {e}")
        status = "error"
        result = None
        metrics = {}

    # Convert pickle to CSV if successful
    pickle_path = None
    csv_path = None
    if status == "success":
        # Find the pickle file
        model_log_dir = f"runs/inference_{model_name}"
        if os.path.exists(model_log_dir):
            pickle_files = list(Path(model_log_dir).rglob("*.pickle"))
            if pickle_files:
                pickle_path = str(pickle_files[0])
                print(f"📦 Converting pickle to CSV: {pickle_path}")
                csv_path = convert_pickle_to_csv(pickle_path)
                if csv_path:
                    print(f"✅ CSV saved: {csv_path}")

    return {
        'model': model_name,
        'config': config_path,
        'status': status,
        'duration_seconds': duration,
        'timestamp': end_time.isoformat(),
        'metrics': metrics,
        'pickle_path': pickle_path,
        'csv_path': csv_path,
    }

def main():
    """Run all model inferences"""
    print("="*70)
    print("🔬 BATCH INFERENCE: Testing 9 PURE-trained Models")
    print("="*70)

    results = []

    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Processing {config}")

        # Check if config exists
        if not os.path.exists(config):
            print(f"⚠️  Config file not found: {config}")
            results.append({
                'model': Path(config).stem,
                'config': config,
                'status': 'config_not_found',
                'duration_seconds': 0,
                'timestamp': datetime.now().isoformat(),
            })
            continue

        # Run inference
        result = run_inference(config)
        results.append(result)

    # Save results summary
    summary_file = 'inference_batch_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("📊 BATCH INFERENCE SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    error_count = sum(1 for r in results if r['status'] == 'error')
    timeout_count = sum(1 for r in results if r['status'] == 'timeout')

    print(f"\n✅ Successful: {success_count}/{len(CONFIGS)}")
    print(f"❌ Failed: {failed_count}/{len(CONFIGS)}")
    print(f"💥 Errors: {error_count}/{len(CONFIGS)}")
    print(f"⏰ Timeouts: {timeout_count}/{len(CONFIGS)}")

    print(f"\n📄 Results saved to: {summary_file}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
