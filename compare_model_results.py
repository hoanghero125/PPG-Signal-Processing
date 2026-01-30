#!/usr/bin/env python3
"""
Parse and compare results from all tested models
Reads metrics from inference_batch_results.json created by test_all_pure_models.py
"""
import os
import re
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Model names and their log directories
MODELS = {
    'TSCAN': 'runs/inference_TSCAN_PURE',
    'DeepPhys': 'runs/inference_DeepPhys_PURE',
    'EfficientPhys': 'runs/inference_EfficientPhys_PURE',
    'PhysNet': 'runs/inference_PhysNet_PURE',
    'PhysFormer': 'runs/inference_PhysFormer_PURE',
    'PhysMamba': 'runs/inference_PhysMamba_PURE',
    'RhythmFormer': 'runs/inference_RhythmFormer_PURE',
    'FactorizePhys': 'runs/inference_FactorizePhys_PURE',
    'iBVPNet': 'runs/inference_iBVPNet_PURE',
}

def parse_metrics_from_user():
    """
    Manually collect metrics from user input
    User should copy-paste the console output containing metrics
    """
    print("\n" + "="*70)
    print("📋 MANUAL METRICS COLLECTION")
    print("="*70)
    print("\nFor each model that ran successfully, please provide the metrics")
    print("from the console output (the lines that say 'FFT MAE', 'FFT RMSE', etc.)\n")

    all_metrics = []

    for model_name, log_dir in MODELS.items():
        if not os.path.exists(log_dir):
            print(f"\n⚠️  {model_name}: Log directory not found, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print("Please enter the metrics (press Enter to skip if not available):")

        try:
            mae = input(f"  MAE (e.g., 13.4765625): ").strip()
            rmse = input(f"  RMSE (e.g., 18.468798564711488): ").strip()
            mape = input(f"  MAPE (e.g., 17.383576697884514): ").strip()
            pearson = input(f"  Pearson (e.g., 0.37737173284765857): ").strip()
            snr = input(f"  SNR (e.g., -6.268942173333062): ").strip()

            all_metrics.append({
                'Model': model_name,
                'MAE': float(mae) if mae else None,
                'RMSE': float(rmse) if rmse else None,
                'MAPE': float(mape) if mape else None,
                'Pearson': float(pearson) if pearson else None,
                'SNR': float(snr) if snr else None,
            })
        except ValueError as e:
            print(f"  ⚠️  Invalid input, skipping {model_name}")
            continue

    return all_metrics

def parse_metrics_auto():
    """
    Try to automatically find metrics from log files or saved outputs
    """
    print("\n" + "="*70)
    print("📋 AUTOMATIC METRICS COLLECTION")
    print("="*70)
    print("\nSearching for metrics in log directories...\n")

    all_metrics = []

    for model_name, log_dir in MODELS.items():
        print(f"📁 Checking: {model_name}")

        if not os.path.exists(log_dir):
            print(f"   ⚠️  Log directory not found")
            continue

        metrics = {
            'Model': model_name,
            'MAE': None,
            'RMSE': None,
            'MAPE': None,
            'Pearson': None,
            'SNR': None,
        }

        # Try to find any log files or text files
        log_files = list(Path(log_dir).rglob("*.log")) + list(Path(log_dir).rglob("*.txt"))

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()

                    # Parse metrics using regex
                    mae_match = re.search(r'(?:FFT )?MAE[:\s]+([0-9.]+)', content)
                    rmse_match = re.search(r'(?:FFT )?RMSE[:\s]+([0-9.]+)', content)
                    mape_match = re.search(r'(?:FFT )?MAPE[:\s]+([0-9.]+)', content)
                    pearson_match = re.search(r'(?:FFT )?Pearson[:\s]+([0-9.]+)', content)
                    snr_match = re.search(r'(?:FFT )?SNR[:\s]+([-.0-9]+)', content)

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
            except Exception as e:
                continue

        if metrics['MAE'] is not None:
            print(f"   ✅ Found metrics: MAE={metrics['MAE']}")
        else:
            print(f"   ⚠️  No metrics found")

        all_metrics.append(metrics)

    return all_metrics

def main():
    """Compare all model results"""
    print("="*70)
    print("📊 COMPARING MODEL RESULTS")
    print("="*70)

    results = []

    # First, try to read from inference_batch_results.json
    batch_results_file = 'inference_batch_results.json'
    if os.path.exists(batch_results_file):
        print(f"\n📄 Reading metrics from {batch_results_file}...")

        with open(batch_results_file, 'r') as f:
            batch_data = json.load(f)

        for item in batch_data:
            model_name = item['model'].replace('CUSTOM_', '').replace('_PURE', '')
            metrics = item.get('metrics', {})

            results.append({
                'Model': model_name,
                'Status': item['status'],
                'MAE': metrics.get('MAE'),
                'RMSE': metrics.get('RMSE'),
                'MAPE': metrics.get('MAPE'),
                'Pearson': metrics.get('Pearson'),
                'SNR': metrics.get('SNR'),
            })

            print(f"  {model_name}: {item['status']}" +
                  (f" - MAE={metrics.get('MAE')}" if metrics.get('MAE') else ""))

    else:
        print(f"\n⚠️  {batch_results_file} not found.")
        print("Did you run 'python test_all_pure_models.py' yet?")
        print("\nTrying to parse from log directories...")
        results = parse_metrics_auto()

    # Check if we got any metrics
    if not results or not any(r.get('MAE') is not None for r in results):
        print("\n⚠️  No metrics found automatically.")
        response = input("\nWould you like to enter metrics manually? (y/n): ").strip().lower()
        if response == 'y':
            results = parse_metrics_from_user()
        else:
            print("\n❌ No metrics to compare. Exiting.")
            return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Remove rows with all None metrics
    df = df.dropna(subset=['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'], how='all')

    # Sort by MAE (lower is better)
    df = df.sort_values('MAE')

    # Save to CSV
    output_file = 'model_comparison_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    # Print comparison table
    print("\n" + "="*70)
    print("📋 PERFORMANCE COMPARISON (sorted by MAE)")
    print("="*70)
    print(df.to_string(index=False))

    # Create visualization
    create_comparison_plots(df)

    print("\n" + "="*70)

def create_comparison_plots(df):
    """Create comparison plots"""
    if df.empty or df['MAE'].isna().all():
        print("⚠️  No data available for plotting")
        return

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    metrics = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']

    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Skip if no data
        if df[metric].isna().all():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(metric)
            continue

        # Sort data for plotting
        plot_df = df.sort_values(metric, ascending=(metric not in ['Pearson', 'SNR']))

        # Create bar plot
        bars = ax.barh(plot_df['Model'], plot_df[metric])

        # Color bars (green for best, red for worst)
        if metric in ['Pearson', 'SNR']:
            colors = plt.cm.RdYlGn(plot_df[metric] / plot_df[metric].max())
        else:
            colors = plt.cm.RdYlGn_r(plot_df[metric] / plot_df[metric].max())

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig('model_comparison_plot.png', dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: model_comparison_plot.png")
    plt.close()

if __name__ == "__main__":
    main()
