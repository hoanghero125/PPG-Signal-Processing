# 📋 Instruction Guide: Testing 9 PURE-Trained Pretrained Models on Custom Dataset

## Goal
Test 9 pretrained models (all trained on PURE dataset) on your custom preprocessed data and compare their performance.

---

## ✅ Prerequisites (Already Done)
- Custom dataset preprocessed: `preprocessed_data/Custom_SizeW72...` ✅
- 48 subjects, 716 chunks ready ✅
- CustomLoader registered in toolbox ✅

---

## 📝 Task 1: Create 9 Inference Config Files

Create these config files in `configs/infer_configs/`:

### 1. `CUSTOM_TSCAN_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_TSCAN_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: Tscan
  TSCAN:
    FRAME_DEPTH: 10

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_TSCAN.pth"
```

### 2. `CUSTOM_DeepPhys_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_DeepPhys_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: DeepPhys

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_DeepPhys.pth"
```

### 3. `CUSTOM_EfficientPhys_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_EfficientPhys_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: EfficientPhys
  EFFICIENTPHYS:
    FRAME_DEPTH: 10

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_EfficientPhys.pth"
```

### 4. `CUSTOM_PhysNet_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_PhysNet_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: Physnet
  PHYSNET:
    FRAME_NUM: 64

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_PhysNet_DiffNormalized.pth"
```

### 5. `CUSTOM_PhysFormer_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_PhysFormer_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: PhysFormer
  PHYSFORMER:
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    PATCH_SIZE: 4
    THETA: 0.7

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_PhysFormer_DiffNormalized.pth"
```

### 6. `CUSTOM_PhysMamba_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_PhysMamba_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: PhysMamba

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_PhysMamba_DiffNormalized.pth"
```

### 7. `CUSTOM_RhythmFormer_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_RhythmFormer_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: RhythmFormer

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_RhythmFormer.pth"
```

### 8. `CUSTOM_FactorizePhys_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_FactorizePhys_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: FactorizePhys
  FactorizePhys:
    CHANNELS: 3
    FRAME_NUM: 160
    MD_FSAM: False
    MD_INFERENCE: True
    MD_R: 1
    MD_RESIDUAL: True
    MD_S: 1
    MD_STEPS: 4
    MD_TRANSFORM: T_KAB
    MD_TYPE: NMF
    TYPE: Standard

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_FactorizePhys_FSAM_Res.pth"
```

### 9. `CUSTOM_iBVPNet_PURE.yaml`
```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

LOG:
  PATH: runs/inference_iBVPNet_PURE

MODEL:
  DROP_RATE: 0.2
  NAME: iBVPNet
  iBVPNet:
    CHANNELS: 3
    FRAME_NUM: 160

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "final_model_release/PURE_iBVPNet.pth"
```

---

## 📝 Task 2: Create Batch Testing Script

Create: `test_all_pure_models.py`

```python
#!/usr/bin/env python3
"""
Batch testing script for all PURE-trained pretrained models
"""
import subprocess
import os
import json
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
        else:
            print(f"❌ {model_name} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            status = "failed"
            
    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"⏰ {model_name} timed out after {duration:.1f}s")
        status = "timeout"
        result = None
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"💥 {model_name} crashed: {e}")
        status = "error"
        result = None
    
    return {
        'model': model_name,
        'config': config_path,
        'status': status,
        'duration_seconds': duration,
        'timestamp': end_time.isoformat(),
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
```

---

## 📝 Task 3: Create Results Comparison Script

Create: `compare_model_results.py`

```python
#!/usr/bin/env python3
"""
Parse and compare results from all tested models
"""
import os
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

def parse_metrics_from_log(log_dir):
    """
    Parse metrics from model log directory
    Look for metric files or output logs
    """
    metrics = {
        'MAE': None,
        'RMSE': None,
        'MAPE': None,
        'Pearson': None,
        'SNR': None,
    }
    
    # Try to find metrics file (adjust path based on actual output structure)
    possible_files = [
        os.path.join(log_dir, 'test_results.txt'),
        os.path.join(log_dir, 'metrics.txt'),
        os.path.join(log_dir, 'results.txt'),
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Parse metrics using regex (adjust patterns based on actual format)
                mae_match = re.search(r'MAE[:\s]+([0-9.]+)', content)
                rmse_match = re.search(r'RMSE[:\s]+([0-9.]+)', content)
                mape_match = re.search(r'MAPE[:\s]+([0-9.]+)', content)
                pearson_match = re.search(r'Pearson[:\s]+([0-9.]+)', content)
                snr_match = re.search(r'SNR[:\s]+([0-9.]+)', content)
                
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
            
            break
    
    return metrics

def main():
    """Compare all model results"""
    print("="*70)
    print("📊 COMPARING MODEL RESULTS")
    print("="*70)
    
    results = []
    
    for model_name, log_dir in MODELS.items():
        print(f"\n📁 Parsing: {model_name}")
        
        if not os.path.exists(log_dir):
            print(f"   ⚠️  Log directory not found: {log_dir}")
            continue
        
        metrics = parse_metrics_from_log(log_dir)
        
        results.append({
            'Model': model_name,
            **metrics
        })
        
        print(f"   MAE: {metrics['MAE']}")
        print(f"   RMSE: {metrics['RMSE']}")
        print(f"   MAPE: {metrics['MAPE']}")
        print(f"   Pearson: {metrics['Pearson']}")
        print(f"   SNR: {metrics['SNR']}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
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
```

---

## 🚀 Execution Steps

### Step 1: Test One Model First
```bash
python main.py --config_file configs/infer_configs/CUSTOM_TSCAN_PURE.yaml
```

**Expected output:**
- Model loads pretrained weights ✅
- Runs inference on 716 chunks ✅
- Outputs metrics (MAE, RMSE, etc.) ✅
- Saves results to `runs/inference_TSCAN_PURE/` ✅

### Step 2: If Successful, Run All Models
```bash
python test_all_pure_models.py
```

**This will:**
- Test all 9 models sequentially
- Save progress log
- Create `inference_batch_results.json`

### Step 3: Compare Results
```bash
python compare_model_results.py
```

**This will:**
- Parse metrics from all model outputs
- Create comparison table
- Generate visualization plots
- Save results as CSV and PNG

---

## 📊 Expected Output

```
model_comparison_results.csv
model_comparison_plot.png
inference_batch_results.json

runs/
├── inference_TSCAN_PURE/
├── inference_DeepPhys_PURE/
├── inference_EfficientPhys_PURE/
├── inference_PhysNet_PURE/
├── inference_PhysFormer_PURE/
├── inference_PhysMamba_PURE/
├── inference_RhythmFormer_PURE/
├── inference_FactorizePhys_PURE/
└── inference_iBVPNet_PURE/
```

---

## ⚠️ Important Notes

1. **Model names are case-sensitive**: `Tscan` not `TSCAN`, `Physnet` not `PhysNet`
2. **All configs use same preprocessed data** - no re-preprocessing needed
3. **Each model outputs to separate directory** to avoid conflicts
4. **GPU memory**: If OOM errors occur, reduce `INFERENCE.BATCH_SIZE` from 4 to 2 or 1
5. **Timeout**: Each model has 30-minute timeout; adjust if needed

---

## ✅ Success Checklist

- [ ] 9 config files created in `configs/infer_configs/`
- [ ] `test_all_pure_models.py` created
- [ ] `compare_model_results.py` created
- [ ] Single model test successful
- [ ] Batch test runs without errors
- [ ] Comparison table generated
- [ ] Results ranked by performance

---

**Ready to execute! Feed this to Claude Code.** 🚀