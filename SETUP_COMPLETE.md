# rPPG-Toolbox: Setup Complete

**Date**: 2026-01-31

---

## What Has Been Created

### 1. Configuration Files (36 total)

All configs created in `configs/infer_configs/`:

**PURE trained (9):**
- CUSTOM_PURE_TSCAN.yaml
- CUSTOM_PURE_DeepPhys.yaml
- CUSTOM_PURE_EfficientPhys.yaml
- CUSTOM_PURE_PhysNet.yaml
- CUSTOM_PURE_PhysFormer.yaml
- CUSTOM_PURE_PhysMamba.yaml
- CUSTOM_PURE_RhythmFormer.yaml
- CUSTOM_PURE_FactorizePhys.yaml
- CUSTOM_PURE_iBVPNet.yaml

**UBFC-rPPG trained (7):**
- CUSTOM_UBFC_TSCAN.yaml
- CUSTOM_UBFC_DeepPhys.yaml
- CUSTOM_UBFC_EfficientPhys.yaml
- CUSTOM_UBFC_PhysNet.yaml
- CUSTOM_UBFC_PhysFormer.yaml
- CUSTOM_UBFC_PhysMamba.yaml
- CUSTOM_UBFC_RhythmFormer.yaml
- CUSTOM_UBFC_FactorizePhys.yaml

**SCAMPS trained (6):**
- CUSTOM_SCAMPS_TSCAN.yaml
- CUSTOM_SCAMPS_DeepPhys.yaml
- CUSTOM_SCAMPS_EfficientPhys.yaml
- CUSTOM_SCAMPS_PhysNet.yaml
- CUSTOM_SCAMPS_PhysFormer.yaml
- CUSTOM_SCAMPS_FactorizePhys.yaml

**iBVP trained (2):**
- CUSTOM_iBVP_EfficientPhys.yaml
- CUSTOM_iBVP_FactorizePhys.yaml

**MA-UBFC trained (4):**
- CUSTOM_MAUBFC_TSCAN.yaml
- CUSTOM_MAUBFC_DeepPhys.yaml
- CUSTOM_MAUBFC_EfficientPhys.yaml
- CUSTOM_MAUBFC_PhysNet.yaml

**BP4D trained (7):**
- CUSTOM_BP4D_TSCAN.yaml
- CUSTOM_BP4D_DeepPhys.yaml
- CUSTOM_BP4D_EfficientPhys.yaml
- CUSTOM_BP4D_PhysNet.yaml
- CUSTOM_BP4D_BigSmall_F1.yaml (commented out in notebook)
- CUSTOM_BP4D_BigSmall_F2.yaml (commented out in notebook)
- CUSTOM_BP4D_BigSmall_F3.yaml (commented out in notebook)

**Note:** BigSmall models may not work with Custom dataset due to special dataset loader requirements.

### 2. Inference Notebook

**File:** `run_all_models_inference.ipynb`

**Features:**
- Run all 36 models sequentially
- Comment/uncomment models to select which to run
- Real-time output streaming
- Automatic metrics capture (MAE, RMSE, MAPE, Pearson, SNR)
- Results summary table
- Exports to JSON and CSV
- Error handling and statistics

### 3. Documentation

**File:** `rPPG_TOOLBOX_DEEP_UNDERSTANDING.md`

**Contents:**
- Complete model-to-config mapping
- Model-specific requirements
- Preprocessing system explanation
- Config creation guidelines
- Testing strategy
- Common issues and solutions

---

## Your Data Structure Required

```
rPPG-Toolbox/
├── raw_data/
│   ├── subject1/
│   │   ├── video.mp4 (or .avi, .mov)
│   │   └── ppg.csv (with 'green' column - ground truth PPG)
│   ├── subject2/
│   │   ├── video.mp4
│   │   └── ppg.csv
│   └── ...
└── preprocessed_data/ (will be created automatically)
```

**ppg.csv format:**
```csv
timestamp,green
0.000,245.3
0.040,248.7
0.080,251.2
...
```

---

## How to Run

### Step 1: Verify Your Data

Check that you have:
- Videos in `raw_data/subject*/`
- PPG ground truth files `ppg.csv` with 'green' column

### Step 2: First Run (Preprocessing)

All configs have `DO_PREPROCESS: True` set for first run.

Open `run_all_models_inference.ipynb` and run all cells.

**What happens:**
- Each model preprocesses data according to its requirements
- Creates separate cached directories in `preprocessed_data/`
- Runs inference
- Saves results to `runs/inference_{MODEL}/`

### Step 3: Subsequent Runs

After first run, change `DO_PREPROCESS: False` in configs you want to rerun.

**To change preprocessing setting:**
```bash
# Example: Turn off preprocessing for PURE_TSCAN
sed -i 's/DO_PREPROCESS: True/DO_PREPROCESS: False/' configs/infer_configs/CUSTOM_PURE_TSCAN.yaml
```

Or edit configs manually.

### Step 4: View Results

**During run:** Real-time metrics displayed in notebook

**After run:**
- `model_comparison_results.json` - Full results in JSON
- `model_comparison_results.csv` - Results table in CSV
- `runs/inference_{MODEL}/saved_test_outputs/*.pickle` - Per-model outputs
- `runs/inference_{MODEL}/plots/` - Bland-Altman plots

---

## Interpreting Results

### Metrics You Get

- **MAE**: Average HR error (BPM) - Lower is better
- **RMSE**: Error with penalty for big mistakes - Lower is better
- **MAPE**: Percentage error - Lower is better
- **Pearson**: Correlation coefficient - Higher is better (max 1.0)
- **SNR**: Signal quality (dB) - Higher is better

### What's Good Performance?

**Excellent:**
- MAE < 2 BPM
- Pearson > 0.95

**Very Good:**
- MAE 2-5 BPM
- Pearson 0.90-0.95

**Good:**
- MAE 5-10 BPM
- Pearson 0.80-0.90

**Best model = Lowest MAE + Highest Pearson**

---

## Customization Options

### Run Specific Models Only

Edit `run_all_models_inference.ipynb`, comment out unwanted models:

```python
CONFIGS = [
    "configs/infer_configs/CUSTOM_PURE_TSCAN.yaml",
    # "configs/infer_configs/CUSTOM_PURE_DeepPhys.yaml",  # Commented out
    "configs/infer_configs/CUSTOM_PURE_PhysNet.yaml",
    ...
]
```

### Change Batch Size

Edit config file:
```yaml
INFERENCE:
  BATCH_SIZE: 8  # Increase for faster inference (if GPU memory allows)
```

### Use Different Evaluation Method

Edit config file:
```yaml
INFERENCE:
  EVALUATION_METHOD: "peak detection"  # or "FFT" (default)
```

---

## Troubleshooting

### Issue: "Preprocessed directory does not exist"

**Solution:** Set `DO_PREPROCESS: True` in the config

### Issue: Model fails with channel mismatch

**Cause:** Wrong DATA_TYPE or DATA_FORMAT in config

**Solution:** Don't modify DATA_TYPE or DATA_FORMAT - they are model-specific

### Issue: BigSmall models fail

**Expected:** BigSmall models use special dataset loader

**Solution:** Skip BigSmall models (already commented out in notebook)

### Issue: Out of memory

**Solution:** Reduce BATCH_SIZE in config:
```yaml
INFERENCE:
  BATCH_SIZE: 2  # or 1
```

---

## Next Steps

1. Verify your data structure matches requirements
2. Open `run_all_models_inference.ipynb`
3. Run all cells
4. Check `model_comparison_results.csv` for best model
5. Analyze results from best performing models

---

## Files Created

- `configs/infer_configs/CUSTOM_*.yaml` (36 files)
- `run_all_models_inference.ipynb`
- `rPPG_TOOLBOX_DEEP_UNDERSTANDING.md`
- This file: `SETUP_COMPLETE.md`

---

**You are ready to run inference on all 36 models!**
