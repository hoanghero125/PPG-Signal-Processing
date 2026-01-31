# rPPG-Toolbox Technical Reference

**Date**: 2026-01-30

---

## Repository Overview

**rPPG-Toolbox**: Open-source platform for camera-based physiological sensing (remote photoplethysmography)

**Supported Neural Models:**
1. DeepPhys (2018)
2. PhysNet (2019)
3. TS-CAN/TSCAN (2020)
4. EfficientPhys (2023)
5. BigSmall (2023)
6. PhysFormer (2022)
7. iBVPNet (2024)
8. PhysMamba (2024)
9. RhythmFormer (2024)
10. FactorizePhys (2024)

**Available Pretrained Models: 36 total**
- PURE trained: 9 models
- UBFC-rPPG trained: 7 models
- SCAMPS trained: 6 models
- iBVP trained: 2 models
- MA-UBFC trained: 4 models
- BP4D trained: 7 models (includes BigSmall and PseudoLabel variants)
- Note: BigSmall models use special multi-task architecture

---

## Configuration System

### YAML Config Structure

```yaml
BASE: ['']
TOOLBOX_MODE: "only_test"  # or "train_and_test"

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  DATA:
    FS: 30
    DATASET: Custom
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW  # or NCDHW
    DATA_PATH: "raw_data"
    CACHED_PATH: "preprocessed_data"
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized']  # or ['DiffNormalized','Standardized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
      RESIZE:
        H: 72
        W: 72

MODEL:
  NAME: Tscan
  DROP_RATE: 0.2

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  MODEL_PATH: "final_model_release/PURE_TSCAN.pth"

LOG:
  PATH: runs/exp
```

---

## Data Format Specifications

### DATA_FORMAT
- **NDCHW**: N=batch, D=depth/time, C=channel, H=height, W=width
- **NCDHW**: N=batch, C=channel, D=depth/time, H=height, W=width

### DATA_TYPE
- `['DiffNormalized']`: Single preprocessing → 3 RGB channels
- `['DiffNormalized','Standardized']`: Dual preprocessing → 6 channels (3 RGB × 2)

---

## Model-Specific Requirements

| Model | DATA_FORMAT | DATA_TYPE | CHUNK_LENGTH | FRAME_NUM/DEPTH | Resize | Notes |
|-------|------------|-----------|--------------|-----------------|--------|-------|
| TSCAN | NDCHW | `['DiffNormalized','Standardized']` | 180 | 10 (FRAME_DEPTH) | 72x72 | 6 channels |
| DeepPhys | NDCHW | `['DiffNormalized','Standardized']` | 180 | - | 72x72 | 6 channels |
| EfficientPhys | NDCHW | `['DiffNormalized','Standardized']` | 180 | 10 (FRAME_DEPTH) | 72x72 | 6 channels |
| PhysNet | NCDHW | `['DiffNormalized']` | 128 | 128 (FRAME_NUM) | 72x72 | 3 channels |
| PhysFormer | NCDHW | `['DiffNormalized']` | 160 | - | 128x128 | 3 channels |
| PhysMamba | NCDHW | `['DiffNormalized']` | 160 | - | 128x128 | 3 channels |
| RhythmFormer | NDCHW | `['DiffNormalized']` | 160 | - | 128x128 | 3 channels |
| FactorizePhys | NDCHW | `['DiffNormalized']` | 180 | 160 (FRAME_NUM) | 128x128 | 3 channels |
| iBVPNet | NDCHW | `['DiffNormalized']` | 180 | 160 (FRAME_NUM) | 128x128 | 3 channels |
| BigSmall | NDCHW | Special (BIG/SMALL) | 3 | 3 (FRAME_DEPTH) | 144x144 & 9x9 | Multi-resolution |

**Reference official configs in `configs/infer_configs/` for complete model settings**

---

## Preprocessing System

### First Run (DO_PREPROCESS: True)
1. Reads raw videos from DATA_PATH
2. Performs face detection (HC or Y5F backend)
3. Crops and resizes faces
4. Applies DATA_TYPE preprocessing
5. Chunks videos into CHUNK_LENGTH frames
6. Saves to CACHED_PATH
7. Creates CSV file lists in CACHED_PATH/DataFileLists/

### Subsequent Runs (DO_PREPROCESS: False)
Loads preprocessed data from cached directory

### Cached Directory Naming
Auto-generated based on preprocessing parameters:
```
Custom_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_...
```

**Changing DATA_TYPE changes directory name**

---

## Inference Workflow

### Testing Pretrained Models

**Step 1**: Prepare config file
- Set `TOOLBOX_MODE: "only_test"`
- Set `MODEL_PATH` to pretrained weights
- Configure preprocessing settings
- Set `DO_PREPROCESS: True` for first run

**Step 2**: Run inference
```bash
python main.py --config_file configs/infer_configs/CONFIG.yaml
```

**Step 3**: Results location
```
runs/exp/  # or LOG.PATH
├── saved_test_outputs/*.pickle
└── plots/
```

---

## Creating Custom Configs

### Minimal Modification Approach

Start with official/train config → Change 4 parameters only:

```yaml
# Change 1: Dataset
DATASET: UBFC-rPPG  →  DATASET: Custom

# Change 2: Raw data path
DATA_PATH: "/path/to/original"  →  DATA_PATH: "raw_data"

# Change 3: Cache path
CACHED_PATH: "/path/to/original"  →  CACHED_PATH: "preprocessed_data"

# Change 4: Log directory
PATH: runs/exp  →  PATH: runs/inference_{DATASET}_{MODEL}
```

**For train configs:** Also set `TOOLBOX_MODE: "only_test"` and update `MODEL_PATH`

Keep all other settings from source config unchanged.

### Complete Model-to-Config Mapping (36 models)

**PURE Trained (9 models):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| PURE_TSCAN.pth | PURE_UBFC-rPPG_TSCAN_BASIC.yaml | CUSTOM_PURE_TSCAN.yaml |
| PURE_DeepPhys.pth | PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml | CUSTOM_PURE_DeepPhys.yaml |
| PURE_EfficientPhys.pth | PURE_UBFC-rPPG_EFFICIENTPHYS.yaml | CUSTOM_PURE_EfficientPhys.yaml |
| PURE_PhysNet_DiffNormalized.pth | PURE_UBFC-rPPG_PHYSNET_BASIC.yaml | CUSTOM_PURE_PhysNet.yaml |
| PURE_PhysFormer_DiffNormalized.pth | PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml | CUSTOM_PURE_PhysFormer.yaml |
| PURE_PhysMamba_DiffNormalized.pth | PURE_UBFC-rPPG_PHYSMAMBA_BASIC.yaml | CUSTOM_PURE_PhysMamba.yaml |
| PURE_RhythmFormer.pth | PURE_UBFC-rPPG_RHYTHMFORMER_BASIC.yaml | CUSTOM_PURE_RhythmFormer.yaml |
| PURE_FactorizePhys_FSAM_Res.pth | PURE_UBFC-rPPG_FactorizePhys_FSAM_Res.yaml | CUSTOM_PURE_FactorizePhys.yaml |
| PURE_iBVPNet.pth | PURE_UBFC-rPPG_iBVPNet_BASIC.yaml | CUSTOM_PURE_iBVPNet.yaml |

**UBFC-rPPG Trained (7 models):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| UBFC-rPPG_TSCAN.pth | UBFC-rPPG_PURE_TSCAN_BASIC.yaml | CUSTOM_UBFC_TSCAN.yaml |
| UBFC-rPPG_DeepPhys.pth | UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml | CUSTOM_UBFC_DeepPhys.yaml |
| UBFC-rPPG_EfficientPhys.pth | UBFC-rPPG_PURE_EFFICIENTPHYS.yaml | CUSTOM_UBFC_EfficientPhys.yaml |
| UBFC-rPPG_PhysNet_DiffNormalized.pth | UBFC-rPPG_PURE_PHYSNET_BASIC.yaml | CUSTOM_UBFC_PhysNet.yaml |
| UBFC-rPPG_PhysFormer_DiffNormalized.pth | UBFC-rPPG_PURE_PHYSFORMER_BASIC.yaml | CUSTOM_UBFC_PhysFormer.yaml |
| UBFC-rPPG_PhysMamba_DiffNormalized.pth | UBFC-rPPG_PURE_PHYSMAMBA_BASIC.yaml | CUSTOM_UBFC_PhysMamba.yaml |
| UBFC-rPPG_RhythmFormer.pth | UBFC-rPPG_PURE_RHYTHMFORMER.yaml | CUSTOM_UBFC_RhythmFormer.yaml |
| UBFC-rPPG_FactorizePhys_FSAM_Res.pth | UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml | CUSTOM_UBFC_FactorizePhys.yaml |

**SCAMPS Trained (6 models):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| SCAMPS_TSCAN.pth | SCAMPS_PURE_TSCAN_BASIC.yaml | CUSTOM_SCAMPS_TSCAN.yaml |
| SCAMPS_DeepPhys.pth | SCAMPS_PURE_DEEPPHYS_BASIC.yaml | CUSTOM_SCAMPS_DeepPhys.yaml |
| SCAMPS_EfficientPhys.pth | SCAMPS_PURE_EFFICIENTPHYS.yaml | CUSTOM_SCAMPS_EfficientPhys.yaml |
| SCAMPS_PhysNet_DiffNormalized.pth | SCAMPS_PURE_PHYSNET_BASIC.yaml | CUSTOM_SCAMPS_PhysNet.yaml |
| SCAMPS_PhysFormer_DiffNormalized.pth | SCAMPS_PURE_PHYSFORMER_BASIC.yaml | CUSTOM_SCAMPS_PhysFormer.yaml |
| SCAMPS_FactorizePhys_FSAM_Res.pth | SCAMPS_PURE_FactorizePhys_FSAM_Res.yaml | CUSTOM_SCAMPS_FactorizePhys.yaml |

**iBVP Trained (2 models):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| iBVP_EfficientPhys.pth | PURE_iBVP_EFFICIENTPHYS.yaml | CUSTOM_iBVP_EfficientPhys.yaml |
| iBVP_FactorizePhys_FSAM_Res.pth | PURE_iBVP_FactorizePhys_FSAM_Res.yaml | CUSTOM_iBVP_FactorizePhys.yaml |

**MA-UBFC Trained (4 models - manual configs required):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| MA-UBFC_tscan.pth | (create from TSCAN template) | CUSTOM_MAUBFC_TSCAN.yaml |
| MA-UBFC_deepphys.pth | (create from DeepPhys template) | CUSTOM_MAUBFC_DeepPhys.yaml |
| MA-UBFC_efficientphys.pth | (create from EfficientPhys template) | CUSTOM_MAUBFC_EfficientPhys.yaml |
| MA-UBFC_physnet.pth | (create from PhysNet template) | CUSTOM_MAUBFC_PhysNet.yaml |

**BP4D Trained (7 models - create from train configs):**

| Model File | Source Config | Custom Config |
|------------|---------------|---------------|
| BP4D_PseudoLabel_TSCAN.pth | BP4D_BP4D_PURE_TSCAN_BASIC.yaml (train) | CUSTOM_BP4D_TSCAN.yaml |
| BP4D_PseudoLabel_DeepPhys.pth | BP4D_BP4D_PURE_DEEPPHYS_BASIC.yaml (train) | CUSTOM_BP4D_DeepPhys.yaml |
| BP4D_PseudoLabel_EfficientPhys.pth | BP4D_BP4D_PURE_EFFICIENTPHYS.yaml (train) | CUSTOM_BP4D_EfficientPhys.yaml |
| BP4D_PseudoLabel_PhysNet_DiffNormalized.pth | BP4D_BP4D_PURE_PHYSNET_BASIC.yaml (train) | CUSTOM_BP4D_PhysNet.yaml |
| BP4D_BigSmall_Multitask_Fold1.pth | BP4D_BP4D_BIGSMALL_FOLD1.yaml (train) | CUSTOM_BP4D_BigSmall_F1.yaml |
| BP4D_BigSmall_Multitask_Fold2.pth | BP4D_BP4D_BIGSMALL_FOLD2.yaml (train) | CUSTOM_BP4D_BigSmall_F2.yaml |
| BP4D_BigSmall_Multitask_Fold3.pth | BP4D_BP4D_BIGSMALL_FOLD3.yaml (train) | CUSTOM_BP4D_BigSmall_F3.yaml |

---

## Special Config Creation Notes

### MA-UBFC Models
No official inference configs exist. Create configs based on model architecture:
- MA-UBFC_tscan.pth → Use TSCAN settings (NDCHW, 6 channels, chunk 180)
- MA-UBFC_deepphys.pth → Use DeepPhys settings (NDCHW, 6 channels, chunk 180)
- MA-UBFC_efficientphys.pth → Use EfficientPhys settings (NDCHW, 6 channels, chunk 180)
- MA-UBFC_physnet.pth → Use PhysNet settings (NCDHW, 3 channels, chunk 128)

### BP4D PseudoLabel Models
Extract TEST section from train configs:
- Use FS: 30 (same as PURE/UBFC-rPPG)
- DATASET: Custom
- USE_PSUEDO_PPG_LABEL not needed for inference
- Keep model-specific DATA_FORMAT and DATA_TYPE from train config

### BP4D BigSmall Models
Special requirements:
- Uses BP4DPlusBigSmall dataset loader
- CHUNK_LENGTH: 3 (very short)
- Special BIGSMALL preprocessing section
- Different resize: BIG (144x144), SMALL (9x9)
- BATCH_SIZE: 180
- May require fold-specific configuration
- Cannot directly test on Custom dataset without modifications

---

## Metrics

**Available metrics:**
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- MAPE: Mean Absolute Percentage Error
- Pearson: Pearson Correlation Coefficient
- SNR: Signal-to-Noise Ratio
- BA: Bland-Altman plot

**Evaluation methods:**
- FFT: Frequency domain analysis
- Peak Detection: Time domain analysis

---

## Key Repository Files

- `main.py`: Entry point
- `config.py`: Config definitions
- `configs/train_configs/`: Training configs
- `configs/infer_configs/`: Inference configs
- `dataset/data_loader/`: Dataset loaders
- `neural_methods/model/`: Model architectures
- `neural_methods/trainer/`: Training/testing logic
- `final_model_release/`: Pretrained weights

---

## Common Issues

**"Preprocessed directory does not exist"**
- Config settings changed, cached path name different
- Solution: Set DO_PREPROCESS: True or match settings to existing data

**"expected input to have 3 channels, but got X"**
- Wrong DATA_FORMAT or DATA_TYPE
- Solution: Use correct settings from official config for that model

**"running_mean should contain X elements not Y"**
- Model expects different number of channels
- Solution: Match DATA_TYPE to model requirements

**BigSmall models may not work with Custom dataset**
- Requires BP4DPlusBigSmall dataset loader
- Special preprocessing with BIG/SMALL dual-resolution
- May need custom dataset loader implementation
- Consider excluding BigSmall models if Custom loader incompatible

---

## Testing Strategy for All 36 Models

### Config Organization

**Priority 1: Models with existing inference configs (24 models)**
1-9: PURE trained models
10-16: UBFC-rPPG trained models
17-22: SCAMPS trained models
23-24: iBVP trained models

**Priority 2: Models requiring manual configs (12 models)**
25-28: MA-UBFC trained models (create from architecture templates)
29-35: BP4D trained models (create from train configs)
36: BP4D BigSmall (special handling required)

### Execution Approach

1. Create 36 custom config files based on source configs
2. Use Jupyter notebook with config list
3. Comment/uncomment configs to run specific models
4. Each model preprocesses its own data independently
5. Results saved to unique log directories per model

### Expected Preprocessing Groups

Different preprocessing settings create different cached directories:
- **Group A** (NDCHW, 6ch, chunk 180): TSCAN, DeepPhys, EfficientPhys variants
- **Group B** (NCDHW, 3ch, chunk 128): PhysNet variants
- **Group C** (NCDHW, 3ch, chunk 160): PhysFormer, PhysMamba variants
- **Group D** (NDCHW, 3ch, chunk 160): RhythmFormer variant
- **Group E** (NDCHW, 3ch, chunk 180): FactorizePhys, iBVPNet variants
- **Group F** (BigSmall special): BigSmall models

---

## References

- GitHub: https://github.com/ubicomplab/rPPG-Toolbox
- Paper: arXiv:2210.00716
- Example configs: configs/infer_configs/
