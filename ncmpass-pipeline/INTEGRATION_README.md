# N-CMAPSS Integration — PetroMind Pipeline

## Architecture

```
PetroMind-main/
├── petromind/                          # Core pipeline (unchanged)
│   └── pipeline/
│       ├── lstm_model.py               # LSTMClassifier
│       ├── models.py                   # LSTMRULModel
│       ├── train_lstm.py               # Original trainer
│       ├── trainer.py                  # Trainer base class
│       ├── features.py                 # FeatureExtractor, SequenceFeatureExtractor
│       ├── windowing.py                # build_sliding_windows
│       ├── dataset.py                  # SensorNormalizer
│       ├── labeling.py                 # compute_classification_label
│       ├── utils.py                    # compute_rul
│       └── ...
│
├── fine-tuning/                        # N-CMAPSS-specific scripts
│   ├── ncmapss_loader.py              # N-CMAPSS HDF5 loader (corrected)
│   ├── train_ncmapss.py               # Training wrapper (uses petromind)
│   ├── test_ncmapss.py                # Testing wrapper (uses petromind)
│   ├── ncmapss.py                     # Unified CLI
│   ├── diagnose_ncmapss.py            # Diagnostic tool
│   ├── benchmark_ncmapss.py           # Benchmark tool
│   ├── checkpoints_ncmapss_rul/      # RUL model checkpoints
│   └── checkpoints_ncmapss_cls/      # Classifier checkpoints
```

## Design Principles

1. **petromind.pipeline** contains all generic ML logic (models, training, features)
2. **fine-tuning/** contains only N-CMAPSS-specific code (data loading, CLI)
3. No code duplication — everything reuses petromind modules
4. Clean separation of concerns

## Usage

### Train RUL Regressor
```bash
cd fine-tuning/
python train_ncmapss.py --task rul --data-dir data/ncmapss/ --epochs 100
```

### Train Classifier
```bash
python train_ncmapss.py --task cls --data-dir data/ncmapss/ --threshold 20 --epochs 100
```

### Test
```bash
# Test classifier
python test_ncmapss.py --task cls --model-path checkpoints_ncmapss_cls/best_model.pt --data-dir data/ncmapss/ --split dev

# Test RUL
python test_ncmapss.py --task rul --model-path checkpoints_ncmapss_rul/best_model.pt --data-dir data/ncmapss/ --split dev
```

### Unified CLI
```bash
# Train
python ncmapss.py train --task rul --data-dir data/ncmapss/ --epochs 100

# Test
python ncmapss.py test --task cls --model-path checkpoints_ncmapss_cls/best_model.pt --data-dir data/ncmapss/
```

## Files

| File | Purpose | Uses petromind? |
|------|---------|-----------------|
| `ncmapss_loader.py` | HDF5 loading with `_dev`/`_test` suffix support | No |
| `train_ncmapss.py` | Training orchestration | Yes — models, windowing, normalizer |
| `test_ncmapss.py` | Testing/evaluation | Yes — models, windowing, normalizer |
| `ncmapss.py` | Unified CLI entry point | Yes — delegates to train/test |
| `diagnose_ncmapss.py` | Data quality inspection | No |
| `benchmark_ncmapss.py` | Config comparison | Yes |

## What Changed from Standalone

| Standalone | Integrated | Benefit |
|------------|------------|---------|
| Embedded LSTM models | Import from `petromind.pipeline` | Single source of truth |
| Embedded `build_sliding_windows` | Import from `petromind.pipeline` | Reuse tested code |
| Embedded `SensorNormalizer` | Import from `petromind.pipeline` | Consistent normalization |
| Embedded loader | `ncmapss_loader.py` | Only N-CMAPSS-specific code remains |
| 3 separate scripts | Unified CLI + clean wrappers | Easier maintenance |

## Adding New Datasets

To add another dataset (e.g., C-MAPSS, PHM2012):

1. Create `cmapss_loader.py` — dataset-specific loading
2. Create `train_cmapss.py` — reuse petromind, just swap loader
3. Done — all model/training logic is already in petromind
