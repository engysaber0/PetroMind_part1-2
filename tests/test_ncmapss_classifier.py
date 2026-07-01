#!/usr/bin/env python3
"""
Test LSTM Classifier on N-CMAPSS dataset.

Mirrors the existing main_test.py workflow but loads N-CMAPSS HDF5 test data.

Usage:
    python test_ncmapss_classifier.py --data-dir data/ncmapss/ --subsets N-CMAPSS_DS02-006.h5

    # Or with specific split
    python test_ncmapss_classifier.py --data-dir data/ncmapss/ --split test
"""
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_model.petromind.pipeline.config import PipelineConfig
from lstm_model.petromind.pipeline.labeling import compute_classification_label
from lstm_model.petromind.pipeline.windowing import build_sliding_windows
from lstm_model.petromind.pipeline.features import SequenceFeatureExtractor
from lstm_model.petromind.pipeline.lstm_model import LSTMClassifier

from ncmapss_loader import load_ncmapss_all_datasets

print("=" * 60)
print("N-CMAPSS CLASSIFIER TEST")
print("=" * 60)

# =========================
# PARSE ARGS
# =========================
import argparse
p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=str, default="data/ncmapss")
p.add_argument("--subsets", type=str, nargs="+", default=None)
p.add_argument("--split", type=str, default="test", choices=["dev", "test"])
p.add_argument("--model-path", type=str, default="ncmapss_model.pth")
p.add_argument("--mean-path", type=str, default="ncmapss_mean.npy")
p.add_argument("--std-path", type=str, default="ncmapss_std.npy")
p.add_argument("--threshold", type=float, default=0.4)
args = p.parse_args()

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# LOAD MODEL
# =========================
print("\n[Load Model]")

mean = np.load(args.mean_path)
std = np.load(args.std_path)

# Load config to get input_dim
config_path = Path(args.model_path).parent / "ncmapss_config.json"
if config_path.exists():
    import json
    with open(config_path) as f:
        config_info = json.load(f)
    input_dim = config_info["input_dim"]
    print(f"Loaded config: input_dim={input_dim}")
else:
    # Infer from mean shape
    # mean shape is (features,) for 2D or (features,) for 3D after flatten
    # We need to know the feature dimension after SequenceFeatureExtractor
    print("WARNING: config not found, input_dim will be inferred from data")
    input_dim = None

model = LSTMClassifier(input_dim=input_dim or 96).to(device)  # 96 is common default
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print(f"Model loaded from: {args.model_path}")

# =========================
# LOAD TEST DATA
# =========================
print("\n[Load Test Data]")

df_test = load_ncmapss_all_datasets(
    data_dir=args.data_dir,
    h5_files=args.subsets,
    split=args.split,
    sample_every=1,
    include_health_params=True,
    include_virtual_sensors=True,
)

if len(df_test) == 0:
    print("ERROR: No test data loaded.")
    sys.exit(1)

df_test = df_test.sort_values(["unit_id", "cycle"])
print(f"Test engines: {df_test['unit_id'].nunique()}")
print(f"Test cycles: {len(df_test)}")

# =========================
# LABELING
# =========================
cfg = PipelineConfig(window_size=30)
df_test = compute_classification_label(df_test, cfg)

# =========================
# WINDOWING
# =========================
print("\n[Windowing]")

feature_cols = [c for c in df_test.columns if c not in ["unit_id", "cycle", "rul", "label"]]

all_engines = sorted(df_test["unit_id"].unique())
valid_engines = set()
X_test = []
y_test = []

for uid in all_engines:
    engine_df = df_test[df_test["unit_id"] == uid]
    vals = engine_df[feature_cols].values
    labels = engine_df["label"].values

    if len(vals) >= cfg.window_size:
        # Take the last window for each engine (like main_test.py does)
        X_test.append(vals[-cfg.window_size:])
        y_test.append(labels[-1])  # label of last cycle
        valid_engines.add(uid)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)
print(f"Test windows: {X_test.shape}")
print(f"Valid engines: {len(valid_engines)}")

# =========================
# FEATURES
# =========================
print("\n[Feature Engineering]")

extractor = SequenceFeatureExtractor(window_size=cfg.window_size)
X_test = extractor.transform(X_test)
print(f"After features: {X_test.shape}")

# Update input_dim if needed
if input_dim is None or X_test.shape[2] != input_dim:
    input_dim = X_test.shape[2]
    print(f"Updated input_dim: {input_dim}")
    # Recreate model with correct input_dim
    model = LSTMClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

# =========================
# NORMALIZATION
# =========================
print("\n[Normalization]")

X_test = (X_test - mean) / std

# =========================
# PREDICTION
# =========================
print("\n[Prediction]")

loader = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=64)
all_probs = []

with torch.no_grad():
    for X_batch in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs[:, 1].cpu().numpy())

all_probs = np.array(all_probs)
preds = (all_probs > args.threshold).astype(int)

print(f"Predictions: {len(preds)}")
print(f"True At Risk : {y_test.sum()} ({y_test.mean():.1%})")
print(f"Pred At Risk : {preds.sum()} ({preds.mean():.1%})")

# =========================
# EVALUATION
# =========================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(classification_report(y_test, preds, target_names=["Healthy", "At Risk"]))
try:
    auc = roc_auc_score(y_test, all_probs)
    print(f"ROC-AUC: {auc:.4f}")
except ValueError:
    print("ROC-AUC: N/A (only one class present)")

# =========================
# PER-ENGINE SUMMARY
# =========================
print("\n[Per-Engine Summary]")
engine_results = []
for i, uid in enumerate(sorted(valid_engines)):
    engine_results.append({
        "unit_id": uid,
        "true_label": int(y_test[i]),
        "pred_label": int(preds[i]),
        "prob_at_risk": float(all_probs[i]),
    })

results_df = pd.DataFrame(engine_results)
print(results_df.to_string(index=False))
