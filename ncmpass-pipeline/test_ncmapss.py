#!/usr/bin/env python3
"""
N-CMAPSS Integrated Testing - Uses petromind.pipeline models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from petromind.pipeline.lstm_model import LSTMClassifier
    from petromind.pipeline.models import LSTMRULModel
    print("[OK] Imported models from petromind.pipeline")
except Exception as e:
    print(f"[FATAL] Cannot import petromind models: {e}")
    sys.exit(1)

from ncmapss_loader import load_ncmapss_smart, get_ncmapss_feature_cols


# Utility implementations
def build_sliding_windows(df, feature_cols, window_size, stride, target_col):
    X, y = [], []
    for unit in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit].sort_values('cycle')
        values = unit_df[feature_cols].values
        targets = unit_df[target_col].values
        for i in range(0, len(values) - window_size + 1, stride):
            X.append(values[i:i+window_size])
            y.append(targets[i+window_size-1])
    return np.array(X), np.array(y)


class SensorNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
        return (X - self.mean) / self.std
    def transform(self, X):
        return (X - self.mean) / self.std


def compute_classification_label(df, threshold=20):
    df = df.copy()
    df['label'] = (df['rul'] > threshold).astype(int)
    return df


def parse_args():
    p = argparse.ArgumentParser(description='Test N-CMAPSS models')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--subsets', nargs='+', default=None)
    p.add_argument('--split', default='test')
    p.add_argument('--sample-every', type=int, default=1)
    p.add_argument('--task', default='cls', choices=['rul', 'cls'])
    p.add_argument('--window-size', type=int, default=30)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--threshold', type=float, default=20)
    p.add_argument('--model-path', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--output-csv', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sep = os.linesep

    print("=" * 60)
    print("N-CMAPSS INTEGRATED TESTING")
    print("=" * 60)
    print(f"Task: {args.task} | Device: {device}")

    print(sep + "[1] Loading model from " + args.model_path + "...")
    if not os.path.exists(args.model_path):
        print("[FATAL] Model not found: " + args.model_path)
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    sensor_cols = checkpoint.get('sensor_cols', None)
    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)

    if mean is not None and isinstance(mean, torch.Tensor):
        mean = mean.numpy()
    if std is not None and isinstance(std, torch.Tensor):
        std = std.numpy()

    print("Loaded from epoch " + str(checkpoint.get('epoch', 'unknown')))

    print(sep + "[2] Loading test data...")
    df = load_ncmapss_smart(
        data_dir=args.data_dir,
        subsets=args.subsets,
        split=args.split,
        sample_every=args.sample_every,
        verbose=True
    )

    if df is None or len(df) == 0:
        print("[FATAL] No data loaded.")
        sys.exit(1)

    if sensor_cols is None:
        sensor_cols = get_ncmapss_feature_cols(df)
    print(sep + "[3] Sensors: " + str(len(sensor_cols)))

    if args.task == 'rul':
        y_col = 'rul'
        print(sep + "[4] RUL range: [" + f"{df['rul'].min():.1f}" + ", " + f"{df['rul'].max():.1f}" + "]")
    else:
        df = compute_classification_label(df, threshold=args.threshold)
        y_col = 'label'
        label_counts = df['label'].value_counts().sort_index()
        print(sep + "[4] Labels (threshold=" + str(args.threshold) + "): " + str(dict(label_counts)))
        if len(label_counts) < 2:
            print("[WARN] Only one class in test set.")

    print(sep + "[5] Building windows...")
    X, y = build_sliding_windows(df, sensor_cols, args.window_size, args.stride, y_col)
    print("Test windows: " + str(X.shape))

    print(sep + "[6] Normalizing...")
    normalizer = SensorNormalizer(mean=mean, std=std)
    if mean is None:
        print("[WARN] No saved normalizer. Computing from test data.")
        X = normalizer.fit_transform(X)
    else:
        X = normalizer.transform(X)

    print(sep + "[7] Building model...")
    input_dim = X.shape[2]
    if args.task == 'rul':
        model = LSTMRULModel(input_dim=input_dim)
    else:
        # FIX: Use correct signature (input_dim, hidden_size, num_layers, dropout)
        model = LSTMClassifier(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Input dim: " + str(input_dim))

    print(sep + "[8] Running inference...")
    test_ds = TensorDataset(torch.FloatTensor(X))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            output = model(xb)
            if args.task == 'rul':
                all_preds.extend(output.cpu().numpy().flatten())
            else:
                probs = torch.softmax(output, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    y = np.array(y)

    print(sep + "[9] Evaluation...")

    if args.task == 'rul':
        rmse = np.sqrt(np.mean((all_preds - y) ** 2))
        mae = np.mean(np.abs(all_preds - y))
        errors = y - all_preds
        score = np.sum(np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1))

        print(sep + "RMSE: " + f"{rmse:.1f}")
        print("MAE: " + f"{mae:.1f}")
        print("NASA Score: " + f"{score:.1f}")

        results_df = pd.DataFrame({'true_rul': y, 'pred_rul': all_preds, 'error': errors})
    else:
        print(sep + "Unique true classes: " + str(np.unique(y)))
        print("Unique predicted classes: " + str(np.unique(all_preds)))

        print(sep + "Per-class Accuracy:")
        for cls in sorted(np.unique(np.concatenate([y, all_preds]))):
            mask = y == cls
            if mask.sum() > 0:
                acc = np.mean(all_preds[mask] == cls)
                print("  Class " + str(cls) + ": " + f"{acc:.4f}" + " (" + str(mask.sum()) + " samples)")
            else:
                print("  Class " + str(cls) + ": N/A")

        accuracy = np.mean(y == all_preds)
        print(sep + "Overall Accuracy: " + f"{accuracy:.4f}")

        if len(np.unique(y)) > 1:
            tp = np.sum((y == 1) & (all_preds == 1))
            fp = np.sum((y == 0) & (all_preds == 1))
            fn = np.sum((y == 1) & (all_preds == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print("Precision: " + f"{precision:.4f}")
            print("Recall: " + f"{recall:.4f}")
            print("F1 Score: " + f"{f1:.4f}")

        tp = np.sum((y == 1) & (all_preds == 1))
        tn = np.sum((y == 0) & (all_preds == 0))
        fp = np.sum((y == 0) & (all_preds == 1))
        fn = np.sum((y == 1) & (all_preds == 0))
        print(sep + "Confusion Matrix:")
        print("  TN=" + str(tn) + "  FP=" + str(fp))
        print("  FN=" + str(fn) + "  TP=" + str(tp))

        all_probs = np.array(all_probs)
        results_df = pd.DataFrame({
            'true_label': y,
            'pred_label': all_preds,
            'prob_class_0': all_probs[:, 0],
            'prob_class_1': all_probs[:, 1],
        })

    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(sep + "Saved predictions to " + args.output_csv)

    print(sep + "Done!")


if __name__ == '__main__':
    main()
