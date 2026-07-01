#!/usr/bin/env python3
"""
N-CMAPSS Classifier Testing — Fixed Version
Handles edge cases: single-class test sets, missing files, etc.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Handle import path robustly
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from petromind.pipeline import (
        compute_classification_label,
        build_sliding_windows,
        SequenceFeatureExtractor,
        SensorNormalizer
    )
    from petromind.pipeline.lstm_model import LSTMClassifier
    print("[OK] Imported from petromind.pipeline")
except Exception as e1:
    print(f"[WARN] petromind.pipeline import failed: {e1}")
    try:
        pipeline_dir = os.path.join(PROJECT_ROOT, 'petromind', 'pipeline')
        if pipeline_dir not in sys.path:
            sys.path.insert(0, pipeline_dir)
        from labeling import compute_classification_label
        from windowing import build_sliding_windows
        from features import SequenceFeatureExtractor
        from dataset import SensorNormalizer
        from lstm_model import LSTMClassifier
        print("[OK] Imported from relative paths")
    except Exception as e2:
        print(f"[FATAL] Cannot import PetroMind modules: {e2}")
        sys.exit(1)

try:
    from ncmapss_loader import load_ncmapss_smart, get_ncmapss_feature_cols
except ImportError:
    sys.path.insert(0, SCRIPT_DIR)
    from ncmapss_loader import load_ncmapss_smart, get_ncmapss_feature_cols


def parse_args():
    p = argparse.ArgumentParser(description='Test LSTMClassifier on N-CMAPSS')
    p.add_argument('--data-dir', required=True, help='Path to N-CMAPSS .h5 files')
    p.add_argument('--subsets', nargs='+', default=None, help='Specific files')
    p.add_argument('--split', default='test', choices=['dev', 'test', 'all'],
                   help='Which split to evaluate on')
    p.add_argument('--sample-every', type=int, default=1)
    p.add_argument('--model-path', default='ncmapss_model.pth',
                   help='Path to saved model .pth or .pt file')
    p.add_argument('--mean-path', default='ncmapss_mean.npy')
    p.add_argument('--std-path', default='ncmapss_std.npy')
    p.add_argument('--window-size', type=int, default=30)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--threshold', type=float, default=0.4,
                   help='Classification threshold (0=At Risk, 1=Healthy)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--output-csv', default=None,
                   help='Save predictions to CSV')
    return p.parse_args()


def load_model_safe(model_path, input_dim, device):
    """Load model handling both .pth and .pt formats."""
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2)

    if not os.path.exists(model_path):
        # Try checkpoint directory
        alt_path = os.path.join('checkpoints_ncmapss', 'best_model.pt')
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"[INFO] Using checkpoint: {model_path}")
        else:
            print(f"[FATAL] Model not found: {model_path}")
            sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint
    else:
        model.load_state_dict(checkpoint)
        metadata = {}

    model = model.to(device)
    model.eval()
    return model, metadata


def evaluate_with_proper_report(y_true, y_pred, target_names=None):
    """
    Compute classification metrics safely, handling single-class cases.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    all_classes = sorted(np.unique(np.concatenate([unique_true, unique_pred])))

    print(f"\nUnique true classes: {unique_true}")
    print(f"Unique predicted classes: {unique_pred}")
    print(f"All classes present: {all_classes}")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        print(f"\nConfusion Matrix (classes {all_classes}):")
        print(cm)
    except Exception as e:
        print(f"[WARN] Could not compute confusion matrix: {e}")
        cm = None

    # Classification report
    if target_names is None:
        target_names = [f"Class_{c}" for c in all_classes]

    print("\nClassification Report:")
    try:
        # Only include labels that actually exist in y_true to avoid warnings
        present_labels = sorted(unique_true)
        present_names = [target_names[i] if i < len(target_names) else f"Class_{i}" 
                        for i in present_labels]

        report = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=present_names,
            digits=4,
            zero_division=0
        )
        print(report)
    except Exception as e:
        print(f"[WARN] Could not generate full report: {e}")
        # Fallback: compute basic metrics manually
        accuracy = np.mean(y_true == y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        if len(unique_true) > 1:
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"Weighted F1: {f1:.4f}")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for cls in all_classes:
        mask = y_true == cls
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == cls)
            print(f"  Class {cls}: {acc:.4f} ({mask.sum()} samples)")
        else:
            print(f"  Class {cls}: N/A (0 samples in ground truth)")

    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    if len(unique_true) > 1:
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
    else:
        print("[INFO] Only one class in ground truth — F1 not meaningful.")

    return cm


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("N-CMAPSS CLASSIFIER TESTING (Fixed)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading test data...")
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

    print(f"\nLoaded: {len(df)} cycles from {df['unit_id'].nunique()} engines")

    # ------------------------------------------------------------------
    # Step 2: Detect columns
    # ------------------------------------------------------------------
    sensor_cols = get_ncmapss_feature_cols(df)
    print(f"\n[Step 2] Sensors: {len(sensor_cols)}")

    # ------------------------------------------------------------------
    # Step 3: Labels
    # ------------------------------------------------------------------
    print("\n[Step 3] Computing classification labels...")
    df = compute_classification_label(df, rul_col='rul', threshold=args.threshold)
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution: {dict(label_counts)}")

    if len(label_counts) < 2:
        print("[WARN] Test set has only ONE class. Metrics will be limited.")
        print("[HINT] Try --split dev or --threshold with a different value.")

    # ------------------------------------------------------------------
    # Step 4: Windows
    # ------------------------------------------------------------------
    print("\n[Step 4] Building windows...")
    X_raw, y = build_sliding_windows(
        df, sensor_cols, args.window_size, args.stride, 'label'
    )
    print(f"Windows: {X_raw.shape}")

    # ------------------------------------------------------------------
    # Step 5: Feature extraction
    # ------------------------------------------------------------------
    print("\n[Step 5] Feature extraction...")
    feat_extractor = SequenceFeatureExtractor(sensor_cols=sensor_cols)
    X = feat_extractor.transform(X_raw)
    print(f"Features: {X.shape}")

    # ------------------------------------------------------------------
    # Step 6: Normalization
    # ------------------------------------------------------------------
    print("\n[Step 6] Normalizing...")
    normalizer = SensorNormalizer()

    # Try to load saved stats, else compute from test data (not ideal but works)
    if os.path.exists(args.mean_path) and os.path.exists(args.std_path):
        normalizer.mean = np.load(args.mean_path)
        normalizer.std = np.load(args.std_path)
        print(f"[OK] Loaded normalizer from {args.mean_path}")
    else:
        print("[WARN] No saved normalizer found. Computing from test data (not ideal).")
        normalizer.fit(X)

    X = normalizer.transform(X)

    # ------------------------------------------------------------------
    # Step 7: Load model
    # ------------------------------------------------------------------
    print("\n[Step 7] Loading model...")
    input_dim = X.shape[2]
    model, metadata = load_model_safe(args.model_path, input_dim, device)
    print(f"[OK] Model loaded. Input dim: {input_dim}")

    # ------------------------------------------------------------------
    # Step 8: Inference
    # ------------------------------------------------------------------
    print("\n[Step 8] Running inference...")
    test_ds = TensorDataset(torch.FloatTensor(X))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ------------------------------------------------------------------
    # Step 9: Evaluation
    # ------------------------------------------------------------------
    print("\n[Step 9] Evaluation...")
    target_names = ['At Risk (0)', 'Healthy (1)']
    cm = evaluate_with_proper_report(y, all_preds, target_names)

    # Save predictions if requested
    if args.output_csv:
        results_df = pd.DataFrame({
            'true_label': y,
            'pred_label': all_preds,
            'prob_class_0': all_probs[:, 0],
            'prob_class_1': all_probs[:, 1],
        })
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved predictions to {args.output_csv}")

    print("\nDone!")


if __name__ == '__main__':
    main()
