#!/usr/bin/env python3
"""
Benchmark N-CMAPSS models against C-MAPSS FD baseline.

Usage:
    python benchmark_models.py --cmapss-model checkpoints/best_model.pt \
                               --ncmapss-model checkpoints_ncmapss/ncmapss_rul_best.pt \
                               --cmapss-data data/All_test_data.xlsx \
                               --ncmapss-data-dir data/ncmapss/

Evaluates both models on their respective test sets and produces
a side-by-side comparison report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error
)

sys.path.insert(0, str(Path(__file__).parent))

from Prediction_Analysis_Results.src_code.petromind.pipeline import (
    PipelineConfig, LSTMRULModel, LSTMClassifier,
    build_dataloaders, build_sliding_windows,
    compute_classification_label, compute_rul,
    validate_dataframe, FeatureExtractor, SensorNormalizer,
)
from Prediction_Analysis_Results.src_code.petromind.pipeline.utils import get_active_feature_cols, load_cmapss_excel_all_sheets
from ncmapss_loader import load_ncmapss_all_datasets, get_ncmapss_feature_cols


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark models")
    p.add_argument("--cmapss-model", type=str, default=None,
                   help="Path to C-MAPSS FD trained model")
    p.add_argument("--ncmapss-model", type=str, default=None,
                   help="Path to N-CMAPSS trained model")
    p.add_argument("--cmapss-data", type=str, default=None,
                   help="Path to C-MAPSS test data (Excel)")
    p.add_argument("--ncmapss-data-dir", type=str, default=None,
                   help="Directory with N-CMAPSS .h5 files")
    p.add_argument("--model-type", type=str, default="rul",
                   choices=["rul", "classifier"])
    p.add_argument("--output", type=str, default="benchmark_report.md")
    return p.parse_args()


def evaluate_cmapss_rul(model_path, data_path, cfg):
    """Evaluate C-MAPSS FD RUL model."""
    print("[C-MAPSS FD] Loading test data...")
    df = load_cmapss_excel_all_sheets(data_path)
    df = validate_dataframe(df, cfg)
    df = compute_rul(df, cfg)
    df = compute_classification_label(df, cfg)

    feature_cols = get_active_feature_cols(df, cfg)
    X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)

    # Load normalizer
    mean = np.load("mean.npy") if Path("mean.npy").exists() else X.mean(axis=(0,1))
    std = np.load("std.npy") if Path("std.npy").exists() else X.std(axis=(0,1)) + 1e-8
    X = (X - mean) / std

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    loader = build_dataloaders(X, y_cls, y_rul, engine_ids, cfg)[1]  # val loader
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            Xb = batch["features"].to(device)
            yb = batch["rul"].numpy()
            pred = model(Xb).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(yb)

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # NASA score
    score = 0.0
    for d in (y_true - y_pred):
        if d < 0:
            score += np.exp(-d / 10.0) - 1
        else:
            score += np.exp(d / 13.0) - 1

    return {"rmse": rmse, "mae": mae, "score": score}


def evaluate_ncmapss_rul(model_path, data_dir, cfg):
    """Evaluate N-CMAPSS RUL model."""
    print("[N-CMAPSS] Loading test data...")
    from ncmapss_loader import load_ncmapss_all_datasets

    df = load_ncmapss_all_datasets(data_dir, split="test")
    df = compute_classification_label(df, cfg)

    feature_cols = get_ncmapss_feature_cols(df)
    X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols)

    # Load normalizer
    mean_path = Path(model_path).parent / "ncmapss_rul_mean.npy"
    std_path = Path(model_path).parent / "ncmapss_rul_std.npy"
    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path)
        std = np.load(std_path)
        X = (X - mean) / std

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRULModel(input_dim=X.shape[2], cfg=cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    loader = build_dataloaders(X, y_cls, y_rul, engine_ids, cfg)[1]
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            Xb = batch["features"].to(device)
            yb = batch["rul"].numpy()
            pred = model(Xb).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(yb)

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    score = 0.0
    for d in (y_true - y_pred):
        if d < 0:
            score += np.exp(-d / 10.0) - 1
        else:
            score += np.exp(d / 13.0) - 1

    return {"rmse": rmse, "mae": mae, "score": score}


def main():
    args = parse_args()
    cfg = PipelineConfig()

    results = {}

    if args.cmapss_model and args.cmapss_data:
        results["C-MAPSS FD"] = evaluate_cmapss_rul(args.cmapss_model, args.cmapss_data, cfg)

    if args.ncmapss_model and args.ncmapss_data_dir:
        results["N-CMAPSS"] = evaluate_ncmapss_rul(args.ncmapss_model, args.ncmapss_data_dir, cfg)

    # Generate report
    report = """
# PetroMind Model Benchmark Report

## RUL Regression Comparison

| Metric | C-MAPSS FD | N-CMAPSS | Winner |
|--------|-----------|----------|--------|
"""
    if "C-MAPSS FD" in results and "N-CMAPSS" in results:
        c = results["C-MAPSS FD"]
        n = results["N-CMAPSS"]
        for metric in ["rmse", "mae", "score"]:
            winner = "N-CMAPSS" if n[metric] < c[metric] else "C-MAPSS FD"
            report += f"| {metric.upper()} | {c[metric]:.2f} | {n[metric]:.2f} | {winner} |\n"
    else:
        for name, metrics in results.items():
            report += f"\n### {name}\n"
            for k, v in metrics.items():
                report += f"- {k.upper()}: {v:.2f}\n"

    report += """
## Interpretation

- **RMSE**: Lower is better. Measures average prediction error in cycles.
- **MAE**: Lower is better. More robust to outliers than RMSE.
- **NASA Score**: Lower is better. Asymmetric penalty (late predictions punished more).

## Recommendation

"""
    if "C-MAPSS FD" in results and "N-CMAPSS" in results:
        c = results["C-MAPSS FD"]
        n = results["N-CMAPSS"]
        wins = sum(1 for m in ["rmse", "mae", "score"] if n[m] < c[m])
        if wins >= 2:
            report += "✅ **Promote N-CMAPSS model** — it outperforms C-MAPSS FD on most metrics.\n"
        else:
            report += "❌ **Keep C-MAPSS FD model** — N-CMAPSS did not improve performance.\n"
            report += "   Consider: more epochs, hyperparameter tuning, or feature engineering.\n"

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {args.output}")
    print(report)


if __name__ == "__main__":
    main()
