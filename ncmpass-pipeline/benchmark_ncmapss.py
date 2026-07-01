#!/usr/bin/env python3
"""
N-CMAPSS Training Benchmark & Comparison Script
Compares different configurations and helps diagnose issues.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from petromind.pipeline import (
        compute_rul, compute_classification_label,
        build_sliding_windows,
        FeatureExtractor, SequenceFeatureExtractor,
        SensorNormalizer, Trainer
    )
    from petromind.pipeline.models import LSTMRULModel
    from petromind.pipeline.lstm_model import LSTMClassifier
except Exception as e:
    print(f"[FATAL] Cannot import PetroMind: {e}")
    sys.exit(1)

# Use fast loader
sys.path.insert(0, SCRIPT_DIR)
from ncmapss_loader_fast import load_ncmapss_smart_fast, get_ncmapss_feature_cols


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark N-CMAPSS training configs')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--subsets', nargs='+', default=None)
    p.add_argument('--output-json', default='benchmark_results.json')
    p.add_argument('--device', default=None)
    return p.parse_args()


def benchmark_config(name, config, df, sensor_cols, device):
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    start_time = time.time()

    # Split
    engines = df['unit_id'].unique()
    np.random.seed(42)
    np.random.shuffle(engines)
    n_train = int(len(engines) * 0.8)
    train_df = df[df['unit_id'].isin(engines[:n_train])]
    val_df = df[df['unit_id'].isin(engines[n_train:])]

    # Task
    if config['task'] == 'rul':
        y_train = train_df['rul'].values
        y_val = val_df['rul'].values
        y_col = 'rul'
    else:
        train_df = compute_classification_label(train_df, rul_col='rul', threshold=0.4)
        val_df = compute_classification_label(val_df, rul_col='rul', threshold=0.4)
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_col = 'label'

    # Windows
    X_train_raw, _ = build_sliding_windows(
        train_df, sensor_cols, config['window_size'], config['stride'], y_col
    )
    X_val_raw, _ = build_sliding_windows(
        val_df, sensor_cols, config['window_size'], config['stride'], y_col
    )

    # Re-extract y aligned with windows
    # (build_sliding_windows returns y aligned, so we use its output)
    _, y_train = build_sliding_windows(
        train_df, sensor_cols, config['window_size'], config['stride'], y_col
    )
    _, y_val = build_sliding_windows(
        val_df, sensor_cols, config['window_size'], config['stride'], y_col
    )

    print(f"  Windows: train={X_train_raw.shape}, val={X_val_raw.shape}")

    # Feature extraction
    if config.get('use_sequence_features', False):
        feat = SequenceFeatureExtractor(sensor_cols=sensor_cols)
        X_train = feat.transform(X_train_raw)
        X_val = feat.transform(X_val_raw)
    else:
        X_train = X_train_raw
        X_val = X_val_raw

    # Normalize
    norm = SensorNormalizer()
    X_train = norm.fit_transform(X_train)
    X_val = norm.transform(X_val)

    # DataLoaders
    if config['task'] == 'rul':
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(-1))
    else:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # Model
    input_dim = X_train.shape[2]
    if config['task'] == 'rul':
        model = LSTMRULModel(input_dim, config['hidden_dim'], config['num_layers'], config['dropout'])
        criterion = nn.MSELoss()
    else:
        model = LSTMClassifier(input_dim, config['hidden_dim'], config['num_layers'], 2, config['dropout'])
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Quick train (5 epochs for benchmark)
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, 6):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val

        print(f"  Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

    elapsed = time.time() - start_time

    result = {
        'name': name,
        'config': config,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(history[-1]['train_loss']),
        'final_val_loss': float(history[-1]['val_loss']),
        'time_seconds': elapsed,
        'n_train_windows': int(X_train.shape[0]),
        'n_val_windows': int(X_val.shape[0]),
        'history': history,
    }

    return result


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("N-CMAPSS TRAINING BENCHMARK")
    print("=" * 60)

    # Load data once
    print("\n[1] Loading data...")
    df = load_ncmapss_smart_fast(
        data_dir=args.data_dir,
        subsets=args.subsets,
        split='dev',
        sample_every=1,
        use_cache=True,
        verbose=True,
        max_engines_per_file=None
    )

    sensor_cols = get_ncmapss_feature_cols(df)
    print(f"\nSensors: {len(sensor_cols)}")

    # Define benchmark configs
    configs = [
        {
            'name': 'RUL_baseline',
            'task': 'rul',
            'window_size': 30,
            'stride': 1,
            'batch_size': 64,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 1e-3,
            'use_sequence_features': False,
        },
        {
            'name': 'RUL_large_window',
            'task': 'rul',
            'window_size': 60,
            'stride': 1,
            'batch_size': 64,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 1e-3,
            'use_sequence_features': False,
        },
        {
            'name': 'RUL_big_model',
            'task': 'rul',
            'window_size': 30,
            'stride': 1,
            'batch_size': 64,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'lr': 1e-3,
            'use_sequence_features': False,
        },
        {
            'name': 'RUL_low_lr',
            'task': 'rul',
            'window_size': 30,
            'stride': 1,
            'batch_size': 64,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 1e-4,
            'use_sequence_features': False,
        },
        {
            'name': 'CLS_baseline',
            'task': 'cls',
            'window_size': 30,
            'stride': 1,
            'batch_size': 64,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 1e-3,
            'use_sequence_features': True,
        },
    ]

    results = []
    for cfg in configs:
        try:
            result = benchmark_config(cfg['name'], cfg, df, sensor_cols, device)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {cfg['name']} failed: {e}")
            results.append({
                'name': cfg['name'],
                'error': str(e),
                'config': cfg
            })

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    summary_data = []
    for r in results:
        if 'error' in r:
            print(f"{r['name']:20s} FAILED: {r['error']}")
        else:
            print(f"{r['name']:20s} best_val={r['best_val_loss']:.4f}  "
                  f"time={r['time_seconds']:.1f}s  "
                  f"windows={r['n_train_windows']}")
            summary_data.append({
                'name': r['name'],
                'best_val_loss': r['best_val_loss'],
                'final_val_loss': r['final_val_loss'],
                'time_seconds': r['time_seconds'],
                'n_train_windows': r['n_train_windows'],
            })

    # Save results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {args.output_json}")

    # Save summary CSV
    if summary_data:
        pd.DataFrame(summary_data).to_csv(args.output_json.replace('.json', '.csv'), index=False)


if __name__ == '__main__':
    main()
