#!/usr/bin/env python3
"""
N-CMAPSS Integrated Training — Uses petromind.pipeline properly
Falls back to standalone implementations for API mismatches.
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

# ------------------------------------------------------------------
# Robust import handling for petromind
# ------------------------------------------------------------------
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

# Try to import petromind utilities, but fall back to our own if API differs
try:
    from petromind.pipeline import build_sliding_windows as _bsw
    from petromind.pipeline import SensorNormalizer as _SN
    from petromind.pipeline import compute_classification_label as _ccl
    # Test the signatures
    import inspect
    bsw_sig = inspect.signature(_bsw)
    ccl_sig = inspect.signature(_ccl)
    print(f"[OK] petromind build_sliding_windows signature: {bsw_sig}")
    print(f"[OK] petromind compute_classification_label signature: {ccl_sig}")

    # Check if signatures match what we need
    bsw_params = list(bsw_sig.parameters.keys())
    ccl_params = list(ccl_sig.parameters.keys())

    if len(bsw_params) >= 5:
        build_sliding_windows = _bsw
        SensorNormalizer = _SN
        compute_classification_label = _ccl
        print("[OK] Using petromind implementations directly")
        USE_PETROMIND_UTILS = True
    else:
        print("[WARN] petromind build_sliding_windows has different signature, using fallback")
        USE_PETROMIND_UTILS = False
except Exception as e:
    print(f"[WARN] Cannot use petromind utilities ({e}), using fallback implementations")
    USE_PETROMIND_UTILS = False

# Fallback implementations
if not USE_PETROMIND_UTILS:
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
        def __init__(self):
            self.mean = None
            self.std = None
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

# Import N-CMAPSS loader
from ncmapss_loader import load_ncmapss_smart, get_ncmapss_feature_cols


# ------------------------------------------------------------------
# NASA C-MAPSS Scoring Function
# ------------------------------------------------------------------
def compute_nasa_score(errors):
    errors = np.array(errors)
    return np.sum(np.where(errors < 0,
                           np.exp(-errors / 13) - 1,
                           np.exp(errors / 10) - 1))


# ------------------------------------------------------------------
# Engine-based split
# ------------------------------------------------------------------
def split_by_engine(df, train_ratio=0.8, shuffle=True, seed=42):
    engines = df['unit_id'].unique()
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(engines)
    n_train = max(1, int(len(engines) * train_ratio))
    train_engines = engines[:n_train]
    val_engines = engines[n_train:]
    return (
        df[df['unit_id'].isin(train_engines)].copy(),
        df[df['unit_id'].isin(val_engines)].copy(),
        train_engines,
        val_engines
    )


# ------------------------------------------------------------------
# Custom Trainer
# ------------------------------------------------------------------
class NCMAPSSTrainer:
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = None

    def set_criterion(self, criterion):
        self.criterion = criterion

    def train_epoch(self, train_loader):
        self.model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def validate(self, val_loader):
        self.model.eval()
        losses = []
        preds = []
        targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                losses.append(loss.item())
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())
        return np.mean(losses), np.array(preds), np.array(targets)


# ------------------------------------------------------------------
# Main Training Script
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train N-CMAPSS models using petromind.pipeline')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--subsets', nargs='+', default=None)
    p.add_argument('--split', default='dev')
    p.add_argument('--sample-every', type=int, default=1)
    p.add_argument('--task', default='rul', choices=['rul', 'cls'])
    p.add_argument('--window-size', type=int, default=30)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--threshold', type=float, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-dir', default=None)
    p.add_argument('--device', default=None)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_rul(args, df, sensor_cols, device):
    print(" " + "=" * 60)
    print("TRAINING RUL REGRESSOR")
    print("=" * 60)

    train_df, val_df, train_engines, val_engines = split_by_engine(df, args.train_ratio)
    print(f"Train engines: {len(train_engines)} ({list(train_engines)})")
    print(f"Val engines: {len(val_engines)} ({list(val_engines)})")

    X_train, y_train = build_sliding_windows(train_df, sensor_cols, args.window_size, args.stride, 'rul')
    X_val, y_val = build_sliding_windows(val_df, sensor_cols, args.window_size, args.stride, 'rul')
    print(f"Train windows: {X_train.shape} | Val windows: {X_val.shape}")

    normalizer = SensorNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)

    checkpoint_dir = args.checkpoint_dir or 'checkpoints_ncmapss_rul'
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.save(os.path.join(checkpoint_dir, 'mean.npy'), normalizer.mean)
    np.save(os.path.join(checkpoint_dir, 'std.npy'), normalizer.std)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(-1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    input_dim = X_train.shape[2]
    model = LSTMRULModel(input_dim, args.hidden_dim, args.num_layers, args.dropout)
    print(f"Model: input={input_dim}, hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = NCMAPSSTrainer(model, device, args.lr)
    trainer.set_criterion(nn.MSELoss())

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_preds, val_targets = trainer.validate(val_loader)

        rmse = np.sqrt(np.mean((val_preds.flatten() - val_targets.flatten()) ** 2))
        mae = np.mean(np.abs(val_preds.flatten() - val_targets.flatten()))
        score = compute_nasa_score(val_targets.flatten() - val_preds.flatten())

        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:8.2f}  val={val_loss:8.2f}  "
              f"lr={current_lr:.1e}  RMSE={rmse:.1f}  MAE={mae:.1f}  Score={score:.1f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'rmse': rmse, 'mae': mae, 'score': score, 'lr': current_lr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss, 'rmse': rmse, 'mae': mae, 'score': score,
                'config': vars(args), 'sensor_cols': sensor_cols,
                'mean': torch.FloatTensor(normalizer.mean),
                'std': torch.FloatTensor(normalizer.std),
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"
Early stopping at epoch {epoch}")
                break

    print("
Final Evaluation...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_preds, final_targets = trainer.validate(val_loader)

    print(f"Best RMSE: {np.sqrt(np.mean((final_preds.flatten() - final_targets.flatten())**2)):.1f}")
    print(f"Best MAE: {np.mean(np.abs(final_preds.flatten() - final_targets.flatten())):.1f}")

    pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'history.csv'), index=False)
    print(f"
Saved to {checkpoint_dir}/")
    return checkpoint_dir


def train_cls(args, df, sensor_cols, device):
    print("\n " + "=" * 60)
    print("TRAINING CLASSIFIER")
    print("=" * 60)

    df = compute_classification_label(df, threshold=args.threshold)
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution: {dict(label_counts)}")

    n_at_risk = label_counts.get(0, 0)
    n_healthy = label_counts.get(1, 0)
    total = n_at_risk + n_healthy
    if total > 0:
        print(f"  At Risk (0):  {n_at_risk} ({100*n_at_risk/total:.1f}%)")
        print(f"  Healthy (1):  {n_healthy} ({100*n_healthy/total:.1f}%)")

    if len(label_counts) < 2:
        print("[FATAL] Only one class present. Adjust --threshold.")
        sys.exit(1)

    train_df, val_df, train_engines, val_engines = split_by_engine(df, args.train_ratio)
    print(f"Train engines: {len(train_engines)} | Val engines: {len(val_engines)}")

    X_train, y_train = build_sliding_windows(train_df, sensor_cols, args.window_size, args.stride, 'label')
    X_val, y_val = build_sliding_windows(val_df, sensor_cols, args.window_size, args.stride, 'label')
    print(f"Train windows: {X_train.shape} | Val windows: {X_val.shape}")

    normalizer = SensorNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)

    checkpoint_dir = args.checkpoint_dir or 'checkpoints_ncmapss_cls'
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.save(os.path.join(checkpoint_dir, 'mean.npy'), normalizer.mean)
    np.save(os.path.join(checkpoint_dir, 'std.npy'), normalizer.std)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"Class weights: {class_weights}")

    input_dim = X_train.shape[2]
    model = LSTMClassifier(input_dim, args.hidden_dim, args.num_layers, 2, args.dropout)
    print(f"Model: input={input_dim}, hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = NCMAPSSTrainer(model, device, args.lr)
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    trainer.set_criterion(nn.CrossEntropyLoss(weight=weight_tensor))

    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_logits, val_targets = trainer.validate(val_loader)

        val_preds = np.argmax(val_logits, axis=-1)
        val_targets = val_targets.flatten()

        tp = np.sum((val_targets == 1) & (val_preds == 1))
        fp = np.sum((val_targets == 0) & (val_preds == 1))
        fn = np.sum((val_targets == 1) & (val_preds == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = np.mean(val_preds == val_targets)

        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={current_lr:.1e}  F1={f1:.4f}  Acc={acc:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'f1': f1, 'accuracy': acc, 'lr': current_lr})

        if val_loss < best_val_loss or f1 > best_f1:
            best_val_loss = min(best_val_loss, val_loss)
            best_f1 = max(best_f1, f1)
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss, 'f1': f1, 'accuracy': acc,
                'config': vars(args), 'sensor_cols': sensor_cols,
                'mean': torch.FloatTensor(normalizer.mean),
                'std': torch.FloatTensor(normalizer.std),
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"
Early stopping at epoch {epoch}")
                break

    print("Final Evaluation...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_logits, final_targets = trainer.validate(val_loader)
    final_preds = np.argmax(final_logits, axis=-1)
    final_targets = final_targets.flatten()

    tp = np.sum((final_targets == 1) & (final_preds == 1))
    tn = np.sum((final_targets == 0) & (final_preds == 0))
    fp = np.sum((final_targets == 0) & (final_preds == 1))
    fn = np.sum((final_targets == 1) & (final_preds == 0))

    print(f"Best F1: {f1:.4f}")
    print(f"Best Accuracy: {np.mean(final_preds == final_targets):.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'history.csv'), index=False)
    print(f"Saved to {checkpoint_dir}/")
    return checkpoint_dir


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("N-CMAPSS INTEGRATED TRAINING")
    print("=" * 60)
    print(f"Task: {args.task} | Device: {device}")
    if not USE_PETROMIND_UTILS:
        print("Using fallback implementations for windowing/normalization/labeling")

    print("
[1] Loading N-CMAPSS data...")
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

    sensor_cols = get_ncmapss_feature_cols(df)
    print(f"
[2] Sensors detected: {len(sensor_cols)}")

    if args.task == 'rul':
        checkpoint_dir = train_rul(args, df, sensor_cols, device)
    else:
        checkpoint_dir = train_cls(args, df, sensor_cols, device)

    print("
" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoint saved to: {checkpoint_dir}/")


if __name__ == '__main__':
    main()
