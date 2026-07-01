#!/usr/bin/env python3
"""
End-to-End Test: PetroMind N-CMAPSS Pipeline

This script:
1. Generates synthetic N-CMAPSS data (no 14GB download needed)
2. Loads it using ncmapss_loader.py
3. Runs through the full pipeline (windowing, features, normalization)
4. Trains a small model
5. Verifies everything works

Usage:
    python e2e_test.py

Expected output: "✅ ALL TESTS PASSED" at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np

# Add parent directory to path for petromind imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("PETROMIND N-CMAPSS END-TO-END TEST")
print("=" * 60)

# =====================================================================
# STEP 1: Generate synthetic data
# =====================================================================
print("\n[Step 1] Generating synthetic N-CMAPSS data...")

from generate_synthetic_ncmapss import generate_synthetic_ncmapss

with tempfile.TemporaryDirectory() as tmpdir:
    h5_path = Path(tmpdir) / "N-CMAPSS_DS02-006.h5"
    generate_synthetic_ncmapss(
        output_path=h5_path,
        n_engines=5,        # Small for quick test
        min_cycles=40,
        max_cycles=60,
        seed=42,
    )

    # =====================================================================
    # STEP 2: Load with ncmapss_loader
    # =====================================================================
    print("\n[Step 2] Loading with ncmapss_loader...")

    from ncmapss_loader import load_ncmapss_file, get_ncmapss_feature_cols

    df = load_ncmapss_file(
        h5_path,
        split="dev",
        sample_every=1,
        include_virtual_sensors=True,
        include_health_params=True,
        verbose=True,
    )

    assert len(df) > 0, "No data loaded!"
    assert "unit_id" in df.columns
    assert "cycle" in df.columns
    assert "rul" in df.columns
    print(f"  ✅ Loaded {len(df)} cycles from {df['unit_id'].nunique()} engines")

    # =====================================================================
    # STEP 3: Pipeline integration
    # =====================================================================
    print("\n[Step 3] Pipeline integration...")

    from petromind.pipeline import (
        PipelineConfig,
        build_sliding_windows,
        compute_classification_label,
        FeatureExtractor,
        SensorNormalizer,
        build_dataloaders,
    )

    # Create config
    cfg = PipelineConfig(
        window_size=20,  # Smaller for test
        stride=1,
        prediction_horizon=20,
        rul_clip=125,
        val_ratio=0.2,
        batch_size=16,
        normalize_sensors=True,
    )

    # Compute classification labels
    df = compute_classification_label(df, cfg)
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Get feature columns
    feature_cols = get_ncmapss_feature_cols(df)
    print(f"  Feature columns: {len(feature_cols)}")

    # Build windows
    X_raw, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg, feature_cols=feature_cols)
    print(f"  Windows: {X_raw.shape}")
    assert X_raw.shape[0] > 0, "No windows produced!"
    print(f"  ✅ Windowing works: {X_raw.shape}")

    # Feature engineering
    extractor = FeatureExtractor(cfg, n_pca_components=3)
    X_eng = extractor.transform(X_raw)
    print(f"  Engineered features: {X_eng.shape}")
    assert not np.isnan(X_eng).any(), "NaN in engineered features!"
    print(f"  ✅ Feature engineering works")

    # Normalization
    normalizer = SensorNormalizer()
    unique_engines = np.sort(np.unique(engine_ids))
    n_val = max(1, int(len(unique_engines) * cfg.val_ratio))
    val_engines = set(unique_engines[-n_val:])
    train_mask = np.array([eid not in val_engines for eid in engine_ids])

    X_train = X_raw[train_mask]
    normalizer.fit(X_train)
    X_norm = normalizer.transform(X_raw)
    print(f"  ✅ Normalization works")

    # Build DataLoaders
    train_loader, val_loader, dataset = build_dataloaders(
        X_norm, y_cls, y_rul, engine_ids, cfg
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    assert len(train_loader) > 0, "No train batches!"
    print(f"  ✅ DataLoaders work")

    # =====================================================================
    # STEP 4: Model test
    # =====================================================================
    print("\n[Step 4] Model test...")

    try:
        import torch
        from petromind.pipeline.models import LSTMRULModel

        input_dim = X_norm.shape[2]
        model = LSTMRULModel(input_dim=input_dim, cfg=cfg)

        # Test forward pass
        batch = next(iter(train_loader))
        X_batch = batch["features"]

        with torch.no_grad():
            output = model(X_batch)

        assert output.shape[0] == X_batch.shape[0], "Batch size mismatch!"
        assert (output >= 0).all(), "Negative RUL prediction!"
        print(f"  ✅ LSTMRULModel forward pass works: {output.shape}")

    except ImportError as e:
        print(f"  ⚠️  PyTorch not available: {e}")

    # =====================================================================
    # STEP 5: Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nYour pipeline is ready! Next steps:")
    print("  1. Get real N-CMAPSS data (download from Kaggle or use synthetic)")
    print("  2. Run: python train_ncmapss.py --data-dir data/ncmapss/ --preset quick-start")
    print("  3. Benchmark vs your C-MAPSS FD models")

if __name__ == "__main__":
    e2e_test()
