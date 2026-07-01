#!/usr/bin/env python3
"""
N-CMAPSS Corrected Loader
Handles actual file structure: A_dev, A_test, Y_dev, Y_test, etc.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib

# Standard N-CMAPSS column names
PHYSICAL_SENSOR_COLS = [
    'Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50',
    'P15', 'P21', 'P24', 'Ps30', 'P40', 'P50'
]

VIRTUAL_SENSOR_COLS = [
    'X_v_0', 'X_v_1', 'X_v_2', 'X_v_3', 'X_v_4', 'X_v_5',
    'X_v_6', 'X_v_7', 'X_v_8', 'X_v_9', 'X_v_10', 'X_v_11',
    'X_v_12', 'X_v_13'
]

HEALTH_PARAM_COLS = [
    'Fan_eff_mod', 'Fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
    'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
    'LPT_eff_mod', 'LPT_flow_mod'
]

SCENARIO_COLS = ['alt', 'Mach', 'TRA', 'T2']
ALL_SENSOR_COLS = PHYSICAL_SENSOR_COLS + VIRTUAL_SENSOR_COLS + HEALTH_PARAM_COLS + SCENARIO_COLS


def get_cache_path(h5_path, split, sample_every):
    """Generate cache file path based on input parameters."""
    h5_path = Path(h5_path)
    cache_dir = h5_path.parent / '.ncmapss_cache'
    cache_dir.mkdir(exist_ok=True)
    mtime = os.path.getmtime(h5_path)
    hash_str = hashlib.md5(f"{h5_path.name}_{mtime}_{split}_{sample_every}".encode()).hexdigest()[:12]
    return cache_dir / f"{h5_path.stem}_{split}_{sample_every}_{hash_str}.pkl"


def load_ncmapss_file(h5_path, split='dev', sample_every=1, use_cache=True, 
                      verbose=False, max_engines=None):
    """
    Load N-CMAPSS file with correct _dev / _test suffix handling.

    Args:
        h5_path: Path to .h5 file
        split: 'dev' or 'test' — determines which suffix to read
        sample_every: Subsample factor (1=keep all, 5=keep every 5th)
        use_cache: Whether to use disk cache
        verbose: Print progress
        max_engines: Limit to N engines (for quick testing)
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    # Check cache
    cache_file = get_cache_path(h5_path, split, sample_every)
    if use_cache and cache_file.exists():
        if verbose:
            print(f"[CACHE] Loading from {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if verbose:
        print(f"[LOAD] Reading {h5_path.name} | split={split} | sample={sample_every}")

    suffix = f"_{split}"

    with h5py.File(h5_path, 'r') as f:
        # Check available keys
        keys = list(f.keys())

        # Determine correct key names
        a_key = f"A{suffix}" if f"A{suffix}" in keys else "A"
        y_key = f"Y{suffix}" if f"Y{suffix}" in keys else "Y"
        w_key = f"W{suffix}" if f"W{suffix}" in keys else "W"
        xs_key = f"X_s{suffix}" if f"X_s{suffix}" in keys else "X_s"
        xv_key = f"X_v{suffix}" if f"X_v{suffix}" in keys else "X_v"
        t_key = f"T{suffix}" if f"T{suffix}" in keys else "T"

        if verbose:
            print(f"  Using keys: A={a_key}, Y={y_key}, W={w_key}, X_s={xs_key}, X_v={xv_key}, T={t_key}")

        # Read arrays
        A = f[a_key][:]
        Y = f[y_key][:].flatten()

        has_W = w_key in keys
        has_Xs = xs_key in keys
        has_Xv = xv_key in keys
        has_T = t_key in keys

        if verbose:
            print(f"  A: {A.shape} | Y: {Y.shape}")
            print(f"  W: {has_W} | X_s: {has_Xs} | X_v: {has_Xv} | T: {has_T}")

        # Apply sampling within each (unit, cycle)
        if sample_every > 1:
            sampled_mask = np.zeros(len(A), dtype=bool)
            for unit in np.unique(A[:, 0]):
                for cycle in np.unique(A[A[:, 0] == unit, 1]):
                    idx = np.where((A[:, 0] == unit) & (A[:, 1] == cycle))[0]
                    sampled_mask[idx[::sample_every]] = True
            row_indices = np.where(sampled_mask)[0]
        else:
            row_indices = np.arange(len(A))

        # Limit engines if requested
        if max_engines is not None:
            units_in_data = np.unique(A[row_indices, 0])
            selected_units = units_in_data[:max_engines]
            row_mask = np.isin(A[:, 0], selected_units)
            row_indices = row_indices[row_mask[row_indices]]

        if verbose:
            print(f"  Selected {len(row_indices):,} / {len(A):,} rows")

        # Read sensor data for selected rows only
        W = f[w_key][row_indices] if has_W else None
        X_s = f[xs_key][row_indices] if has_Xs else None
        X_v = f[xv_key][row_indices] if has_Xv else None
        T = f[t_key][row_indices] if has_T else None

        A_filtered = A[row_indices]
        Y_filtered = Y[row_indices]

    # Aggregate per cycle
    records = []
    units = np.unique(A_filtered[:, 0])

    if verbose:
        print(f"  Engines: {len(units)} (sample_every={sample_every})")

    for unit in units:
        unit_mask = A_filtered[:, 0] == unit
        cycles = np.unique(A_filtered[unit_mask, 1])

        for cycle in cycles:
            cyc_mask = unit_mask & (A_filtered[:, 1] == cycle)
            cyc_indices = np.where(cyc_mask)[0]

            if len(cyc_indices) == 0:
                continue

            # RUL is the last Y value in the cycle
            rul = Y_filtered[cyc_indices[-1]]

            record = {
                'unit_id': int(unit),
                'cycle': int(cycle),
                'rul': float(rul),
            }

            # Add scenario data (mean per cycle)
            if W is not None:
                for i, col in enumerate(SCENARIO_COLS):
                    record[col] = float(np.mean(W[cyc_indices, i]))

            # Add physical sensors
            if X_s is not None:
                for i, col in enumerate(PHYSICAL_SENSOR_COLS):
                    record[col] = float(np.mean(X_s[cyc_indices, i]))

            # Add virtual sensors
            if X_v is not None:
                for i, col in enumerate(VIRTUAL_SENSOR_COLS):
                    record[col] = float(np.mean(X_v[cyc_indices, i]))

            # Add health parameters
            if T is not None:
                for i, col in enumerate(HEALTH_PARAM_COLS):
                    record[col] = float(np.mean(T[cyc_indices, i]))

            records.append(record)

    df = pd.DataFrame(records)

    # Cache result
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        if verbose:
            print(f"[CACHE] Saved to {cache_file.name}")

    if verbose:
        print(f"  Loaded: {len(df)} cycles from {df['unit_id'].nunique()} engines")
        print(f"  RUL range: [{df['rul'].min():.1f}, {df['rul'].max():.1f}]")

    return df


def load_ncmapss_smart(data_dir, subsets=None, split='dev', sample_every=1,
                       use_cache=True, verbose=False, max_engines_per_file=None):
    """Load multiple N-CMAPSS files."""
    data_dir = Path(data_dir)

    if subsets is None:
        subsets = sorted(data_dir.glob('N-CMAPSS_*.h5'))
        if not subsets:
            raise FileNotFoundError(f"No N-CMAPSS_*.h5 files found in {data_dir}")
    else:
        subsets = [data_dir / s if not os.path.isabs(s) else Path(s) for s in subsets]

    all_dfs = []
    for h5_path in subsets:
        if not h5_path.exists():
            print(f"[SKIP] Not found: {h5_path}")
            continue
        try:
            df = load_ncmapss_file(
                h5_path, split=split, sample_every=sample_every,
                use_cache=use_cache, verbose=verbose,
                max_engines=max_engines_per_file
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {h5_path.name}: {e}")
            continue

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    if verbose:
        print(f"\n[Summary] {combined['unit_id'].nunique()} engines, {len(combined)} cycles total")
        print(f"Shape: {combined.shape} | Engines: {combined['unit_id'].nunique()}")
    return combined


def get_ncmapss_feature_cols(df):
    """Get sensor columns from DataFrame."""
    exclude = ['unit_id', 'cycle', 'rul', 'label']
    return [c for c in df.columns if c not in exclude]


def clear_cache(data_dir):
    """Clear all cached files."""
    cache_dir = Path(data_dir) / '.ncmapss_cache'
    if cache_dir.exists():
        for f in cache_dir.glob('*.pkl'):
            f.unlink()
        print(f"[CACHE] Cleared {cache_dir}")


# Backward compatibility aliases
load_ncmapss_file_fast = load_ncmapss_file
load_ncmapss_smart_fast = load_ncmapss_smart
load_ncmapss_all_datasets = load_ncmapss_smart


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--file', default=None)
    p.add_argument('--split', default='dev')
    p.add_argument('--sample-every', type=int, default=1)
    p.add_argument('--max-engines', type=int, default=None)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--clear-cache', action='store_true')
    args = p.parse_args()

    if args.clear_cache:
        clear_cache(args.data_dir)
        sys.exit(0)

    subsets = [args.file] if args.file else None
    df = load_ncmapss_smart(
        args.data_dir, subsets=subsets, split=args.split,
        sample_every=args.sample_every, use_cache=not args.no_cache,
        verbose=True, max_engines_per_file=args.max_engines
    )
    print(f"\nColumns: {list(df.columns)}")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
