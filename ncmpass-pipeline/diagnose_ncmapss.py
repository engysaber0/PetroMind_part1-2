#!/usr/bin/env python3
"""
N-CMAPSS Data Quality Diagnostic — Corrected for _dev/_test suffixes
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Diagnose N-CMAPSS data quality')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--file', default=None, help='Specific file to analyze')
    p.add_argument('--output-dir', default='diagnostic_plots')
    p.add_argument('--max-engines', type=int, default=10)
    return p.parse_args()


def inspect_h5_structure(h5_path):
    """Print detailed structure of HDF5 file with suffix handling."""
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(h5_path)}")
    print(f"{'='*60}")

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"\nDatasets: {keys}")

        for key in keys:
            ds = f[key]
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

        # Analyze both dev and test splits
        for split in ['dev', 'test']:
            suffix = f"_{split}"
            a_key = f"A{suffix}" if f"A{suffix}" in keys else "A"
            y_key = f"Y{suffix}" if f"Y{suffix}" in keys else "Y"

            if a_key not in keys or y_key not in keys:
                print(f"\n[SKIP] Split '{split}' not found")
                continue

            print(f"\n--- SPLIT: {split.upper()} ---")
            A = f[a_key][:]
            Y = f[y_key][:].flatten()

            print(f"A ({a_key}) shape: {A.shape}")
            print(f"Y ({y_key}) shape: {Y.shape}")
            print(f"A columns: [unit, cycle, flight_class, health_state]")
            print(f"  unit range: [{A[:,0].min():.0f}, {A[:,0].max():.0f}]")
            print(f"  cycle range: [{A[:,1].min():.0f}, {A[:,1].max():.0f}]")
            print(f"  flight_class: {np.unique(A[:,2])}")
            print(f"  health_state: {np.unique(A[:,3])}")

            # Engine stats
            units = np.unique(A[:, 0])
            print(f"Engines: {len(units)}")

            engine_stats = []
            for unit in units:
                mask = A[:, 0] == unit
                cycles = A[mask, 1]
                health = A[mask, 3]
                rul_vals = Y[mask]

                stats = {
                    'unit': int(unit),
                    'n_rows': int(mask.sum()),
                    'n_cycles': int(np.unique(cycles).size),
                    'health_states': np.unique(health).tolist(),
                    'rul_min': float(rul_vals.min()),
                    'rul_max': float(rul_vals.max()),
                    'rul_mean': float(rul_vals.mean()),
                    'rul_std': float(rul_vals.std()),
                }
                engine_stats.append(stats)

            df_stats = pd.DataFrame(engine_stats)
            print(f"\nEngine Statistics:")
            print(df_stats.to_string())

            # Issue detection
            print(f"\n--- ISSUE DETECTION ({split}) ---")

            # Issue 1: All engines have same health state
            all_health = df_stats['health_states'].apply(lambda x: tuple(sorted(x)))
            unique_health_patterns = all_health.unique()
            print(f"Health state patterns: {unique_health_patterns}")
            if len(unique_health_patterns) == 1:
                print("[WARN] All engines have identical health state pattern!")

            # Issue 2: RUL doesn't vary
            if df_stats['rul_std'].sum() < 0.1:
                print("[WARN] RUL has near-zero variance across all engines!")

            # Issue 3: Too few cycles per engine
            min_cycles = df_stats['n_cycles'].min()
            if min_cycles < 10:
                print(f"[WARN] Some engines have very few cycles (min={min_cycles})")

            # Issue 4: Check sensor data
            xs_key = f"X_s{suffix}" if f"X_s{suffix}" in keys else "X_s"
            if xs_key in keys:
                X_s = f[xs_key][:]
                print(f"\nX_s ({xs_key}) shape: {X_s.shape}")
                sensor_means = np.mean(X_s, axis=0)
                sensor_stds = np.std(X_s, axis=0)
                print(f"Sensor means: min={sensor_means.min():.4f}, max={sensor_means.max():.4f}")
                print(f"Sensor stds:  min={sensor_stds.min():.4f}, max={sensor_stds.max():.4f}")

                constant_sensors = np.where(sensor_stds < 1e-6)[0]
                if len(constant_sensors) > 0:
                    print(f"[WARN] Constant sensors (no variance): {constant_sensors}")

            return df_stats, A, Y, split


def plot_rul_curves(A, Y, output_dir, split_name, max_engines=10):
    """Plot RUL curves for each engine."""
    os.makedirs(output_dir, exist_ok=True)

    units = np.unique(A[:, 0])
    if max_engines > 0:
        units = units[:max_engines]

    fig, axes = plt.subplots(len(units), 1, figsize=(12, 2*len(units)))
    if len(units) == 1:
        axes = [axes]

    for idx, unit in enumerate(units):
        mask = A[:, 0] == unit
        rul = Y[mask]

        ax = axes[idx]
        ax.plot(range(len(rul)), rul, linewidth=0.5)
        ax.set_ylabel(f'Unit {int(unit)}')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)

        # Mark health state changes
        health = A[mask, 3]
        state_changes = np.where(np.diff(health) != 0)[0]
        for sc in state_changes:
            ax.axvline(x=sc, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f'rul_curves_{split_name}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    print(f"\nSaved RUL curves to {output_dir}/{fname}")
    plt.close()


def plot_sensor_distributions(h5_path, A, output_dir, split_name, max_engines=10):
    """Plot sensor distributions."""
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        suffix = f"_{split_name}"
        xs_key = f"X_s{suffix}" if f"X_s{suffix}" in keys else "X_s"

        if xs_key not in keys:
            print("[SKIP] No X_s data for sensor plots")
            return

        X_s = f[xs_key][:]
        units = np.unique(A[:, 0])
        if max_engines > 0:
            units = units[:max_engines]

        n_sensors = min(4, X_s.shape[1])
        fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2*n_sensors))
        if n_sensors == 1:
            axes = [axes]

        sensor_names = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50',
                        'P15', 'P21', 'P24', 'Ps30', 'P40', 'P50']

        for s in range(n_sensors):
            ax = axes[s]
            name = sensor_names[s] if s < len(sensor_names) else f'Sensor_{s}'

            for unit in units:
                mask = A[:, 0] == unit
                data = X_s[mask, s]
                ax.plot(range(len(data)), data, alpha=0.5, label=f'Unit {int(unit)}')

            ax.set_ylabel(name)
            ax.set_xlabel('Time Step')
            ax.grid(True, alpha=0.3)
            if len(units) <= 5:
                ax.legend()

        plt.tight_layout()
        fname = f'sensor_traces_{split_name}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        print(f"Saved sensor traces to {output_dir}/{fname}")
        plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("N-CMAPSS DATA QUALITY DIAGNOSTIC (Corrected)")
    print("=" * 60)

    data_dir = args.data_dir
    if args.file:
        files = [os.path.join(data_dir, args.file)]
    else:
        files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.endswith('.h5')])

    if not files:
        print(f"[ERROR] No .h5 files found in {data_dir}")
        sys.exit(1)

    all_stats = []
    for h5_path in files:
        if not os.path.exists(h5_path):
            print(f"[SKIP] Not found: {h5_path}")
            continue

        try:
            stats, A, Y, split = inspect_h5_structure(h5_path)
            all_stats.append(stats)

            plot_rul_curves(A, Y, args.output_dir, split, args.max_engines)
            plot_sensor_distributions(h5_path, A, args.output_dir, split, args.max_engines)

        except Exception as e:
            print(f"[ERROR] Failed to process {h5_path}: {e}")
            import traceback
            traceback.print_exc()

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY")
        print(f"{'='*60}")
        print(f"Total engines: {len(combined)}")
        print(f"Total rows: {combined['n_rows'].sum():,}")
        print(f"Total cycles: {combined['n_cycles'].sum():,}")
        print(f"\nRUL statistics:")
        print(combined[['rul_min', 'rul_max', 'rul_mean', 'rul_std']].describe())

        summary_path = os.path.join(args.output_dir, 'summary.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        combined.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
