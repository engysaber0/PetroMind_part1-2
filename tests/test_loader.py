#!/usr/bin/env python3
"""
Test N-CMAPSS loader.

Usage:
    python test_loader.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ncmapss_loader import load_ncmapss_file

h5_path = "data/ncmapss/N-CMAPSS_DS02-006.h5"

print(f"Testing: {h5_path}")
print("=" * 50)

try:
    df = load_ncmapss_file(h5_path, split="dev", sample_every=5, verbose=True)
    print(f"\n✅ SUCCESS!")
    print(f"   Cycles: {len(df)}")
    print(f"   Engines: {df['unit_id'].nunique()}")
    print(f"   RUL range: [{df['rul'].min():.1f}, {df['rul'].max():.1f}]")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
except FileNotFoundError:
    print(f"\n⚠️  File not found: {h5_path}")
    print("   Generate synthetic data first:")
    print("   python generate_synthetic_ncmapss.py --output-dir data/ncmapss/")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
