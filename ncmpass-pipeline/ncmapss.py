#!/usr/bin/env python3
"""
N-CMAPSS CLI - Unified interface for training and testing
Usage:
    python ncmapss.py train --task rul --data-dir data/ncmapss/
    python ncmapss.py test --task cls --model-path checkpoints_ncmapss_cls/best_model.pt --data-dir data/ncmapss/
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ncmapss.py [train|test] [options...]")
        print("")
        print("Examples:")
        print("  python ncmapss.py train --task rul --data-dir data/ncmapss/ --epochs 100")
        print("  python ncmapss.py train --task cls --data-dir data/ncmapss/ --threshold 20")
        print("  python ncmapss.py test --task cls --model-path checkpoints_ncmapss_cls/best_model.pt --data-dir data/ncmapss/")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'train':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        import train_ncmapss
        train_ncmapss.main()
    elif command == 'test':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        import test_ncmapss
        test_ncmapss.main()
    else:
        print("Unknown command: " + command)
        print("Use 'train' or 'test'")
        sys.exit(1)

if __name__ == '__main__':
    main()
