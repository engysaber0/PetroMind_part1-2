#!/bin/bash
# N-CMAPSS Integration Setup
# Copies integrated files to fine-tuning/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "N-CMAPSS Integration Setup"
echo "=========================================="

# Check petromind exists
if [ ! -d "$SCRIPT_DIR/../petromind/pipeline" ]; then
    echo "[ERROR] petromind/pipeline not found"
    exit 1
fi

echo "[OK] Found petromind/pipeline"

# Check ncmapss_loader exists
if [ ! -f "$SCRIPT_DIR/ncmapss_loader.py" ]; then
    echo "[ERROR] ncmapss_loader.py not found in fine-tuning/"
    echo "[HINT] Make sure ncmapss_loader.py is in the same directory"
    exit 1
fi

echo "[OK] Found ncmapss_loader.py"

# Verify imports work
echo ""
echo "[Step] Verifying imports..."
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/..')
try:
    from petromind.pipeline import build_sliding_windows, SensorNormalizer
    from petromind.pipeline.lstm_model import LSTMClassifier
    from petromind.pipeline.models import LSTMRULModel
    print('[OK] petromind.pipeline imports successful')
except Exception as e:
    print(f'[ERROR] Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "[Step] Verifying ncmapss_loader..."
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
try:
    from ncmapss_loader import load_ncmapss_smart
    print('[OK] ncmapss_loader import successful')
except Exception as e:
    print(f'[ERROR] Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Setup complete! You can now run:"
echo ""
echo "  python train_ncmapss.py --task rul --data-dir data/ncmapss/ --epochs 100"
echo "  python train_ncmapss.py --task cls --data-dir data/ncmapss/ --threshold 20"
echo "  python test_ncmapss.py --task cls --model-path checkpoints_ncmapss_cls/best_model.pt --data-dir data/ncmapss/"
echo "=========================================="
