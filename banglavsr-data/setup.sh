#!/usr/bin/env bash
set -e

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found on PATH. Install it:"
  echo "  Ubuntu/Debian: sudo apt-get install -y ffmpeg"
  echo "  macOS:         brew install ffmpeg"
fi

echo
echo "Setup complete."
echo "Activate later with:  source .venv/bin/activate"
echo "For GPU detection, replace onnxruntime with onnxruntime-gpu:"
echo "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
echo
echo "Run on one video:"
echo "  python run.py --url \"https://www.youtube.com/watch?v=XXXX\" --out data --limit 1"
