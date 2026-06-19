#!/usr/bin/env bash
# Deploy a trained PiChip detector ON the Raspberry Pi.
#
# Workflow:
#   1) On your Mac, after training (train.py produces models/pichip_detector.pt):
#        scp pichip_detector.pt asoong91@pichip.local:~/pichip_viewer/models/
#   2) On the Pi:
#        ./deploy_model.sh        # exports ONNX (faster CPU inference) + points .env at it
#        ./run.sh                 # start the viewer
#
# Pass a different .pt as $1 if needed.
set -euo pipefail
cd "$(dirname "$0")"

PT="${1:-models/pichip_detector.pt}"
if [ ! -f "$PT" ]; then
  echo "ERROR: model not found at $PT"
  echo "Copy your trained weights over first, e.g.:"
  echo "  scp pichip_detector.pt $(whoami)@$(hostname).local:~/pichip_viewer/models/"
  exit 1
fi

echo "Exporting $PT -> ONNX (≈1.7x faster than the raw .pt on this Pi)..."
.venv/bin/yolo export model="$PT" format=onnx imgsz=640

ONNX="${PT%.pt}.onnx"

# Point the viewer at the ONNX model.
if grep -q '^PICHIP_DETECTOR_PATH=' .env 2>/dev/null; then
  sed -i "s|^PICHIP_DETECTOR_PATH=.*|PICHIP_DETECTOR_PATH=$ONNX|" .env
else
  echo "PICHIP_DETECTOR_PATH=$ONNX" >> .env
fi

echo
echo "Ready. Detector: $ONNX"
echo "Run it with:  ./run.sh"
echo "  - over SSH (headless): prints running chip counts"
echo "  - from the Pi's desktop (monitor attached): shows the live annotated window"
