#!/usr/bin/env bash
# Deploy a trained PiChip detector ON the Raspberry Pi.
#
# Workflow:
#   1) On your Mac, after training (train.py produces models/pichip_detector.pt):
#        scp pichip_detector.pt <pi-user>@<pi-host>.local:~/pichip_viewer/models/
#   2) On the Pi:
#        ./deploy_model.sh            # export ONNX @640, point .env at it
#        ./deploy_model.sh "" 416     # smaller imgsz = faster, less accurate on small chips
#        ./run.sh                     # start the viewer
#
# Args: $1 = path to .pt (default models/pichip_detector.pt), $2 = inference imgsz (default 640).
set -euo pipefail
cd "$(dirname "$0")"

PT="${1:-models/pichip_detector.pt}"
IMGSZ="${2:-640}"
if [ ! -f "$PT" ]; then
  echo "ERROR: model not found at $PT"
  echo "Copy your trained weights over first, e.g.:"
  echo "  scp pichip_detector.pt $(whoami)@$(hostname).local:~/pichip_viewer/models/"
  exit 1
fi

echo "Exporting $PT -> ONNX at imgsz=$IMGSZ (faster than the raw .pt on this Pi)..."
.venv/bin/yolo export model="$PT" format=onnx imgsz="$IMGSZ"

ONNX="${PT%.pt}.onnx"

# Point the viewer at the ONNX model and keep its inference size in sync (a static-shape
# ONNX model only accepts the size it was exported at).
set_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" .env 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" .env
  else
    echo "${key}=${val}" >> .env
  fi
}
set_env PICHIP_DETECTOR_PATH "$ONNX"
set_env PICHIP_IMGSZ "$IMGSZ"

echo
echo "Ready. Detector: $ONNX  (imgsz=$IMGSZ)"
echo "Run it with:  ./run.sh"
echo "  - over SSH (headless): prints running chip counts; MJPEG stream stays live"
echo "  - from the Pi's desktop (monitor attached): shows the live annotated window"
