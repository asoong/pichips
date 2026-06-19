# PiChip Viewer

Real-time poker chip stack detection, counting, and value calculation using a **custom
YOLO detector** trained on synthetic data from the Raya Labs PiChip pipeline. Designed to
work with a Raspberry Pi video stream.

## Features

- **Custom YOLO detector**: color (white/red/blue/yellow) and orientation (face/edge) come
  straight from a model you train — no fragile HSV color guessing
- **Edge-based counting**: counts individual chips in side-lying stacks via Canny edge
  detection
- **Real-time display**: OpenCV window with detection boxes and value HUD

## End-to-end pipeline

This repo is the **training + inference** end of the PiChip pipeline. The full loop is
documented in **[TRAINING.md](TRAINING.md)** (includes how many images to use):

```
PiChip web client:  generate synthetic images → segment/curate → export dataset.zip
        │
        ▼
pichip_viewer:  train.py  →  models/pichip_detector.pt  →  overlay_viewer.py
```

## Project Structure

```
pichip_viewer/
├── overlay_viewer.py      # Main viewer (loads the custom YOLO detector)
├── train.py               # Train a detector from an exported dataset
├── TRAINING.md            # End-to-end runbook (generate → train → run)
├── models/                # Model weights (gitignored)
│   └── pichip_detector.pt # Your trained detector (produced by train.py)
├── web/                   # Streamlit debug/experimentation app
│   ├── app.py
│   └── requirements.txt
├── .env.example
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
git clone <repo-url> pichip_viewer
cd pichip_viewer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a model

Train one from a PiChip training-set export (see **[TRAINING.md](TRAINING.md)**):

```bash
python train.py --dataset datasets/<token> --model yolo26n.pt \
  --epochs 100 --imgsz 1024 --batch 16 --device mps
# writes models/pichip_detector.pt
```

### 3. Configure & run

```bash
cp .env.example .env       # set PICHIP_STREAM_URL to your Pi's address
python overlay_viewer.py
```

Press `q` to quit.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PICHIP_STREAM_URL` | `tcp://pichip.local:8888` | Raspberry Pi video stream URL |
| `PICHIP_DETECTOR_PATH` | `models/pichip_detector.pt` | Trained detector loaded by the viewer |
| `PICHIP_YOLO_PATH` | `models/pichip_detector.pt` | Model for the Streamlit app (a `*-worldv2.pt` enables YOLO-World mode) |
| `PICHIP_DETECT_INTERVAL` | `5` | Run detection every N frames |
| `PICHIP_CONFIDENCE` | `0.3` | Detection confidence threshold |
| `PICHIP_DEVICE` | `auto` | Compute device: `auto`, `cuda`, `mps`, `cpu` |

## Streamlit Debug App

A separate web UI for experimentation. By default it loads the same custom detector; point
`PICHIP_YOLO_PATH` at a `yolov8s-worldv2.pt` to fall back to open-vocab YOLO-World.

```bash
pip install -r web/requirements.txt
cd web
streamlit run app.py
```

## Architecture

```
Video Stream (Pi) -> Custom YOLO detector -> per-class boxes (color + face/edge)
                  -> edge-count each side-lying stack -> value HUD
```

1. **Detection**: the trained YOLO model localizes chips and labels each as
   `<color>_<face|edge>`.
2. **Counting**: face detections count as 1; edge (side-lying stack) detections are
   counted via Canny edges + peak detection.
3. **Visualization**: overlay boxes, per-detection counts, and the value HUD.

## Chip Configuration

| Color | Value |
|-------|-------|
| White | $1 |
| Red | $5 |
| Blue | $10 |
| Yellow | $25 |

Adjust values live in the Streamlit sidebar, or edit `CHIP_VALUES` in the source.

## Hardware Requirements

- **Video Source**: Raspberry Pi with camera streaming via TCP (e.g., FFmpeg)
- **Compute**: Apple Silicon (MPS) recommended on Mac; NVIDIA GPU (CUDA) for best
  performance; CPU works but is slower.

## Tuning

- `PICHIP_DETECT_INTERVAL`: increase for higher FPS, decrease for more responsive updates.
- `PICHIP_CONFIDENCE`: raise to drop low-confidence detections.
- Retrain with more data (see TRAINING.md) if detection/counting is weak on real chips.
