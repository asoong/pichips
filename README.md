# PiChip Viewer

Real-time poker chip stack detection, counting, and value calculation using computer vision. Designed to work with a Raspberry Pi video stream.

## Features

- **MobileSAM Segmentation**: Precise chip stack segmentation using point prompts
- **Edge-based Counting**: Counts individual chips in stacks via Canny edge detection
- **HSV Color Classification**: Identifies chip colors (white, red, blue, yellow)
- **Real-time Display**: OpenCV window with mask overlays and value HUD

## Project Structure

```
pichip_viewer/
├── overlay_viewer.py      # Main viewer application
├── models/                # Model weights (download separately)
│   ├── mobile_sam.pt
│   └── yolov8s-worldv2.pt
├── web/                   # Streamlit debug/experimentation app
│   ├── app.py
│   └── requirements.txt
├── .env.example           # Environment variable template
└── requirements.txt       # Core dependencies
```

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repo-url>
cd pichip_viewer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Create models directory (if not exists)
mkdir -p models

# Download MobileSAM (~39MB) - required for main viewer
wget -O models/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
# Most important: set PICHIP_STREAM_URL to your Pi's address
```

### 4. Run

```bash
python overlay_viewer.py
```

Press `q` to quit.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PICHIP_STREAM_URL` | `tcp://pichip.local:8888` | Raspberry Pi video stream URL |
| `PICHIP_MOBILESAM_PATH` | `models/mobile_sam.pt` | Path to MobileSAM weights |
| `PICHIP_SAM_INTERVAL` | `10` | Run segmentation every N frames |
| `PICHIP_MIN_REGION_AREA` | `1000` | Minimum pixel area for chip detection |
| `PICHIP_DEVICE` | `auto` | Compute device: `auto`, `cuda`, `mps`, `cpu` |

## Streamlit Debug App

A separate web-based app for experimentation with YOLO-World detection.

### Setup

```bash
# Install additional dependencies
pip install -r web/requirements.txt

# Download YOLO-World model (optional - auto-downloads on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt')"
mv yolov8s-worldv2.pt models/
```

### Run

```bash
cd web
streamlit run app.py
```

The web UI provides:
- Adjustable confidence threshold
- Editable chip values
- Real-time count metrics

## Hardware Requirements

- **Video Source**: Raspberry Pi with camera streaming via TCP (e.g., using FFmpeg)
- **Compute**:
  - Apple Silicon (MPS): Recommended for Mac
  - NVIDIA GPU (CUDA): Best performance
  - CPU: Works but slower

## Chip Configuration

| Color | Value | HSV Range |
|-------|-------|-----------|
| White | $1 | Low saturation, high value |
| Red | $5 | Hue 0-10, 165-180 |
| Blue | $10 | Hue 95-125 |
| Yellow | $25 | Hue 15-35 |

## Architecture

```
Video Stream (Pi) -> Color Detection -> MobileSAM Segmentation -> Edge Counting -> Display
```

1. **Color Detection**: HSV thresholding finds chip regions
2. **MobileSAM**: Point prompts generate precise masks for each region
3. **Edge Counting**: Canny edges + peak detection counts chips per stack
4. **Visualization**: Overlay masks, contours, and value HUD

## Tuning

Adjust these values in `.env` for your setup:

- `PICHIP_SAM_INTERVAL`: Increase for better FPS, decrease for more responsive detection
- `PICHIP_MIN_REGION_AREA`: Increase if detecting noise, decrease if missing small stacks

HSV color ranges can be tuned in `overlay_viewer.py` (`CHIP_HSV_RANGES` dict) for different lighting conditions.
