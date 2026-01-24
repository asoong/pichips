# PiChip Viewer

Real-time poker/casino chip detection and value calculation using YOLO11 with a Raspberry Pi 4 video stream.

## Architecture

- **Video Source**: Raspberry Pi 4 streaming via `tcp://pichip.local:8888`
- **Model**: `chips_yolo11.onnx` - YOLO11 detection model
- **Inference**: ONNX Runtime (CPU)
- **Display**: OpenCV window with bounding boxes, counts, and value HUD

## Chip Configuration

| Class ID | Color  | Value |
|----------|--------|-------|
| 0        | white  | $1    |
| 1        | red    | $5    |
| 2        | blue   | $10   |
| 3        | green  | $25   |
| 4        | black  | $100  |

## Files

- `overlay_viewer.py` - Main application
- `chips_yolo11.onnx` - YOLO11 detection model

## Usage

```bash
source .venv/bin/activate
python overlay_viewer.py
```

Press `q` to quit.

## Dependencies

- opencv-python
- numpy
- onnxruntime

## Model Details

- Input: `[1, 3, 640, 640]` (RGB, normalized to 0-1)
- Output: `[1, 84, 8400]` (detection format: 4 bbox coords + 80 class scores per prediction)
- Image size: 640x640
- Confidence threshold: 0.25
