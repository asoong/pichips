# PiChip Viewer

Real-time poker-chip detection, counting, and value calculation using a **custom YOLO
detector** trained on synthetic data from the Raya Labs PiChip pipeline. The detector runs
**on a Raspberry Pi** reading its camera directly, and you can watch the annotated feed in a
browser on another machine (or on a monitor plugged into the Pi).

> See **[README.html](README.html)** for a visual architecture diagram, and
> **[TRAINING.md](TRAINING.md)** for the end-to-end runbook (including how many images to use).

## How it works

The system spans three places:

```
┌─────────────── Raya Labs PiChip web client ───────────────┐
│ generate synthetic images (Gemini) → segment/label (SAM3) │
│ → curate masks → export dataset.zip (YOLO format) to S3   │
└────────────────────────────┬──────────────────────────────┘
                             │  download dataset.zip
                             ▼
┌──────────────────── Your Mac — training ──────────────────┐
│ train.py (Ultralytics, MPS) → models/pichip_detector.pt   │
└────────────────────────────┬──────────────────────────────┘
                             │  scp the .pt to the Pi
                             ▼
┌──────────────── Raspberry Pi — live inference ────────────┐
│ CSI camera (picamera2) → overlay_viewer.py:               │
│   YOLO/ONNX detect → color + face/edge per chip           │
│   → edge-count side-lying stacks → value HUD              │
│   → MJPEG server on :8090                                  │
└───────┬───────────────────────────────────┬───────────────┘
        │ MJPEG over HTTP                    │ HDMI
        ▼                                    ▼
   Mac browser                          Monitor on the Pi
 http://pichip.local:8090/              (live window)
```

1. **Detection** — the trained YOLO model localizes each chip and labels it
   `<color>_<face|edge>` (color ∈ white/red/blue/yellow). Class names are embedded in the
   weights, so the viewer reads them via `model.names` and never hardcodes a list.
2. **Counting** — a face-on chip counts as 1; a side-lying *stack* (one `edge` detection) is
   counted with Canny edges + peak detection.
3. **Output** — boxes + per-chip counts + a running value HUD, shown in a local window and/or
   served as an MJPEG HTTP stream.

Inference runs in a **background thread**, so capture/display/streaming stay at full camera
frame rate (~15 fps on a Pi 4) while detections refresh asynchronously a couple of times a
second — the feed never freezes waiting on the model.

## Project structure

```
pichip_viewer/
├── overlay_viewer.py   # Main viewer: camera → detector → HUD → (window | MJPEG)
├── train.py            # Train a detector from an exported dataset (run on the Mac)
├── deploy_model.sh     # On the Pi: export a trained .pt → ONNX + point .env at it
├── run.sh              # On the Pi: launch the viewer using the venv + .env
├── TRAINING.md         # End-to-end runbook (generate → train → run)
├── README.html         # Architecture diagram (open in a browser)
├── models/             # Model weights (gitignored)
├── web/                # Streamlit debug app (YOLO-World / experimentation)
├── .env.example
└── requirements.txt
```

## Setup

### On the Raspberry Pi (where inference runs)

Requires a Pi with a CSI camera and 64-bit Raspberry Pi OS / Debian. `picamera2` must be
available at the system level (preinstalled on Raspberry Pi OS, else `sudo apt install -y
python3-picamera2`).

```bash
git clone <repo-url> pichip_viewer
cd pichip_viewer

# --system-site-packages so the venv can see the system picamera2 / libcamera
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# IMPORTANT: install CPU-only torch from the CPU index — the default pulls ~1.3 GB of
# unused CUDA wheels on ARM.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python

cp .env.example .env        # defaults already target the Pi camera + MJPEG stream
```

### On your Mac (where training runs)

```bash
pip install ultralytics      # pulls an MPS-capable torch on macOS
```

## Commands

### 1. Train (on the Mac)

After exporting a training set from the web client (see [TRAINING.md](TRAINING.md)):

```bash
python train.py --dataset datasets/<token> --model yolo11n.pt \
  --epochs 100 --imgsz 1024 --batch 16 --device mps
# → models/pichip_detector.pt   (yolo26n.pt also supported)
```

### 2. Deploy to the Pi

```bash
# from the Mac
scp models/pichip_detector.pt <pi-user>@pichip.local:~/pichip_viewer/models/

# on the Pi
./deploy_model.sh    # exports models/pichip_detector.onnx (faster) + updates .env
```

### 3. Run on the Pi

```bash
./run.sh
```
- **Over SSH (headless):** prints running chip counts; the MJPEG stream stays live.
- **On a monitor attached to the Pi:** shows the live annotated window (press `q` to quit).

### 4. Watch from your Mac

With the viewer running on the Pi, open in a browser:

```
http://pichip.local:8090/
```
(Also works in VLC: *Open Network Stream* → that URL. Off-network, tunnel it:
`ssh -L 8090:localhost:8090 <pi-user>@pichip.local` then open `http://localhost:8090/`.)

### Self-test before you have a trained model

A stock model is handy for checking the camera + stream end-to-end:

```bash
PICHIP_DETECTOR_PATH=yolo11n.onnx ./run.sh   # then open the URL (it just won't count chips)
```

### Streamlit debug app (optional)

```bash
pip install -r web/requirements.txt
cd web && streamlit run app.py
```
Loads the custom detector by default; point `PICHIP_YOLO_PATH` at a `*-worldv2.pt` for
open-vocab YOLO-World mode instead.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PICHIP_SOURCE` | auto | `picamera2` (Pi camera) or `stream` (TCP). Auto-detects the Pi camera. |
| `PICHIP_DETECTOR_PATH` | `models/pichip_detector.pt` | Detector loaded by the viewer (`.pt` or ONNX) |
| `PICHIP_MJPEG_PORT` | `8090` | Serve the annotated feed at `http://<pi>:<port>/` (0 = off) |
| `PICHIP_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` (use `cpu` on the Pi) |
| `PICHIP_IMGSZ` | `0` | Inference resolution (0 = model default). Lower (416/320) = faster, worse on small chips. Must match the ONNX export size — `deploy_model.sh` keeps them in sync. |
| `PICHIP_CONFIDENCE` | `0.3` | Detection confidence threshold |
| `PICHIP_CAMERA_WIDTH` / `PICHIP_CAMERA_HEIGHT` | `1280` / `720` | picamera2 capture size |
| `PICHIP_CAMERA_SWAP_RB` | `0` | Set `1` if camera colors look red/blue-swapped |
| `PICHIP_HEADLESS` | auto | Force no-window mode (auto-on when there's no display) |
| `PICHIP_STREAM_URL` | `tcp://pichip.local:8888` | Remote TCP source, used only when `PICHIP_SOURCE=stream` |

## Performance (Raspberry Pi 4, CPU)

There are two distinct frame rates:

- **Display / stream FPS** — how smooth the video is. ~**15 fps**, because inference runs in
  a background thread (see "How it works"). This is what you watch.
- **Detection refresh** — how often boxes/counts update. CPU-bound by the detector:

  | Runtime | Detection speed | Notes |
  |---------|-----------------|-------|
  | PyTorch `.pt` @640 | ~1.3/s | works, but slow |
  | **ONNX @640** | **~2.2/s** | recommended — what `deploy_model.sh` produces |
  | ONNX @416 | ~4–5/s | `./deploy_model.sh "" 416` — faster, worse on small chips |
  | NCNN | — | segfaults on this aarch64 / Python 3.13 build; not used |

For a mostly-static chip tray, a couple of detection updates per second is plenty. To make
**detection itself** real-time, the Pi 4 CPU is the ceiling — use a hardware accelerator
(Coral USB Accelerator on the Pi 4, or a Pi 5 + Raspberry Pi AI Kit / Hailo).

Other knobs: lower `PICHIP_IMGSZ`, drop `PICHIP_CAMERA_WIDTH`/`HEIGHT` or `PICHIP_MJPEG_QUALITY`
for higher display fps, lower `PICHIP_CONFIDENCE`, or retrain with more data (see TRAINING.md).

## Chip values

| Color | Value |
|-------|-------|
| White | $1 |
| Red | $5 |
| Blue | $10 |
| Yellow | $25 |

Edit `CHIP_VALUES` in `overlay_viewer.py` (or the Streamlit sidebar) to change these.
