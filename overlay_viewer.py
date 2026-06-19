import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from threading import Thread
from queue import Queue

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Configuration from environment variables
STREAM_URL = os.getenv("PICHIP_STREAM_URL", "tcp://pichip.local:8888")
# The custom detector trained from a PiChip training-set export (see train.py). Its
# class names (white_face, red_edge, ...) are embedded in the weights and read back via
# model.names, so this file never hardcodes the class list.
DETECTOR_PATH = Path(os.getenv("PICHIP_DETECTOR_PATH", "models/pichip_detector.pt"))
DETECT_INTERVAL = int(os.getenv("PICHIP_DETECT_INTERVAL", "5"))
CONFIDENCE = float(os.getenv("PICHIP_CONFIDENCE", "0.3"))
DEVICE_PREF = os.getenv("PICHIP_DEVICE", "auto")

# Chip configuration - 4 types: white, red, blue, yellow
CHIP_VALUES = {
    "white": 1,
    "red": 5,
    "blue": 10,
    "yellow": 25,
}

# Display colors (BGR)
CHIP_COLORS_BGR = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


class VideoCapture:
    """Threaded video capture to avoid blocking."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = Queue(maxsize=2)
        self.stopped = False
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()
                    self.q.put(frame)
                except:
                    pass

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def release(self):
        self.stopped = True
        self.cap.release()


def parse_label(label):
    """Split a detector class name into (color, orientation).

    Labels are "<color>_<orientation>" (e.g. "red_face", "white_edge"). Anything without
    a recognized color (e.g. "unknown") returns (None, None) and is ignored.
    """
    if "_" not in label:
        return None, None
    color, orientation = label.rsplit("_", 1)
    if color not in CHIP_VALUES:
        return None, None
    return color, orientation


def count_chips_in_stack(frame_gray, box):
    """Estimate how many chips are in a side-lying stack via edge detection.

    A single side-lying stack ("_edge") shows up as one detection but contains many
    chips; we count the repeating edge markings along its long axis. Face-on chips are
    counted as 1 by the caller and never reach this function.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 1

    h, w = roi.shape
    if h == 0 or w == 0:
        return 1

    edges = cv2.Canny(roi, 30, 100)

    # Project edges along the stack's long axis.
    if h > w:  # Vertical stack
        profile = np.sum(edges, axis=1).astype(float)
    else:  # Horizontal stack
        profile = np.sum(edges, axis=0).astype(float)

    if profile.max() == 0:
        return 1

    profile = cv2.GaussianBlur(profile.reshape(1, -1), (5, 1), 0).flatten()
    threshold = profile.max() * 0.3
    peaks = np.sum(np.diff((profile > threshold).astype(int)) == 1)
    count = max(1, (peaks + 1) // 2)

    # Sanity bound: chips are at least ~3px thick.
    stack_len = max(h, w)
    max_possible = stack_len // 3
    return min(count, max(1, max_possible))


def detect(model, frame_gray, results):
    """Turn raw YOLO results into chip detections with per-stack counts."""
    names = model.names
    detections = []
    counts = defaultdict(int)

    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, classes):
            label = names.get(int(cls), str(cls)) if isinstance(names, dict) else names[int(cls)]
            color, orientation = parse_label(label)
            if color is None:
                continue

            count = count_chips_in_stack(frame_gray, box) if orientation == "edge" else 1

            detections.append(
                {
                    "box": box,
                    "color": color,
                    "orientation": orientation,
                    "count": count,
                    "score": float(conf),
                }
            )
            counts[color] += count

    return detections, counts


def draw_visualization(frame, detections, chip_counts):
    """Draw detection boxes, per-chip labels, and the value HUD."""
    vis = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        color = det["color"]
        bgr = CHIP_COLORS_BGR.get(color, (0, 255, 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2)

        label = f"{color} {det['orientation']}: {det['count']} ({det['score']:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(
            vis, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), (0, 0, 0), -1
        )
        cv2.putText(
            vis,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Draw HUD
    total_value = sum(
        count * CHIP_VALUES.get(color, 0) for color, count in chip_counts.items()
    )

    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (170, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    y_pos = 25
    cv2.putText(
        vis, "Chip Counts:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    y_pos += 24

    for color in ["white", "red", "blue", "yellow"]:
        count = chip_counts.get(color, 0)
        value = count * CHIP_VALUES.get(color, 0)
        text = f"{color}: {count} (${value})"
        bgr = CHIP_COLORS_BGR.get(color, (255, 255, 255))
        cv2.putText(vis, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)
        y_pos += 22

    y_pos += 5
    cv2.putText(
        vis,
        f"Total: ${total_value}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    return vis


def get_device():
    """Select compute device based on preference and availability."""
    if DEVICE_PREF != "auto":
        return DEVICE_PREF
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    if not DETECTOR_PATH.exists():
        raise SystemExit(
            f"Detector model not found at {DETECTOR_PATH}.\n"
            "Train one first: export a training set from the PiChip web client, then run\n"
            "  python train.py --dataset datasets/<token>\n"
            "or set PICHIP_DETECTOR_PATH to an existing .pt file."
        )

    device = get_device()
    print(f"Loading PiChip detector ({DETECTOR_PATH}) on {device}...")
    model = YOLO(str(DETECTOR_PATH))
    print(f"Classes: {', '.join(model.names.values())}")

    print("Starting video capture...")
    cap = VideoCapture(STREAM_URL)

    print("PiChip Viewer (custom YOLO detector) - Press 'q' to quit")

    frame_count = 0
    cached_detections = []
    cached_counts = defaultdict(int)

    while True:
        frame = cap.read()
        if frame is None:
            continue

        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run detection periodically to keep the display smooth.
        if frame_count % DETECT_INTERVAL == 1 or not cached_detections:
            results = model.predict(
                frame, conf=CONFIDENCE, device=device, verbose=False
            )
            cached_detections, cached_counts = detect(model, frame_gray, results)

        vis = draw_visualization(frame, cached_detections, cached_counts)
        cv2.imshow("PiChip Viewer", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
