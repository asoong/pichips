import os
import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
import time
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

st.set_page_config(page_title="PiChip Viewer", layout="wide")

# Configuration from environment variables
DEFAULT_STREAM_URL = os.getenv("PICHIP_STREAM_URL", "tcp://pichip.local:8888")
# Defaults to the custom detector trained from a PiChip export (train.py). A YOLO-World
# model (filename contains "world") still works and falls back to open-vocab detection.
YOLO_PATH = os.getenv(
    "PICHIP_YOLO_PATH",
    os.getenv("PICHIP_DETECTOR_PATH", "models/pichip_detector.pt"),
)
DEVICE_PREF = os.getenv("PICHIP_DEVICE", "auto")

CHIP_VALUES = {"white": 1, "red": 5, "blue": 10, "yellow": 25}
CHIP_BGR = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


def _resolve_model_path():
    model_path = Path(YOLO_PATH)
    if not model_path.exists():
        parent_path = Path("..") / YOLO_PATH
        if parent_path.exists():
            return parent_path
    return model_path


def is_world_model(path: Path) -> bool:
    """YOLO-World (open-vocab) models need set_classes; custom detectors must not."""
    return "world" in path.name.lower()


@st.cache_resource
def load_model():
    model_path = _resolve_model_path()
    model = YOLO(str(model_path))
    world = is_world_model(model_path)
    if world:
        # Open-vocab model: tell it what to look for.
        model.set_classes(["poker chip", "casino chip", "chip stack"])
    return model, world


def parse_label(label):
    """Split a custom-detector class name into (color, orientation)."""
    if "_" not in label:
        return None, None
    color, orientation = label.rsplit("_", 1)
    if color not in CHIP_VALUES:
        return None, None
    return color, orientation


def classify_chip_color(frame_rgb, box):
    """Classify chip color from the detected region (used for YOLO-World only)."""
    x1, y1, x2, y2 = map(int, box)
    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return "white"

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    sat_mean = np.mean(s)

    if sat_mean < 40:  # Low saturation = white
        return "white"

    hue_mean = np.mean(h[s > 30]) if np.any(s > 30) else 0

    if hue_mean < 10 or hue_mean > 170:
        return "red"
    elif 20 <= hue_mean <= 35:
        return "yellow"
    elif 100 <= hue_mean <= 130:
        return "blue"
    else:
        return "white"


def count_chips_in_stack(frame_gray, box):
    """Count chips in a side-lying stack using edge detection."""
    x1, y1, x2, y2 = map(int, box)
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 1

    h, w = roi.shape
    if h == 0 or w == 0:
        return 1

    edges = cv2.Canny(roi, 30, 100)

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

    stack_len = max(h, w)
    max_possible = stack_len // 3
    return min(count, max(1, max_possible))


def draw_detections(frame, detections):
    """Draw boxes and labels"""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        color = CHIP_BGR.get(det["color"], (0, 255, 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{det['color']}: {det['count']} ({det['score']:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return vis


def main():
    st.title("PiChip Viewer")
    st.caption("Custom YOLO detector (or YOLO-World) + edge counting")

    with st.sidebar:
        st.header("Settings")
        stream_url = st.text_input("Stream URL", DEFAULT_STREAM_URL)

        st.subheader("Detection")
        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.3)

        st.subheader("Chip Values ($)")
        values = {
            c: st.number_input(c.capitalize(), value=CHIP_VALUES[c]) for c in CHIP_VALUES
        }

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Counts")
        displays = {c: st.empty() for c in CHIP_VALUES}
        st.divider()
        total_disp = st.empty()

    with col1:
        start = st.button("Start", type="primary")
        stop = st.button("Stop")
        video = st.empty()
        status = st.empty()

    if start:
        st.session_state["run"] = True
    if stop:
        st.session_state["run"] = False

    if st.session_state.get("run"):
        status.info("Loading model...")
        model, world = load_model()
        names = model.names
        status.success(
            "Model ready (YOLO-World)" if world else "Model ready (custom detector)"
        )

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            status.error(f"Cannot connect to {stream_url}")
            return
        status.success("Connected!")

        while st.session_state.get("run"):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = model.predict(rgb, conf=confidence, verbose=False)

            detections = []
            counts = defaultdict(int)

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls in zip(boxes, confs, classes):
                    if world:
                        # Open-vocab: model only knows "chip", so guess color via HSV
                        # and treat every detection as a countable stack.
                        color = classify_chip_color(rgb, box)
                        chip_count = count_chips_in_stack(gray, box)
                    else:
                        # Custom detector: color + orientation come from the class name.
                        color, orientation = parse_label(names[int(cls)])
                        if color is None:
                            continue
                        chip_count = (
                            count_chips_in_stack(gray, box)
                            if orientation == "edge"
                            else 1
                        )

                    detections.append(
                        {
                            "box": box,
                            "color": color,
                            "count": chip_count,
                            "score": float(conf),
                        }
                    )
                    counts[color] += chip_count

            vis = draw_detections(frame, detections)
            video.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

            total = sum(counts.get(c, 0) * values[c] for c in CHIP_VALUES)
            for c in CHIP_VALUES:
                displays[c].metric(
                    c.capitalize(),
                    f"{counts.get(c, 0)} chips",
                    f"${counts.get(c, 0) * values[c]}",
                )
            total_disp.metric("Total", f"${total}")

            time.sleep(0.03)

        cap.release()


if __name__ == "__main__":
    main()
