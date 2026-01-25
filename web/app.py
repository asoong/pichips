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
YOLO_PATH = os.getenv("PICHIP_YOLO_PATH", "models/yolov8s-worldv2.pt")
DEVICE_PREF = os.getenv("PICHIP_DEVICE", "auto")

CHIP_VALUES = {"white": 1, "red": 5, "blue": 10, "yellow": 25}
CHIP_BGR = {"white": (255,255,255), "red": (0,0,255), "blue": (255,0,0), "yellow": (0,255,255)}


@st.cache_resource
def load_model():
    # Check parent directory for model if not found locally
    model_path = Path(YOLO_PATH)
    if not model_path.exists():
        parent_path = Path("..") / YOLO_PATH
        if parent_path.exists():
            model_path = parent_path
    model = YOLO(str(model_path))
    return model


def classify_chip_color(frame_rgb, box):
    """Classify chip color from the detected region"""
    x1, y1, x2, y2 = map(int, box)
    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return "white"

    # Get dominant color via HSV analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Mask out low saturation (white/gray)
    sat_mean = np.mean(s)
    val_mean = np.mean(v)

    if sat_mean < 40:  # Low saturation = white
        return "white"

    hue_mean = np.mean(h[s > 30]) if np.any(s > 30) else 0

    # Hue ranges: red=0-10 or 170-180, yellow=20-35, blue=100-130
    if hue_mean < 10 or hue_mean > 170:
        return "red"
    elif 20 <= hue_mean <= 35:
        return "yellow"
    elif 100 <= hue_mean <= 130:
        return "blue"
    else:
        return "white"


def count_chips_in_stack(frame_gray, box):
    """Count chips in a stack using edge detection"""
    x1, y1, x2, y2 = map(int, box)
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 1

    h, w = roi.shape
    if h == 0 or w == 0:
        return 1

    edges = cv2.Canny(roi, 30, 100)

    # Profile along the longer axis (stack direction)
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

    # Sanity check based on size
    stack_len = max(h, w)
    max_possible = stack_len // 3  # Chips are at least 3px thick
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
        cv2.rectangle(vis, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return vis


def main():
    st.title("PiChip Viewer")
    st.caption("YOLO-World (semantic detection) + Edge counting")

    with st.sidebar:
        st.header("Settings")
        stream_url = st.text_input("Stream URL", DEFAULT_STREAM_URL)

        st.subheader("Detection")
        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.3)

        st.subheader("Chip Values ($)")
        values = {c: st.number_input(c.capitalize(), value=CHIP_VALUES[c]) for c in CHIP_VALUES}

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
        status.info("Loading YOLO-World...")
        model = load_model()

        # Set classes to detect poker chips
        model.set_classes(["poker chip", "casino chip", "chip stack"])
        status.success("Model ready")

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

            # Run YOLO-World detection
            results = model.predict(rgb, conf=confidence, verbose=False)

            detections = []
            counts = defaultdict(int)

            for r in results:
                for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    color = classify_chip_color(rgb, box)
                    chip_count = count_chips_in_stack(gray, box)

                    detections.append({
                        "box": box,
                        "color": color,
                        "count": chip_count,
                        "score": float(conf)
                    })
                    counts[color] += chip_count

            vis = draw_detections(frame, detections)
            video.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

            total = sum(counts.get(c, 0) * values[c] for c in CHIP_VALUES)
            for c in CHIP_VALUES:
                displays[c].metric(c.capitalize(), f"{counts.get(c, 0)} chips", f"${counts.get(c, 0) * values[c]}")
            total_disp.metric("Total", f"${total}")

            time.sleep(0.03)

        cap.release()


if __name__ == "__main__":
    main()
