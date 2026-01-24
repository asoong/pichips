import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import Counter

STREAM_URL = "tcp://pichip.local:8888"
MODEL_PATH = Path("./chips_yolo11.onnx")

IMGSZ = 640
CONF = 0.25
IOU_THRESH = 0.45

# Chip configuration: class_id -> {name, value, color (BGR)}
CHIP_CONFIG = {
    0: {"name": "white", "value": 1, "color": (255, 255, 255)},
    1: {"name": "red", "value": 5, "color": (0, 0, 255)},
    2: {"name": "blue", "value": 10, "color": (255, 0, 0)},
    3: {"name": "green", "value": 25, "color": (0, 255, 0)},
    4: {"name": "black", "value": 100, "color": (50, 50, 50)},
}

sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]


def preprocess(bgr):
    """Preprocess frame for YOLO11 inference."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (IMGSZ, IMGSZ))
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x, (h, w)


def postprocess(outputs, orig_hw):
    """Process YOLO11 detection outputs.

    YOLO11 output format: [1, 4+num_classes, num_predictions]
    - First 4 values: x_center, y_center, width, height
    - Remaining values: class scores
    """
    preds = outputs[0]  # Shape: [1, 84, 8400]

    # Transpose to [num_predictions, features]
    preds = preds[0].T  # Shape: [8400, 84]

    h, w = orig_hw
    sx, sy = w / IMGSZ, h / IMGSZ

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        # Extract bbox: x_center, y_center, width, height
        x_c, y_c, bw, bh = pred[:4]

        # Extract class scores and find best class
        class_scores = pred[4:]
        cls_id = np.argmax(class_scores)
        conf = class_scores[cls_id]

        if conf < CONF:
            continue

        # Only process chip classes (0-4)
        if cls_id not in CHIP_CONFIG:
            continue

        # Convert center coords to corner coords and scale
        x1 = (x_c - bw / 2) * sx
        y1 = (y_c - bh / 2) * sy
        x2 = (x_c + bw / 2) * sx
        y2 = (y_c + bh / 2) * sy

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return []

    # Apply NMS
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        boxes_np.tolist(), scores_np.tolist(), CONF, IOU_THRESH
    )

    dets = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            dets.append({
                "bbox": boxes[i],
                "conf": float(scores[i]),
                "cls": int(class_ids[i]),
            })

    return dets


def count_chips(dets):
    """Count chips by type and calculate total value."""
    counts = Counter([d["cls"] for d in dets])
    total_value = sum(
        count * CHIP_CONFIG[cls]["value"]
        for cls, count in counts.items()
        if cls in CHIP_CONFIG
    )
    return counts, total_value


def draw_boxes(frame, dets):
    """Draw bounding boxes with chip colors."""
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        conf = d["conf"]
        cls = d["cls"]

        config = CHIP_CONFIG.get(cls, {"name": str(cls), "color": (0, 255, 0)})
        color = config["color"]
        label = f"{config['name']} {conf:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)

        # Draw label text (black for light colors, white for dark)
        text_color = (0, 0, 0) if cls in [0, 3] else (255, 255, 255)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return frame


def draw_hud(frame, counts, total_value):
    """Draw chip counts and total value on screen."""
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (150, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 25
    cv2.putText(frame, "Chip Count:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20

    # Show counts for each chip type
    for cls in sorted(CHIP_CONFIG.keys()):
        config = CHIP_CONFIG[cls]
        count = counts.get(cls, 0)
        text = f"{config['name']}: {count}"
        color = config["color"]
        # Make white text visible
        if cls == 0:
            color = (200, 200, 200)
        cv2.putText(frame, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18

    # Draw total value
    y += 5
    cv2.putText(frame, f"Total: ${total_value}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def main():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {STREAM_URL}")

    print("PiChip Viewer - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Run inference
        x, orig_hw = preprocess(frame)
        outputs = sess.run(out_names, {in_name: x})
        dets = postprocess(outputs, orig_hw)

        # Count chips and calculate value
        counts, total_value = count_chips(dets)

        # Draw visualization
        vis = draw_boxes(frame, dets)
        vis = draw_hud(vis, counts, total_value)

        cv2.imshow("PiChip Viewer", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
