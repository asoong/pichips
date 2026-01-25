import os
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from threading import Thread
from queue import Queue

from dotenv import load_dotenv
from mobile_sam import sam_model_registry, SamPredictor

load_dotenv()

# Configuration from environment variables
STREAM_URL = os.getenv("PICHIP_STREAM_URL", "tcp://pichip.local:8888")
MOBILESAM_PATH = Path(os.getenv("PICHIP_MOBILESAM_PATH", "models/mobile_sam.pt"))
SAM_INTERVAL = int(os.getenv("PICHIP_SAM_INTERVAL", "10"))
MIN_REGION_AREA = int(os.getenv("PICHIP_MIN_REGION_AREA", "1000"))
DEVICE_PREF = os.getenv("PICHIP_DEVICE", "auto")

# Chip configuration - 4 types: white, red, blue, yellow
CHIP_VALUES = {
    "white": 1,
    "red": 5,
    "blue": 10,
    "yellow": 25,
}

# HSV ranges for detecting chip colors
CHIP_HSV_RANGES = {
    "white": ((0, 0, 180), (180, 40, 255)),
    "red": ((0, 120, 100), (10, 255, 255)),
    "red2": ((165, 120, 100), (180, 255, 255)),
    "blue": ((95, 100, 100), (125, 255, 255)),
    "yellow": ((15, 100, 100), (35, 255, 255)),
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


def find_chip_regions(frame):
    """Find chip regions using color detection. Returns center points and colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    regions = []

    for color_name, (lower, upper) in CHIP_HSV_RANGES.items():
        if color_name == "red2":
            continue

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Combine red ranges
        if color_name == "red":
            lower2, upper2 = CHIP_HSV_RANGES["red2"]
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_REGION_AREA:
                continue

            # Get center point for SAM prompt
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(contour)

            regions.append({
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "color": color_name,
                "area": area,
            })

    return regions


def segment_with_sam(predictor, frame_rgb, regions):
    """Use SAM to get precise masks for each detected region."""
    predictor.set_image(frame_rgb)

    segmented = []

    for region in regions:
        cx, cy = region["center"]

        # Use point prompt
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Take the mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # Get bounding box of mask
        coords = np.column_stack(np.where(mask))
        if len(coords) == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        segmented.append({
            "mask": mask,
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "color": region["color"],
            "score": scores[best_idx],
        })

    return segmented


def count_chips_in_segment(mask, frame_gray):
    """Count individual chips in a segmented region using edge detection."""
    # Get bounding box
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return 1

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min

    if w == 0 or h == 0:
        return 1

    # Extract ROI
    roi = frame_gray[y_min:y_max, x_min:x_max].copy()
    roi_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8)

    # Apply mask
    roi = cv2.bitwise_and(roi, roi, mask=roi_mask)

    # Edge detection
    edges = cv2.Canny(roi, 30, 100)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    # Determine orientation - horizontal or vertical stack
    is_horizontal = w > h

    if is_horizontal:
        # Project edges vertically
        profile = np.sum(edges, axis=0).astype(np.float32)
    else:
        # Project edges horizontally
        profile = np.sum(edges, axis=1).astype(np.float32)

    if len(profile) == 0 or np.max(profile) == 0:
        return 1

    # Smooth
    k = max(3, len(profile) // 30)
    if k % 2 == 0:
        k += 1
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).flatten()

    # Count peaks
    threshold = np.max(profile) * 0.3
    above = profile > threshold
    crossings = np.diff(above.astype(int))
    num_peaks = np.sum(crossings == 1)

    # Each chip typically has 1-2 edge lines
    chip_count = max(1, (num_peaks + 1) // 2)

    # Sanity bounds
    length = w if is_horizontal else h
    min_chips = max(1, length // 25)  # Max 25px per chip
    max_chips = length // 4  # Min 4px per chip

    return max(min_chips, min(chip_count, max_chips))


def draw_visualization(frame, segments, chip_counts):
    """Draw segmentation masks, labels, and HUD."""
    vis = frame.copy()

    # Draw each segment
    for seg in segments:
        mask = seg["mask"]
        color = seg["color"]
        count = seg.get("count", 0)
        x, y, w, h = seg["bbox"]
        bgr = CHIP_COLORS_BGR.get(color, (0, 255, 0))

        # Draw semi-transparent mask
        overlay = vis.copy()
        overlay[mask] = bgr
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, bgr, 2)

        # Draw label
        label = f"{color}: {count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x, label_y = x, max(y - 5, th + 5)
        cv2.rectangle(vis, (label_x, label_y - th - 4), (label_x + tw + 4, label_y + 2), (0, 0, 0), -1)
        cv2.putText(vis, label, (label_x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw HUD
    total_value = sum(
        count * CHIP_VALUES.get(color, 0)
        for color, count in chip_counts.items()
    )

    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (170, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    y_pos = 25
    cv2.putText(vis, "Chip Counts:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 24

    for color in ["white", "red", "blue", "yellow"]:
        count = chip_counts.get(color, 0)
        value = count * CHIP_VALUES.get(color, 0)
        text = f"{color}: {count} (${value})"
        bgr = CHIP_COLORS_BGR.get(color, (255, 255, 255))
        cv2.putText(vis, text, (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)
        y_pos += 22

    y_pos += 5
    cv2.putText(vis, f"Total: ${total_value}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return vis


def get_device():
    """Select compute device based on preference and availability."""
    if DEVICE_PREF != "auto":
        return DEVICE_PREF
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # Load MobileSAM
    device = get_device()
    print(f"Loading MobileSAM on {device}...")

    sam = sam_model_registry["vit_t"](checkpoint=str(MOBILESAM_PATH))
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    print("Starting video capture...")
    cap = VideoCapture(STREAM_URL)

    print("PiChip Viewer (MobileSAM) - Press 'q' to quit")

    frame_count = 0
    cached_segments = []
    cached_counts = defaultdict(int)

    while True:
        frame = cap.read()
        if frame is None:
            continue

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run detection + SAM periodically
        if frame_count % SAM_INTERVAL == 1 or not cached_segments:
            # Step 1: Find chip regions by color
            regions = find_chip_regions(frame)

            # Step 2: Segment with SAM
            if regions:
                with torch.no_grad():
                    segments = segment_with_sam(predictor, frame_rgb, regions)
            else:
                segments = []

            # Step 3: Count chips in each segment
            chip_counts = defaultdict(int)
            for seg in segments:
                count = count_chips_in_segment(seg["mask"], frame_gray)
                seg["count"] = count
                chip_counts[seg["color"]] += count

            cached_segments = segments
            cached_counts = chip_counts

        # Draw
        vis = draw_visualization(frame, cached_segments, cached_counts)
        cv2.imshow("PiChip Viewer", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
