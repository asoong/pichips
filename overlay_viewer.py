import os
import math
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from threading import Thread, Lock, Condition
from queue import Queue

from dotenv import load_dotenv
from ultralytics import YOLO

# Pillow renders the HUD text in a real TTF typeface (OpenCV can only draw its blocky
# Hershey vector fonts). It's always present on the Pi (a transitive dep of ultralytics);
# if it's somehow missing the overlay falls back to the legacy Hershey renderer.
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_OK = True
except Exception:  # pragma: no cover
    PIL_OK = False

load_dotenv()

# Configuration from environment variables
STREAM_URL = os.getenv("PICHIP_STREAM_URL", "tcp://pichip.local:8888")
# The custom detector trained from a PiChip training-set export (see train.py). Its
# class names (white_face, red_edge, ...) are embedded in the weights and read back via
# model.names, so this file never hardcodes the class list.
DETECTOR_PATH = Path(os.getenv("PICHIP_DETECTOR_PATH", "models/pichip_detector.pt"))
DETECT_INTERVAL = int(
    os.getenv("PICHIP_DETECT_INTERVAL", "5")
)  # legacy; unused (async now)
CONFIDENCE = float(os.getenv("PICHIP_CONFIDENCE", "0.3"))
DEVICE_PREF = os.getenv("PICHIP_DEVICE", "auto")
# Inference resolution. 0 = the model's native size. Lower (e.g. 416/320) is much faster
# on a Pi CPU but worse on small objects. With a static-shape ONNX model this must match
# the size it was exported at (deploy_model.sh keeps them in sync).
IMGSZ = int(os.getenv("PICHIP_IMGSZ", "0"))

# Frame source: "stream" reads PICHIP_STREAM_URL over TCP (viewer runs on a separate
# machine pulling the Pi's stream); "picamera2" captures the Pi's CSI camera directly
# (viewer runs ON the Pi). Default to the Pi camera when picamera2 is importable.
SOURCE = os.getenv("PICHIP_SOURCE", "").lower()
CAMERA_WIDTH = int(os.getenv("PICHIP_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("PICHIP_CAMERA_HEIGHT", "720"))
# picamera2 "RGB888" is already BGR-ordered for OpenCV; flip this if colors look swapped.
CAMERA_SWAP_RB = os.getenv("PICHIP_CAMERA_SWAP_RB", "0").lower() in ("1", "true", "yes")
# Headless = no GUI window. Auto-on when there's no display (e.g. over SSH).
HEADLESS = os.getenv("PICHIP_HEADLESS", "").lower() in ("1", "true", "yes") or not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
# Stop after N processed frames (0 = run forever) — handy for a headless smoke test.
MAX_FRAMES = int(os.getenv("PICHIP_MAX_FRAMES", "0"))
# When set, write the latest annotated frame here (headless verification).
SNAPSHOT_PATH = os.getenv("PICHIP_SNAPSHOT_PATH", "")
# Stream annotated frames as MJPEG over HTTP on this port (0 = off). View from another
# machine at http://<pi-host>:<port>/ — ideal when the Pi runs headless over SSH.
MJPEG_PORT = int(os.getenv("PICHIP_MJPEG_PORT", "0"))
MJPEG_QUALITY = int(os.getenv("PICHIP_MJPEG_QUALITY", "80"))


def _resolve_source():
    """Pick the frame source: explicit PICHIP_SOURCE wins, else auto-detect the Pi cam."""
    if SOURCE in ("picamera2", "camera", "csi"):
        return "picamera2"
    if SOURCE in ("stream", "tcp"):
        return "stream"
    # Auto: use the Pi camera if picamera2 is available, else the TCP stream.
    try:
        import picamera2  # noqa: F401

        return "picamera2"
    except Exception:
        return "stream"


# ---- Optional MJPEG-over-HTTP output (watch the live feed from another machine) ----


class _FrameBuffer:
    """Thread-safe holder for the latest annotated JPEG frame."""

    def __init__(self):
        self._cond = Condition(Lock())
        self._jpeg = None

    def update(self, jpeg_bytes):
        with self._cond:
            self._jpeg = jpeg_bytes
            self._cond.notify_all()

    def wait_for_frame(self, timeout=5.0):
        with self._cond:
            self._cond.wait(timeout)
            return self._jpeg


_frame_buffer = _FrameBuffer()


def start_mjpeg_server(port):
    """Serve _frame_buffer as multipart MJPEG at http://0.0.0.0:<port>/ in a daemon thread."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/stream", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            try:
                while True:
                    jpeg = _frame_buffer.wait_for_frame()
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(jpeg)).encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (browser/VLC) disconnected

        def log_message(self, *args):
            pass  # silence per-request logging

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


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

# ---- Overlay typography ----
# Bundled variable TTFs (offline-safe on the Pi); the per-role weight is set via the
# font's `wght` axis. Override either path with an absolute path to swap typefaces.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_DISPLAY = os.getenv("PICHIP_FONT_DISPLAY", "") or str(FONT_DIR / "Sora.ttf")
FONT_MONO = os.getenv("PICHIP_FONT_MONO", "") or str(FONT_DIR / "JetBrainsMono.ttf")
# Text backend: auto = PIL/TTF when available, else cv2 Hershey; cv2 forces the legacy
# look; pil forces PIL (degrades to PIL's default font if the TTF can't load).
TEXT_BACKEND = os.getenv("PICHIP_TEXT_BACKEND", "auto").lower()
# HUD palette: league = gold-on-charcoal redesign; classic = the original colors.
HUD_THEME = os.getenv("PICHIP_HUD_THEME", "league").lower()

# role -> (font path, pixel size, weight-axis value)
_FONT_SPECS = {
    "title": (FONT_DISPLAY, 22, 600),
    "label": (FONT_DISPLAY, 16, 500),
    "caption": (FONT_DISPLAY, 21, 600),
    "num": (FONT_MONO, 16, 500),
    "total": (FONT_MONO, 27, 600),
    "tag": (FONT_MONO, 15, 600),
}

# ---- Overlay theme (BGR) ----
if HUD_THEME == "classic":
    PANEL_BG = (28, 28, 30)
    PANEL_ALPHA = 0.55
    PANEL_BORDER = (90, 90, 95)
    ACCENT = (90, 200, 255)  # warm amber — title
    ACCENT_TOTAL = (90, 200, 255)  # total value
    TEXT_PRIMARY = (240, 240, 240)
    TEXT_MUTED = (165, 165, 170)
    GUIDE_COLOR = (150, 230, 120)  # soft green viewfinder
else:  # "league" — gold on charcoal
    PANEL_BG = (23, 21, 20)  # deep charcoal
    PANEL_ALPHA = 0.62
    PANEL_BORDER = (40, 78, 90)  # faint gold hairline
    ACCENT = (60, 200, 235)  # brand gold (#EBC83C)
    ACCENT_TOTAL = (78, 210, 244)  # brighter gold (#F4D24E)
    TEXT_PRIMARY = (240, 238, 236)
    TEXT_MUTED = (126, 120, 120)
    GUIDE_COLOR = (74, 184, 216)  # soft gold viewfinder (#D8B84A)

# ---- Placement guide config ----
GUIDE_ON = os.getenv("PICHIP_GUIDE", "1").lower() not in ("0", "false", "no", "off")
GUIDE_SHAPE = os.getenv("PICHIP_GUIDE_SHAPE", "wide").lower()  # wide | square
GUIDE_SCALE = float(os.getenv("PICHIP_GUIDE_SCALE", "0.7"))  # fraction of frame width
GUIDE_DIM = float(
    os.getenv("PICHIP_GUIDE_DIM", "0.25")
)  # 0..1 dim outside guide (0 = off)
COUNT_IN_GUIDE = os.getenv("PICHIP_COUNT_IN_GUIDE", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


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


class Picamera2Capture:
    """Capture directly from the Pi CSI camera (imx708 etc.) via picamera2.

    Returns BGR frames to match the rest of the OpenCV pipeline. Used when the viewer runs
    ON the Pi (PICHIP_SOURCE=picamera2) instead of pulling a remote TCP stream — OpenCV's
    VideoCapture can't read the libcamera CSI camera directly.
    """

    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        # Camera Module 3 has autofocus — enable continuous AF so a chip tray placed at
        # arm's length stays sharp. Harmless no-op on fixed-focus cameras. (AfMode 2 =
        # Continuous; set numerically to avoid a hard libcamera.controls import.)
        try:
            self.picam2.set_controls({"AfMode": 2})
        except Exception:
            pass

    def read(self):
        # picamera2 "RGB888" yields a buffer already in BGR byte order for OpenCV, so we
        # return it as-is unless the user asked to swap (PICHIP_CAMERA_SWAP_RB).
        frame = self.picam2.capture_array()
        if CAMERA_SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def release(self):
        try:
            self.picam2.stop()
        except Exception:
            pass


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


def guide_rect(shape):
    """Centered placement-guide rectangle (x1, y1, x2, y2) for a frame of this shape.

    Wide (tray-shaped) by default; square when PICHIP_GUIDE_SHAPE=square. Width is
    GUIDE_SCALE of the frame width; height follows the shape, both clamped to the frame.
    This is the single source of truth used by both counting and drawing.
    """
    h, w = shape[:2]
    cx, cy = w // 2, h // 2
    gw = int(max(40, min(w * GUIDE_SCALE, w - 20)))
    # Keep the guide clear of the top-left HUD panel so its corner isn't occluded.
    gw = min(gw, 2 * max(60, cx - 240))
    ratio = 1.0 if GUIDE_SHAPE == "square" else 0.45
    gh = int(gw * ratio)
    if gh > h - 20:
        gh = h - 20
        if GUIDE_SHAPE == "square":
            gw = gh
    x1, y1 = cx - gw // 2, cy - gh // 2
    return (x1, y1, x1 + gw, y1 + gh)


def _center_in_rect(box, rect):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]


def detect(model, frame_gray, results, guide=None):
    """Turn raw YOLO results into chip detections with per-stack counts.

    When a guide rect is given and PICHIP_COUNT_IN_GUIDE is on, detections whose center
    falls outside the guide are dropped (so both counts and drawn markers stay scoped to
    the placement box).
    """
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
            label = (
                names.get(int(cls), str(cls))
                if isinstance(names, dict)
                else names[int(cls)]
            )
            color, orientation = parse_label(label)
            if color is None:
                continue
            if guide is not None and COUNT_IN_GUIDE and not _center_in_rect(box, guide):
                continue

            count = (
                count_chips_in_stack(frame_gray, box) if orientation == "edge" else 1
            )

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


def _text_cv2(img, text, org, scale, color, thickness=1, shadow=True):
    """Legacy Hershey text with a dark shadow — the fallback when PIL is unavailable."""
    if shadow:
        cv2.putText(
            img,
            text,
            (org[0] + 1, org[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


_font_cache = {}


def _font(role):
    """Cached PIL ImageFont for a role; weight set via the variable-font `wght` axis.

    Degrades to PIL's built-in font if the TTF can't be opened, so a misdeploy renders in
    a plain face instead of crashing.
    """
    if role in _font_cache:
        return _font_cache[role]
    path, size, weight = _FONT_SPECS[role]
    try:
        f = ImageFont.truetype(path, size)
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass  # static (non-variable) font — use its default instance
    except Exception:
        try:
            f = ImageFont.load_default(size)
        except TypeError:
            f = ImageFont.load_default()
    _font_cache[role] = f
    return f


def _bgr2rgb(color):
    """OpenCV palette colors are BGR; PIL wants RGB."""
    return (int(color[2]), int(color[1]), int(color[0]))


class _TextLayer:
    """Collects overlay text during compose and rasterizes it in one PIL pass per frame.

    Shapes are drawn with cv2 first; each text string is queued via add() and painted by
    flush() after a single BGR<->RGB conversion of the frame. When PIL is unavailable (or
    PICHIP_TEXT_BACKEND=cv2), add() renders immediately with the legacy Hershey font and
    flush() is a no-op — the overlay always draws, worst case in the old font.
    """

    _use_pil = PIL_OK and TEXT_BACKEND != "cv2"

    # Legacy cv2 (scale, thickness) per role, for the fallback path.
    _CV2 = {
        "title": (0.6, 1),
        "label": (0.5, 1),
        "caption": (0.7, 2),
        "num": (0.5, 1),
        "total": (0.8, 2),
        "tag": (0.5, 1),
    }
    # OpenCV's Hershey font is ASCII-only; map the few non-ASCII glyphs we use so the
    # fallback doesn't render them as "??". (The PIL path renders them natively.)
    _ASCII = str.maketrans({"×": "x"})

    def __init__(self):
        self._items = []

    def add(self, img, xy, text, role, color, anchor="lm", stroke=1):
        """Queue (PIL) or immediately draw (cv2) `text`. anchor is a PIL anchor string
        (horizontal l/m/r + vertical m=middle); we only use vertically-centered text."""
        if self._use_pil:
            self._items.append((xy, str(text), role, _bgr2rgb(color), anchor, stroke))
        else:
            self._add_cv2(img, xy, str(text), role, color, anchor)

    def measure(self, text, role):
        """Advance width of `text` in px — replaces cv2.getTextSize for right-alignment."""
        if self._use_pil:
            return _font(role).getlength(str(text))
        scale, thick = self._CV2[role]
        safe = str(text).translate(self._ASCII)
        (tw, _), _ = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        return tw

    def flush(self, img):
        if not self._use_pil or not self._items:
            return img
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))  # BGR->RGB
        draw = ImageDraw.Draw(pil)
        for xy, text, role, rgb, anchor, stroke in self._items:
            draw.text(
                (int(xy[0]), int(xy[1])),
                text,
                font=_font(role),
                fill=rgb,
                anchor=anchor,
                stroke_width=stroke,
                stroke_fill=(0, 0, 0),
            )
        img[:, :, :] = np.asarray(pil)[:, :, ::-1]  # RGB->BGR, in place
        return img

    def _add_cv2(self, img, xy, text, role, color, anchor):
        scale, thick = self._CV2[role]
        text = text.translate(self._ASCII)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = xy
        h = anchor[0] if anchor else "l"
        v = anchor[1] if len(anchor) > 1 else "m"
        if h == "m":
            x -= tw / 2
        elif h == "r":
            x -= tw
        if v == "m":
            y += th / 2  # cv2 origin is the baseline; center the cap height on y
        elif v in ("a", "t"):
            y += th
        _text_cv2(img, text, (int(x), int(y)), scale, color, thick)


def _draw_chip_swatch(vis, center, r, color, active):
    """A small poker-chip glyph: filled disc + rim + edge spots; ring-only when inactive."""
    cx, cy = int(center[0]), int(center[1])
    if not active:
        dim = tuple(int(c * 0.40 + 18) for c in color)
        cv2.circle(vis, (cx, cy), r, dim, 1, cv2.LINE_AA)
        return
    cv2.circle(vis, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r, (35, 35, 40), 1, cv2.LINE_AA)
    for ang in range(0, 360, 45):
        a = math.radians(ang)
        x1 = int(cx + (r - 2) * math.cos(a))
        y1 = int(cy + (r - 2) * math.sin(a))
        x2 = int(cx + (r + 1) * math.cos(a))
        y2 = int(cy + (r + 1) * math.sin(a))
        cv2.line(vis, (x1, y1), (x2, y2), (245, 245, 245), 1, cv2.LINE_AA)


def _rounded_rect(img, p1, p2, color, radius, thickness=-1):
    """Filled (thickness=-1) or outlined rounded rectangle."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in (
            (x1 + r, y1 + r),
            (x2 - r, y1 + r),
            (x1 + r, y2 - r),
            (x2 - r, y2 - r),
        ):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        for cx, cy, ang in (
            (x1 + r, y1 + r, 180),
            (x2 - r, y1 + r, 270),
            (x1 + r, y2 - r, 90),
            (x2 - r, y2 - r, 0),
        ):
            cv2.ellipse(
                img, (cx, cy), (r, r), ang, 0, 90, color, thickness, cv2.LINE_AA
            )


def _alpha_panel(vis, p1, p2, color, alpha, radius):
    """Blend a translucent rounded panel onto vis over the given ROI (corners untouched)."""
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(vis.shape[1], p2[0]), min(vis.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = vis[y1:y2, x1:x2]
    panel = roi.copy()
    _rounded_rect(panel, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, radius, -1)
    cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0, roi)


def draw_placement_guide(vis, layer, show_caption=False):
    """Dim the area outside the guide and draw a viewfinder reticle (+ optional caption)."""
    if not GUIDE_ON:
        return
    h, w = vis.shape[:2]
    x1, y1, x2, y2 = guide_rect(vis.shape)

    # Dim everything outside the guide to focus attention on the tray area.
    if GUIDE_DIM > 0:
        dark = (vis * (1.0 - GUIDE_DIM)).astype(np.uint8)
        dark[y1:y2, x1:x2] = vis[y1:y2, x1:x2]
        vis[:] = dark

    # Faint full outline + bright L-shaped corner brackets.
    cv2.rectangle(vis, (x1, y1), (x2, y2), GUIDE_COLOR, 1, cv2.LINE_AA)
    arm = max(18, int(min(x2 - x1, y2 - y1) * 0.08))
    for cx, sx in ((x1, 1), (x2, -1)):
        for cy, sy in ((y1, 1), (y2, -1)):
            cv2.line(vis, (cx, cy), (cx + sx * arm, cy), GUIDE_COLOR, 3, cv2.LINE_AA)
            cv2.line(vis, (cx, cy), (cx, cy + sy * arm), GUIDE_COLOR, 3, cv2.LINE_AA)

    if show_caption:
        cap = "Place tray(s) here"
        tx = (x1 + x2) // 2
        ty = y1 - 18 if y1 - 18 > 18 else y2 + 18
        layer.add(vis, (tx, ty), cap, "caption", GUIDE_COLOR, anchor="mm", stroke=2)


def _draw_detection(vis, det, layer):
    """Corner-bracket marker in the chip color + a compact rounded count tag."""
    x1, y1, x2, y2 = (int(v) for v in det["box"])
    bgr = CHIP_COLORS_BGR.get(det["color"], (0, 255, 0))

    arm = max(8, int(min(x2 - x1, y2 - y1) * 0.28))
    for cx, sx in ((x1, 1), (x2, -1)):
        for cy, sy in ((y1, 1), (y2, -1)):
            cv2.line(vis, (cx, cy), (cx + sx * arm, cy), bgr, 2, cv2.LINE_AA)
            cv2.line(vis, (cx, cy), (cx, cy + sy * arm), bgr, 2, cv2.LINE_AA)

    tag = str(det["count"])
    pad = 5
    bw = int(layer.measure(tag, "tag")) + 2 * pad
    bh = 16 + 2 * pad
    ty1 = max(0, y1 - bh)
    _rounded_rect(vis, (x1, ty1), (x1 + bw, ty1 + bh), bgr, 5, -1)
    # Dark text on light swatches (white/yellow), white otherwise, for contrast.
    txt = (20, 20, 20) if det["color"] in ("white", "yellow") else (255, 255, 255)
    layer.add(vis, (x1 + bw / 2, ty1 + bh / 2), tag, "tag", txt, anchor="mm", stroke=0)


def _draw_hud(vis, chip_counts, layer):
    """Charcoal panel: gold title, per-color tabular rows (chip + ×count + $value), total."""
    colors = ["white", "red", "blue", "yellow"]
    total_value = sum(c * CHIP_VALUES.get(col, 0) for col, c in chip_counts.items())

    x0, y0, pad, panel_w = 12, 12, 14, 206
    row_h = 28

    # Precompute vertical anchors (text is vertically centered on each *_cy).
    title_cy = y0 + pad + 15
    rule1_y = y0 + pad + 30 + 6
    rows_top = rule1_y + 10
    rule2_y = rows_top + len(colors) * row_h + 6
    total_cy = rule2_y + 10 + 19
    panel_h = (total_cy + 19 + pad) - y0

    _alpha_panel(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, PANEL_ALPHA, 14)
    _rounded_rect(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BORDER, 14, 1)

    left = x0 + pad
    right = x0 + panel_w - pad
    x_value = right  # right edge of the $value column
    x_count = right - 66  # right edge of the ×count column

    layer.add(vis, (left, title_cy), "PICHIP", "title", ACCENT, anchor="lm")
    cv2.line(vis, (left, rule1_y), (right, rule1_y), PANEL_BORDER, 1, cv2.LINE_AA)

    for i, col in enumerate(colors):
        cnt = chip_counts.get(col, 0)
        val = cnt * CHIP_VALUES.get(col, 0)
        active = cnt > 0
        cy = rows_top + row_h // 2 + i * row_h
        _draw_chip_swatch(vis, (left + 8, cy), 8, CHIP_COLORS_BGR[col], active)
        tcol = TEXT_PRIMARY if active else TEXT_MUTED
        layer.add(vis, (left + 26, cy), col, "label", tcol, anchor="lm")
        layer.add(vis, (x_count, cy), f"×{cnt}", "num", tcol, anchor="rm")
        layer.add(vis, (x_value, cy), f"${val}", "num", tcol, anchor="rm")

    cv2.line(vis, (left, rule2_y), (right, rule2_y), PANEL_BORDER, 1, cv2.LINE_AA)
    layer.add(vis, (left, total_cy), "TOTAL", "label", TEXT_MUTED, anchor="lm")
    layer.add(
        vis,
        (x_value, total_cy),
        f"${total_value}",
        "total",
        ACCENT_TOTAL,
        anchor="rm",
        stroke=2,
    )


def draw_visualization(frame, detections, chip_counts):
    """Compose the overlay: placement guide, per-chip markers, and the value HUD.

    cv2 shapes are drawn first; all text is collected in `layer` and rasterized in a
    single PIL pass by layer.flush() so the frame is converted BGR<->RGB only once.
    """
    vis = frame.copy()
    total_chips = sum(chip_counts.values())
    layer = _TextLayer()

    draw_placement_guide(vis, layer, show_caption=(total_chips == 0))
    for det in detections:
        _draw_detection(vis, det, layer)
    _draw_hud(vis, chip_counts, layer)
    layer.flush(vis)
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


class InferenceWorker(Thread):
    """Runs detection on the latest frame in a background thread.

    Decouples inference (~0.3-0.5s/frame on a Pi CPU) from capture/display/streaming, so
    the feed runs at full camera frame rate and detections refresh asynchronously a couple
    of times per second instead of freezing the whole loop on every inference.
    """

    def __init__(self, model, device):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self._lock = Lock()
        self._frame = None
        self._detections = []
        self._counts = defaultdict(int)
        self._stop = False

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def latest(self):
        with self._lock:
            return self._detections, self._counts

    def stop(self):
        self._stop = True

    def run(self):
        predict_kw = {"conf": CONFIDENCE, "device": self.device, "verbose": False}
        if IMGSZ:
            predict_kw["imgsz"] = IMGSZ
        while not self._stop:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.003)  # nothing new yet
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.model.predict(frame, **predict_kw)
            dets, counts = detect(self.model, gray, results, guide_rect(frame.shape))
            with self._lock:
                self._detections, self._counts = dets, counts


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

    source = _resolve_source()
    if source == "picamera2":
        print("Starting Pi camera (picamera2)...")
        cap = Picamera2Capture()
    else:
        print(f"Connecting to stream {STREAM_URL}...")
        cap = VideoCapture(STREAM_URL)

    print(
        "PiChip Viewer (custom YOLO detector) — "
        + ("headless" if HEADLESS else "press 'q' to quit")
    )

    if MJPEG_PORT:
        start_mjpeg_server(MJPEG_PORT)
        print(
            f"MJPEG stream live on port {MJPEG_PORT} — open "
            f"http://<pi-host>:{MJPEG_PORT}/ in a browser on the same network."
        )

    worker = InferenceWorker(model, device)
    worker.start()

    frame_count = 0
    processed = 0
    last_print = time.time()
    vis = None

    try:
        while True:
            frame = cap.read()
            if frame is None:
                continue

            frame_count += 1
            # Inference runs in the background on the latest frame; this loop never blocks
            # on it, so capture/display/stream stay at full camera frame rate.
            worker.submit(frame)
            cached_detections, cached_counts = worker.latest()

            vis = draw_visualization(frame, cached_detections, cached_counts)
            processed += 1

            if MJPEG_PORT:
                ok, buf = cv2.imencode(
                    ".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
                )
                if ok:
                    _frame_buffer.update(buf.tobytes())

            if not HEADLESS:
                cv2.imshow("PiChip Viewer", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            elif processed % 60 == 0:
                # No window over SSH — print live display FPS + counts so it's observable.
                now = time.time()
                fps = 60.0 / (now - last_print) if now > last_print else 0.0
                last_print = now
                total = sum(
                    c * CHIP_VALUES.get(col, 0) for col, c in cached_counts.items()
                )
                print(
                    f"[{processed} frames] {fps:.1f} fps display | "
                    f"counts={dict(cached_counts)} total=${total}",
                    flush=True,
                )

            if MAX_FRAMES and processed >= MAX_FRAMES:
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        worker.stop()
        if SNAPSHOT_PATH and vis is not None:
            cv2.imwrite(SNAPSHOT_PATH, vis)
            print(f"Saved annotated snapshot to {SNAPSHOT_PATH}")
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
