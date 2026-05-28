"""
Tennis Ball 3D Detection — ZED Stereo Camera on Raspberry Pi
============================================================
Uses the ZED as a standard USB stereo camera (no ZED SDK / CUDA needed).
Detects a tennis ball with a model-first hybrid detector plus OpenCV fallback,
then tracks and triangulates real-world (X, Y, Z) coordinates in meters.

Coordinate frame (camera-centered):
  +X → right
  +Y → down
  +Z → forward (away from camera)

Requirements:
  sudo apt install python3-opencv python3-numpy v4l-utils

Usage:
  python3 ball_detection.py
  python3 ball_detection.py --calibration my_calibration.npz
  python3 ball_detection.py --no-display      (headless / SSH mode)
  python3 ball_detection.py --width 1280 --height 360 --fps 15
"""

import cv2
import numpy as np
import argparse
import time
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from birdseye import BirdsEyeOverlay
    _BIRDSEYE_AVAILABLE = True
    _BIRDSEYE_IMPORT_ERROR = None
except ImportError as exc:
    BirdsEyeOverlay = None
    _BIRDSEYE_AVAILABLE = False
    _BIRDSEYE_IMPORT_ERROR = exc


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    """
    Intrinsic and extrinsic parameters for the ZED stereo pair.

    Default values are for ZED (not ZED Mini) at 720p (1280×720 per eye).
    Run calibrate_stereo.py with a checkerboard to get accurate values.
    """
    # ── Intrinsics (shared approximation for both eyes at 720p) ──────────────
    fx: float = 700.0       # Focal length in pixels (x)
    fy: float = 700.0       # Focal length in pixels (y)
    cx: float = 640.0       # Principal point x  (≈ width/2)
    cy: float = 360.0       # Principal point y  (≈ height/2)

    # ── Extrinsics ────────────────────────────────────────────────────────────
    baseline: float = 0.12  # Stereo baseline in metres (ZED = 120 mm)

    # ── Frame geometry ────────────────────────────────────────────────────────
    frame_width:  int = 2560  # Full side-by-side width from USB capture
    frame_height: int = 720   # Full frame height
    fps: float = 30.0
    fourcc: Optional[str] = "MJPG"
    exposure: Optional[float] = None
    gain: Optional[float] = None

    # Full stereo calibration. These are required for rectification.
    calibration_image_size: Optional[Tuple[int, int]] = None  # (eye_width, height)
    K_l: Optional[np.ndarray] = None
    D_l: Optional[np.ndarray] = None
    K_r: Optional[np.ndarray] = None
    D_r: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    T: Optional[np.ndarray] = None

    @property
    def eye_width(self) -> int:
        return self.frame_width // 2

    @property
    def has_full_calibration(self) -> bool:
        return all(
            v is not None
            for v in (self.K_l, self.D_l, self.K_r, self.D_r, self.R, self.T)
        )


# ── Tennis ball HSV colour range ──────────────────────────────────────────────
# Tennis balls are yellow-green; tweak if lighting differs significantly.
HSV_LOWER = np.array([22,  80,  80])
HSV_UPPER = np.array([65, 255, 255])

# ── Detection / geometry limits ───────────────────────────────────────────────
MIN_RADIUS_PX = 4     # At 2 m a tennis ball is roughly 8-10 px radius at 720p.
MAX_RADIUS_PX = 180   # Ignore circles larger than this.
MIN_STEREO_RADIUS_RATIO = 0.45
MAX_TRACK_MISSES = 5

TENNIS_BALL_RADIUS_M = 0.0335   # Physical radius ≈ 33.5 mm (ITF standard)

# Rectified stereo pairs should have almost the same y coordinate.
MAX_EPIPOLAR_Y_DIFF_PX = 10.0


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fourcc_to_string(value: float) -> str:
    code = int(value)
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00") else "unknown"


class ZEDCamera:
    """
    Opens the ZED as a standard USB UVC device and splits each frame into
    left / right eye images.  No ZED SDK or CUDA required.
    """

    def __init__(self, device_id: int = 0, config: CameraConfig = CameraConfig()):
        self.cfg = config
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(device_id, backend)

        if config.fourcc:
            fourcc = config.fourcc.upper()
            if len(fourcc) != 4:
                raise ValueError("--fourcc must be exactly four characters, such as MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        if config.exposure is not None:
            # V4L2 uses 1.0 for manual mode; DirectShow commonly accepts 0.25.
            manual_auto_exposure = 0.25 if sys.platform.startswith("win") else 1.0
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_auto_exposure)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
        if config.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, config.gain)

        if not self.cap.isOpened():
            raise RuntimeError(camera_open_error(device_id))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(f"[ZED] Opened at {actual_w}×{actual_h} "
              f"(eye: {actual_w // 2}×{actual_h}) "
              f"fps={actual_fps:.1f} fourcc={actual_fourcc}")

        # Update config to match what the camera actually gave us
        self.cfg.frame_width  = actual_w
        self.cfg.frame_height = actual_h
        if not self.cfg.has_full_calibration:
            self.cfg.cx = actual_w // 4   # cx for each eye ≈ eye_width / 2
            self.cfg.cy = actual_h // 2

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left, right) BGR frames, or (None, None) on error."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        w = self.cfg.eye_width
        return frame[:, :w].copy(), frame[:, w:].copy()

    def release(self):
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  BALL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BallDetection:
    """A single image-space ball candidate or tracker prediction."""
    cx: float
    cy: float
    radius: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    source: str
    score: float = 0.0
    tracked: bool = False
    predicted: bool = False
    mask_area: Optional[float] = None

    @property
    def center(self) -> Tuple[float, float]:
        return self.cx, self.cy

    def to_circle(self) -> Tuple[int, int, int]:
        return int(round(self.cx)), int(round(self.cy)), int(round(self.radius))

    def to_dict(self):
        x, y, w, h = self.bbox
        return {
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
            "radius": round(self.radius, 2),
            "bbox": [x, y, w, h],
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 3),
            "source": self.source,
            "tracked": self.tracked,
            "predicted": self.predicted,
        }


Detection = Optional[BallDetection]


def clip_bbox(x: float, y: float, w: float, h: float,
              frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(frame_w - 1, int(round(x))))
    y1 = max(0, min(frame_h - 1, int(round(y))))
    x2 = max(0, min(frame_w, int(round(x + w))))
    y2 = max(0, min(frame_h, int(round(y + h))))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def bbox_from_circle(cx: float, cy: float, radius: float,
                     frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    return clip_bbox(cx - radius, cy - radius, radius * 2, radius * 2, frame_w, frame_h)


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def offset_detection(det: BallDetection, dx: int, dy: int,
                     frame_w: int, frame_h: int) -> BallDetection:
    x, y, w, h = det.bbox
    det.cx += dx
    det.cy += dy
    det.bbox = clip_bbox(x + dx, y + dy, w, h, frame_w, frame_h)
    return det


def suppress_overlapping(candidates: List[BallDetection],
                         iou_threshold: float = 0.55) -> List[BallDetection]:
    ordered = sorted(candidates, key=lambda d: d.score or d.confidence, reverse=True)
    kept: List[BallDetection] = []
    for det in ordered:
        if all(bbox_iou(det.bbox, old.bbox) < iou_threshold for old in kept):
            kept.append(det)
    return kept


class ColorShapeBallDetector:
    """
    OpenCV fallback detector. It is intentionally looser than the original
    HSV+Hough pass so seams, logos, shadows, and small 2 m balls can still
    produce candidates for stereo/tracker scoring.
    """

    def __init__(self):
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.last_mask: Optional[np.ndarray] = None

    def detect_candidates(self, bgr_frame: np.ndarray) -> List[BallDetection]:
        h, w = bgr_frame.shape[:2]
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # Wider than the old close-range threshold: the real ball has worn felt,
        # darker print, shadows, and can be motion-blurred at 2 m.
        lower = np.array([18, 40, 45])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        self.last_mask = mask.copy()

        candidates: List[BallDetection] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(10.0, np.pi * MIN_RADIUS_PX * MIN_RADIUS_PX * 0.45)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
                continue
            circularity = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))
            fill_ratio = min(1.0, area / max(np.pi * radius * radius, 1.0))
            if circularity < 0.18 or fill_ratio < 0.16:
                continue

            confidence = min(0.78, 0.22 + 0.30 * circularity + 0.30 * fill_ratio)
            bbox = bbox_from_circle(cx, cy, radius, w, h)
            candidates.append(BallDetection(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                bbox=bbox,
                confidence=float(confidence),
                source="opencv",
                score=float(confidence),
                mask_area=float(area),
            ))

        blurred = cv2.GaussianBlur(mask, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=24,
            param1=45,
            param2=10,
            minRadius=MIN_RADIUS_PX,
            maxRadius=MAX_RADIUS_PX,
        )
        if circles is not None:
            for cx, cy, radius in np.round(circles[0]).astype(int):
                if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
                    continue
                x1, y1, bw, bh = bbox_from_circle(cx, cy, radius, w, h)
                patch = mask[y1:y1 + bh, x1:x1 + bw]
                support = float(cv2.countNonZero(patch)) / max(float(bw * bh), 1.0)
                if support < 0.08:
                    continue
                confidence = min(0.70, 0.35 + support)
                candidates.append(BallDetection(
                    cx=float(cx),
                    cy=float(cy),
                    radius=float(radius),
                    bbox=(x1, y1, bw, bh),
                    confidence=float(confidence),
                    source="opencv-hough",
                    score=float(confidence),
                ))

        return suppress_overlapping(candidates)


class ONNXBallDetector:
    """Tiny YOLO-style ONNX detector using OpenCV DNN for Raspberry Pi runtime."""

    def __init__(self, model_path: Optional[str], input_size: int = 320,
                 conf_threshold: float = 0.35, nms_threshold: float = 0.45):
        self.model_path = Path(model_path) if model_path else None
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None

        if self.model_path and self.model_path.exists():
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[Detector] Loaded ONNX model: {self.model_path}")

    @property
    def available(self) -> bool:
        return self.net is not None

    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _rows_from_output(output) -> np.ndarray:
        if isinstance(output, (tuple, list)):
            output = output[0]
        rows = np.squeeze(output)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim == 2 and rows.shape[0] < rows.shape[1] and rows.shape[0] <= 128:
            rows = rows.T
        return rows

    @staticmethod
    def _confidence(row: np.ndarray) -> float:
        if row.size < 5:
            return 0.0
        if row.size == 5:
            return float(row[4])
        if row.size == 6:
            tail = float(row[5])
            if tail == round(tail):
                return float(row[4])
            return float(row[4] * tail) if tail <= 1.0 else float(row[4])

        obj = float(row[4])
        class_scores = row[5:]
        best_class = float(np.max(class_scores)) if class_scores.size else 1.0
        yolo_v5_conf = obj * best_class
        yolo_v8_conf = float(np.max(row[4:]))
        return yolo_v8_conf if yolo_v8_conf > 0.80 and yolo_v5_conf < 0.20 else yolo_v5_conf

    def detect_candidates(self, bgr_frame: np.ndarray) -> List[BallDetection]:
        if not self.available:
            return []

        h, w = bgr_frame.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(bgr_frame)
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        rows = self._rows_from_output(self.net.forward())

        boxes: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []
        for row in rows:
            conf = self._confidence(row)
            if conf < self.conf_threshold:
                continue
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
                cx *= self.input_size
                cy *= self.input_size
                bw *= self.input_size
                bh *= self.input_size
            x = (cx - bw / 2.0 - pad_x) / scale
            y = (cy - bh / 2.0 - pad_y) / scale
            bw /= scale
            bh /= scale
            box = clip_bbox(x, y, bw, bh, w, h)
            if box[2] < MIN_RADIUS_PX * 2 or box[3] < MIN_RADIUS_PX * 2:
                continue
            boxes.append(box)
            confidences.append(float(conf))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        candidates: List[BallDetection] = []
        for idx in np.array(indices).flatten():
            x, y, bw, bh = boxes[int(idx)]
            radius = max(1.0, (bw + bh) / 4.0)
            candidates.append(BallDetection(
                cx=float(x + bw / 2.0),
                cy=float(y + bh / 2.0),
                radius=float(radius),
                bbox=(x, y, bw, bh),
                confidence=float(confidences[int(idx)]),
                source="model",
                score=float(confidences[int(idx)] + 0.20),
            ))
        return candidates


class ImageBallTracker:
    """Constant-velocity Kalman tracker for one eye: x, y, radius."""

    def __init__(self, max_missed: int = MAX_TRACK_MISSES):
        self.max_missed = max_missed
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kalman.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        self.kalman.measurementMatrix[0, 0] = 1.0
        self.kalman.measurementMatrix[1, 1] = 1.0
        self.kalman.measurementMatrix[2, 2] = 1.0
        self.kalman.processNoiseCov = np.diag([4, 4, 2, 25, 25, 8]).astype(np.float32)
        self.kalman.measurementNoiseCov = np.diag([12, 12, 6]).astype(np.float32)
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
        self.initialized = False
        self.missed = 0
        self.last_prediction: Detection = None
        self._last_t: Optional[float] = None

    def _set_dt(self, now: float):
        if self._last_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1.0 / 120.0, min(0.20, now - self._last_t))
        self._last_t = now
        self.kalman.transitionMatrix[:] = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float32)

    def predict(self, frame_shape: Tuple[int, int, int]) -> Detection:
        if not self.initialized:
            self.last_prediction = None
            return None
        self._set_dt(time.time())
        pred = self.kalman.predict()
        h, w = frame_shape[:2]
        cx = float(pred[0, 0])
        cy = float(pred[1, 0])
        radius = float(max(MIN_RADIUS_PX, pred[2, 0]))
        if cx < -radius or cy < -radius or cx > w + radius or cy > h + radius:
            self.missed += 1
            self.last_prediction = None
            return None
        confidence = max(0.10, 0.55 * (1.0 - self.missed / max(self.max_missed + 1, 1)))
        self.last_prediction = BallDetection(
            cx=cx,
            cy=cy,
            radius=radius,
            bbox=bbox_from_circle(cx, cy, radius, w, h),
            confidence=float(confidence),
            source="tracker",
            score=float(confidence * 0.75),
            tracked=True,
            predicted=True,
        )
        self.missed += 1
        return self.last_prediction if self.missed <= self.max_missed else None

    def correct(self, det: BallDetection, frame_shape: Tuple[int, int, int]) -> BallDetection:
        h, w = frame_shape[:2]
        measurement = np.array([[det.cx], [det.cy], [det.radius]], dtype=np.float32)
        if not self.initialized:
            self.kalman.statePost = np.array(
                [[det.cx], [det.cy], [det.radius], [0], [0], [0]], dtype=np.float32
            )
            self.initialized = True
        else:
            corrected = self.kalman.correct(measurement)
            det.cx = float(corrected[0, 0])
            det.cy = float(corrected[1, 0])
            det.radius = float(max(MIN_RADIUS_PX, corrected[2, 0]))
            det.bbox = bbox_from_circle(det.cx, det.cy, det.radius, w, h)
        det.tracked = True
        det.predicted = False
        self.missed = 0
        self.last_prediction = det
        return det

    def search_roi(self, frame_shape: Tuple[int, int, int],
                   padding_scale: float = 3.0) -> Optional[Tuple[int, int, int, int]]:
        pred = self.last_prediction
        if pred is None or self.missed > self.max_missed:
            return None
        h, w = frame_shape[:2]
        pad = max(56.0, pred.radius * padding_scale + self.missed * 18.0)
        return clip_bbox(pred.cx - pad, pred.cy - pad, pad * 2, pad * 2, w, h)


class TennisBallDetector:
    """Hybrid model-first detector with OpenCV fallback and tracker-aware scoring."""

    def __init__(self, mode: str = "hybrid", model_path: Optional[str] = None,
                 conf_threshold: float = 0.35, nms_threshold: float = 0.45,
                 model_input_size: int = 320):
        self.mode = mode
        self.model = ONNXBallDetector(model_path, model_input_size, conf_threshold, nms_threshold)
        self.fallback = ColorShapeBallDetector()
        self.last_debug_mask: Optional[np.ndarray] = None

        if mode in ("hybrid", "model") and not self.model.available:
            message = f"[Detector] ONNX model not found or unavailable: {model_path}"
            if mode == "model":
                raise FileNotFoundError(message)
            print(message + " — using OpenCV fallback until a model is added.")

    def _score(self, det: BallDetection, prediction: Detection = None) -> BallDetection:
        source_bonus = {
            "model": 0.28,
            "opencv": 0.05,
            "opencv-hough": 0.03,
            "tracker": -0.10,
        }.get(det.source, 0.0)
        score = det.confidence + source_bonus
        if prediction is not None:
            dist = np.hypot(det.cx - prediction.cx, det.cy - prediction.cy)
            gate = max(24.0, prediction.radius * 4.0)
            score += 0.30 * max(0.0, 1.0 - dist / gate)
        det.score = float(score)
        return det

    def detect_candidates(self, bgr_frame: np.ndarray,
                          prediction: Detection = None,
                          roi: Optional[Tuple[int, int, int, int]] = None
                          ) -> Tuple[List[BallDetection], Optional[np.ndarray]]:
        h, w = bgr_frame.shape[:2]
        search_frame = bgr_frame
        offset_x = 0
        offset_y = 0
        if roi is not None:
            offset_x, offset_y, roi_w, roi_h = roi
            search_frame = bgr_frame[offset_y:offset_y + roi_h, offset_x:offset_x + roi_w]

        candidates: List[BallDetection] = []
        if self.mode in ("hybrid", "model") and self.model.available:
            candidates.extend(self.model.detect_candidates(search_frame))
        if self.mode in ("hybrid", "opencv"):
            candidates.extend(self.fallback.detect_candidates(search_frame))

        debug_mask = self.fallback.last_mask
        if roi is not None:
            candidates = [offset_detection(det, offset_x, offset_y, w, h) for det in candidates]
            if not candidates:
                # Fall back to a full-frame scan if the tracker ROI lost the ball.
                candidates, debug_mask = self.detect_candidates(bgr_frame, prediction, roi=None)
                return candidates, debug_mask

        scored = [self._score(det, prediction) for det in candidates]
        return suppress_overlapping(scored), debug_mask

    @staticmethod
    def annotate(frame: np.ndarray, det: Detection,
                 circle_color=(0, 255, 0), text: str = "") -> np.ndarray:
        if det is None:
            return frame
        cx, cy, r = det.to_circle()
        x, y, w, h = det.bbox
        if det.predicted:
            circle_color = (0, 180, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), circle_color, 1)
        cv2.circle(frame, (cx, cy), r, circle_color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        label = text
        if det.source:
            label = f"{label} {det.source}:{det.confidence:.2f}".strip()
        if det.predicted:
            label += " pred"
        if label:
            cv2.putText(frame, label, (min(cx + r + 5, frame.shape[1] - 180), max(18, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, circle_color, 1)
        return frame


# ══════════════════════════════════════════════════════════════════════════════
#  RECTIFICATION + TRIANGULATION
# ══════════════════════════════════════════════════════════════════════════════

class StereoRectifier:
    """
    Rectifies raw ZED UVC eye images so same-world points lie on the same row.
    Without this step, simple x-disparity triangulation is not reliable.
    """

    def __init__(self, config: CameraConfig):
        self.cfg = config
        self.enabled = False
        self._maps_l = None
        self._maps_r = None

        if not config.has_full_calibration:
            print("[Rectification] No full calibration matrices found; using raw frames.")
            return

        image_size = (config.eye_width, config.frame_height)
        K_l, K_r = self._scaled_intrinsics(image_size)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l,
            config.D_l,
            K_r,
            config.D_r,
            image_size,
            config.R,
            config.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        self._maps_l = cv2.initUndistortRectifyMap(
            K_l, config.D_l, R1, P1, image_size, cv2.CV_16SC2
        )
        self._maps_r = cv2.initUndistortRectifyMap(
            K_r, config.D_r, R2, P2, image_size, cv2.CV_16SC2
        )

        config.fx = float(P1[0, 0])
        config.fy = float(P1[1, 1])
        config.cx = float(P1[0, 2])
        config.cy = float(P1[1, 2])
        config.baseline = abs(float(P2[0, 3]) / config.fx)

        self.enabled = True
        print("[Rectification] Enabled.")
        print(
            f"  Rectified fx={config.fx:.1f} fy={config.fy:.1f} "
            f"cx={config.cx:.1f} cy={config.cy:.1f} B={config.baseline*1000:.1f} mm"
        )

    def _scaled_intrinsics(self, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.calibration_image_size is None:
            return self.cfg.K_l.copy(), self.cfg.K_r.copy()

        cal_w, cal_h = self.cfg.calibration_image_size
        cur_w, cur_h = image_size
        sx = cur_w / cal_w
        sy = cur_h / cal_h

        def scale(K: np.ndarray) -> np.ndarray:
            K2 = K.copy()
            K2[0, 0] *= sx
            K2[0, 2] *= sx
            K2[1, 1] *= sy
            K2[1, 2] *= sy
            return K2

        if abs(sx - 1.0) > 0.001 or abs(sy - 1.0) > 0.001:
            print(
                "[Rectification] Scaling calibration intrinsics from "
                f"{cal_w}x{cal_h} to {cur_w}x{cur_h}."
            )

        return scale(self.cfg.K_l), scale(self.cfg.K_r)

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return left, right

        left_rect = cv2.remap(left, self._maps_l[0], self._maps_l[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self._maps_r[0], self._maps_r[1], cv2.INTER_LINEAR)
        return left_rect, right_rect


def validate_stereo_pair(left: Detection, right: Detection,
                         max_y_diff: float = MAX_EPIPOLAR_Y_DIFF_PX) -> Tuple[bool, str]:
    if left is None or right is None:
        missing = []
        if left is None:
            missing.append("LEFT")
        if right is None:
            missing.append("RIGHT")
        return False, "Ball not detected in " + " ".join(missing)

    xl, yl, rl = left.cx, left.cy, left.radius
    xr, yr, rr = right.cx, right.cy, right.radius
    y_diff = abs(float(yl - yr))
    if y_diff > max_y_diff:
        return False, f"Rejected pair: epipolar y-diff {y_diff:.1f}px > {max_y_diff:.1f}px"

    disparity = float(xl - xr)
    if disparity <= 1.0:
        return False, f"Rejected pair: invalid disparity {disparity:.1f}px"

    radius_ratio = min(rl, rr) / max(rl, rr)
    if radius_ratio < MIN_STEREO_RADIUS_RATIO:
        return False, f"Rejected pair: radius mismatch L={rl:.1f}px R={rr:.1f}px"

    return True, "OK"


def choose_stereo_pair(left_candidates: Sequence[BallDetection],
                       right_candidates: Sequence[BallDetection],
                       config: CameraConfig,
                       max_y_diff: float = MAX_EPIPOLAR_Y_DIFF_PX
                       ) -> Tuple[Detection, Detection, bool, str]:
    best_pair: Tuple[Detection, Detection] = (None, None)
    best_score = -1e9
    best_rejection = "Ball not detected"

    if not left_candidates or not right_candidates:
        missing = []
        if not left_candidates:
            missing.append("LEFT")
        if not right_candidates:
            missing.append("RIGHT")
        return None, None, False, "Ball not detected in " + " ".join(missing)

    for left in left_candidates:
        for right in right_candidates:
            y_diff = abs(float(left.cy - right.cy))
            disparity = float(left.cx - right.cx)
            if disparity <= 1.0:
                best_rejection = f"Rejected pair: invalid disparity {disparity:.1f}px"
                continue
            if y_diff > max_y_diff:
                best_rejection = f"Rejected pair: epipolar y-diff {y_diff:.1f}px > {max_y_diff:.1f}px"
                continue
            radius_ratio = min(left.radius, right.radius) / max(left.radius, right.radius)
            if radius_ratio < MIN_STEREO_RADIUS_RATIO:
                best_rejection = (
                    f"Rejected pair: radius mismatch L={left.radius:.1f}px "
                    f"R={right.radius:.1f}px"
                )
                continue

            stereo_z = (config.fx * config.baseline) / disparity
            avg_radius_px = max((left.radius + right.radius) / 2.0, 1.0)
            size_z = (config.fx * TENNIS_BALL_RADIUS_M) / avg_radius_px
            size_ratio = min(stereo_z, size_z) / max(stereo_z, size_z)
            if size_ratio < 0.22:
                best_rejection = (
                    f"Rejected pair: depth/size disagreement Z={stereo_z:.2f}m "
                    f"size_Z={size_z:.2f}m"
                )
                continue

            epipolar_score = max(0.0, 1.0 - y_diff / max(max_y_diff, 1.0))
            score = (
                left.score + right.score
                + 0.35 * epipolar_score
                + 0.25 * radius_ratio
                + 0.20 * size_ratio
            )
            if left.predicted or right.predicted:
                score -= 0.30
            if score > best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair[0] is None or best_pair[1] is None:
        return None, None, False, best_rejection

    left, right = best_pair
    sources = f"{left.source}+{right.source}"
    status = (
        f"OK {sources} conf={min(left.confidence, right.confidence):.2f}"
        if not (left.predicted or right.predicted)
        else f"TRACKED {sources} conf={min(left.confidence, right.confidence):.2f}"
    )
    return left, right, True, status

@dataclass
class BallPosition:
    X: float          # metres, positive = right of camera centre
    Y: float          # metres, positive = below camera centre
    Z: float          # metres, positive = forward
    disparity: float  # pixels (diagnostic)
    size_Z: float     # metres, depth estimated from apparent ball size (cross-check)
    epipolar_y_diff: float  # pixels (diagnostic after rectification)
    known_Z: Optional[float] = None
    depth_error: Optional[float] = None
    depth_error_pct: Optional[float] = None
    confidence: float = 0.0
    source: str = ""
    tracked: bool = False
    predicted: bool = False

    def __str__(self):
        mode = "pred" if self.predicted else "meas"
        msg = (f"X={self.X:+.3f} m  Y={self.Y:+.3f} m  Z={self.Z:.3f} m  "
               f"(disp={self.disparity:.1f} px | ydiff={self.epipolar_y_diff:.1f} px "
               f"| size_Z={self.size_Z:.3f} m | {mode} {self.source} "
               f"conf={self.confidence:.2f})")
        if self.depth_error is not None and self.depth_error_pct is not None:
            msg += (f" | known_Z={self.known_Z:.3f} m "
                    f"err={self.depth_error:+.3f} m ({self.depth_error_pct:+.1f}%)")
        return msg

    def to_dict(self):
        record = {
            "X": round(self.X, 4),
            "Y": round(self.Y, 4),
            "Z": round(self.Z, 4),
            "disparity": round(self.disparity, 2),
            "size_Z": round(self.size_Z, 4),
            "epipolar_y_diff": round(self.epipolar_y_diff, 2),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "tracked": self.tracked,
            "predicted": self.predicted,
        }
        if self.known_Z is not None:
            record["known_Z"] = round(self.known_Z, 4)
            record["depth_error"] = round(self.depth_error, 4)
            record["depth_error_pct"] = round(self.depth_error_pct, 2)
        return record


class StereoTriangulator:
    """
    Converts (left_detection, right_detection) pixel pairs into a metric
    3-D position using the standard stereo disparity formula.

    Formulas
    --------
        disparity  d  = x_left − x_right           (pixels, must be > 0)
        depth      Z  = (fx × B) / d               (metres)
        lateral    X  = (x_left − cx) × Z / fx     (metres)
        vertical   Y  = (y_left − cy) × Z / fy     (metres)

    A secondary size-based depth estimate is computed as a sanity check:
        Z_size = (fx × R_real) / r_pixels
    """

    def __init__(self, config: CameraConfig = CameraConfig()):
        self.cfg = config

    def triangulate(self, left: Detection, right: Detection,
                    known_depth_m: Optional[float] = None) -> Optional[BallPosition]:
        if left is None or right is None:
            return None

        xl, yl, rl = left.cx, left.cy, left.radius
        xr, yr, rr = right.cx, right.cy, right.radius

        disparity = float(xl - xr)
        if disparity <= 1.0:
            # Disparity too small → unreliable or behind camera
            return None

        Z = (self.cfg.fx * self.cfg.baseline) / disparity
        X = (xl - self.cfg.cx) * Z / self.cfg.fx
        Y = (yl - self.cfg.cy) * Z / self.cfg.fy

        avg_radius_px = max((rl + rr) / 2.0, 1.0)
        size_Z = (self.cfg.fx * TENNIS_BALL_RADIUS_M) / avg_radius_px
        epipolar_y_diff = abs(float(yl - yr))

        depth_error = None
        depth_error_pct = None
        known_Z = None
        if known_depth_m is not None and known_depth_m > 0:
            known_Z = known_depth_m
            depth_error = Z - known_depth_m
            depth_error_pct = (depth_error / known_depth_m) * 100.0

        return BallPosition(
            X=X,
            Y=Y,
            Z=Z,
            disparity=disparity,
            size_Z=size_Z,
            epipolar_y_diff=epipolar_y_diff,
            known_Z=known_Z,
            depth_error=depth_error,
            depth_error_pct=depth_error_pct,
            confidence=min(left.confidence, right.confidence),
            source=f"{left.source}+{right.source}",
            tracked=left.tracked or right.tracked,
            predicted=left.predicted or right.predicted,
        )


class PositionTracker3D:
    """Constant-velocity tracker for X, Y, Z position smoothing and short gaps."""

    def __init__(self, max_missed: int = MAX_TRACK_MISSES):
        self.max_missed = max_missed
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kalman.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        self.kalman.measurementMatrix[0, 0] = 1.0
        self.kalman.measurementMatrix[1, 1] = 1.0
        self.kalman.measurementMatrix[2, 2] = 1.0
        self.kalman.processNoiseCov = np.diag([0.01, 0.01, 0.02, 1.0, 1.0, 1.2]).astype(np.float32)
        self.kalman.measurementNoiseCov = np.diag([0.03, 0.03, 0.06]).astype(np.float32)
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False
        self.missed = 0
        self._last_t: Optional[float] = None
        self._last_template: Optional[BallPosition] = None

    def _set_dt(self, now: float):
        if self._last_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1.0 / 120.0, min(0.25, now - self._last_t))
        self._last_t = now
        self.kalman.transitionMatrix[:] = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float32)

    def correct(self, pos: BallPosition) -> BallPosition:
        self._set_dt(time.time())
        measurement = np.array([[pos.X], [pos.Y], [pos.Z]], dtype=np.float32)
        if not self.initialized:
            self.kalman.statePost = np.array(
                [[pos.X], [pos.Y], [pos.Z], [0], [0], [0]], dtype=np.float32
            )
            self.initialized = True
            smoothed = self.kalman.statePost
        else:
            smoothed = self.kalman.correct(measurement)

        pos.X = float(smoothed[0, 0])
        pos.Y = float(smoothed[1, 0])
        pos.Z = float(max(0.0, smoothed[2, 0]))
        pos.tracked = True
        pos.predicted = False
        self.missed = 0
        self._last_template = pos
        return pos

    def predict(self) -> Optional[BallPosition]:
        if not self.initialized or self.missed >= self.max_missed:
            return None
        self._set_dt(time.time())
        pred = self.kalman.predict()
        self.missed += 1
        template = self._last_template
        if template is None:
            return None
        return BallPosition(
            X=float(pred[0, 0]),
            Y=float(pred[1, 0]),
            Z=float(max(0.0, pred[2, 0])),
            disparity=template.disparity,
            size_Z=template.size_Z,
            epipolar_y_diff=template.epipolar_y_diff,
            known_Z=template.known_Z,
            depth_error=template.depth_error,
            depth_error_pct=template.depth_error_pct,
            confidence=max(0.10, template.confidence * (1.0 - self.missed / (self.max_missed + 1))),
            source="3d-tracker",
            tracked=True,
            predicted=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(path: str, config: CameraConfig) -> CameraConfig:
    """
    Load stereo calibration saved by calibrate_stereo.py and update config.
    File format: NumPy .npz with keys  fx, fy, cx, cy, baseline,
    K_l, D_l, K_r, D_r, R, T, and optional image_width/image_height.
    """
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}\n"
            "Create it first with:\n"
            "  python3 calibrate_stereo.py --device 0\n"
            "Then run detection with:\n"
            "  python3 ball_detection.py --device 0 --calibration calibration.npz\n"
            "For an uncalibrated quick camera test, omit the --calibration option."
        )

    data = np.load(path)
    config.fx       = float(data["fx"])
    config.fy       = float(data["fy"])
    config.cx       = float(data["cx"])
    config.cy       = float(data["cy"])
    config.baseline = float(data["baseline"])

    required = ("K_l", "D_l", "K_r", "D_r", "R", "T")
    if all(k in data for k in required):
        config.K_l = data["K_l"]
        config.D_l = data["D_l"]
        config.K_r = data["K_r"]
        config.D_r = data["D_r"]
        config.R = data["R"]
        config.T = data["T"]
        if "image_width" in data and "image_height" in data:
            config.calibration_image_size = (
                int(data["image_width"]),
                int(data["image_height"]),
            )
        config.baseline = abs(float(config.T[0]))
    else:
        missing = ", ".join(k for k in required if k not in data)
        print(f"[Calibration] Warning: missing {missing}; rectification disabled.")

    print(f"[Calibration] Loaded from {path}")
    print(f"  fx={config.fx:.1f}  fy={config.fy:.1f}  "
          f"cx={config.cx:.1f}  cy={config.cy:.1f}  B={config.baseline*1000:.1f} mm")
    return config


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def overlay_info(frame: np.ndarray, pos: Optional[BallPosition],
                 fps: float, status_message: str):
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    if pos:
        state = "PRED" if pos.predicted else "MEAS"
        txt = (
            f"{state} X:{pos.X:+.3f}m  Y:{pos.Y:+.3f}m  Z:{pos.Z:.3f}m  "
            f"disp:{pos.disparity:.1f}px {pos.source} c:{pos.confidence:.2f}"
        )
        if pos.depth_error is not None:
            txt += f"  err:{pos.depth_error:+.3f}m"
        cv2.putText(frame, txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 80), 2)
    else:
        cv2.putText(frame, status_message[:100], (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Divider line between eyes
    cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.putText(frame, "LEFT",  (10,       h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "RIGHT", (w // 2 + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Tennis ball 3D detection via ZED stereo")
    p.add_argument("--device",      type=int,   default=0,
                   help="V4L2 device index (default 0 -> /dev/video0)")
    p.add_argument("--width",       type=int,   default=2560,
                   help="Requested full side-by-side capture width (default 2560)")
    p.add_argument("--height",      type=int,   default=720,
                   help="Requested capture height (default 720)")
    p.add_argument("--fps",         type=float, default=30.0,
                   help="Requested camera frame rate (default 30)")
    p.add_argument("--fourcc",      type=str,   default="MJPG",
                   help="Pixel format (default MJPG; avoids green-screen on ZED)")
    p.add_argument("--exposure", type=float, default=None,
                   help="Optional manual camera exposure value. Lower values reduce motion blur.")
    p.add_argument("--gain", type=float, default=None,
                   help="Optional manual camera gain value.")
    p.add_argument("--detector", choices=["hybrid", "model", "opencv"], default="hybrid",
                   help="Detection mode: model-first hybrid, model only, or OpenCV fallback only")
    p.add_argument("--model", type=str, default="models/tennis_ball.onnx",
                   help="Path to YOLO-style ONNX ball detector for hybrid/model mode")
    p.add_argument("--conf-threshold", type=float, default=0.35,
                   help="Minimum ONNX detector confidence")
    p.add_argument("--nms-threshold", type=float, default=0.45,
                   help="ONNX detector non-maximum suppression IoU threshold")
    p.add_argument("--model-input-size", type=int, default=320,
                   help="Square ONNX detector input size, e.g. 320 or 416")
    p.add_argument("--debug-mask", action="store_true",
                   help="Show the OpenCV fallback mask for debugging")
    p.add_argument("--max-track-misses", type=int, default=MAX_TRACK_MISSES,
                   help="Frames to keep predicting through short detector misses")
    p.add_argument("--zed-calibration", type=str, default=None,
                   help="Path to Stereolabs factory .conf file (e.g. SN28837104.conf). "
                        "Enables full rectification without a checkerboard.")
    p.add_argument("--zed-resolution", type=str, default="HD",
                   choices=["2K", "FHD", "HD", "VGA"],
                   help="Resolution key in the .conf file (default HD = 1280x720)")
    p.add_argument("--calibration", type=str,   default=None,
                   help="Path to .npz calibration file from calibrate_stereo.py")
    p.add_argument("--no-display",  action="store_true",
                   help="Disable OpenCV window (use for headless SSH sessions)")
    p.add_argument("--birdseye", action="store_true",
                   help="Show bird's eye 2D tracking view")
    p.add_argument("--intercept-z", type=float, default=0.30,
                   help="Z distance in metres where panel intercept is predicted")
    p.add_argument("--log",         type=str,   default=None,
                   help="Append JSON position records to this file")
    p.add_argument("--known-distance", type=float, default=None,
                   help="Known measured ball distance in metres for depth-error checks")
    p.add_argument("--max-epipolar-y-diff", type=float, default=MAX_EPIPOLAR_Y_DIFF_PX,
                   help="Reject left/right matches with larger rectified y difference")
    return p.parse_args()


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.width % 2 != 0:
        raise ValueError("--width must be even because the stereo frame is split in half")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.known_distance is not None and args.known_distance <= 0:
        raise ValueError("--known-distance must be a positive distance in metres")
    if not 0.0 < args.conf_threshold <= 1.0:
        raise ValueError("--conf-threshold must be in the range (0, 1]")
    if not 0.0 < args.nms_threshold <= 1.0:
        raise ValueError("--nms-threshold must be in the range (0, 1]")
    if args.model_input_size <= 0:
        raise ValueError("--model-input-size must be positive")
    if args.max_track_misses < 0:
        raise ValueError("--max-track-misses must be non-negative")
    if args.intercept_z <= 0:
        raise ValueError("--intercept-z must be positive")

    # ── Setup ──────────────────────────────────────────────────────────────
    config = CameraConfig(
        frame_width=args.width,
        frame_height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        exposure=args.exposure,
        gain=args.gain,
    )
    if args.calibration:
        config = load_calibration(args.calibration, config)

    camera       = ZEDCamera(device_id=args.device, config=config)
    rectifier    = StereoRectifier(camera.cfg)
    detector     = TennisBallDetector(
        mode=args.detector,
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        model_input_size=args.model_input_size,
    )
    left_tracker = ImageBallTracker(max_missed=args.max_track_misses)
    right_tracker = ImageBallTracker(max_missed=args.max_track_misses)
    position_tracker = PositionTracker3D(max_missed=args.max_track_misses)
    triangulator = StereoTriangulator(config=camera.cfg)  # use updated config

    bev = None
    if args.birdseye:
        if _BIRDSEYE_AVAILABLE:
            bev = BirdsEyeOverlay(
                arena_x_m=4.0,
                arena_z_m=4.0,
                intercept_z=args.intercept_z,
            )
            print("[Bird's Eye] Enabled.")
        else:
            print(f"[Bird's Eye] birdseye.py not available: {_BIRDSEYE_IMPORT_ERROR}")

    log_file = open(args.log, "a") if args.log else None

    # ── FPS tracking ───────────────────────────────────────────────────────
    fps        = 0.0
    t_fps      = time.time()
    frame_idx  = 0

    print("\n[Ready] Press 'q' to quit, 's' to save current frame.\n")
    if args.known_distance is not None:
        print(f"[Depth Check] Comparing Z against known distance {args.known_distance:.3f} m")

    while True:
        left, right = camera.read()
        if left is None:
            print("[Error] Failed to capture frame.")
            time.sleep(0.1)
            continue

        left, right = rectifier.rectify(left, right)

        # ── Detect + track ─────────────────────────────────────────────
        left_pred = left_tracker.predict(left.shape)
        right_pred = right_tracker.predict(right.shape)
        left_roi = left_tracker.search_roi(left.shape)
        right_roi = right_tracker.search_roi(right.shape)

        left_candidates, left_mask = detector.detect_candidates(left, left_pred, left_roi)
        right_candidates, right_mask = detector.detect_candidates(right, right_pred, right_roi)
        if left_pred is not None:
            left_candidates.append(left_pred)
        if right_pred is not None:
            right_candidates.append(right_pred)

        left_det, right_det, pair_ok, status_message = choose_stereo_pair(
            left_candidates,
            right_candidates,
            config=camera.cfg,
            max_y_diff=args.max_epipolar_y_diff,
        )

        # ── Triangulate + smooth ────────────────────────────────────────
        position = None
        if pair_ok and left_det is not None and right_det is not None:
            if not left_det.predicted:
                left_det = left_tracker.correct(left_det, left.shape)
            if not right_det.predicted:
                right_det = right_tracker.correct(right_det, right.shape)

            position = triangulator.triangulate(
                left_det,
                right_det,
                known_depth_m=args.known_distance,
            )
            if position is not None and not position.predicted:
                position = position_tracker.correct(position)
            elif position is not None:
                tracked_position = position_tracker.predict()
                if tracked_position is not None:
                    tracked_position.disparity = position.disparity
                    tracked_position.size_Z = position.size_Z
                    tracked_position.epipolar_y_diff = position.epipolar_y_diff
                    position = tracked_position
        else:
            position = position_tracker.predict()
            if position is not None:
                status_message = "TRACKED 3d-tracker prediction through detector miss"

        # ── Bird's eye view / intercept ─────────────────────────────────
        if bev is not None:
            bev.update(position)
            ix = bev.intercept_x
            if ix is not None and position is not None and frame_idx % 10 == 0:
                print(f"  -> Panel intercept X={ix:+.3f} m")

        # ── Log / print ─────────────────────────────────────────────────
        if position:
            print(f"[Frame {frame_idx:05d}] {position}")
            if log_file:
                record = {
                    "frame": frame_idx,
                    "t": time.time(),
                    "rectified": rectifier.enabled,
                    "status": status_message,
                    "left_detection": left_det.to_dict() if left_det else None,
                    "right_detection": right_det.to_dict() if right_det else None,
                    **position.to_dict(),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
        elif frame_idx % 30 == 0:
            print(f"[Frame {frame_idx:05d}] {status_message}")

        # ── FPS ─────────────────────────────────────────────────────────
        frame_idx += 1
        if frame_idx % 30 == 0:
            fps   = 30.0 / (time.time() - t_fps)
            t_fps = time.time()

        # ── Display ─────────────────────────────────────────────────────
        if not args.no_display:
            # Annotate each eye
            detector.annotate(left,  left_det,  circle_color=(0, 255, 0),   text="L")
            detector.annotate(right, right_det, circle_color=(0, 200, 255), text="R")
            if left_pred is not None and left_det is not left_pred:
                detector.annotate(left, left_pred, circle_color=(0, 160, 255), text="L")
            if right_pred is not None and right_det is not right_pred:
                detector.annotate(right, right_pred, circle_color=(0, 160, 255), text="R")

            # Combine side by side and downscale for Pi's display
            combined = np.hstack([left, right])
            combined = cv2.resize(combined, (1280, 360))
            overlay_info(combined, position, fps, status_message)

            cv2.imshow("ZED Ball Detection", combined)
            if bev is not None:
                cv2.imshow("Bird's Eye View", bev.frame)
            if args.debug_mask and left_mask is not None and right_mask is not None:
                mask_h = min(left_mask.shape[0], right_mask.shape[0])
                mask_l = left_mask[:mask_h]
                mask_r = right_mask[:mask_h]
                mask_combined = np.hstack([
                    cv2.cvtColor(mask_l, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR),
                ])
                mask_combined = cv2.resize(mask_combined, (1280, 360))
                cv2.imshow("OpenCV Fallback Mask", mask_combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                fname = f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(fname, combined)
                print(f"[Saved] {fname}")
        else:
            # Headless: just run until Ctrl-C
            pass

    # ── Cleanup ─────────────────────────────────────────────────────────────
    camera.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
    print("Done.")


if __name__ == "__main__":
    main()
