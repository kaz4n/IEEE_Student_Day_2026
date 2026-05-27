"""
Tennis Ball 3D Detection — ZED Stereo Camera on Raspberry Pi
============================================================
Uses the ZED as a standard USB stereo camera (no ZED SDK / CUDA needed).
Detects a tennis ball via HSV colour filtering or optional YOLO tracking in
both eyes, then triangulates to get real-world (X, Y, Z) coordinates in meters.

Coordinate frame (camera-centered):
  +X → right
  +Y → down
  +Z → forward (away from camera)

Requirements:
  pip install opencv-python numpy

Usage:
  python3 ball_detection.py
  python3 ball_detection.py --detector hsv
  python3 ball_detection.py --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball
  python3 ball_detection.py --zed-calibration SN28837104.conf
  python3 ball_detection.py --zed-calibration SN28837104.conf --calibration calibration.npz
  python3 ball_detection.py --no-display      (headless / SSH mode)
"""

import cv2
import numpy as np
import argparse
import configparser
import time
import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


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
    # ── Intrinsics — left eye, HD 1280×720 (from SN28837104 factory cal) ───────
    fx: float = 532.935     # Focal length x
    fy: float = 532.470     # Focal length y
    cx: float = 642.825     # Principal point x
    cy: float = 368.369     # Principal point y

    # ── Extrinsics ────────────────────────────────────────────────────────────
    baseline: float = 0.120144  # Stereo baseline in metres (ZED 2 SN28837104)

    # ── Frame geometry ────────────────────────────────────────────────────────
    frame_width:  int = 2560  # Full side-by-side width from USB capture
    frame_height: int = 720   # Full frame height
    fps: float = 30.0
    fourcc: Optional[str] = "MJPG"

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


def create_zed2_sn28837104_config(
    frame_width: int = 2560,
    frame_height: int = 720,
    fps: float = 30.0,
    fourcc: Optional[str] = "MJPG",
    verbose: bool = True,
) -> CameraConfig:
    """Return hardcoded ZED 2 SN28837104 factory calibration for HD mode."""
    fx_l, fy_l, cx_l, cy_l = 532.935, 532.470, 642.825, 368.3685
    fx_r, fy_r, cx_r, cy_r = 531.920, 531.910, 648.270, 364.320
    baseline_mm = 120.144
    ty_mm = -0.0708224
    tz_mm = -0.0801476
    rx = 0.00161995
    cv = -0.00199086
    rz = 0.000540406

    config = CameraConfig(
        fx=fx_l,
        fy=fy_l,
        cx=cx_l,
        cy=cy_l,
        baseline=baseline_mm / 1000.0,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        fourcc=fourcc,
        calibration_image_size=(1280, 720),
        K_l=np.array([[fx_l, 0, cx_l],
                      [0, fy_l, cy_l],
                      [0,    0,    1]], dtype=np.float64),
        D_l=np.array([
            -0.0644799,
            0.0414727,
            2.32191e-05,
            0.000663395,
            -0.0160775,
        ], dtype=np.float64),
        K_r=np.array([[fx_r, 0, cx_r],
                      [0, fy_r, cy_r],
                      [0,    0,    1]], dtype=np.float64),
        D_r=np.array([
            -0.0648757,
            0.0410955,
            -0.000163562,
            -0.000147387,
            -0.0159065,
        ], dtype=np.float64),
        T=np.array([
            [baseline_mm / 1000.0],
            [ty_mm / 1000.0],
            [tz_mm / 1000.0],
        ], dtype=np.float64),
    )
    config.R, _ = cv2.Rodrigues(np.array([rx, cv, rz], dtype=np.float64))

    if verbose:
        print("[ZED Factory Cal] Using hardcoded ZED 2 SN28837104 HD factory calibration")
        print(f"  Left  fx={fx_l:.3f} fy={fy_l:.3f} cx={cx_l:.3f} cy={cy_l:.3f}")
        print(f"  Right fx={fx_r:.3f} fy={fy_r:.3f} cx={cx_r:.3f} cy={cy_r:.3f}")
        print(f"  Baseline={baseline_mm:.3f} mm  RX={rx:.6f}  CV={cv:.6f}  RZ={rz:.6f}")
    return config


# ── Tennis ball HSV colour range ──────────────────────────────────────────────
# Tennis balls are yellow-green; tweak if lighting differs significantly.
HSV_LOWER = np.array([22,  80,  80])
HSV_UPPER = np.array([65, 255, 255])

# ── Hough / geometry limits ───────────────────────────────────────────────────
MIN_RADIUS_PX = 8    # Ignore circles smaller than this (noise)
MAX_RADIUS_PX = 150  # Ignore circles larger than this

TENNIS_BALL_RADIUS_M = 0.0335   # Physical radius ≈ 33.5 mm (ITF standard)

# Rectified stereo pairs should have almost the same y coordinate.
MAX_EPIPOLAR_Y_DIFF_PX = 4.0


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fourcc_to_string(value: float) -> str:
    code = int(value)
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00") else "unknown"


def camera_open_error(device_id: int) -> str:
    if sys.platform.startswith("win"):
        return (
            f"Cannot open camera at device {device_id}. "
            "Check the USB connection and camera permissions."
        )

    device_path = f"/dev/video{device_id}"
    return (
        f"Cannot open camera at device {device_id} ({device_path}). "
        "Check the USB connection, verify the correct node with "
        "`v4l2-ctl --list-devices`, inspect formats with "
        f"`v4l2-ctl -d {device_path} --list-formats-ext`, and make sure your "
        "user is in the video group with `sudo usermod -aG video \"$USER\"` "
        "then log out and back in."
    )


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
        # Disable auto-exposure to keep colour stable
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(camera_open_error(device_id))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(
            f"[ZED] Opened at {actual_w}×{actual_h} "
            f"(eye: {actual_w // 2}×{actual_h}) "
            f"fps={actual_fps:.1f} fourcc={actual_fourcc}"
        )

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

Detection = Optional[Tuple[int, int, int]]   # (cx_px, cy_px, radius_px)


@dataclass
class DetectionCandidate:
    """Internal detector result with metadata used before stereo triangulation."""
    cx: float
    cy: float
    radius: float
    confidence: float = 1.0
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    track_id: Optional[int] = None
    source: str = ""

    def as_detection(self) -> Tuple[int, int, int]:
        return (
            int(round(self.cx)),
            int(round(self.cy)),
            max(1, int(round(self.radius))),
        )


@dataclass
class StereoPairSelection:
    left: Detection
    right: Detection
    pair_ok: bool
    status: str
    hold_left: Detection = None
    hold_right: Detection = None


class TennisBallDetector:
    """
    Detects a single tennis ball in a BGR image using:
      1. HSV colour masking  →  isolates yellow-green blobs
      2. Morphological clean-up  →  removes noise
      3. HoughCircles  →  fits a circle to the blob

    Returns the best (most prominent) circle, or None.
    """

    def __init__(self):
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    def _tennis_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Patterned balls often have black/white seams that punch holes in the
        # colour mask, so close gaps more aggressively before fitting circles.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        return mask

    def detect_candidates(self, bgr_frame: np.ndarray,
                          stream_id: Optional[str] = None) -> List[DetectionCandidate]:
        mask = self._tennis_mask(bgr_frame)
        candidates: List[DetectionCandidate] = []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(20.0, math.pi * (MIN_RADIUS_PX ** 2) * 0.18)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < MIN_RADIUS_PX or radius > MAX_RADIUS_PX:
                continue

            circle_area = math.pi * radius * radius
            fill_ratio = area / max(circle_area, 1.0)
            circularity = (4.0 * math.pi * area) / max(perimeter * perimeter, 1.0)
            if circularity < 0.35 or fill_ratio < 0.18:
                continue

            size_score = min(1.0, radius / 40.0)
            confidence = min(
                1.0,
                0.55 * min(circularity, 1.0) +
                0.35 * min(fill_ratio, 1.0) +
                0.10 * size_score,
            )
            candidates.append(DetectionCandidate(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                confidence=confidence,
                class_name="tennis_ball",
                source="hsv_contour",
            ))

        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=50,
            param2=18,
            minRadius=MIN_RADIUS_PX,
            maxRadius=MAX_RADIUS_PX,
        )

        if circles is not None:
            for cx, cy, r in np.round(circles[0]).astype(int):
                if any(math.hypot(cx - c.cx, cy - c.cy) < max(5.0, r * 0.35)
                       for c in candidates):
                    continue
                candidates.append(DetectionCandidate(
                    cx=float(cx),
                    cy=float(cy),
                    radius=float(r),
                    confidence=min(0.90, 0.45 + (float(r) / MAX_RADIUS_PX) * 0.45),
                    class_name="tennis_ball",
                    source="hsv_hough",
                ))

        candidates.sort(
            key=lambda c: (c.confidence, c.radius),
            reverse=True,
        )
        return candidates

    def detect(self, bgr_frame: np.ndarray) -> Detection:
        candidates = self.detect_candidates(bgr_frame)
        return candidates[0].as_detection() if candidates else None

    @staticmethod
    def annotate(frame: np.ndarray, det: Detection,
                 circle_color=(0, 255, 0), text: str = "") -> np.ndarray:
        if det is None:
            return frame
        cx, cy, r = det
        cv2.circle(frame, (cx, cy), r, circle_color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        if text:
            cv2.putText(frame, text, (cx + r + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, circle_color, 2)
        return frame


class YOLOTennisBallDetector:
    """
    Optional Ultralytics YOLO detector. Returns the same (cx, cy, radius)
    tuple as TennisBallDetector so the stereo geometry stays unchanged.
    """

    def __init__(self, model_path: str, class_name: str,
                 confidence: float = 0.20, image_size: int = 640,
                 use_tracking: bool = True, tracker: str = "bytetrack.yaml"):
        if not model_path:
            raise ValueError("--yolo-model is required when using --detector yolo")

        self._validate_model_path(model_path)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Install it before using "
                "--detector yolo, or use the default --detector hsv mode."
            ) from exc

        self._YOLO = YOLO
        self.model_path = model_path
        self.model = YOLO(model_path)
        self._models = {"left": self.model}
        self.confidence = confidence
        self.image_size = image_size
        self.use_tracking = use_tracking
        self.tracker = tracker
        self._tracking_failed = False
        self._tracking_warning_printed = False
        self.target_class_id = self._resolve_class_id(class_name)

    @staticmethod
    def _validate_model_path(model_path: str):
        known_auto_models = {
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        }
        path = Path(model_path)
        looks_local = (
            path.parent != Path(".")
            or model_path.startswith(".")
            or model_path.startswith("/")
            or model_path not in known_auto_models
        )
        if model_path.endswith(".pt") and looks_local and not path.exists():
            raise ValueError(
                f"YOLO model file not found: {model_path}. "
                "Use a pretrained Ultralytics model such as `--yolo-model yolo11n.pt "
                "--yolo-class sports_ball`, or copy/train your custom model and pass "
                "its real path, for example `--yolo-model runs/detect/train/weights/best.pt`."
            )

    def _resolve_class_id(self, class_name: str) -> Optional[int]:
        if class_name is None or class_name.strip().lower() in ("", "any", "all"):
            return None
        if class_name.isdigit():
            return int(class_name)

        wanted = self._normalise_name(class_name)
        names = self.model.names
        if isinstance(names, dict):
            items = list(names.items())
        else:
            items = list(enumerate(names))

        for class_id, name in items:
            if self._normalise_name(str(name)) == wanted:
                return int(class_id)

        available = ", ".join(str(name) for _, name in items)
        raise ValueError(f"YOLO class '{class_name}' was not found. Available classes: {available}")

    @staticmethod
    def _normalise_name(name: str) -> str:
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    def _model_for_stream(self, stream_id: Optional[str]):
        if not self.use_tracking or self._tracking_failed:
            return self.model

        key = stream_id or "left"
        if key not in self._models:
            self._models[key] = self._YOLO(self.model_path)
        return self._models[key]

    def _run_model(self, bgr_frame: np.ndarray, stream_id: Optional[str]):
        model = self._model_for_stream(stream_id)
        if self.use_tracking and not self._tracking_failed:
            try:
                return model.track(
                    bgr_frame,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    persist=True,
                    tracker=self.tracker,
                    verbose=False,
                )
            except Exception as exc:
                self._tracking_failed = True
                if not self._tracking_warning_printed:
                    print(
                        "[YOLO] Tracking unavailable; falling back to per-frame "
                        f"detection. Reason: {exc}"
                    )
                    self._tracking_warning_printed = True

        return model.predict(
            bgr_frame,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )

    def detect_candidates(self, bgr_frame: np.ndarray,
                          stream_id: Optional[str] = None) -> List[DetectionCandidate]:
        results = self._run_model(bgr_frame, stream_id)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates: List[DetectionCandidate] = []
        track_ids = getattr(boxes, "id", None)
        names = self.model.names
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if self.target_class_id is not None and cls_id != self.target_class_id:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = max(w, h) / 2.0

            track_id = None
            if track_ids is not None:
                try:
                    track_id = int(track_ids[idx].item())
                except (AttributeError, TypeError, ValueError, IndexError):
                    track_id = None

            class_name = None
            if isinstance(names, dict):
                class_name = str(names.get(cls_id, cls_id))
            elif 0 <= cls_id < len(names):
                class_name = str(names[cls_id])

            candidates.append(DetectionCandidate(
                cx=cx,
                cy=cy,
                radius=radius,
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
                track_id=track_id,
                source="yolo_track" if track_id is not None else "yolo",
            ))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect(self, bgr_frame: np.ndarray) -> Detection:
        candidates = self.detect_candidates(bgr_frame)
        return candidates[0].as_detection() if candidates else None

    @staticmethod
    def annotate(frame: np.ndarray, det: Detection,
                 circle_color=(0, 255, 0), text: str = "") -> np.ndarray:
        return TennisBallDetector.annotate(frame, det, circle_color, text)


class StereoDetectionSmoother:
    """EMA smoothing for valid stereo detections only; reset on invalid/lost pairs."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.left = None
        self.right = None

    def reset(self):
        self.left = None
        self.right = None

    def _smooth_one(self, previous, current):
        if previous is None:
            return tuple(float(v) for v in current)
        return tuple(
            self.alpha * float(cur) + (1.0 - self.alpha) * float(prev)
            for prev, cur in zip(previous, current)
        )

    def update(self, left: Detection, right: Detection) -> Tuple[Detection, Detection]:
        self.left = self._smooth_one(self.left, left)
        self.right = self._smooth_one(self.right, right)
        return self._to_detection(self.left), self._to_detection(self.right)

    @staticmethod
    def _to_detection(values) -> Detection:
        if values is None:
            return None
        cx, cy, r = values
        return int(round(cx)), int(round(cy)), int(round(r))


class StereoCandidateMatcher:
    """
    Chooses the most plausible left/right ball pair from detector candidates.
    A short memory helps reacquire the same moving ball without outputting
    stale 3-D coordinates during detection loss.
    """

    def __init__(self, hold_frames: int = 6):
        self.hold_frames = hold_frames
        self.last_left: Detection = None
        self.last_right: Detection = None
        self.lost_frames = 0

    def reset(self):
        self.last_left = None
        self.last_right = None
        self.lost_frames = 0

    def select(self,
               left_candidates: Sequence[DetectionCandidate],
               right_candidates: Sequence[DetectionCandidate],
               max_y_diff: float) -> StereoPairSelection:
        if not left_candidates or not right_candidates:
            missing = []
            if not left_candidates:
                missing.append("LEFT")
            if not right_candidates:
                missing.append("RIGHT")
            return self._lost("Ball not detected in " + " ".join(missing))

        best = None
        best_score = -1e9
        last_reject = "Rejected pair: no valid stereo candidate"

        for left in left_candidates:
            left_det = left.as_detection()
            for right in right_candidates:
                right_det = right.as_detection()
                pair_ok, reason = validate_stereo_pair(
                    left_det,
                    right_det,
                    max_y_diff=max_y_diff,
                )
                if not pair_ok:
                    last_reject = reason
                    continue

                score = self._score_pair(left, right, left_det, right_det, max_y_diff)
                if score > best_score:
                    best_score = score
                    best = (left_det, right_det)

        if best is None:
            return self._lost(last_reject)

        self.last_left, self.last_right = best
        self.lost_frames = 0
        return StereoPairSelection(
            left=self.last_left,
            right=self.last_right,
            pair_ok=True,
            status="OK",
        )

    def _score_pair(self, left: DetectionCandidate, right: DetectionCandidate,
                    left_det: Tuple[int, int, int],
                    right_det: Tuple[int, int, int],
                    max_y_diff: float) -> float:
        xl, yl, rl = left_det
        xr, yr, rr = right_det
        y_diff = abs(float(yl - yr))
        radius_ratio = min(rl, rr) / max(rl, rr)

        score = left.confidence + right.confidence
        score += 0.25 * radius_ratio
        score -= 0.10 * (y_diff / max(max_y_diff, 1.0))

        if self.last_left is not None and self.last_right is not None:
            lpx, lpy, lpr = self.last_left
            rpx, rpy, rpr = self.last_right
            motion = math.hypot(xl - lpx, yl - lpy) + math.hypot(xr - rpx, yr - rpy)
            avg_radius = max((rl + rr + lpr + rpr) / 4.0, 1.0)
            score -= min(2.0, motion / (avg_radius * 8.0))

        return score

    def _lost(self, status: str) -> StereoPairSelection:
        self.lost_frames += 1
        hold_left = None
        hold_right = None
        if (
            self.hold_frames > 0
            and self.last_left is not None
            and self.last_right is not None
            and self.lost_frames <= self.hold_frames
        ):
            hold_left = self.last_left
            hold_right = self.last_right
            status = f"Reacquiring ({self.lost_frames}/{self.hold_frames}): {status}"
        elif self.lost_frames > self.hold_frames:
            self.last_left = None
            self.last_right = None

        return StereoPairSelection(
            left=None,
            right=None,
            pair_ok=False,
            status=status,
            hold_left=hold_left,
            hold_right=hold_right,
        )


def create_detector(args):
    if args.detector == "hsv":
        return TennisBallDetector()
    if args.detector == "yolo":
        return YOLOTennisBallDetector(
            model_path=args.yolo_model,
            class_name=args.yolo_class,
            confidence=args.yolo_conf,
            image_size=args.yolo_imgsz,
            use_tracking=not args.no_yolo_track,
            tracker=args.yolo_tracker,
        )
    raise ValueError(f"Unknown detector mode: {args.detector}")


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

    xl, yl, rl = left
    xr, yr, rr = right
    y_diff = abs(float(yl - yr))
    if y_diff > max_y_diff:
        return False, f"Rejected pair: epipolar y-diff {y_diff:.1f}px > {max_y_diff:.1f}px"

    disparity = float(xl - xr)
    if disparity <= 1.0:
        return False, f"Rejected pair: invalid disparity {disparity:.1f}px"

    radius_ratio = min(rl, rr) / max(rl, rr)
    if radius_ratio < 0.65:
        return False, f"Rejected pair: radius mismatch L={rl}px R={rr}px"

    return True, "OK"

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

    def __str__(self):
        msg = (f"X={self.X:+.3f} m  Y={self.Y:+.3f} m  Z={self.Z:.3f} m  "
               f"(disp={self.disparity:.1f} px | ydiff={self.epipolar_y_diff:.1f} px "
               f"| size_Z={self.size_Z:.3f} m)")
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
        }
        if self.known_Z is not None:
            record["known_Z"] = round(self.known_Z, 4)
            record["depth_error"] = round(self.depth_error, 4)
            record["depth_error_pct"] = round(self.depth_error_pct, 2)
        return record


class PositionSmoother:
    """EMA smoothing for valid 3-D positions only; reset on invalid/lost pairs."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.position = None

    def reset(self):
        self.position = None

    def update(self, current: BallPosition) -> BallPosition:
        if self.position is None:
            self.position = current
            return current

        prev = self.position
        a = self.alpha
        smoothed = BallPosition(
            X=a * current.X + (1.0 - a) * prev.X,
            Y=a * current.Y + (1.0 - a) * prev.Y,
            Z=a * current.Z + (1.0 - a) * prev.Z,
            disparity=a * current.disparity + (1.0 - a) * prev.disparity,
            size_Z=a * current.size_Z + (1.0 - a) * prev.size_Z,
            epipolar_y_diff=current.epipolar_y_diff,
            known_Z=current.known_Z,
        )
        if current.depth_error is not None and current.depth_error_pct is not None:
            smoothed.depth_error = smoothed.Z - current.known_Z
            smoothed.depth_error_pct = (smoothed.depth_error / current.known_Z) * 100.0

        self.position = smoothed
        return smoothed


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

        xl, yl, rl = left
        xr, yr, rr = right

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
        # Use only the horizontal component — T[0] is the true stereo baseline.
        # norm(T) would include tiny vertical/depth offsets and slightly overestimate.
        config.baseline = abs(float(config.T[0]))
    else:
        missing = ", ".join(k for k in required if k not in data)
        print(f"[Calibration] Warning: missing {missing}; rectification disabled.")

    print(f"[Calibration] Loaded from {path}")
    print(f"  fx={config.fx:.1f}  fy={config.fy:.1f}  "
          f"cx={config.cx:.1f}  cy={config.cy:.1f}  B={config.baseline*1000:.1f} mm")
    return config


def load_zed_factory_calibration(conf_path: str, config: CameraConfig,
                                  resolution: str = "HD") -> CameraConfig:
    """
    Parse a Stereolabs factory .conf file (e.g. SN28837104.conf) and populate
    the full set of intrinsics, distortion, and stereo extrinsics in OpenCV format.

    resolution choices: '2K'  (2208x1242)
                        'FHD' (1920x1080)
                        'HD'  (1280x720)   <- default, matches our capture size
                        'VGA' (672x376)

    After loading this file, full stereo rectification is enabled without
    needing to run calibrate_stereo.py first.
    """
    conf_file = Path(conf_path)
    if not conf_file.exists():
        raise FileNotFoundError(
            f"ZED factory calibration file not found: {conf_path}\n"
            "Download it from https://calib.stereolabs.com/?SN=<your_serial_number>"
        )

    parser = configparser.ConfigParser()
    parser.read(conf_path)

    left_key  = f"LEFT_CAM_{resolution}"
    right_key = f"RIGHT_CAM_{resolution}"

    available = parser.sections()
    if left_key not in available or right_key not in available:
        raise ValueError(
            f"Resolution '{resolution}' not found in {conf_path}.\n"
            f"Available sections: {available}\n"
            f"Choose from: 2K, FHD, HD, VGA"
        )

    def get(section, key):
        return float(parser[section][key])

    # -- Intrinsics -----------------------------------------------------------
    fx_l = get(left_key,  "fx");  fy_l = get(left_key,  "fy")
    cx_l = get(left_key,  "cx");  cy_l = get(left_key,  "cy")
    fx_r = get(right_key, "fx");  fy_r = get(right_key, "fy")
    cx_r = get(right_key, "cx");  cy_r = get(right_key, "cy")

    config.fx = fx_l
    config.fy = fy_l
    config.cx = cx_l
    config.cy = cy_l

    # -- Distortion (standard 5-parameter OpenCV model: k1,k2,p1,p2,k3) ------
    def distortion(section):
        return np.array([
            get(section, "k1"),
            get(section, "k2"),
            get(section, "p1"),
            get(section, "p2"),
            get(section, "k3"),
        ], dtype=np.float64)

    config.K_l = np.array([[fx_l, 0, cx_l],
                            [0, fy_l, cy_l],
                            [0,    0,    1]], dtype=np.float64)
    config.K_r = np.array([[fx_r, 0, cx_r],
                            [0, fy_r, cy_r],
                            [0,    0,    1]], dtype=np.float64)
    config.D_l = distortion(left_key)
    config.D_r = distortion(right_key)

    # -- Stereo extrinsics from [STEREO] section ------------------------------
    stereo = parser["STEREO"]
    baseline_mm = float(stereo["Baseline"])         # 120.144 mm
    ty_mm       = float(stereo.get("TY", "0"))      # vertical offset mm
    tz_mm       = float(stereo.get("TZ", "0"))      # depth offset mm

    # Rotation angles (radians): RX=pitch, CV=yaw/convergence, RZ=roll
    rx = float(stereo.get(f"RX_{resolution}", "0"))
    cv = float(stereo.get(f"CV_{resolution}", "0"))
    rz = float(stereo.get(f"RZ_{resolution}", "0"))

    # Build rotation matrix using Rodrigues (Stereolabs convention: pitch, yaw, roll)
    R_vec = np.array([rx, cv, rz], dtype=np.float64)
    config.R, _ = cv2.Rodrigues(R_vec)

    # Translation: [Baseline, TY, TZ] in metres (right camera relative to left)
    config.T = np.array([
        [baseline_mm / 1000.0],
        [ty_mm       / 1000.0],
        [tz_mm       / 1000.0],
    ], dtype=np.float64)

    config.baseline = baseline_mm / 1000.0

    # Image size hint for the rectifier intrinsic scaler
    res_sizes = {"2K": (2208, 1242), "FHD": (1920, 1080),
                 "HD": (1280, 720),  "VGA": (672, 376)}
    if resolution in res_sizes:
        config.calibration_image_size = res_sizes[resolution]

    print(f"[ZED Factory Cal] Loaded {conf_path} at {resolution}")
    print(f"  Left  fx={fx_l:.3f} fy={fy_l:.3f} cx={cx_l:.3f} cy={cy_l:.3f}")
    print(f"  Right fx={fx_r:.3f} fy={fy_r:.3f} cx={cx_r:.3f} cy={cy_r:.3f}")
    print(f"  Baseline={baseline_mm:.3f} mm  RX={rx:.6f}  CV={cv:.6f}  RZ={rz:.6f}")
    print(f"  Distortion D_l=[{', '.join(f'{v:.6f}' for v in config.D_l)}]")
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
        txt = f"X:{pos.X:+.3f}m  Y:{pos.Y:+.3f}m  Z:{pos.Z:.3f}m  disp:{pos.disparity:.1f}px"
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
                   help="Pixel format (default MJPG — avoids green-screen on ZED)")
    p.add_argument("--zed-calibration", type=str, default=None,
                   help="Path to Stereolabs factory .conf file (e.g. SN28837104.conf). "
                        "Enables full rectification without a checkerboard.")
    p.add_argument("--zed-resolution", type=str, default="HD",
                   choices=["2K", "FHD", "HD", "VGA"],
                   help="Resolution key in the .conf file (default HD = 1280x720)")
    p.add_argument("--calibration", type=str,   default=None,
                   help="Path to .npz calibration file from calibrate_stereo.py "
                        "(applied on top of --zed-calibration if both are given)")
    p.add_argument("--no-display",  action="store_true",
                   help="Disable OpenCV window (use for headless SSH sessions)")
    p.add_argument("--log",         type=str,   default=None,
                   help="Append JSON position records to this file")
    p.add_argument("--known-distance", type=float, default=None,
                   help="Known measured ball distance in metres for depth-error checks")
    p.add_argument("--max-epipolar-y-diff", type=float, default=MAX_EPIPOLAR_Y_DIFF_PX,
                   help="Reject left/right matches with larger rectified y difference")
    p.add_argument("--detector", choices=("hsv", "yolo"), default="hsv",
                   help="Detector backend: hsv is lightweight fallback, yolo is optional")
    p.add_argument("--yolo-model", type=str, default=None,
                   help="Path/name of YOLO model for --detector yolo, such as best.pt")
    p.add_argument("--yolo-conf", type=float, default=0.20,
                   help="YOLO confidence threshold (default 0.20)")
    p.add_argument("--yolo-class", type=str, default="sports_ball",
                   help="YOLO class name or id to use (default sports_ball)")
    p.add_argument("--yolo-imgsz", type=int, default=640,
                   help="YOLO inference image size (default 640)")
    p.add_argument("--no-yolo-track", action="store_true",
                   help="Disable Ultralytics tracking and use per-frame YOLO prediction")
    p.add_argument("--yolo-tracker", type=str, default="bytetrack.yaml",
                   help="Ultralytics tracker config for YOLO mode (default bytetrack.yaml)")
    p.add_argument("--smooth-alpha", type=float, default=0.35,
                   help="EMA smoothing factor for valid detections/positions (0..1, default 0.35)")
    p.add_argument("--track-hold-frames", type=int, default=6,
                   help="Frames to show last detection while reacquiring; no stale 3-D output (default 6)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.width % 2 != 0:
        raise ValueError("--width must be even because the stereo frame is split in half")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0.0 <= args.yolo_conf <= 1.0:
        raise ValueError("--yolo-conf must be between 0 and 1")
    if args.yolo_imgsz <= 0:
        raise ValueError("--yolo-imgsz must be positive")
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError("--smooth-alpha must be between 0 and 1")
    if args.track_hold_frames < 0:
        raise ValueError("--track-hold-frames must be zero or greater")
    if args.known_distance is not None and args.known_distance <= 0:
        raise ValueError("--known-distance must be a positive distance in metres")

    try:
        detector = create_detector(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"[Detector] {exc}") from None

    # ── Setup ──────────────────────────────────────────────────────────────
    # Priority: hardcoded ZED 2 factory defaults -> optional factory .conf
    # override -> optional checkerboard .npz override.
    config = create_zed2_sn28837104_config(
        frame_width=args.width,
        frame_height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        verbose=not args.zed_calibration,
    )
    if args.zed_calibration:
        config = load_zed_factory_calibration(
            args.zed_calibration, config, resolution=args.zed_resolution
        )
    if args.calibration:
        config = load_calibration(args.calibration, config)

    camera       = ZEDCamera(device_id=args.device, config=config)
    rectifier    = StereoRectifier(camera.cfg)
    triangulator = StereoTriangulator(config=camera.cfg)  # use updated config
    detection_smoother = StereoDetectionSmoother(args.smooth_alpha)
    position_smoother = PositionSmoother(args.smooth_alpha)
    stereo_matcher = StereoCandidateMatcher(args.track_hold_frames)

    log_file = open(args.log, "a") if args.log else None

    # ── FPS tracking ───────────────────────────────────────────────────────
    fps        = 0.0
    t_fps      = time.time()
    frame_idx  = 0

    print(
        f"\n[Ready] detector={args.detector} smooth_alpha={args.smooth_alpha:.2f} "
        f"hold_frames={args.track_hold_frames}"
    )
    print("Press 'q' to quit, 's' to save current frame.\n")
    if args.known_distance is not None:
        print(f"[Depth Check] Comparing Z against known distance {args.known_distance:.3f} m")

    try:
      while True:
        left, right = camera.read()
        if left is None:
            print("[Error] Failed to capture frame.")
            time.sleep(0.1)
            continue

        left, right = rectifier.rectify(left, right)

        # ── Detect ──────────────────────────────────────────────────────
        left_candidates = detector.detect_candidates(left, stream_id="left")
        right_candidates = detector.detect_candidates(right, stream_id="right")

        selection = stereo_matcher.select(
            left_candidates,
            right_candidates,
            max_y_diff=args.max_epipolar_y_diff,
        )
        left_raw_det = selection.left
        right_raw_det = selection.right
        pair_ok = selection.pair_ok
        status_message = selection.status

        left_det = None
        right_det = None
        if pair_ok:
            left_det, right_det = detection_smoother.update(left_raw_det, right_raw_det)
            pair_ok, status_message = validate_stereo_pair(
                left_det,
                right_det,
                max_y_diff=args.max_epipolar_y_diff,
            )
            if not pair_ok:
                detection_smoother.reset()
                position_smoother.reset()
        else:
            detection_smoother.reset()
            position_smoother.reset()

        # ── Triangulate ─────────────────────────────────────────────────
        raw_position = (
            triangulator.triangulate(
                left_det,
                right_det,
                known_depth_m=args.known_distance,
            )
            if pair_ok else None
        )
        position = position_smoother.update(raw_position) if raw_position else None

        # ── Log / print ─────────────────────────────────────────────────
        if position:
            print(f"[Frame {frame_idx:05d}] {position}")
            if log_file:
                record = {
                    "frame": frame_idx,
                    "t": time.time(),
                    "rectified": rectifier.enabled,
                    "status": status_message,
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
            detector.annotate(left,  left_raw_det,  circle_color=(120, 120, 120), text="raw L")
            detector.annotate(right, right_raw_det, circle_color=(120, 120, 120), text="raw R")
            if not pair_ok:
                detector.annotate(left,  selection.hold_left,  circle_color=(255, 180, 0), text="hold L")
                detector.annotate(right, selection.hold_right, circle_color=(255, 180, 0), text="hold R")
            detector.annotate(left,  left_det,  circle_color=(0, 255, 0),   text="L")
            detector.annotate(right, right_det, circle_color=(0, 200, 255), text="R")

            # Combine side by side and downscale for Pi's display
            combined = np.hstack([left, right])
            combined = cv2.resize(combined, (1280, 360))
            overlay_info(combined, position, fps, status_message)

            cv2.imshow("ZED Ball Detection", combined)
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

    except KeyboardInterrupt:
        print("\n[Interrupted] Shutting down.")
    finally:
        # ── Cleanup ──────────────────────────────────────────────────────────
        camera.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        print("Done.")


if __name__ == "__main__":
    main()
