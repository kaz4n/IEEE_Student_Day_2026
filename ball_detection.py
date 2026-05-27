"""
Tennis Ball 3D Detection — ZED Stereo Camera on Raspberry Pi
============================================================
Uses the ZED as a standard USB stereo camera (no ZED SDK / CUDA needed).
Detects tennis ball via HSV color filtering + Hough circles in both eyes,
then triangulates to get real-world (X, Y, Z) coordinates in meters.

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
from typing import Optional, Tuple


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
    fourcc: Optional[str] = None

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


class TennisBallDetector:
    """
    Detects a single tennis ball in a BGR image using:
      1. HSV colour masking  →  isolates yellow-green blobs
      2. Morphological clean-up  →  removes noise
      3. HoughCircles  →  fits a circle to the blob

    Returns the best (most prominent) circle, or None.
    """

    def __init__(self):
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, bgr_frame: np.ndarray) -> Detection:
        hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Clean up small holes and salt-and-pepper noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        mask = cv2.GaussianBlur(mask, (9, 9), 2)

        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=50,
            param2=18,
            minRadius=MIN_RADIUS_PX,
            maxRadius=MAX_RADIUS_PX,
        )

        if circles is None:
            return None

        # Pick the largest circle (most likely the real ball, not glare)
        circles = np.round(circles[0]).astype(int)
        cx, cy, r = max(circles, key=lambda c: c[2])
        return int(cx), int(cy), int(r)

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
                   help="V4L2 device index (default 0 → /dev/video0)")
    p.add_argument("--width",       type=int,   default=2560,
                   help="Requested full side-by-side capture width (default 2560)")
    p.add_argument("--height",      type=int,   default=720,
                   help="Requested capture height (default 720)")
    p.add_argument("--fps",         type=float, default=30.0,
                   help="Requested camera frame rate (default 30)")
    p.add_argument("--fourcc",      type=str,   default=None,
                   help="Optional four-character pixel format request, such as MJPG")
    p.add_argument("--calibration", type=str,   default=None,
                   help="Path to .npz calibration file from calibrate_stereo.py")
    p.add_argument("--no-display",  action="store_true",
                   help="Disable OpenCV window (use for headless SSH sessions)")
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

    # ── Setup ──────────────────────────────────────────────────────────────
    config = CameraConfig(
        frame_width=args.width,
        frame_height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    if args.calibration:
        config = load_calibration(args.calibration, config)

    camera       = ZEDCamera(device_id=args.device, config=config)
    rectifier    = StereoRectifier(camera.cfg)
    detector     = TennisBallDetector()
    triangulator = StereoTriangulator(config=camera.cfg)  # use updated config

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

        # ── Detect ──────────────────────────────────────────────────────
        left_det  = detector.detect(left)
        right_det = detector.detect(right)

        pair_ok, status_message = validate_stereo_pair(
            left_det,
            right_det,
            max_y_diff=args.max_epipolar_y_diff,
        )

        # ── Triangulate ─────────────────────────────────────────────────
        position = (
            triangulator.triangulate(
                left_det,
                right_det,
                known_depth_m=args.known_distance,
            )
            if pair_ok else None
        )

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

    # ── Cleanup ─────────────────────────────────────────────────────────────
    camera.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
    print("Done.")


if __name__ == "__main__":
    main()
