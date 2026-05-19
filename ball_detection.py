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
  pip install opencv-python numpy

Usage:
  python ball_detection.py
  python ball_detection.py --calibration my_calibration.npz
  python ball_detection.py --no-display      (headless / SSH mode)
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

    @property
    def eye_width(self) -> int:
        return self.frame_width // 2


# ── Tennis ball HSV colour range ──────────────────────────────────────────────
# Tennis balls are yellow-green; tweak if lighting differs significantly.
HSV_LOWER = np.array([22,  80,  80])
HSV_UPPER = np.array([65, 255, 255])

# ── Hough / geometry limits ───────────────────────────────────────────────────
MIN_RADIUS_PX = 8    # Ignore circles smaller than this (noise)
MAX_RADIUS_PX = 150  # Ignore circles larger than this

TENNIS_BALL_RADIUS_M = 0.0335   # Physical radius ≈ 33.5 mm (ITF standard)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LAYER
# ══════════════════════════════════════════════════════════════════════════════

class ZEDCamera:
    """
    Opens the ZED as a standard USB UVC device and splits each frame into
    left / right eye images.  No ZED SDK or CUDA required.
    """

    def __init__(self, device_id: int = 0, config: CameraConfig = CameraConfig()):
        self.cfg = config
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(device_id, backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Disable auto-exposure to keep colour stable
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at device {device_id}. "
                "Check USB connection and camera permissions."
            )

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ZED] Opened at {actual_w}×{actual_h} "
              f"(eye: {actual_w // 2}×{actual_h})")

        # Update config to match what the camera actually gave us
        self.cfg.frame_width  = actual_w
        self.cfg.frame_height = actual_h
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
#  TRIANGULATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BallPosition:
    X: float          # metres, positive = right of camera centre
    Y: float          # metres, positive = below camera centre
    Z: float          # metres, positive = forward
    disparity: float  # pixels (diagnostic)
    size_Z: float     # metres, depth estimated from apparent ball size (cross-check)

    def __str__(self):
        return (f"X={self.X:+.3f} m  Y={self.Y:+.3f} m  Z={self.Z:.3f} m  "
                f"(disp={self.disparity:.1f} px | size_Z={self.size_Z:.3f} m)")

    def to_dict(self):
        return {"X": round(self.X, 4), "Y": round(self.Y, 4),
                "Z": round(self.Z, 4), "disparity": round(self.disparity, 2),
                "size_Z": round(self.size_Z, 4)}


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

    def triangulate(self, left: Detection, right: Detection) -> Optional[BallPosition]:
        if left is None or right is None:
            return None

        xl, yl, rl = left
        xr, yr, _  = right

        disparity = float(xl - xr)
        if disparity <= 1.0:
            # Disparity too small → unreliable or behind camera
            return None

        Z = (self.cfg.fx * self.cfg.baseline) / disparity
        X = (xl - self.cfg.cx) * Z / self.cfg.fx
        Y = (yl - self.cfg.cy) * Z / self.cfg.fy

        size_Z = (self.cfg.fx * TENNIS_BALL_RADIUS_M) / max(rl, 1)

        return BallPosition(X=X, Y=Y, Z=Z, disparity=disparity, size_Z=size_Z)


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(path: str, config: CameraConfig) -> CameraConfig:
    """
    Load stereo calibration saved by calibrate_stereo.py and update config.
    File format: NumPy .npz with keys  fx, fy, cx, cy, baseline
    """
    data = np.load(path)
    config.fx       = float(data["fx"])
    config.fy       = float(data["fy"])
    config.cx       = float(data["cx"])
    config.cy       = float(data["cy"])
    config.baseline = float(data["baseline"])
    print(f"[Calibration] Loaded from {path}")
    print(f"  fx={config.fx:.1f}  fy={config.fy:.1f}  "
          f"cx={config.cx:.1f}  cy={config.cy:.1f}  B={config.baseline*1000:.1f} mm")
    return config


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def overlay_info(frame: np.ndarray, pos: Optional[BallPosition],
                 fps: float, left_det: Detection, right_det: Detection):
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    if pos:
        txt = f"X:{pos.X:+.3f}m  Y:{pos.Y:+.3f}m  Z:{pos.Z:.3f}m  disp:{pos.disparity:.1f}px"
        cv2.putText(frame, txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 80), 2)
    else:
        reason = "Ball not detected in "
        reason += "LEFT " if left_det  is None else ""
        reason += "RIGHT" if right_det is None else ""
        cv2.putText(frame, reason, (10, 22),
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
    p.add_argument("--calibration", type=str,   default=None,
                   help="Path to .npz calibration file from calibrate_stereo.py")
    p.add_argument("--no-display",  action="store_true",
                   help="Disable OpenCV window (use for headless SSH sessions)")
    p.add_argument("--log",         type=str,   default=None,
                   help="Append JSON position records to this file")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    config = CameraConfig()
    if args.calibration:
        config = load_calibration(args.calibration, config)

    camera       = ZEDCamera(device_id=args.device, config=config)
    detector     = TennisBallDetector()
    triangulator = StereoTriangulator(config=camera.cfg)  # use updated config

    log_file = open(args.log, "a") if args.log else None

    # ── FPS tracking ───────────────────────────────────────────────────────
    fps        = 0.0
    t_fps      = time.time()
    frame_idx  = 0

    print("\n[Ready] Press 'q' to quit, 's' to save current frame.\n")

    while True:
        left, right = camera.read()
        if left is None:
            print("[Error] Failed to capture frame.")
            time.sleep(0.1)
            continue

        # ── Detect ──────────────────────────────────────────────────────
        left_det  = detector.detect(left)
        right_det = detector.detect(right)

        # ── Triangulate ─────────────────────────────────────────────────
        position = triangulator.triangulate(left_det, right_det)

        # ── Log / print ─────────────────────────────────────────────────
        if position:
            print(f"[Frame {frame_idx:05d}] {position}")
            if log_file:
                record = {"frame": frame_idx, "t": time.time(), **position.to_dict()}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

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
            overlay_info(combined, position, fps, left_det, right_det)

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
