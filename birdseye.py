"""
Bird's Eye View for tennis-ball tracking.

This module consumes the BallPosition objects produced by ball_detection.py and
renders an overhead X/Z map. The vertical Y axis is intentionally ignored.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


CANVAS_W = 500
CANVAS_H = 600
TRAIL_LENGTH = 60
MIN_POINTS_FOR_PREDICTION = 4
VELOCITY_WINDOW = 8
INTERCEPT_Z_M = 0.30
MAX_MISSED_FRAMES = 8

BG_COLOR = (24, 31, 28)
GRID_COLOR = (68, 82, 76)
AXIS_COLOR = (120, 145, 135)
TEXT_COLOR = (230, 238, 235)
MUTED_TEXT_COLOR = (156, 170, 164)
C_BALL = (255, 220, 70)
C_TRAIL_OLD = (44, 100, 58)
C_TRAIL_NEW = (90, 240, 130)
C_VELOCITY = (0, 220, 255)
C_INTERCEPT_LINE = (255, 145, 60)
C_INTERCEPT = (70, 95, 255)
C_ROBOT = (245, 245, 245)


@dataclass
class TrackPoint:
    x: float
    z: float
    t: float


@dataclass
class TrajectoryEstimate:
    vx: float
    vz: float
    intercept_x: Optional[float]
    intercept_valid: bool
    speed_mps: float


class BallTracker:
    """Stores X/Z history and estimates motion in the floor plane."""

    def __init__(self, trail_length: int = TRAIL_LENGTH,
                 velocity_window: int = VELOCITY_WINDOW,
                 intercept_z: float = INTERCEPT_Z_M):
        self.trail_length = trail_length
        self.velocity_window = velocity_window
        self.intercept_z = intercept_z
        self.trail: List[TrackPoint] = []
        self.trajectory: Optional[TrajectoryEstimate] = None
        self.missed_frames = 0

    @property
    def has_data(self) -> bool:
        return bool(self.trail)

    @property
    def last_position(self) -> Optional[TrackPoint]:
        return self.trail[-1] if self.trail else None

    def update(self, x: Optional[float], z: Optional[float],
               timestamp: Optional[float] = None):
        if x is None or z is None:
            self.missed_frames += 1
            if self.missed_frames >= MAX_MISSED_FRAMES:
                self.trail.clear()
                self.trajectory = None
            return

        t = time.monotonic() if timestamp is None else timestamp
        self.missed_frames = 0
        self.trail.append(TrackPoint(float(x), float(z), float(t)))
        if len(self.trail) > self.trail_length:
            self.trail = self.trail[-self.trail_length:]
        self.trajectory = self._estimate_trajectory()

    def _estimate_trajectory(self) -> Optional[TrajectoryEstimate]:
        if len(self.trail) < MIN_POINTS_FOR_PREDICTION:
            return None

        points = self.trail[-self.velocity_window:]
        if len(points) < MIN_POINTS_FOR_PREDICTION:
            return None

        t0 = points[0].t
        ts = np.array([p.t - t0 for p in points], dtype=np.float64)
        xs = np.array([p.x for p in points], dtype=np.float64)
        zs = np.array([p.z for p in points], dtype=np.float64)
        if float(ts[-1] - ts[0]) < 1e-3:
            return None

        vx = float(np.polyfit(ts, xs, 1)[0])
        vz = float(np.polyfit(ts, zs, 1)[0])
        speed = float(math.hypot(vx, vz))

        last = self.last_position
        intercept_x = None
        intercept_valid = False
        if last is not None and abs(vz) > 1e-6:
            t_intercept = (self.intercept_z - last.z) / vz
            if t_intercept > 0:
                intercept_x = last.x + vx * t_intercept
                intercept_valid = True

        return TrajectoryEstimate(
            vx=vx,
            vz=vz,
            intercept_x=intercept_x,
            intercept_valid=intercept_valid,
            speed_mps=speed,
        )


class CoordMapper:
    """Converts camera-centred X/Z metres into canvas pixels."""

    def __init__(self, arena_x_m: float, arena_z_m: float,
                 canvas_w: int, canvas_h: int, margin: int = 44):
        self.arena_x_m = arena_x_m
        self.arena_z_m = arena_z_m
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.margin = margin
        usable_w = max(1, canvas_w - 2 * margin)
        usable_h = max(1, canvas_h - 2 * margin)
        self.scale = min(usable_w / arena_x_m, usable_h / arena_z_m)
        self.origin_u = canvas_w // 2
        self.origin_v = canvas_h - margin

    def to_px(self, x_m: float, z_m: float) -> Tuple[int, int]:
        u = int(round(self.origin_u + x_m * self.scale))
        v = int(round(self.origin_v - z_m * self.scale))
        return u, v


class BirdsEyeRenderer:
    """Draws the tracker state into an OpenCV BGR image."""

    def __init__(self, arena_x_m: float = 4.0, arena_z_m: float = 4.0,
                 canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H,
                 intercept_z: float = INTERCEPT_Z_M):
        self.arena_x_m = arena_x_m
        self.arena_z_m = arena_z_m
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.intercept_z = intercept_z
        self.mapper = CoordMapper(arena_x_m, arena_z_m, canvas_w, canvas_h)

    def render(self, tracker: BallTracker) -> np.ndarray:
        frame = np.full((self.canvas_h, self.canvas_w, 3), BG_COLOR, dtype=np.uint8)
        self._draw_grid(frame)
        self._draw_intercept_line(frame)
        self._draw_robot(frame)
        self._draw_trail(frame, tracker.trail)
        self._draw_current_ball(frame, tracker.last_position)
        self._draw_trajectory(frame, tracker)
        self._draw_hud(frame, tracker)
        return frame

    def _draw_grid(self, frame: np.ndarray):
        half_x = self.arena_x_m / 2.0
        x = -math.floor(half_x)
        while x <= half_x + 1e-6:
            u1, v1 = self.mapper.to_px(x, 0.0)
            u2, v2 = self.mapper.to_px(x, self.arena_z_m)
            color = AXIS_COLOR if abs(x) < 1e-6 else GRID_COLOR
            cv2.line(frame, (u1, v1), (u2, v2), color, 1)
            cv2.putText(frame, f"{x:+.0f}m", (u1 - 18, v1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, MUTED_TEXT_COLOR, 1)
            x += 1.0

        z = 0.0
        while z <= self.arena_z_m + 1e-6:
            u1, v1 = self.mapper.to_px(-half_x, z)
            u2, v2 = self.mapper.to_px(half_x, z)
            color = AXIS_COLOR if abs(z) < 1e-6 else GRID_COLOR
            cv2.line(frame, (u1, v1), (u2, v2), color, 1)
            cv2.putText(frame, f"Z={z:.0f}m", (8, v1 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED_TEXT_COLOR, 1)
            z += 1.0

    def _draw_intercept_line(self, frame: np.ndarray):
        half_x = self.arena_x_m / 2.0
        start = self.mapper.to_px(-half_x, self.intercept_z)
        end = self.mapper.to_px(half_x, self.intercept_z)
        self._dashed_line(frame, start, end, C_INTERCEPT_LINE, dash_len=10, gap_len=7)
        cv2.putText(frame, f"panel Z={self.intercept_z:.2f}m",
                    (start[0] + 6, start[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_INTERCEPT_LINE, 1)

    def _draw_robot(self, frame: np.ndarray):
        u, v = self.mapper.to_px(0.0, 0.0)
        cv2.rectangle(frame, (u - 34, v - 18), (u + 34, v + 18), C_ROBOT, 2)
        cv2.line(frame, (u, v - 24), (u, v - 44), C_ROBOT, 2)
        cv2.putText(frame, "ROBOT", (u - 27, v + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    def _draw_trail(self, frame: np.ndarray, trail: List[TrackPoint]):
        if len(trail) < 2:
            return
        count = len(trail)
        for i in range(1, count):
            p0 = trail[i - 1]
            p1 = trail[i]
            alpha = i / max(count - 1, 1)
            color = tuple(
                int(C_TRAIL_OLD[c] * (1.0 - alpha) + C_TRAIL_NEW[c] * alpha)
                for c in range(3)
            )
            cv2.line(frame, self.mapper.to_px(p0.x, p0.z),
                     self.mapper.to_px(p1.x, p1.z), color, 2)

    def _draw_current_ball(self, frame: np.ndarray, point: Optional[TrackPoint]):
        if point is None:
            return
        u, v = self.mapper.to_px(point.x, point.z)
        cv2.circle(frame, (u, v), 8, C_BALL, -1)
        cv2.circle(frame, (u, v), 11, (35, 35, 35), 2)

    def _draw_trajectory(self, frame: np.ndarray, tracker: BallTracker):
        point = tracker.last_position
        traj = tracker.trajectory
        if point is None or traj is None:
            return

        start = self.mapper.to_px(point.x, point.z)
        end = self.mapper.to_px(point.x + traj.vx * 0.5, point.z + traj.vz * 0.5)
        cv2.arrowedLine(frame, start, end, C_VELOCITY, 2, tipLength=0.25)

        if traj.intercept_valid and traj.intercept_x is not None:
            intercept = self.mapper.to_px(traj.intercept_x, self.intercept_z)
            self._dashed_line(frame, start, intercept, C_INTERCEPT, dash_len=8, gap_len=6)
            diamond = np.array([
                [intercept[0], intercept[1] - 9],
                [intercept[0] + 9, intercept[1]],
                [intercept[0], intercept[1] + 9],
                [intercept[0] - 9, intercept[1]],
            ], dtype=np.int32)
            cv2.fillConvexPoly(frame, diamond, C_INTERCEPT)
            cv2.putText(frame, f"X={traj.intercept_x:+.2f}m",
                        (intercept[0] + 10, intercept[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_INTERCEPT, 1)

    def _draw_hud(self, frame: np.ndarray, tracker: BallTracker):
        y0 = self.canvas_h - 70
        cv2.rectangle(frame, (0, y0), (self.canvas_w, self.canvas_h), (18, 23, 21), -1)
        point = tracker.last_position
        traj = tracker.trajectory
        if point is None:
            cv2.putText(frame, "No ball position", (14, y0 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, MUTED_TEXT_COLOR, 1)
            return

        cv2.putText(frame, f"Ball X={point.x:+.3f}m  Z={point.z:.3f}m",
                    (14, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1)
        if traj is None:
            cv2.putText(frame, "Collecting trajectory...",
                        (14, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED_TEXT_COLOR, 1)
            return

        state = "APPROACHING" if traj.vz < 0 else "RECEDING"
        intercept_text = (
            f"intercept X={traj.intercept_x:+.3f}m"
            if traj.intercept_valid and traj.intercept_x is not None
            else "intercept X=None"
        )
        cv2.putText(frame,
                    f"Vx={traj.vx:+.2f}  Vz={traj.vz:+.2f} m/s  {state}",
                    (14, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
        cv2.putText(frame, intercept_text,
                    (14, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_INTERCEPT_LINE, 1)

    @staticmethod
    def _dashed_line(frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
                     color: Tuple[int, int, int], dash_len: int = 8, gap_len: int = 5):
        x1, y1 = start
        x2, y2 = end
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1.0:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        travelled = 0.0
        while travelled < dist:
            a = travelled
            b = min(travelled + dash_len, dist)
            p0 = (int(round(x1 + dx * a)), int(round(y1 + dy * a)))
            p1 = (int(round(x1 + dx * b)), int(round(y1 + dy * b)))
            cv2.line(frame, p0, p1, color, 1)
            travelled += dash_len + gap_len


class BirdsEyeOverlay:
    """Small wrapper used by ball_detection.py."""

    def __init__(self, arena_x_m: float = 4.0, arena_z_m: float = 4.0,
                 intercept_z: float = INTERCEPT_Z_M):
        self.tracker = BallTracker(intercept_z=intercept_z)
        self.renderer = BirdsEyeRenderer(
            arena_x_m=arena_x_m,
            arena_z_m=arena_z_m,
            intercept_z=intercept_z,
        )
        self.frame = self.renderer.render(self.tracker)

    @property
    def intercept_x(self) -> Optional[float]:
        traj = self.tracker.trajectory
        if traj is None or not traj.intercept_valid:
            return None
        return traj.intercept_x

    def update(self, position, timestamp: Optional[float] = None) -> np.ndarray:
        if position is None:
            self.tracker.update(None, None)
        else:
            self.tracker.update(position.X, position.Z, timestamp=timestamp)
        self.frame = self.renderer.render(self.tracker)
        return self.frame


class _DemoPosition:
    def __init__(self, x: float, z: float):
        self.X = x
        self.Z = z


def _position_from_record(record: dict) -> Tuple[Optional[_DemoPosition], Optional[float]]:
    source = record.get("position", record)
    if not isinstance(source, dict):
        return None, None

    x = source.get("X", source.get("x"))
    z = source.get("Z", source.get("z"))
    if x is None or z is None:
        return None, None

    timestamp = record.get("t", source.get("t"))
    try:
        timestamp = float(timestamp) if timestamp is not None else None
        return _DemoPosition(float(x), float(z)), timestamp
    except (TypeError, ValueError):
        return None, None


def _read_jsonl_position(line: str) -> Tuple[Optional[_DemoPosition], Optional[float]]:
    line = line.strip()
    if not line:
        return None, None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, None
    return _position_from_record(record)


def _show_overlay_frame(overlay: BirdsEyeOverlay, window_title: str,
                        wait_ms: int = 30) -> int:
    cv2.imshow(window_title, overlay.frame)
    return cv2.waitKey(wait_ms) & 0xFF


def run_log_follow(path: str, replay: bool = False, intercept_z: float = INTERCEPT_Z_M,
                   arena_x_m: float = 4.0, arena_z_m: float = 4.0,
                   stale_timeout_s: float = 0.75):
    log_path = Path(path)
    overlay = BirdsEyeOverlay(
        arena_x_m=arena_x_m,
        arena_z_m=arena_z_m,
        intercept_z=intercept_z,
    )
    window_title = "Bird's Eye View - live log"
    print(f"[Bird's Eye] Reading real positions from {log_path}")
    print("[Bird's Eye] Press 'q' to quit.")

    while not log_path.exists():
        key = _show_overlay_frame(overlay, window_title, wait_ms=100)
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        print(f"[Bird's Eye] Waiting for log file: {log_path}")
        time.sleep(0.5)

    with log_path.open("r", encoding="utf-8") as file:
        if not replay:
            file.seek(0, 2)

        last_update = time.monotonic()
        while True:
            line = file.readline()
            if line:
                position, timestamp = _read_jsonl_position(line)
                if position is not None:
                    overlay.update(position, timestamp=timestamp)
                    last_update = time.monotonic()
            else:
                if replay:
                    break
                if time.monotonic() - last_update > stale_timeout_s:
                    overlay.update(None)
                    last_update = time.monotonic()
                time.sleep(0.03)

            key = _show_overlay_frame(overlay, window_title, wait_ms=1)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


def run_demo(intercept_z: float = INTERCEPT_Z_M,
             arena_x_m: float = 4.0, arena_z_m: float = 4.0):
    overlay = BirdsEyeOverlay(
        arena_x_m=arena_x_m,
        arena_z_m=arena_z_m,
        intercept_z=intercept_z,
    )
    t0 = time.monotonic()
    print("[Bird's Eye Demo] SIMULATED path only. It does not read the camera.")
    print("[Bird's Eye Demo] Press 'q' to quit, 'r' to reset.")
    offset_x = -0.8
    speed_x = 0.18
    speed_z = -0.75
    start_z = 3.7

    while True:
        t = time.monotonic() - t0
        x = offset_x + speed_x * t
        z = start_z + speed_z * t
        if z < 0.05:
            t0 = time.monotonic()
            continue
        frame = overlay.update(_DemoPosition(x, z))
        cv2.putText(frame, "SIMULATED DEMO - no camera input", (92, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 180, 255), 1)
        overlay.frame = frame
        cv2.imshow("Bird's Eye View - demo", overlay.frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            t0 = time.monotonic()

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Bird's eye view overlay")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true",
                        help="Run a simulated rolling-ball demo; does not read the camera")
    source.add_argument("--follow-log", type=str, default=None,
                        help="Follow a JSONL log produced by ball_detection.py --log")
    source.add_argument("--replay-log", type=str, default=None,
                        help="Replay positions from an existing JSONL log")
    parser.add_argument("--intercept-z", type=float, default=INTERCEPT_Z_M,
                        help="Z distance in metres where panel intercept is predicted")
    parser.add_argument("--arena-x", type=float, default=4.0,
                        help="Arena width shown in metres")
    parser.add_argument("--arena-z", type=float, default=4.0,
                        help="Arena depth shown in metres")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.intercept_z <= 0:
        raise ValueError("--intercept-z must be positive")
    if args.arena_x <= 0 or args.arena_z <= 0:
        raise ValueError("--arena-x and --arena-z must be positive")

    if args.demo:
        run_demo(
            intercept_z=args.intercept_z,
            arena_x_m=args.arena_x,
            arena_z_m=args.arena_z,
        )
    elif args.follow_log:
        run_log_follow(
            args.follow_log,
            replay=False,
            intercept_z=args.intercept_z,
            arena_x_m=args.arena_x,
            arena_z_m=args.arena_z,
        )
    elif args.replay_log:
        run_log_follow(
            args.replay_log,
            replay=True,
            intercept_z=args.intercept_z,
            arena_x_m=args.arena_x,
            arena_z_m=args.arena_z,
        )
    else:
        print("No live data source selected.")
        print("For real camera data, run: python ball_detection.py --birdseye")
        print("For a separate viewer, run ball_detection.py with --log positions.jsonl,")
        print("then run: python birdseye.py --follow-log positions.jsonl")
        print("For fake motion only, run: python birdseye.py --demo")


if __name__ == "__main__":
    main()
