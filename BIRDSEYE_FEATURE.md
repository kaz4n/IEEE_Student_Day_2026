# Bird's Eye View — Feature Documentation

**Module:** `birdseye.py`  
**Depends on:** `ball_detection.py`, OpenCV, NumPy  
**Camera:** Stereolabs ZED 2 (SN28837104) via USB, no ZED SDK required

---

## Table of Contents

1. [Overview](#1-overview)
2. [How It Works](#2-how-it-works)
3. [File Structure](#3-file-structure)
4. [Installation and Requirements](#4-installation-and-requirements)
5. [Running the Feature](#5-running-the-feature)
6. [Visual Reference — What You See on Screen](#6-visual-reference--what-you-see-on-screen)
7. [Configuration Constants](#7-configuration-constants)
8. [API Reference](#8-api-reference)
9. [Integration into ball_detection.py](#9-integration-into-ball_detectionpy)
10. [Using intercept_x for Panel Control](#10-using-intercept_x-for-panel-control)
11. [Coordinate System](#11-coordinate-system)
12. [Tuning Guide](#12-tuning-guide)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Overview

The Bird's Eye View feature takes the 3D ball coordinates produced by `ball_detection.py`
and projects them onto a 2D overhead (top-down) map of the arena. It discards the vertical
axis (Y) and works entirely in the floor plane using X (lateral) and Z (depth/forward distance).

The result is a live window that shows:

- Where the ball currently is, relative to the robot
- The path the ball has taken (trail)
- The direction and speed the ball is moving
- A **predicted intercept point** — the X position where the ball will reach the robot's panel

This intercept prediction is the key output for the robot's panel motor. Once you know the
predicted X, you can calculate the panel angle needed to deflect the ball toward the goal.

---

## 2. How It Works

### Pipeline

```
ZED Camera
    |
    v
ball_detection.py
  - Captures stereo frame
  - Detects tennis ball in both eyes
  - Triangulates 3D position: X, Y, Z (metres)
    |
    v
birdseye.py — BirdsEyeOverlay.update(position)
  |
  +-- BallTracker
  |     - Stores rolling history of (X, Z) points with timestamps
  |     - Computes velocity (Vx, Vz) from the most recent detections
  |     - Predicts intercept X at a configured Z threshold
  |
  +-- BirdsEyeRenderer
        - Draws the overhead arena grid
        - Renders ball trail, current position, velocity arrow
        - Marks the intercept point on the panel Z line
        - Shows HUD with live numeric readout
            |
            v
    cv2.imshow("Bird's Eye View")
```

### Intercept Prediction Math

The tracker uses simple linear extrapolation from the most recent ball detections.

Given:
- Current ball position: `(x0, z0)`
- Estimated velocity: `(Vx, Vz)` metres per second
- Panel Z threshold: `intercept_z` (default `0.30 m`)

Time until ball reaches the panel:

```
t_intercept = (intercept_z - z0) / Vz
```

Predicted X at the panel:

```
intercept_x = x0 + Vx * t_intercept
```

The prediction is only marked as **valid** when `t_intercept > 0`, meaning the ball
is still moving toward the robot. If the ball is moving away, `intercept_valid = False`
and no marker is drawn.

---

## 3. File Structure

After adding this feature, your project contains the following files:

```
project/
├── ball_detection.py       # Main detection loop — updated to call birdseye
├── birdseye.py             # Bird's eye view module (new)
├── calibrate_stereo.py     # Stereo checkerboard calibration tool
├── tune_hsv.py             # HSV colour range tuner
├── SN28837104.conf         # ZED 2 factory calibration (your camera)
└── calibration.npz         # Output of calibrate_stereo.py (after you run it)
```

`birdseye.py` must be in the **same directory** as `ball_detection.py`.
If it is missing, the feature is silently skipped and a warning is printed at startup.

---

## 4. Installation and Requirements

No additional packages are needed. The feature uses only what is already installed:

```bash
sudo apt install python3-opencv python3-numpy
```

Verify the import works correctly from your project directory:

```bash
python3 -c "from birdseye import BirdsEyeOverlay; print('OK')"
```

---

## 5. Running the Feature

### Standard run with bird's eye view window

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --birdseye
```

Two windows will open side by side:
- `ZED Ball Detection` — the stereo camera feed with detection circles
- `Bird's Eye View` — the overhead 2D tracking map

### Demo mode — no camera needed

Runs a simulated ball rolling across the arena so you can test the drawing code
on any computer without connecting the ZED. This mode does **not** read camera
or detector output, so it will always show a fake motion path:

```bash
python3 birdseye.py --demo
```

Controls in demo mode:
- `q` — quit
- `r` — reset ball to starting position with a random angle

### Standalone live viewer from detector logs

If you want to run `birdseye.py` as a separate process, feed it the JSONL log
written by `ball_detection.py`:

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --log positions.jsonl \
    --no-display
```

Then in another terminal:

```bash
python3 birdseye.py --follow-log positions.jsonl
```

The recommended live mode is still direct integration:

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --birdseye
```

### Headless mode — no display, over SSH

The bird's eye tracker still runs and computes `intercept_x` even without windows.
The predicted intercept is printed to the terminal every 10 frames:

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --birdseye \
    --no-display
```

Output looks like:

```
[Frame 00042]  X=+0.312 m  Y=-0.041 m  Z=1.847 m  ...
  -> Panel intercept X=+0.089 m
[Frame 00052]  X=+0.278 m  Y=-0.039 m  Z=1.612 m  ...
  -> Panel intercept X=+0.091 m
```

### Changing the intercept Z distance

By default the panel intercept is computed at `Z = 0.30 m`.
Set this to match where your physical panel is positioned relative to the camera:

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --birdseye \
    --intercept-z 0.40
```

### Logging with bird's eye enabled

```bash
python3 ball_detection.py \
    --zed-calibration SN28837104.conf \
    --birdseye \
    --log positions.jsonl
```

Each line of the log file is a JSON record with full 3D coordinates, disparity,
epipolar y-diff, and depth-error fields. The intercept prediction is printed to
the terminal separately.

---

## 6. Visual Reference — What You See on Screen

```
┌─────────────────────────────────────────┐
│                                         │
│  Z=4m ──────────────────────────────── │  ← far end of arena
│       │         .                       │
│       │        . .  ← ball trail        │
│  Z=3m ─ ─ ─ ─ ─ .─ ─ ─ ─ ─ ─ ─ ─ ─ │
│       │          ●  ← current position  │
│       │          ↓  ← velocity arrow    │
│  Z=2m ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│       │                                 │
│  Z=1m ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│       │                                 │
│  Z=0.30m ···◆··· ← intercept marker    │  ← panel Z line (dashed)
│       │    X=+0.09m                     │
│       │   [ROBOT]                       │  ← robot / camera position
│─────────────────────────────────────────│
│  Ball  X=+0.278m   Z=1.612m            │
│  Vx=+0.04  Vz=-0.82 m/s  [APPROACHING] │  ← HUD strip
│  Predicted intercept  X=+0.091m        │
└─────────────────────────────────────────┘
```

### Colour legend

| Colour | Element |
|---|---|
| Dark green → Bright green | Ball trail — fades from old (dim) to new (bright) |
| Cyan filled circle | Current ball position |
| Yellow arrow | Velocity vector — direction and speed over 0.5 s |
| Blue dashed horizontal line | Intercept Z plane (where the panel sits) |
| Orange/red diamond | Predicted intercept X on the panel line |
| Orange dashed line | Trajectory projection from ball to intercept |
| White rectangle | Robot / camera symbol |
| Grey grid lines | 1 m × 1 m floor grid |

---

## 7. Configuration Constants

These are defined at the top of `birdseye.py`. Edit them directly if you need
to change the default behaviour.

| Constant | Default | Description |
|---|---|---|
| `CANVAS_W` | `500` | Width of the bird's eye canvas in pixels |
| `CANVAS_H` | `600` | Height of the bird's eye canvas in pixels |
| `TRAIL_LENGTH` | `60` | Number of past positions kept in the trail |
| `MIN_POINTS_FOR_PREDICTION` | `4` | Minimum detections before trajectory is estimated |
| `VELOCITY_WINDOW` | `8` | Number of recent points used for velocity estimation |
| `INTERCEPT_Z_M` | `0.30` | Default Z distance for panel intercept (metres) |

All colour constants (`C_BALL`, `C_TRAIL_NEW`, etc.) are BGR tuples that can be
changed to match any display preference.

---

## 8. API Reference

### `TrackPoint`

A single timestamped detection in the floor plane.

```python
@dataclass
class TrackPoint:
    x: float    # lateral position in metres
    z: float    # depth position in metres
    t: float    # time.monotonic() timestamp in seconds
```

---

### `TrajectoryEstimate`

The result of the linear velocity fit over recent track points.

```python
@dataclass
class TrajectoryEstimate:
    vx: float             # lateral velocity in metres/second
    vz: float             # depth velocity in metres/second (negative = approaching)
    intercept_x: float    # predicted X at intercept_z plane
    intercept_valid: bool # True only when ball is approaching (t_intercept > 0)
    speed_mps: float      # scalar speed sqrt(vx² + vz²)
```

---

### `BallTracker`

Maintains history and computes trajectory.

```python
BallTracker(
    trail_length: int = 60,       # max stored positions
    velocity_window: int = 8,     # points used for velocity fit
    intercept_z: float = 0.30,    # panel Z in metres
)
```

| Method / Property | Description |
|---|---|
| `.update(x, z)` | Feed a new detection. Pass `(None, None)` on missed frames. |
| `.trail` | `List[TrackPoint]` — all stored positions, oldest first |
| `.last_position` | Most recent `TrackPoint`, or `None` |
| `.trajectory` | Current `TrajectoryEstimate`, or `None` if not enough data |
| `.has_data` | `True` if the trail contains at least one point |

After 8 consecutive missed frames the trail is automatically cleared and
`trajectory` is reset to `None`.

---

### `BirdsEyeRenderer`

Renders the overhead canvas from a tracker's state.

```python
BirdsEyeRenderer(
    arena_x_m: float = 4.0,      # lateral width shown (metres)
    arena_z_m: float = 4.0,      # depth shown (metres)
    canvas_w: int = 500,          # output image width in pixels
    canvas_h: int = 600,          # output image height in pixels
    intercept_z: float = 0.30,   # Z line position (metres)
)
```

| Method | Returns | Description |
|---|---|---|
| `.render(tracker)` | `np.ndarray` (BGR image) | Draws the full bird's eye view from the tracker's current state. |

---

### `BirdsEyeOverlay` ← recommended for integration

The simplest way to add the feature. Wraps `BallTracker` and `BirdsEyeRenderer`
into a single object.

```python
BirdsEyeOverlay(
    arena_x_m: float = 4.0,
    arena_z_m: float = 4.0,
    intercept_z: float = 0.30,
)
```

| Method / Property | Returns | Description |
|---|---|---|
| `.update(position)` | `np.ndarray` | Feed a `BallPosition` (or `None`). Returns the rendered frame. Also stores it in `.frame`. |
| `.frame` | `np.ndarray` | The most recently rendered canvas. Safe to read at any time. |
| `.intercept_x` | `float` or `None` | Predicted X intercept in metres. `None` if ball is not approaching or not enough data. |

---

### `CoordMapper`

Internal helper that converts real-world metres to canvas pixels.
You do not need to use this directly unless you are extending the renderer.

```python
mapper = CoordMapper(arena_x_m, arena_z_m, canvas_w, canvas_h, margin=40)
u, v = mapper.to_px(x_metres, z_metres)
```

The camera / robot is placed at bottom-centre of the canvas.
Z increases upward on the canvas (farther away = higher on screen).

---

## 9. Integration into ball_detection.py

The integration is already done in the updated `ball_detection.py`. This section
documents what was changed so you can understand or replicate it.

### Import (top of file)

```python
try:
    from birdseye import BirdsEyeOverlay
    _BIRDSEYE_AVAILABLE = True
except ImportError:
    _BIRDSEYE_AVAILABLE = False
```

The `try/except` means the rest of `ball_detection.py` works normally even if
`birdseye.py` is missing — it just skips the feature.

### New CLI arguments (`parse_args`)

```python
p.add_argument("--birdseye", action="store_true",
               help="Show bird's eye 2D tracking view")

p.add_argument("--intercept-z", type=float, default=0.30,
               help="Z distance (metres) at which panel intercept is predicted")
```

### Setup (`main`, before the loop)

```python
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
        print("[Bird's Eye] birdseye.py not found.")
```

### Inside the main loop (after triangulation)

```python
if bev is not None:
    bev.update(position)           # position is BallPosition or None
    if not args.no_display:
        cv2.imshow("Bird's Eye View", bev.frame)
    ix = bev.intercept_x
    if ix is not None and position and frame_idx % 10 == 0:
        print(f"  -> Panel intercept X={ix:+.3f} m")
```

---

## 10. Using intercept_x for Panel Control

`bev.intercept_x` is the single output value that your motor controller needs.
It tells you, in metres, where the ball will cross the panel Z plane relative
to the camera centre.

```
intercept_x  < 0   →  ball arriving to the LEFT  of centre
intercept_x  = 0   →  ball arriving dead centre
intercept_x  > 0   →  ball arriving to the RIGHT of centre
```

### Connecting to panel angle

In the competition, the robot must redirect the ball toward the goal.
Given the predicted intercept X and the known goal position, compute the
required panel reflection angle:

```python
import numpy as np

def compute_panel_angle(intercept_x: float,
                        goal_x: float = 2.0,
                        goal_z: float = 2.0) -> float:
    """
    Returns the panel angle in degrees that reflects a ball arriving at
    intercept_x toward the goal position (goal_x, goal_z).

    The panel normal bisects the incoming and outgoing direction vectors.
    """
    # Ball arrives from the rolling area (positive Z, moving toward camera)
    v_in = np.array([0.0, -1.0])   # normalised incoming direction in XZ plane

    # Direction from intercept point to goal
    intercept_pos = np.array([intercept_x, 0.0])
    goal_pos      = np.array([goal_x,      goal_z])
    v_out = goal_pos - intercept_pos
    v_out = v_out / np.linalg.norm(v_out)

    # Panel normal = bisector of -v_in and v_out
    bisector = (-v_in) + v_out
    if np.linalg.norm(bisector) < 1e-6:
        return 0.0
    bisector = bisector / np.linalg.norm(bisector)

    # Panel angle from the X axis
    angle_deg = np.degrees(np.arctan2(bisector[1], bisector[0]))
    return angle_deg


# Example usage in the detection loop:
if bev.intercept_x is not None:
    angle = compute_panel_angle(bev.intercept_x, goal_x=2.0, goal_z=2.0)
    send_to_motor(angle)   # your motor control function
```

### Checking whether to act

Only command the motor when the intercept is valid and the ball is close enough
to act on. Use the tracker's trajectory to filter:

```python
traj = bev.tracker.trajectory

if traj and traj.intercept_valid and traj.speed_mps > 0.1:
    angle = compute_panel_angle(traj.intercept_x)
    send_to_motor(angle)
```

---

## 11. Coordinate System

The coordinate frame used throughout the project is camera-centred:

```
          +Z (forward, into arena)
           ↑
           │
           │    Ball rolling area
           │
           │         ● ball
           │
    ───────┼──────→  +X (right)
         robot
         (0,0)
```

| Axis | Direction | Typical range |
|---|---|---|
| `X` | Positive = right of camera centre | −2.0 m to +2.0 m |
| `Y` | Positive = below camera centre | Not used in bird's eye view |
| `Z` | Positive = forward, away from camera | 0.0 m to 4.0 m |

The bird's eye view displays X on the horizontal axis and Z on the vertical axis,
with Z increasing upward on screen so that "farther away = higher in the image"
matches the natural perspective of looking down at the arena.

---

## 12. Tuning Guide

### The ball disappears from the view while still visible in the camera

The trail auto-clears after `_max_misses = 8` consecutive frames without a
stereo detection. This happens when:

- Epipolar Y-diff is too tight → loosen with `--max-epipolar-y-diff 12`
- Ball moves too fast and HSV mask misses it → check `tune_hsv.py`
- Ball is very close (large apparent radius) → raise `MAX_RADIUS_PX` in `ball_detection.py`

### Intercept prediction is erratic or jumpy

The velocity estimate uses `VELOCITY_WINDOW = 8` points. On a Raspberry Pi running at
~10 fps this covers the last ~0.8 seconds, which is appropriate. If your frame rate is
higher, increase `VELOCITY_WINDOW` to smooth the estimate:

```python
# In birdseye.py
VELOCITY_WINDOW = 12
```

### Intercept always shows `None`

This means either:
1. There are fewer than `MIN_POINTS_FOR_PREDICTION = 4` points in the trail — wait for more detections.
2. `intercept_valid` is `False` — the ball's Vz is positive (moving away from the camera, not toward it).
3. `abs(Vz) < 0.01` — the ball is nearly stationary in Z.

### Arena edges are cut off

Increase the arena dimensions when creating `BirdsEyeOverlay`:

```python
bev = BirdsEyeOverlay(arena_x_m=5.0, arena_z_m=5.0)
```

Or reduce canvas margins in `CoordMapper` by lowering `margin` (default `40` pixels).

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[Bird's Eye] birdseye.py not found` | Wrong working directory | Run both scripts from the same folder |
| Bird's eye window opens but is always empty | Ball never detected in both eyes | Check stereo detection first with `--no-display` off |
| Intercept marker never appears | Ball always receding or not enough trail | Roll the ball directly toward the camera; ensure Z is decreasing |
| Velocity arrow points wrong direction | Ball path is noisy | Increase `VELOCITY_WINDOW`; improve calibration |
| Window is too small / too large | Default canvas size | Change `CANVAS_W` and `CANVAS_H` in `birdseye.py` |
| `ImportError: No module named cv2` | OpenCV not installed | `sudo apt install python3-opencv` |
| Demo runs but real camera does not | birdseye.py works but ball_detection.py fails | Check ZED camera connection and run `ball_detection.py` without `--birdseye` first |
