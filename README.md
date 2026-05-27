# IEEE Student Day 2026 - Tennis Ball 3D Detection

This project is a Python/OpenCV prototype for detecting a tennis ball with a Stereolabs ZED stereo camera and estimating the ball position in 3D coordinates. It is intended for a Raspberry Pi setup where the ZED is used as a standard side-by-side USB camera stream, without relying on the ZED SDK or CUDA depth pipeline.

## Current Situation

The project currently contains three scripts:

- `ball_detection.py` - main runtime program for camera capture, ball detection, stereo rectification, validation, triangulation, display, and optional logging.
- `calibrate_stereo.py` - stereo checkerboard calibration tool that saves a `calibration.npz` file.
- `tune_hsv.py` - HSV color tuning utility for finding better tennis-ball color thresholds under local lighting.

The code now starts with the factory ZED 2 calibration for camera serial `SN28837104`, so the main detector can rectify the left and right images without a checkerboard step. If `calibration.npz` includes stereo matrices saved by `calibrate_stereo.py`, that local checkerboard calibration can still be passed as an override.

The current local Python compile check passes:

```bash
python3 -m py_compile ball_detection.py calibrate_stereo.py tune_hsv.py
```

## Ubuntu / Raspberry Pi 5 Setup

On Ubuntu Desktop for Raspberry Pi 5, use the Ubuntu OpenCV packages instead of `pip install opencv-python`. The system packages are built for ARM and include the OpenCV GUI support used by the calibration and HSV tuning windows.

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy v4l-utils
```

Verify Python can import OpenCV:

```bash
python3 -c "import cv2, numpy; print(cv2.__version__)"
```

The ZED appears as one or more `/dev/video*` devices. Find the correct node and supported formats with:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

If the camera opens only with `sudo` or fails with a permission error, add your user to the `video` group, then log out and back in:

```bash
sudo usermod -aG video "$USER"
```

## Hardware Assumptions

- Camera: Stereolabs ZED stereo camera.
- Compute device: Raspberry Pi or another computer able to read the ZED as a USB video device.
- Target object: tennis ball.
- Camera stream: side-by-side left/right image, expected around `2560x720` total for two `1280x720` eyes.

If the camera negotiates a different resolution, the code reads the actual capture size and adjusts frame splitting. Calibration metadata is also stored so intrinsics can be scaled when the runtime resolution differs from the calibration resolution.

## Main Features

- Opens the ZED as a normal OpenCV `VideoCapture` device.
- Splits each frame into left and right eye images.
- Detects a yellow-green tennis ball using HSV color masking, contour blobs, and Hough circle fallback.
- Can optionally use a YOLO model for tracked 2-D tennis-ball detection with `--detector yolo`.
- Scores multiple left/right candidates before triangulation so the stereo pair is more stable.
- Smooths valid stereo detections and positions to reduce jitter, with a short display-only reacquisition hold.
- Uses hardcoded ZED 2 `SN28837104` factory calibration by default.
- Can override calibration from a Stereolabs `.conf` file or local `calibration.npz`.
- Rectifies both camera images before triangulation when full calibration data is available.
- Validates left/right detections before accepting a 3D result.
- Computes ball coordinates in metres:
  - `X`: left/right offset from the camera center.
  - `Y`: vertical offset from the camera center.
  - `Z`: forward distance from the camera.
- Prints and optionally logs JSON position records.
- Supports a known-distance check for testing depth accuracy.

## How It Works

1. `ball_detection.py` opens the ZED camera and requests a side-by-side stereo frame.
2. The frame is split into left and right images.
3. The built-in ZED 2 factory calibration undistorts and rectifies both images with OpenCV stereo rectification.
4. The selected detector runs on each rectified eye image.
5. In default `hsv` mode, a yellow-green HSV mask finds contour blobs first, then falls back to Hough circles.
6. In optional `yolo` mode, YOLO tracking boxes are converted into `(cx, cy, radius)` candidates.
7. Multiple left/right candidates are matched with the stereo validation rules, preferring the pair closest to the last valid ball.
8. Valid stereo detections and positions are smoothed to reduce jitter; brief losses show a hold/reacquire marker but do not print or log stale 3D coordinates.
9. The pair is rejected if:
   - either eye has no detection,
   - the rectified y coordinates differ too much,
   - disparity is invalid or too small,
   - the detected ball radii are very different.
10. If the pair is valid, stereo disparity is used:

```text
disparity = x_left - x_right
Z = fx * baseline / disparity
X = (x_left - cx) * Z / fx
Y = (y_left - cy) * Z / fy
```

The script also estimates depth from apparent ball size as a rough cross-check.

## Optional Checkerboard Calibration Workflow

The default detector already uses the factory calibration for ZED 2 serial `SN28837104`. Run checkerboard calibration only if you want to compare against a local calibration or replace the factory values.

Print or display a checkerboard with known square size. The current default is:

- 9 by 6 inner corners.
- 18 mm square size.
- At least 15 valid stereo captures.

Run:

```bash
python3 calibrate_stereo.py --device 0
```

The default capture request is `2560x720 @ 30fps`, which is the full side-by-side stereo frame for two `1280x720` eyes. If the camera requires MJPEG or the Pi needs a lighter stream, pass capture options explicitly:

```bash
python3 calibrate_stereo.py --device 0 --width 2560 --height 720 --fps 30 --fourcc MJPG
python3 calibrate_stereo.py --device 0 --width 1280 --height 360 --fps 15 --fourcc MJPG
```

For best accuracy, calibrate at the same resolution you plan to use for detection.

Controls:

- `SPACE` captures a checkerboard pose when both cameras see the board.
- `c` runs calibration after enough captures.
- `q` exits.

The output is:

```text
calibration.npz
```

Use it as an override with:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz
```

## Running Ball Detection

Basic run:

```bash
python3 ball_detection.py --device 0
```

Explicit HSV fallback mode:

```bash
python3 ball_detection.py --device 0 --detector hsv
```

Optional YOLO mode:

```bash
python3 -c "from ultralytics import YOLO; print('ok')"
python3 ball_detection.py --device 0 --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --yolo-imgsz 640 --yolo-conf 0.20
```

If you train or download a custom tennis-ball model, point to the real file path:

```bash
python3 ball_detection.py --device 0 --detector yolo --yolo-model runs/detect/train/weights/best.pt --yolo-class tennis_ball --yolo-imgsz 640 --yolo-conf 0.20
```

Headless run over SSH:

```bash
python3 ball_detection.py --device 0 --no-display
```

Use a Stereolabs `.conf` file instead of the hardcoded default:

```bash
python3 ball_detection.py --device 0 --zed-calibration SN28837104.conf
```

Log detections as JSON lines:

```bash
python3 ball_detection.py --device 0 --log positions.jsonl
```

Compare computed depth against a measured test distance:

```bash
python3 ball_detection.py --device 0 --known-distance 1.0
```

Tune the HSV color range:

```bash
python3 tune_hsv.py --device 0
```

Press `s` in the tuner to print updated `HSV_LOWER` and `HSV_UPPER` values, then copy those values into `ball_detection.py`.

If the Pi cannot keep up at the default capture size, reduce the capture request:

```bash
python3 ball_detection.py --device 0 --width 1280 --height 360 --fps 15
python3 ball_detection.py --device 0 --width 1280 --height 360 --fps 15 --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --yolo-imgsz 480 --yolo-conf 0.20
python3 tune_hsv.py --device 0 --width 1280 --height 360 --fps 15 --fourcc MJPG
```

YOLO mode uses Ultralytics tracking by default. If the tracker causes trouble, disable it and use per-frame detection:

```bash
python3 ball_detection.py --device 0 --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --no-yolo-track
```

## Coordinate Frame

The output position uses a camera-centered coordinate frame:

- Positive `X`: right of the camera center.
- Positive `Y`: below the camera center.
- Positive `Z`: forward, away from the camera.

All position values are reported in metres.

## Current Limitations

- HSV color detection is sensitive to lighting, shadows, and backgrounds with similar yellow-green colors.
- HSV detection may still fail with motion blur, partial occlusion, or very small/far balls.
- YOLO mode is optional and requires Ultralytics plus a suitable model; `--yolo-imgsz 640` is better for far balls but may be slower on Raspberry Pi 5.
- Accurate coordinates depend heavily on calibration quality and matching the active ZED camera.
- The Raspberry Pi may need reduced resolution or frame rate for smooth real-time performance.
- The ZED SDK depth engine is not used; this is an OpenCV stereo geometry approach.
- The hardcoded factory calibration is specific to ZED 2 serial `SN28837104`; use `--zed-calibration` or `--calibration` for another camera.

## Suggested Next Steps

- Test calibrated depth against measured distances such as 0.5 m, 1.0 m, 1.5 m, and 2.0 m.
- Record error statistics from `--known-distance` runs.
- Tune HSV thresholds in the actual competition lighting.
- Add frame-rate measurements on the Raspberry Pi.
- Train a custom tennis-ball YOLO model if COCO `sports_ball` misses your patterned ball at longer distances.
- Add saved sample frames for repeatable offline testing.
