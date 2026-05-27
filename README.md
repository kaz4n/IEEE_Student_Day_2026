# IEEE Student Day 2026 - Tennis Ball 3D Detection

This project is a Python/OpenCV prototype for detecting a tennis ball with a Stereolabs ZED stereo camera and estimating the ball position in 3D coordinates. It is intended for a Raspberry Pi setup where the ZED is used as a standard side-by-side USB camera stream, without relying on the ZED SDK or CUDA depth pipeline.

## Current Situation

The project currently contains three scripts:

- `ball_detection.py` - main runtime program for camera capture, ball detection, stereo rectification, validation, triangulation, display, and optional logging.
- `calibrate_stereo.py` - stereo checkerboard calibration tool that saves a `calibration.npz` file.
- `tune_hsv.py` - HSV color tuning utility for finding better tennis-ball color thresholds under local lighting.

The code now supports full stereo calibration handling. If `calibration.npz` includes the stereo matrices saved by `calibrate_stereo.py`, the main detector rectifies the left and right images before using disparity. This is important because raw ZED UVC side-by-side images should not be treated as already rectified.

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
- Detects the tennis ball using HSV color masking and Hough circle detection.
- Loads stereo calibration from `calibration.npz`.
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
3. If a full calibration file is provided, both images are undistorted and rectified with OpenCV stereo rectification.
4. Each rectified image is converted to HSV.
5. A yellow-green HSV mask isolates likely tennis-ball pixels.
6. Morphological cleanup and Gaussian blur reduce noise.
7. Hough circle detection finds candidate circular blobs.
8. The largest detected circle is selected in each eye.
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

## Calibration Workflow

Print or display a checkerboard with known square size. The current default is:

- 9 by 6 inner corners.
- 25 mm square size.
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

Use it with:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz
```

## Running Ball Detection

Basic run:

```bash
python3 ball_detection.py --device 0
```

Recommended calibrated run:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz
```

Headless run over SSH:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz --no-display
```

Log detections as JSON lines:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz --log positions.jsonl
```

Compare computed depth against a measured test distance:

```bash
python3 ball_detection.py --device 0 --calibration calibration.npz --known-distance 1.0
```

Tune the HSV color range:

```bash
python3 tune_hsv.py --device 0
```

Press `s` in the tuner to print updated `HSV_LOWER` and `HSV_UPPER` values, then copy those values into `ball_detection.py`.

If the Pi cannot keep up at the default capture size, reduce the capture request:

```bash
python3 ball_detection.py --device 0 --width 1280 --height 360 --fps 15 --calibration calibration.npz
python3 tune_hsv.py --device 0 --width 1280 --height 360 --fps 15 --fourcc MJPG
```

## Coordinate Frame

The output position uses a camera-centered coordinate frame:

- Positive `X`: right of the camera center.
- Positive `Y`: below the camera center.
- Positive `Z`: forward, away from the camera.

All position values are reported in metres.

## Current Limitations

- HSV color detection is sensitive to lighting, shadows, and backgrounds with similar yellow-green colors.
- Hough circle detection may fail with motion blur, partial occlusion, or very small/far balls.
- Accurate coordinates depend heavily on calibration quality.
- The Raspberry Pi may need reduced resolution or frame rate for smooth real-time performance.
- The ZED SDK depth engine is not used; this is an OpenCV stereo geometry approach.
- If no full calibration is supplied, rectification is disabled and 3D coordinates are less trustworthy.

## Suggested Next Steps

- Test calibrated depth against measured distances such as 0.5 m, 1.0 m, 1.5 m, and 2.0 m.
- Record error statistics from `--known-distance` runs.
- Tune HSV thresholds in the actual competition lighting.
- Add frame-rate measurements on the Raspberry Pi.
- Consider contour-based detection or tracking if Hough circles are unstable.
- Add saved sample frames for repeatable offline testing.
