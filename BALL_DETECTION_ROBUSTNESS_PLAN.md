# Robust Tennis Ball Detection Plan

## Summary

The old detector used a fixed HSV mask and Hough circles. That works for a nearby, clean, mostly yellow-green ball, but it breaks when the ball is around 2 m away, moving, partially covered by a hand, motion-blurred, or showing dark logos/seams.

The implemented direction is a hybrid runtime:

- A lightweight YOLO-style ONNX detector is the primary path when `models/tennis_ball.onnx` exists.
- A wider OpenCV color/shape detector remains available as a fallback and debug aid.
- Stereo pair selection scores multiple candidates instead of accepting the single largest circle.
- Per-eye Kalman trackers and a 3D Kalman tracker bridge short misses caused by blur or occlusion.
- Camera FPS, FOURCC, exposure, and gain are configurable so the Raspberry Pi 5 can reduce motion blur.

## Runtime Changes

`ball_detection.py` now supports:

```bash
python3 ball_detection.py \
  --zed-calibration SN28837104.conf \
  --detector hybrid \
  --model models/tennis_ball.onnx \
  --conf-threshold 0.35 \
  --nms-threshold 0.45 \
  --model-input-size 320
```

Useful variants:

```bash
# OpenCV fallback only, useful before the ONNX model is trained.
python3 ball_detection.py --zed-calibration SN28837104.conf --detector opencv

# Model only, fails early if the model file is missing.
python3 ball_detection.py --zed-calibration SN28837104.conf --detector model --model models/tennis_ball.onnx

# Debug the fallback mask.
python3 ball_detection.py --zed-calibration SN28837104.conf --debug-mask

# Reduce motion blur on cameras that support manual exposure.
python3 ball_detection.py --zed-calibration SN28837104.conf --exposure -6 --gain 0
```

The log records now include the final 3D position plus left/right detection metadata:

- `source`: `model`, `opencv`, `opencv-hough`, `tracker`, or `3d-tracker`
- `confidence`
- `bbox`
- `tracked`
- `predicted`

## Dataset And Training Workflow

Train the ONNX detector off the Raspberry Pi, then copy the exported model into `models/tennis_ball.onnx`.

Capture labeled frames from the actual ZED view:

- Distances: 10 cm, 35 cm, 50 cm, 1 m, 1.5 m, 2 m, and slightly beyond 2 m.
- Static ball and moving ball.
- Ball moving sideways, toward the camera, and away from the camera.
- Hand-held ball with fingers covering 25-50%.
- Seams, logos, dark marks, worn felt, shadows, and different backgrounds.
- Both left and right eye frames after rectification if possible.

Label the full ball extent, not only the visible yellow-green pixels. If a hand covers part of the ball, draw the bounding box around the estimated full ball. This is what teaches the model to stay locked on the object during partial occlusion.

Recommended model target:

- Small single-class YOLO model.
- Input size: 320 or 416.
- Export format: ONNX.
- Class list: one class, `tennis_ball`.
- Runtime: OpenCV DNN CPU backend on Raspberry Pi 5.

## Detection And Tracking Flow

Each frame follows this sequence:

1. Split ZED side-by-side capture into left and right eyes.
2. Rectify both images with the existing ZED factory or checkerboard calibration.
3. Predict the next left/right ball locations using per-eye Kalman trackers.
4. Search near the predicted ROI first; if the ROI fails, fall back to the full frame.
5. Run the ONNX model when available.
6. Run the OpenCV color/shape fallback in hybrid or fallback mode.
7. Score all left/right candidate pairs by confidence, epipolar y agreement, positive disparity, radius consistency, depth-vs-size sanity, and tracker proximity.
8. Correct the image trackers with measured detections, or keep tracker predictions for short misses.
9. Triangulate the stereo pair into `X, Y, Z`.
10. Smooth or predict the 3D position with a constant-velocity Kalman filter.

## Acceptance Tests

Run the compile check after changes:

```bash
python -m py_compile ball_detection.py calibrate_stereo.py tune_hsv.py
```

Field-test with measured distances:

```bash
python3 ball_detection.py --zed-calibration SN28837104.conf --known-distance 0.35
python3 ball_detection.py --zed-calibration SN28837104.conf --known-distance 1.0
python3 ball_detection.py --zed-calibration SN28837104.conf --known-distance 1.5
python3 ball_detection.py --zed-calibration SN28837104.conf --known-distance 2.0
```

Expected behavior after a trained model is added:

- Detects/tracks the ball at around 2 m in normal lighting.
- Continues tracking through brief motion blur and hand occlusion.
- Recovers within about 3-5 frames after a short miss.
- Keeps 2 m depth error near the calibration limit, roughly 10-15 cm with good rectification.

## Practical Notes

- The OpenCV fallback is intentionally broader than the old HSV detector, but it is still not a replacement for training a real ball model.
- If the ONNX model is missing in `hybrid` mode, the program prints a warning and continues with the fallback.
- If `--detector model` is used and the ONNX model is missing, startup fails so the test does not silently run the weaker fallback.
- Manual exposure is camera-driver dependent. Test values with the ZED on the Raspberry Pi and keep the lowest exposure that still gives usable brightness.
