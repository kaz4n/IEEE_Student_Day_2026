# Project Description

## General Idea

The goal of this project is to use a Stereolabs ZED stereo camera and a Raspberry Pi to detect a tennis ball and estimate its 3D position relative to the camera. The system should identify the ball in both stereo images, match the detections, and use stereo geometry to calculate the ball coordinates.

This project is focused on a practical robotics/computer-vision task: turning camera images into useful real-world position data. The final output should be a live estimate of where the tennis ball is, expressed as `X`, `Y`, and `Z` coordinates in metres.

## Project Goal

Build a working tennis-ball localization pipeline that can:

- Capture synchronized left and right images from the ZED stereo camera.
- Detect a tennis ball in each image.
- Use stereo calibration to correct camera distortion and align the stereo pair.
- Match the left and right detections reliably.
- Triangulate the ball position from stereo disparity.
- Report the ball coordinates in a form that could later be used by a robot, launcher, tracker, or control system.

## Why Stereo Vision

A single camera can detect where the ball appears in an image, but it cannot directly know how far away the ball is without extra assumptions. A stereo camera has two viewpoints separated by a known baseline. The ball appears at slightly different horizontal positions in the left and right images. This difference is called disparity.

Objects that are closer have larger disparity. Objects that are farther away have smaller disparity. With a calibrated stereo camera, the disparity can be converted into real-world depth.

## Main Concept

The system follows this pipeline:

```text
ZED camera stream
    -> split into left and right images
    -> rectify images using stereo calibration
    -> detect tennis ball in both images
    -> validate that detections are a matching stereo pair
    -> calculate disparity
    -> triangulate X, Y, Z coordinates
    -> display, print, or log the result
```

## Detection Method

The tennis ball is detected using its color and shape:

- Tennis balls are usually yellow-green, so HSV color filtering is used to isolate pixels in that color range.
- Noise is reduced with morphological image processing.
- A circle detector is used to find round objects in the filtered image.
- The most likely circle is treated as the ball.

This method is simple and fast enough for a Raspberry Pi prototype, but it depends on lighting and background conditions.

## Coordinate Estimation Method

After the ball is detected in both images, the system calculates disparity:

```text
disparity = x_left - x_right
```

Then it estimates depth:

```text
Z = focal_length * stereo_baseline / disparity
```

The horizontal and vertical coordinates are then calculated from the ball position in the rectified left image:

```text
X = (x_left - principal_point_x) * Z / focal_length_x
Y = (y_left - principal_point_y) * Z / focal_length_y
```

The result is the ball position relative to the camera.

## Expected Final Outcome

The desired final result is a system that can run on the Raspberry Pi, detect a tennis ball in real time or near real time, and output coordinates like:

```text
X=+0.120 m  Y=-0.040 m  Z=1.350 m
```

These coordinates could then be used by another system for decision-making, aiming, navigation, tracking, or visualization.

## Current Development Stage

The project is currently a functional prototype. It has:

- Camera capture code.
- Ball detection code.
- Stereo calibration code.
- Stereo rectification support.
- Basic left/right detection validation.
- Triangulation from disparity.
- Optional depth-error checking against known measured distances.

The most important remaining work is real-world testing with the actual ZED camera and Raspberry Pi, especially calibration accuracy, lighting robustness, and frame-rate performance.

## Success Criteria

The project can be considered successful when:

- The ball is detected consistently in the expected environment.
- Stereo calibration produces a low reprojection error.
- Rectified detections have nearly matching y coordinates.
- Measured depth error is acceptable for the project needs.
- The Raspberry Pi can run the pipeline at a usable frame rate.
- The output coordinates are stable enough to be used by the next stage of the system.
