"""
HSV Colour Tuner — Tennis Ball
================================
Run this script to find the correct HSV range for your tennis ball
under your specific lighting conditions, then paste the values into
ball_detection.py.

Usage: python3 tune_hsv.py
Controls: trackbars for H/S/V min/max — adjust until only the ball is white.
Press 's' to print the current values.
"""

import cv2
import numpy as np
import sys
import argparse

DEVICE_ID  = 0
FRAME_W    = 2560
FRAME_H    = 720
FPS        = 30.0

def nothing(_): pass


def fourcc_to_string(value: float) -> str:
    code = int(value)
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00") else "unknown"


def camera_open_error(device_id: int) -> str:
    if sys.platform.startswith("win"):
        return "ERROR: Cannot open camera. Check USB connection and camera permissions."

    device_path = f"/dev/video{device_id}"
    return (
        f"ERROR: Cannot open camera at {device_path}.\n"
        "Check the USB connection and find the ZED node with:\n"
        "  v4l2-ctl --list-devices\n"
        "Inspect supported formats with:\n"
        f"  v4l2-ctl -d {device_path} --list-formats-ext\n"
        "If permissions fail, add your user to the video group, then log out/in:\n"
        "  sudo usermod -aG video \"$USER\""
    )


def parse_args():
    parser = argparse.ArgumentParser(description="HSV tuner for tennis ball")
    parser.add_argument("--device", type=int, default=DEVICE_ID,
                        help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=FRAME_W,
                        help="Requested full side-by-side capture width (default 2560)")
    parser.add_argument("--height", type=int, default=FRAME_H,
                        help="Requested capture height (default 720)")
    parser.add_argument("--fps", type=float, default=FPS,
                        help="Requested camera frame rate (default 30)")
    parser.add_argument("--fourcc", type=str, default=None,
                        help="Optional four-character pixel format request, such as MJPG")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.width % 2 != 0:
        raise ValueError("--width must be even because the stereo frame is split in half")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(args.device, backend)
    if args.fourcc:
        fourcc = args.fourcc.upper()
        if len(fourcc) != 4:
            raise ValueError("--fourcc must be exactly four characters, such as MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print(camera_open_error(args.device))
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    print(
        f"[ZED] Opened at {actual_w}×{actual_h} "
        f"(eye: {actual_w // 2}×{actual_h}) "
        f"fps={actual_fps:.1f} fourcc={actual_fourcc}"
    )

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 1280, 600)

    # Trackbars — defaults match tennis ball yellow-green
    for name, val, maxv in [
        ("H min", 22, 179), ("H max", 65, 179),
        ("S min", 80, 255), ("S max", 255, 255),
        ("V min", 80, 255), ("V max", 255, 255),
    ]:
        cv2.createTrackbar(name, "HSV Tuner", val, maxv, nothing)

    print("Adjust trackbars until ONLY the tennis ball appears white in the mask.")
    print("Press 's' to print values | 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Use left eye only
        left = frame[:, :frame.shape[1] // 2]

        h_min = cv2.getTrackbarPos("H min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        hsv  = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Show original + mask side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display  = np.hstack([left, mask_bgr])
        display  = cv2.resize(display, (1280, 360))

        cv2.putText(display, "Original (left eye)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "HSV Mask", (650, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("HSV Tuner", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"\n# Paste these into ball_detection.py:")
            print(f"HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
