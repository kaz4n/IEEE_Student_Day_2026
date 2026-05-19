"""
HSV Colour Tuner — Tennis Ball
================================
Run this script to find the correct HSV range for your tennis ball
under your specific lighting conditions, then paste the values into
ball_detection.py.

Usage: python tune_hsv.py
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

def nothing(_): pass


def parse_args():
    parser = argparse.ArgumentParser(description="HSV tuner for tennis ball")
    parser.add_argument("--device", type=int, default=DEVICE_ID,
                        help="Camera index (default 0)")
    return parser.parse_args()

def main():
    args = parse_args()
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(args.device, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

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
        left = frame[:, :FRAME_W // 2]

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
