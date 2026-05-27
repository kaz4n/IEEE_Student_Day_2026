"""
Stereo Calibration for ZED Camera on Raspberry Pi
==================================================
Captures frames from the ZED (as a plain USB camera) and runs
OpenCV stereo calibration using a printed checkerboard.

Output: calibration.npz  — loaded by ball_detection.py via --calibration flag

Steps
-----
1. Print or display a checkerboard (default: 9×6 inner corners, 25 mm squares)
2. Run this script:  python3 calibrate_stereo.py
3. Hold the checkerboard in ~20 different positions/angles in front of the camera
4. Press SPACE to capture each pose  (need at least 15 good captures)
5. Press 'c' when done collecting to run calibration
6. Results saved to calibration.npz

Requirements: sudo apt install python3-opencv python3-numpy v4l-utils
"""

import cv2
import numpy as np
import sys
import time
import argparse
from typing import Optional

# ── Checkerboard configuration ────────────────────────────────────────────────
# Inner corners (columns - 1, rows - 1) of your printed checkerboard
BOARD_W       = 9       # number of inner corners along width
BOARD_H       = 6       # number of inner corners along height
SQUARE_SIZE_M = 0.018   # physical square size in metres (18 mm)
MIN_CAPTURES  = 15      # minimum good captures before calibration is allowed
OUTPUT_FILE   = "calibration.npz"

# ── Camera capture settings ───────────────────────────────────────────────────
DEVICE_ID     = 0
FRAME_W       = 2560    # Full side-by-side width
FRAME_H       = 720
FPS           = 30.0


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


def open_camera(device_id: int, width: int, height: int, fps: float,
                fourcc: Optional[str]):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(device_id, backend)
    if fourcc:
        fourcc = fourcc.upper()
        if len(fourcc) != 4:
            raise ValueError("--fourcc must be exactly four characters, such as MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        print(camera_open_error(device_id))
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
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo calibration for ZED")
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


def split_frame(frame):
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


def find_corners(gray, board_size):
    """Return refined corners or None."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def run_calibration(obj_points, img_pts_l, img_pts_r, img_size):
    """Run full stereo calibration and return parameters dict."""
    board_size = (BOARD_W, BOARD_H)

    print("\n[Calibration] Running — this may take ~30 s on Raspberry Pi ...")

    # Individual camera calibration
    flags_single = (cv2.CALIB_RATIONAL_MODEL)
    _, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_pts_l, img_size,
                                              None, None, flags=flags_single)
    _, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_pts_r, img_size,
                                              None, None, flags=flags_single)

    # Stereo calibration
    flags_stereo = (cv2.CALIB_FIX_INTRINSIC)
    rms, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_pts_l, img_pts_r,
        K_l, D_l, K_r, D_r,
        img_size,
        flags=flags_stereo,
    )

    # T is in same units as SQUARE_SIZE_M (metres), so just take the magnitude
    baseline_m = float(np.linalg.norm(T))

    fx = float(K_l[0, 0])
    fy = float(K_l[1, 1])
    cx = float(K_l[0, 2])
    cy = float(K_l[1, 2])

    print(f"  RMS reprojection error: {rms:.4f} px  (good if < 1.0)")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  Baseline: {baseline_m*1000:.2f} mm")

    return dict(fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline_m, rms=float(rms),
                K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r, R=R, T=T)


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.width % 2 != 0:
        raise ValueError("--width must be even because the stereo frame is split in half")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    board_size = (BOARD_W, BOARD_H)

    # 3-D object points for one checkerboard pose
    objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE_M

    obj_points  = []   # 3-D points (same for every capture)
    img_pts_l   = []   # 2-D points in left eye
    img_pts_r   = []   # 2-D points in right eye

    cap        = open_camera(args.device, args.width, args.height, args.fps, args.fourcc)
    n_captures = 0

    print(f"\nCheckerboard: {BOARD_W}×{BOARD_H} inner corners, {SQUARE_SIZE_M*1000:.0f} mm squares")
    print("Controls:  SPACE = capture | c = calibrate | q = quit\n")

    while True:
        cap_ret, frame = cap.read()
        if not cap_ret:
            continue

        left, right = split_frame(frame)
        left_g  = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Live corner preview
        preview_l = left.copy()
        preview_r = right.copy()
        cl = find_corners(left_g,  board_size)
        cr = find_corners(right_g, board_size)

        status_color = (0, 80, 255)   # red = not both found
        if cl is not None:
            cv2.drawChessboardCorners(preview_l, board_size, cl, True)
        if cr is not None:
            cv2.drawChessboardCorners(preview_r, board_size, cr, True)
        if cl is not None and cr is not None:
            status_color = (0, 220, 0)  # green = both found

        combined = np.hstack([preview_l, preview_r])
        combined = cv2.resize(combined, (1280, 360))

        # Status overlay
        cv2.rectangle(combined, (0, 0), (1280, 40), (20, 20, 20), -1)
        msg = (f"Captures: {n_captures}/{MIN_CAPTURES}  |  "
               f"Board {'FOUND' if (cl is not None and cr is not None) else 'NOT FOUND'}  |  "
               "SPACE=capture  c=calibrate  q=quit")
        cv2.putText(combined, msg, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

        cv2.imshow("Stereo Calibration", combined)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if cl is None or cr is None:
                print("[Skip] Board not detected in both eyes.")
            else:
                obj_points.append(objp)
                img_pts_l.append(cl)
                img_pts_r.append(cr)
                n_captures += 1
                print(f"[Captured] {n_captures} poses collected.")

                # Flash green to confirm
                flash = np.zeros_like(combined)
                flash[:] = (0, 180, 0)
                alpha_frame = cv2.addWeighted(combined, 0.5, flash, 0.5, 0)
                cv2.imshow("Stereo Calibration", alpha_frame)
                cv2.waitKey(200)

        elif key == ord('c'):
            if n_captures < MIN_CAPTURES:
                print(f"[Warning] Need at least {MIN_CAPTURES} captures "
                      f"(have {n_captures}). Keep going.")
            else:
                img_size = (left_g.shape[1], left_g.shape[0])
                params   = run_calibration(obj_points, img_pts_l, img_pts_r, img_size)

                np.savez(
                    OUTPUT_FILE,
                    fx=params["fx"], fy=params["fy"],
                    cx=params["cx"], cy=params["cy"],
                    baseline=params["baseline"],
                    rms=params["rms"],
                    image_width=img_size[0],
                    image_height=img_size[1],
                    K_l=params["K_l"], D_l=params["D_l"],
                    K_r=params["K_r"], D_r=params["D_r"],
                    R=params["R"],    T=params["T"],
                )
                print(f"\n[Saved] {OUTPUT_FILE}")
                print(
                    f"Run detection with:  python3 ball_detection.py "
                    f"--device {args.device} --calibration {OUTPUT_FILE}"
                )
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
