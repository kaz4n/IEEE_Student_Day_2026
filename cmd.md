# Best option — hardcoded ZED 2 SN28837104 factory cal
python3 ball_detection.py

# Explicit HSV fallback mode
python3 ball_detection.py --detector hsv

# Optional YOLO quick test with a COCO model
python3 ball_detection.py --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --yolo-imgsz 640 --yolo-conf 0.20

# Optional YOLO mode with a custom tennis-ball model
python3 ball_detection.py --detector yolo --yolo-model runs/detect/train/weights/best.pt --yolo-class tennis_ball --yolo-imgsz 640 --yolo-conf 0.20

# Optional YOLO without tracker if ByteTrack causes problems
python3 ball_detection.py --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --no-yolo-track

# Lower-resolution tennis-ball detector
python3 ball_detection.py --width 1280 --height 360 --fps 15

# Lower-resolution YOLO detector
python3 ball_detection.py --width 1280 --height 360 --fps 15 --detector yolo --yolo-model yolo11n.pt --yolo-class sports_ball --yolo-imgsz 480 --yolo-conf 0.20

# Verify depth at 1 metre
python3 ball_detection.py --known-distance 1.0

# Headless over SSH
python3 ball_detection.py --no-display --log positions.jsonl

# Optional: override from the Stereolabs .conf file
python3 ball_detection.py --zed-calibration SN28837104.conf
