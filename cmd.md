# Best option — factory cal only (no checkerboard needed)
python3 ball_detection.py --zed-calibration SN28837104.conf

# Verify depth at 1 metre
python3 ball_detection.py --zed-calibration SN28837104.conf --known-distance 1.0

# Headless over SSH
python3 ball_detection.py --zed-calibration SN28837104.conf --no-display --log positions.jsonl