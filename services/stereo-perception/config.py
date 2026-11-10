"""
Centralized configuration for the stereo perception pipeline.

All values can be overridden via environment variables.
Where appropriate, defaults are conservative for the Jetson Nano (4 GB RAM).
"""
from __future__ import annotations
import os

# ── Camera ────────────────────────────────────────────────────────────────────
# CAMERA_TYPE: oak  — Luxonis OAK-D Lite via DepthAI SDK (USB3, recommended)
#              csi  — Jetson CSI cameras via GStreamer / nvarguscamerasrc
#              usb  — USB cameras via V4L2 / OpenCV VideoCapture
#              file — Pre-recorded stereo image pairs (development / CI)
#
# For oak: run calibration ONCE before first use:
#   python3 scripts/calibrate_stereo.py oak-calib --output config/stereo_calibration.yaml
# OAK-D Lite supported heights: 400 (640×400), 480 (640×480), 720 (1280×720), 800 (1280×800)
CAMERA_TYPE      = os.getenv("CAMERA_TYPE",       "oak")
LEFT_CAMERA_ID   = int(os.getenv("LEFT_CAMERA_ID", "0"))
RIGHT_CAMERA_ID  = int(os.getenv("RIGHT_CAMERA_ID", "1"))
CAMERA_WIDTH     = int(os.getenv("CAMERA_WIDTH",   "640"))
CAMERA_HEIGHT    = int(os.getenv("CAMERA_HEIGHT",  "400"))  # 400 for OAK THE_400_P preset
CAMERA_FPS       = int(os.getenv("CAMERA_FPS",     "30"))

# For file-based source (development / testing without hardware)
LEFT_IMAGE_DIR   = os.getenv("LEFT_IMAGE_DIR",  "data/test_stereo/left")
RIGHT_IMAGE_DIR  = os.getenv("RIGHT_IMAGE_DIR", "data/test_stereo/right")
LOOP_FILE_SOURCE = os.getenv("LOOP_FILE_SOURCE", "true").lower() == "true"

# ── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_FILE = os.getenv("CALIBRATION_FILE", "config/stereo_calibration.yaml")
# Allow pipeline to run uncalibrated in development (depth will be wrong)
REQUIRE_CALIBRATION = os.getenv("REQUIRE_CALIBRATION", "false").lower() == "true"

# ── Stereo disparity algorithm ────────────────────────────────────────────────
# STEREO_ALGORITHM: sgbm — Semi-Global Block Matching (better quality, slower)
#                   bm   — Block Matching (faster, lower quality)
STEREO_ALGORITHM  = os.getenv("STEREO_ALGORITHM",  "sgbm")
MIN_DISPARITY     = int(os.getenv("MIN_DISPARITY",   "0"))
NUM_DISPARITIES   = int(os.getenv("NUM_DISPARITIES", "64"))  # must be divisible by 16
BLOCK_SIZE        = int(os.getenv("BLOCK_SIZE",      "7"))   # must be odd, >= 1
UNIQUENESS_RATIO  = int(os.getenv("UNIQUENESS_RATIO","10"))
SPECKLE_WIN_SIZE  = int(os.getenv("SPECKLE_WIN_SIZE","100"))
SPECKLE_RANGE     = int(os.getenv("SPECKLE_RANGE",   "32"))
DISP12_MAX_DIFF   = int(os.getenv("DISP12_MAX_DIFF",  "1"))
PRE_FILTER_CAP    = int(os.getenv("PRE_FILTER_CAP",  "63"))
# P1 = P1_MULT * 3 * BLOCK_SIZE^2  (penalty for 1-pixel disparity change)
# P2 = P2_MULT * 3 * BLOCK_SIZE^2  (penalty for larger disparity change)
P1_MULT = int(os.getenv("P1_MULT", "8"))
P2_MULT = int(os.getenv("P2_MULT", "32"))
USE_GPU  = os.getenv("USE_GPU", "false").lower() == "true"

# WLS post-filter (reduces disparity noise; adds CPU cost)
USE_WLS_FILTER   = os.getenv("USE_WLS_FILTER",  "false").lower() == "true"
WLS_LAMBDA       = float(os.getenv("WLS_LAMBDA", "8000.0"))
WLS_SIGMA        = float(os.getenv("WLS_SIGMA",  "1.5"))

# ── Depth limits ──────────────────────────────────────────────────────────────
MIN_DEPTH_M = float(os.getenv("MIN_DEPTH_M", "0.25"))   # below this = invalid
MAX_DEPTH_M = float(os.getenv("MAX_DEPTH_M", "10.0"))   # above this = invalid

# ── Obstacle detection ────────────────────────────────────────────────────────
# Region widths as fractions of the image (left + center + right = 1.0)
LEFT_REGION_FRAC   = float(os.getenv("LEFT_REGION_FRAC",   "0.33"))
CENTER_REGION_FRAC = float(os.getenv("CENTER_REGION_FRAC", "0.34"))
# RIGHT_REGION_FRAC is 1 - LEFT - CENTER

# Which depth percentile to use as the "nearest reliable obstacle" per region.
# Lower percentile = more sensitive but more noise; 10th is a good default.
OBSTACLE_PERCENTILE = float(os.getenv("OBSTACLE_PERCENTILE", "10.0"))

# Vertical crop: only use the middle N% of image rows (ignore sky/ground near edges)
DEPTH_ROI_TOP_FRAC    = float(os.getenv("DEPTH_ROI_TOP_FRAC",    "0.25"))
DEPTH_ROI_BOTTOM_FRAC = float(os.getenv("DEPTH_ROI_BOTTOM_FRAC", "0.75"))

# ── Safety thresholds ─────────────────────────────────────────────────────────
WARNING_DISTANCE_M   = float(os.getenv("WARNING_DISTANCE_M",   "3.0"))
STOP_DISTANCE_M      = float(os.getenv("STOP_DISTANCE_M",      "1.5"))
EMERGENCY_DISTANCE_M = float(os.getenv("EMERGENCY_DISTANCE_M", "0.5"))

# Stop and wait for fresh perception if stereo vision is older than this
VISION_TIMEOUT_S = float(os.getenv("VISION_TIMEOUT_S", "1.0"))

# ── Motor commands ────────────────────────────────────────────────────────────
MAX_SPEED_MS        = float(os.getenv("MAX_SPEED_MS",        "1.0"))   # m/s forward max
TURN_SPEED_MS       = float(os.getenv("TURN_SPEED_MS",       "0.3"))   # m/s during turn
TURN_RATE           = float(os.getenv("TURN_RATE",           "0.6"))   # rad/s turn rate
MOTOR_CONTROLLER_ADDR = os.getenv("MOTOR_CONTROLLER_ADDR",  "localhost:50052")
MOTOR_COMMAND_TIMEOUT_S = float(os.getenv("MOTOR_COMMAND_TIMEOUT_S", "0.5"))

# ── Pipeline thread queues ────────────────────────────────────────────────────
CAMERA_QUEUE_SIZE     = int(os.getenv("CAMERA_QUEUE_SIZE",     "2"))
PERCEPTION_QUEUE_SIZE = int(os.getenv("PERCEPTION_QUEUE_SIZE", "2"))
CONTROL_RATE_HZ       = float(os.getenv("CONTROL_RATE_HZ",    "20.0"))  # target loop rate

# ── Kafka / telemetry ─────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TELEMETRY_TOPIC   = os.getenv("KAFKA_TELEMETRY_TOPIC",   "rover.telemetry")
TELEMETRY_RATE_HZ       = float(os.getenv("TELEMETRY_RATE_HZ", "5.0"))
ROVER_ID                = os.getenv("ROVER_ID",                 "PHYSICAL-ROVER-001")

# ── FDIR events ───────────────────────────────────────────────────────────────
KAFKA_FDIR_TOPIC = os.getenv("KAFKA_FDIR_TOPIC", "rover.fdir.events")

# ── Observability ─────────────────────────────────────────────────────────────
METRICS_PORT  = int(os.getenv("METRICS_PORT",   "9095"))
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT",      "http://localhost:4317")

# ── Logging / debug ───────────────────────────────────────────────────────────
LOG_LEVEL   = os.getenv("LOG_LEVEL",   "INFO")
DEBUG_VIZ   = os.getenv("DEBUG_VIZ",  "false").lower() == "true"  # show CV2 windows
DEBUG_SAVE  = os.getenv("DEBUG_SAVE", "false").lower() == "true"  # save debug frames
DEBUG_DIR   = os.getenv("DEBUG_DIR",  "debug_output")
DEBUG_EVERY_N_FRAMES = int(os.getenv("DEBUG_EVERY_N_FRAMES", "30"))  # save every Nth frame
