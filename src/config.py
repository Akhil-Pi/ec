UR3_IP = "192.168.1.10"        

SAFE_BOUNDS = {
    "x": (0.05, 0.35),  
    "y": (0.10, 0.50),
    "z": (0.05, 0.70),
}

# Movement parameters
MOVE_SPEED        = 0.1  # m/s
MOVE_ACCELERATION = 0.1  # m/s^2

# Joint Config
DESIRED_JOINTS  = [1.42351, -1.38754, -1.14521, -1.22057, 1.61412, 1.52513]

# Adjustment steps
Z_ADJUST_STEP    = 0.02
X_ADJUST_STEP    = 0.02
Y_ADJUST_STEP    = 0.02
TILT_ADJUST_STEP = 0.05
ROTATION_ADJUST_STEP = 0.08
PARTICIPANT_POSTURE = "standing"

# POSTURAL STRAIN SCORE (PSS) CONFIGURATION
TRUNK_LOW_RISK_DEG   = 20.0 
TRUNK_HIGH_RISK_DEG  = 60.0
CERVICAL_NEUTRAL_CM  = 2.5
CERVICAL_MAX_CM      = 5.0
PSS_THRESHOLD        = 0.25
PSS_HYSTERESIS       = 0.08
PSS_SMOOTHING_WINDOW = 30
CALIBRATION_DURATION_S = 10

# VISION CONFIGURATION
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

# Camera position relative to participant 
CAMERA_POSITION = "left"
CERVICAL_SENSITIVITY_COMPENSATE = {
    "left": {
        "positive": 1.0,   # head tilts RIGHT = away from camera, harder to detect
        "negative": 1.4,   # head tilts LEFT = toward camera, easy to detect
    },
    "center": {
        "positive": 1.0,
        "negative": 1.0,
    },
    "right": {
        "positive": 1.4,
        "negative": 1.0,
    },
}

# MediaPipe Pose: 0=lite, 1=full, 2=heavy
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE = 0.6

# EXPERIMENT CONFIGURATION
SESSION_DURATION_MIN  = 5
N_PARTICIPANTS_TARGET = 6
LOG_FREQUENCY_HZ      = 10

# FILE PATHS
DATA_DIR = "data"
LOG_DIR = "data/sessions"
RESULTS_DIR = "results"
