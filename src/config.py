# UR3 ROBOT CONFIGURATION
UR3_IP = "192.168.1.10"        

# Safety bounds for the TCP (Tool Center Point) in meters.
SAFE_BOUNDS = {
    "x": (0.10, 0.25),  
    "y": (0.20, 0.40),
    "z": (0.10, 0.60),
}

# Movement parameters
MOVE_SPEED        = 0.03  # m/s
MOVE_ACCELERATION = 0.05  # m/s^2

# Joint Config
HOME_JOINTS    = [1.42383, -2.61049, -2.11889, 1.60235, 1.71181, 1.57638]
POINT_1_JOINTS = [1.42507, -2.03308, -2.1191, 0.67063, 1.71179, 1.57622]
POINT_2_JOINTS = [1.42504, -1.67897, -1.84024, -0.27436, 1.7118, 1.57618]
POINT_3_JOINTS = [1.42485, -1.51767, -1.55773, -0.86834, 1.61473, 1.52516]
DESIRED_JOINTS = [1.42351, -1.38754, -1.14521, -1.22057, 1.61412, 1.52513]

JOINT_PATH = [HOME_JOINTS, POINT_1_JOINTS, POINT_2_JOINTS,
              POINT_3_JOINTS, DESIRED_JOINTS]

# Gradual transition
TRANSITION_STEPS      = 20     # robot moves in 20 increments
TRANSITION_STEP_DELAY = 3.0    # seconds between each step (~1 min total)
TRANSITION_SPEED      = 0.02   # 2 cm/s for home→desired gradual move
TRANSITION_ACCEL      = 0.03   # very gentle acceleration

# Adjustment steps
Z_ADJUST_STEP    = 0.02   # 2cm vertical
X_ADJUST_STEP    = 0.02   # 2cm lateral
Y_ADJUST_STEP    = 0.02   # 2cm depth
TILT_ADJUST_STEP = 0.05   # ~3 degrees

PARTICIPANT_POSTURE = "standing"


# POSTURAL STRAIN SCORE (PSS) CONFIGURATION

TRUNK_LOW_RISK_DEG   = 20.0   # below this = low risk (RULA action 1-2)
TRUNK_HIGH_RISK_DEG  = 60.0   # above this = high risk (RULA action 4)

CERVICAL_NEUTRAL_CM  = 2.5    # neutral zone (Hansraj 2014)
CERVICAL_MAX_CM      = 5.0    # maps to 1.0 score

PSS_THRESHOLD        = 0.40   # cobot intervention trigger
PSS_HYSTERESIS       = 0.10   # PSS must drop below 0.30 to "reset"
PSS_SMOOTHING_WINDOW = 30     # frames (~1 sec at 30 FPS)

CALIBRATION_DURATION_S = 15


# VISION CONFIGURATION

CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

# MediaPipe Pose: 0=lite, 1=full, 2=heavy
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE = 0.6


# EXPERIMENT CONFIGURATION

SESSION_DURATION_MIN  = 45    # match proposal (use --duration to override)
N_PARTICIPANTS_TARGET = 6     # realistic given 5 days at Pilotfabrik
LOG_FREQUENCY_HZ      = 10


# FILE PATHS

DATA_DIR = "data"
LOG_DIR = "data/sessions"
RESULTS_DIR = "results"
