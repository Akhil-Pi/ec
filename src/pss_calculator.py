"""
pss_calculator.py
=================
Computes the Postural Strain Score (PSS) from MediaPipe landmarks.

PSS is the equal-weighted mean of two biomechanically grounded sub-scores:
  1. Trunk Inclination Score (RULA/REBA-anchored)
  2. Cervical Displacement Score (Hansraj-anchored)

References:
  - RULA: McAtamney & Corlett, Appl. Ergon. 1993
  - REBA: Hignett & McAtamney, Appl. Ergon. 2000
  - Hansraj: Surg. Technol. Int. 2014
"""
import numpy as np
from collections import deque
import config


def angle_between(v1, v2):
    """Angle in degrees between two 2D vectors."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def midpoint(p1, p2):
    return tuple((a + b) / 2.0 for a, b in zip(p1, p2))


class PSSCalculator:
    """Computes and smooths PSS over time. Maintains a rolling buffer."""

    def __init__(self, smoothing_window=None):
        self.window = smoothing_window or config.PSS_SMOOTHING_WINDOW
        self.buffer = deque(maxlen=self.window)
        self._neutral_cervical_offset = None
        self._neutral_lean_ratio = None

    # ---------- SUB-SCORES ----------

    def trunk_inclination_score(self, landmarks):
        """
        Trunk angle from vertical (hip-midpoint -> shoulder-midpoint vector).
        Linear scaling: 20 deg -> 0.2, 60 deg -> 1.0 (matches proposal IV.B).
        Includes low-angle camera compensation (仰视补偿).
        Returns (angle_deg, score_in_[0,1]).
        """
        if landmarks is None:
            return 0.0, 0.0
        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]
        lh = landmarks["LEFT_HIP"][:2]
        rh = landmarks["RIGHT_HIP"][:2]

        shoulder_mid = midpoint(ls, rs)
        hip_mid = midpoint(lh, rh)

        # Image y-axis points DOWN, so vertical-up is (0, -1)
        trunk_vec = (shoulder_mid[0] - hip_mid[0],
                     shoulder_mid[1] - hip_mid[1])
        vertical = (0.0, -1.0)
        angle_deg = angle_between(trunk_vec, vertical)

        # Low-angle camera compensation: 仰视会使倾斜角看起来更小
        # 当摄像头在参与者下方时，添加校正因子
        # ⚠️ 调整策略: 从5度开始，根据测试结果增大
        # 如果一直raise → 减小这个值
        # 如果检测不到倾斜 → 增大这个值
        CAMERA_ELEVATION_CORRECTION = 5.0  # 从15.0降到5.0
        angle_deg = angle_deg + CAMERA_ELEVATION_CORRECTION

        low = config.TRUNK_LOW_RISK_DEG
        high = config.TRUNK_HIGH_RISK_DEG
        if angle_deg <= low:
            score = max(0.0, 0.2 * (angle_deg / low))
        elif angle_deg >= high:
            score = 1.0
        else:
            # Linear from (20, 0.2) to (60, 1.0)
            score = 0.2 + 0.8 * ((angle_deg - low) / (high - low))
        return angle_deg, float(np.clip(score, 0.0, 1.0))

    def head_tilt_angle(self, landmarks):
        """
        Detect head left/right tilt angle relative to shoulders.
        Uses ear positions to determine head tilt direction.

        SIGNED version:
          - positive angle = head tilted to the right
          - negative angle = head tilted to the left

        Robust detection:
          - Checks visibility of all landmarks (min 0.6)
          - Returns (tilt_angle_deg, tilt_score_in_[0,1])
          - Filters occlusion when arm obscures ear/shoulder
        """
        if landmarks is None:
            return 0.0, 0.0

        # Extract landmarks with visibility check
        le = landmarks.get("LEFT_EAR")
        re = landmarks.get("RIGHT_EAR")
        ls = landmarks.get("LEFT_SHOULDER")
        rs = landmarks.get("RIGHT_SHOULDER")

        if not all([le, re, ls, rs]):
            return 0.0, 0.0

        # Robust visibility check: 0.6 threshold to handle arm occlusion
        MIN_VISIBILITY = 0.6
        if any(lm[3] < MIN_VISIBILITY for lm in [le, re, ls, rs]):
            return 0.0, 0.0

        # Extract 2D positions (normalize [0,1] in image space)
        le_xy = le[:2]
        re_xy = re[:2]
        ls_xy = ls[:2]
        rs_xy = rs[:2]

        # Calculate head midpoint and shoulder midpoint
        head_mid = midpoint(le_xy, re_xy)
        shoulder_mid = midpoint(ls_xy, rs_xy)

        # Vector from shoulder to head (lateral component)
        shoulder_width = abs(rs_xy[0] - ls_xy[0])
        if shoulder_width < 1e-6:
            return 0.0, 0.0

        # Horizontal offset (positive = head to right, negative = head to left)
        head_offset_x = head_mid[0] - shoulder_mid[0]

        # Calculate tilt angle in degrees
        # Normalize by shoulder width to get proportion
        tilt_proportion = head_offset_x / shoulder_width
        # Map proportion to angle: ±shoulder_width offset = ±30 degrees
        tilt_angle_deg = np.arctan(tilt_proportion * 1.0) * 180.0 / np.pi

        # Score mapping: 0 deg -> 0, ±20 deg -> ±1.0
        score = abs(tilt_angle_deg) / 20.0
        score = float(np.clip(score, 0.0, 1.0))

        return tilt_angle_deg, score

    def cervical_displacement_score(self, landmarks):
        """
        Forward head displacement = horizontal distance between ear-midpoint
        and shoulder-midpoint, scaled by user's shoulder width (assumed 38 cm).

        SIGNED version: positive = head forward to right, negative = head forward to left
        Map: 0 cm -> 0, ±5 cm -> ±1.0 (proposal IV.B).
        Returns (displacement_cm_signed, score_in_[0,1]).
        """
        if landmarks is None:
            return 0.0, 0.0

        # Extract landmarks with visibility check (0-1 where 1 is confident)
        le = landmarks["LEFT_EAR"]
        re = landmarks["RIGHT_EAR"]
        ls = landmarks["LEFT_SHOULDER"]
        rs = landmarks["RIGHT_SHOULDER"]

        # Only use if visibility > 0.7 (confident detection)
        # Visibility is at index 3 for each landmark (x, y, z, visibility)
        MIN_VISIBILITY = 0.7
        if (le[3] < MIN_VISIBILITY or re[3] < MIN_VISIBILITY or
            ls[3] < MIN_VISIBILITY or rs[3] < MIN_VISIBILITY):
            return 0.0, 0.0

        # Extract Z-depth before slicing to 2D
        ear_z = (le[2] + re[2]) / 2.0
        shoulder_z = (ls[2] + rs[2]) / 2.0

        # Extract 2D positions
        le_xy = le[:2]
        re_xy = re[:2]
        ls_xy = ls[:2]
        rs_xy = rs[:2]

        ear_mid_xy = midpoint(le_xy, re_xy)
        shoulder_mid_xy = midpoint(ls_xy, rs_xy)

        shoulder_width = abs(ls_xy[0] - rs_xy[0])
        if shoulder_width < 1e-6:
            return 0.0, 0.0

        # 🎬 相机侧面位置支持 (Camera on the side):
        # 当相机在参与者侧面时，左右倾斜表现为Z深度变化而不是X位移
        # 使用综合方法: X位移 + Z深度差

        # 方法1: X方向位移（适合正面相机）
        x_offset_signed = ear_mid_xy[0] - shoulder_mid_xy[0]
        ASSUMED_SHOULDER_WIDTH_CM = 38.0
        displacement_from_x = ((x_offset_signed / shoulder_width)
                               * ASSUMED_SHOULDER_WIDTH_CM)

        # 方法2: Z深度差（适合侧面相机）
        z_diff = shoulder_z - ear_z  # 正值=头向前，负值=头向后

        # 综合两种方法（权重可根据相机位置调整）
        # 如果相机正面: 用X为主
        # 如果相机侧面: 用Z为主
        # 这里用 50:50 折中（可改为 30:70 或 0:100）
        displacement_cm = (displacement_from_x * 0.5 + z_diff * 50.0 * 0.5)

        # ⚠️ Z_DEPTH_SCALE 可能需要调整，根据你的相机标定
        # 如果Z差值通常是 0.01~0.05，那么乘以50让单位统一为cm

        # Subtract individual neutral baseline if calibrated
        if self._neutral_cervical_offset is not None:
            displacement_cm = displacement_cm - self._neutral_cervical_offset

        # Score uses absolute value (magnitude), but displacement keeps sign
        score = abs(displacement_cm) / config.CERVICAL_MAX_CM
        return displacement_cm, float(np.clip(score, 0.0, 1.0))

    # ---------- COMPOSITE PSS ----------

    def forward_lean_score(self, landmarks):
        """
        Proxy for forward lean using front-facing camera.

        For LOW-ANGLE cameras (仰视): Uses elbow/wrist extension instead of foreshortening.
        When upright: arms hang down naturally
        When leaning forward: arms come forward (Y-axis compression in image coords)

        Returns (lean_delta, score_in_[0,1]).
        """
        if landmarks is None:
            return 0.0, 0.0

        # Get landmarks with visibility check
        ls = landmarks.get("LEFT_SHOULDER")
        rs = landmarks.get("RIGHT_SHOULDER")
        le = landmarks.get("LEFT_ELBOW")
        re = landmarks.get("RIGHT_ELBOW")

        if not all([ls, rs, le, re]):
            return 0.0, 0.0

        MIN_VISIBILITY = 0.6
        if any(lm[3] < MIN_VISIBILITY for lm in [ls, rs, le, re]):
            return 0.0, 0.0

        # For low-angle camera: measure horizontal arm extension
        # When leaning forward, elbows move closer to body (X compression)
        ls_xy, rs_xy = ls[:2], rs[:2]
        le_xy, re_xy = le[:2], re[:2]

        # Left arm: shoulder-to-elbow horizontal distance
        left_arm_spread = abs(le_xy[0] - ls_xy[0])
        # Right arm: shoulder-to-elbow horizontal distance
        right_arm_spread = abs(re_xy[0] - rs_xy[0])

        current_spread = (left_arm_spread + right_arm_spread) / 2.0

        # Store neutral spread during calibration
        if self._neutral_lean_ratio is None:
            return 0.0, 0.0

        # How much has arm spread reduced from neutral?
        lean_delta = self._neutral_lean_ratio - current_spread
        # Positive delta = more forward lean (arms compressed inward)

        # Map: 0 delta = 0 score, 0.1 spread reduction = 1.0 score
        score = lean_delta / 0.1
        return lean_delta, float(np.clip(score, 0.0, 1.0))


    def compute(self, landmarks):
        trunk_angle,  trunk_score     = self.trunk_inclination_score(landmarks)
        tilt_angle_deg, tilt_score    = self.head_tilt_angle(landmarks)
        cervical_cm,  cervical_score  = self.cervical_displacement_score(landmarks)
        lean_delta,   lean_score      = self.forward_lean_score(landmarks)

        # PSS = weighted mean: raise + tilt_direction
        # tilt_angle: positive=right tilt, negative=left tilt
        pss_raw = (trunk_score * 0.40 +
                   tilt_score * 0.30 +
                   lean_score * 0.30)

        self.buffer.append(pss_raw)
        pss_smooth = float(np.mean(self.buffer))

        return {
            "trunk_angle":    trunk_angle,
            "trunk_score":    trunk_score,
            "tilt_angle_deg": tilt_angle_deg,
            "tilt_score":     tilt_score,
            "cervical_cm":    cervical_cm,
            "cervical_score": cervical_score,
            "lean_delta":     lean_delta,
            "lean_score":     lean_score,
            "pss_raw":        pss_raw,
            "pss_smooth":     pss_smooth,
        }

    # ---------- CALIBRATION ----------

    def calibrate_neutral(self, samples):
        """
        Set the participant's neutral cervical offset from displacement_cm
        samples collected while they sit upright.
        """
        if not samples:
            self._neutral_cervical_offset = 0.0
            return
        self._neutral_cervical_offset = float(np.median(samples))
        print(f"[PSS] Calibrated neutral cervical offset: "
              f"{self._neutral_cervical_offset:.2f} cm")

    def calibrate_neutral_lean(self, lean_samples):
        if not lean_samples:
            self._neutral_lean_ratio = 0.3   # sensible default
            return
        self._neutral_lean_ratio = float(np.median(lean_samples))
        print(f"[PSS] Neutral lean ratio: {self._neutral_lean_ratio:.3f}")

    def reset(self):
        self.buffer.clear()
        self._neutral_cervical_offset = None
        self._neutral_lean_ratio = None
