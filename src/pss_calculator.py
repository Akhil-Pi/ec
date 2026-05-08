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
        # 当摄像头在参与者下方时，添加+15度的校正因子
        # 可根据实际拍摄角度调整（15-25度范围）
        CAMERA_ELEVATION_CORRECTION = 15.0
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
        # Visibility is at index 2 for each landmark (x, y, z, visibility)
        MIN_VISIBILITY = 0.7
        if (le[3] < MIN_VISIBILITY or re[3] < MIN_VISIBILITY or
            ls[3] < MIN_VISIBILITY or rs[3] < MIN_VISIBILITY):
            return 0.0, 0.0

        le = le[:2]
        re = re[:2]
        ls = ls[:2]
        rs = rs[:2]

        ear_mid = midpoint(le, re)
        shoulder_mid = midpoint(ls, rs)

        shoulder_width = abs(ls[0] - rs[0])
        if shoulder_width < 1e-6:
            return 0.0, 0.0

        # Preserve direction: positive if head forward-right, negative if forward-left
        x_offset_signed = ear_mid[0] - shoulder_mid[0]
        ASSUMED_SHOULDER_WIDTH_CM = 38.0
        displacement_cm = ((x_offset_signed / shoulder_width)
                           * ASSUMED_SHOULDER_WIDTH_CM)

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
        trunk_angle,  trunk_score    = self.trunk_inclination_score(landmarks)
        cervical_cm,  cervical_score = self.cervical_displacement_score(landmarks)
        lean_delta,   lean_score     = self.forward_lean_score(landmarks)

    # PSS = weighted mean of all three
    # Forward lean gets highest weight since that's what we want the cobot to respond to
        pss_raw = (trunk_score * 0.25 +
                   cervical_score * 0.25 +
                   lean_score * 0.50)

        self.buffer.append(pss_raw)
        pss_smooth = float(np.mean(self.buffer))

        return {
            "trunk_angle":    trunk_angle,
            "trunk_score":    trunk_score,
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
