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
        le = landmarks["LEFT_EAR"][:2]
        re = landmarks["RIGHT_EAR"][:2]
        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]

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
    
        When standing upright: shoulder-to-hip vertical distance is large.
        When leaning forward: torso foreshortens, distance shrinks.
    
        Also uses nose-to-shoulder vertical drop as secondary signal.
        Returns (lean_ratio, score_in_[0,1]).
        """
        if landmarks is None:
            return 0.0, 0.0

        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]
        lh = landmarks["LEFT_HIP"][:2]
        rh = landmarks["RIGHT_HIP"][:2]
        nose = landmarks["NOSE"][:2]

        shoulder_mid = midpoint(ls, rs)
        hip_mid      = midpoint(lh, rh)

    # Vertical distance shoulder to hip (normalized image coords)
        torso_height = abs(hip_mid[1] - shoulder_mid[1])

    # Nose to shoulder vertical drop
        nose_to_shoulder = abs(shoulder_mid[1] - nose[1])

    # Ratio: nose_to_shoulder / torso_height
    # Upright: ~0.25-0.30 (head well above shoulders)
    # Forward lean: ratio drops as shoulders rise toward nose
        if torso_height < 1e-6:
            return 0.0, 0.0

        ratio = nose_to_shoulder / torso_height

    # Store neutral ratio during calibration
        if self._neutral_lean_ratio is None:
            return ratio, 0.0

    # How much has ratio dropped from neutral?
        lean_delta = self._neutral_lean_ratio - ratio

    # Map: 0 drop = 0 score, 0.15 drop = 1.0 score
        score = lean_delta / 0.15
        return lean_delta, float(np.clip(score, 0.0, 1.0))


    def compute(self, landmarks):
        trunk_angle,  trunk_score    = self.trunk_inclination_score(landmarks)
        cervical_cm,  cervical_score = self.cervical_displacement_score(landmarks)
        lean_delta,   lean_score     = self.forward_lean_score(landmarks)

    # PSS = weighted mean of all three
    # Forward lean gets highest weight since that's what we want the cobot to respond to
        pss_raw = (trunk_score * 0.30 +
                   cervical_score * 0.30 +
                   lean_score * 0.40)

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
