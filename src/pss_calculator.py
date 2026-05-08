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
        if landmarks is None:
            return 0.0, 0.0

        # Pick the more visible side
        l_ear_vis = landmarks["LEFT_EAR"][3]
        r_ear_vis = landmarks["RIGHT_EAR"][3]
        l_sh_vis  = landmarks["LEFT_SHOULDER"][3]
        r_sh_vis  = landmarks["RIGHT_SHOULDER"][3]

        if l_ear_vis > r_ear_vis and l_sh_vis > r_sh_vis:
            ear      = landmarks["LEFT_EAR"][:2]
            shoulder = landmarks["LEFT_SHOULDER"][:2]
        elif r_ear_vis > l_ear_vis and r_sh_vis > l_sh_vis:
            ear      = landmarks["RIGHT_EAR"][:2]
            shoulder = landmarks["RIGHT_SHOULDER"][:2]
        else:
        # fallback to midpoints
            le = landmarks["LEFT_EAR"][:2]
            re = landmarks["RIGHT_EAR"][:2]
            ls = landmarks["LEFT_SHOULDER"][:2]
            rs = landmarks["RIGHT_SHOULDER"][:2]
            ear      = midpoint(le, re)
            shoulder = midpoint(ls, rs)

        # From side-on view, forward head = ear is AHEAD of shoulder in x
        # shoulder width trick doesn't work side-on, use image width fraction instead
        x_offset = abs(ear[0] - shoulder[0])

        # From side, a 5cm forward head displacement ≈ 8% of frame width at 1.5m
        # Calibration handles the rest — this maps raw offset to approximate cm
        ASSUMED_SHOULDER_WIDTH_CM = 38.0
        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]
        shoulder_width = abs(ls[0] - rs[0])

        # Side-on: shoulder width appears very small (near zero)
        # Use a fixed fraction of frame width instead
        if shoulder_width < 0.05:
            # Side-on camera — shoulder width not reliable
            # Use 15% of frame width as proxy for 5cm
            displacement_cm = (x_offset / 0.15) * 5.0
        else:
            # Angled camera — shoulder width visible
            displacement_cm = (x_offset / shoulder_width) * ASSUMED_SHOULDER_WIDTH_CM

        if self._neutral_cervical_offset is not None:
            displacement_cm = max(0.0, displacement_cm - self._neutral_cervical_offset)

        score = displacement_cm / config.CERVICAL_MAX_CM
        return displacement_cm, float(np.clip(score, 0.0, 1.0))

    # ---------- COMPOSITE PSS ----------

    def compute(self, landmarks):
        """Compute PSS and components for a single frame."""
        trunk_angle, trunk_score = self.trunk_inclination_score(landmarks)
        cervical_cm, cervical_score = self.cervical_displacement_score(landmarks)

        pss_raw = (trunk_score + cervical_score) / 2.0
        self.buffer.append(pss_raw)
        pss_smooth = float(np.mean(self.buffer))

        return {
            "trunk_angle": trunk_angle,
            "trunk_score": trunk_score,
            "cervical_cm": cervical_cm,
            "cervical_score": cervical_score,
            "pss_raw": pss_raw,
            "pss_smooth": pss_smooth,
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

    def reset(self):
        self.buffer.clear()
        self._neutral_cervical_offset = None
