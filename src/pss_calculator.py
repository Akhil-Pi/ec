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
        self._neutral_cervical_offset  = None
        self._neutral_lean_ratio       = None
        self._neutral_torso_height     = None
        self._neutral_nose_drop        = None   # kept for compatibility
        self._neutral_shoulder_y       = None   # NEW: shoulder height in frame
        self._neutral_shoulder_width   = None   # NEW: shoulder width when upright
    # ---------- SUB-SCORES ----------

    def trunk_inclination_score(self, landmarks):
        """
        For downward-angled camera (30-45 deg above shoulder):
        
        When upright: shoulders appear in lower portion of frame,
                    shoulder width appears normal.
        When bending forward: shoulders rise in frame (move toward camera),
                            shoulder width appears wider (foreshortening).
        
        Primary signal: shoulder Y position rising toward top of frame.
        Secondary signal: shoulder width increasing (foreshortening effect).
        """
        if landmarks is None:
            return 0.0, 0.0

        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]
        lh = landmarks["LEFT_HIP"][:2]
        rh = landmarks["RIGHT_HIP"][:2]

        shoulder_mid = midpoint(ls, rs)
        hip_mid      = midpoint(lh, rh)

        # Signal 1: shoulder Y in frame
        # Upright: shoulder_mid[1] is relatively low (large Y value in image coords)
        # Bending forward: shoulders rise toward camera, Y decreases
        shoulder_y = shoulder_mid[1]

        # Signal 2: shoulder width in normalized image coords
        # Upright: normal width
        # Bending: shoulders foreshorten, appear wider
        shoulder_width = abs(ls[0] - rs[0])

        # Signal 3: torso compression — hip to shoulder vertical distance
        # Still useful even from above
        torso_height = hip_mid[1] - shoulder_mid[1]

        if (self._neutral_torso_height is None
                or self._neutral_nose_drop is None):
            # Return raw signals for calibration collection
            return shoulder_y, 0.0

        # How much have shoulders risen (Y decreased)?
        shoulder_rise = self._neutral_shoulder_y - shoulder_y
        rise_score    = shoulder_rise / 0.06   # 6% frame height = significant bend

        # How much has torso compressed?
        torso_delta   = self._neutral_torso_height - torso_height
        torso_score   = torso_delta / 0.06

        # How much has shoulder width increased (foreshortening)?
        width_delta   = shoulder_width - self._neutral_shoulder_width
        width_score   = width_delta / 0.04

        # Combined — shoulder rise is most reliable from above
        combined = (rise_score  * 0.50 +
                    torso_score * 0.30 +
                    width_score * 0.20)

        # Angle for logging (approximate)
        trunk_vec = (shoulder_mid[0] - hip_mid[0],
                    shoulder_mid[1] - hip_mid[1])
        vertical  = (0.0, -1.0)
        angle_deg = angle_between(trunk_vec, vertical)

        return angle_deg, float(np.clip(combined, 0.0, 1.0))

    def cervical_displacement_score(self, landmarks):
        """
        From overhead-left camera, cervical displacement is unreliable
        because the ear-shoulder horizontal offset is dominated by camera angle.
        Use ONLY for directional info (left/right), not magnitude for PSS.
        Returns (displacement_cm, score) where score is heavily dampened.
        """
        if landmarks is None:
            return 0.0, 0.0

        le = landmarks["LEFT_EAR"]
        re = landmarks["RIGHT_EAR"]
        ls = landmarks["LEFT_SHOULDER"]
        rs = landmarks["RIGHT_SHOULDER"]

        MIN_VISIBILITY = 0.7
        if (le[3] < MIN_VISIBILITY or re[3] < MIN_VISIBILITY or
                ls[3] < MIN_VISIBILITY or rs[3] < MIN_VISIBILITY):
            return 0.0, 0.0

        le = le[:2]
        re = re[:2]
        ls = ls[:2]
        rs = rs[:2]

        ear_mid      = midpoint(le, re)
        shoulder_mid = midpoint(ls, rs)
        shoulder_width = abs(ls[0] - rs[0])
        if shoulder_width < 1e-6:
            return 0.0, 0.0

        x_offset_signed = ear_mid[0] - shoulder_mid[0]
        ASSUMED_SHOULDER_WIDTH_CM = 38.0
        displacement_cm = (x_offset_signed / shoulder_width) * ASSUMED_SHOULDER_WIDTH_CM

        if self._neutral_cervical_offset is not None:
            displacement_cm = displacement_cm - self._neutral_cervical_offset

        # Dampen heavily — from this camera angle cervical is noisy
        # Only contribute to PSS if displacement is very large (> 2x neutral range)
        DAMPENING = 0.3
        score = (abs(displacement_cm) / config.CERVICAL_MAX_CM) * DAMPENING

        # Camera compensation for sign only
        camera_pos = getattr(config, 'CAMERA_POSITION', 'center')
        if camera_pos in config.CERVICAL_SENSITIVITY_COMPENSATE:
            mult = (config.CERVICAL_SENSITIVITY_COMPENSATE[camera_pos]["positive"]
                    if displacement_cm > 0
                    else config.CERVICAL_SENSITIVITY_COMPENSATE[camera_pos]["negative"])
            displacement_cm *= mult

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

    def gaze_direction_score(self, landmarks):
        """
        Detects HEAD ROTATION only — not body lean.
        Uses ear asymmetry: when head turns right, right ear becomes
        less visible and left ear moves further right in frame.
        From any camera angle, ear asymmetry is a reliable head-turn signal.
        """
        if landmarks is None:
            return 0.0, 0.0

        le = landmarks["LEFT_EAR"]
        re = landmarks["RIGHT_EAR"]
        ls = landmarks["LEFT_SHOULDER"][:2]
        rs = landmarks["RIGHT_SHOULDER"][:2]

        shoulder_width = abs(ls[0] - rs[0])
        if shoulder_width < 1e-6:
            return 0.0, 0.0

        le_vis = le[3]
        re_vis = re[3]
        le_x   = le[0]
        re_x   = re[0]

        # Method: visibility asymmetry
        # Head turns right → right ear visibility drops, left ear stays visible
        # Head turns left  → left ear visibility drops, right ear stays visible
        vis_diff = le_vis - re_vis   # positive = left more visible = head turned right
        # Scale to [-1, 1]: 0.3 difference = significant turn
        vis_score = np.clip(vis_diff / 0.3, -1.0, 1.0)

        # Method 2: ear X position relative to shoulder midpoint
        shoulder_mid_x = (ls[0] + rs[0]) / 2.0
        ear_mid_x      = (le_x + re_x) / 2.0
        pos_offset     = (ear_mid_x - shoulder_mid_x) / shoulder_width
        pos_score      = np.clip(pos_offset / 0.2, -1.0, 1.0)

        # Combine: visibility asymmetry is more reliable for head turn
        combined = vis_score * 0.6 + pos_score * 0.4

        # Camera compensation for left-positioned camera
        if hasattr(config, 'CAMERA_POSITION'):
            cam = config.CAMERA_POSITION
            if cam == "left" and combined > 0:
                combined *= 1.5
            elif cam == "right" and combined < 0:
                combined *= 1.5

        score = float(np.clip(combined, -1.0, 1.0))
        return combined, score
    
    def compute(self, landmarks):
        trunk_angle,  trunk_score    = self.trunk_inclination_score(landmarks)
        cervical_cm,  cervical_score = self.cervical_displacement_score(landmarks)
        lean_delta,   lean_score     = self.forward_lean_score(landmarks)
        gaze_offset,  gaze_score     = self.gaze_direction_score(landmarks)

        # From overhead-left camera:
        # - trunk_score and lean_score are unreliable (always 0)
        # - cervical_score is noisy (now dampened)
        # - gaze magnitude IS the primary postural effort signal
        # PSS driven mainly by gaze magnitude + dampened cervical
        gaze_magnitude = abs(gaze_score)   # 0 = centered, 1 = extreme lateral

        pss_raw = (gaze_magnitude  * 0.55 +
                cervical_score  * 0.30 +
                lean_score      * 0.10 +
                trunk_score     * 0.05)

        self.buffer.append(pss_raw)
        pss_smooth = float(np.mean(self.buffer))

        return {
            "trunk_angle":    trunk_angle,
            "trunk_score":    trunk_score,
            "cervical_cm":    cervical_cm,
            "cervical_score": cervical_score,
            "lean_delta":     lean_delta,
            "lean_score":     lean_score,
            "gaze_offset":    gaze_offset,
            "gaze_score":     gaze_score,
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

    def calibrate_neutral_trunk(self, torso_samples, nose_samples,
                                shoulder_y_samples=None,
                                shoulder_width_samples=None):
        if not torso_samples:
            self._neutral_torso_height   = 0.3
            self._neutral_nose_drop      = 0.15
            self._neutral_shoulder_y     = 0.4
            self._neutral_shoulder_width = 0.3
            return

        self._neutral_torso_height   = float(np.median(torso_samples))
        self._neutral_nose_drop      = float(np.median(nose_samples)) \
                                    if nose_samples else 0.15
        self._neutral_shoulder_y     = float(np.median(shoulder_y_samples)) \
                                    if shoulder_y_samples else 0.4
        self._neutral_shoulder_width = float(np.median(shoulder_width_samples)) \
                                    if shoulder_width_samples else 0.3

        print(f"[PSS] Neutral torso height:    {self._neutral_torso_height:.3f}")
        print(f"[PSS] Neutral shoulder Y:      {self._neutral_shoulder_y:.3f}")
        print(f"[PSS] Neutral shoulder width:  {self._neutral_shoulder_width:.3f}")
    
    
    def calibrate_neutral_lean(self, lean_samples):
        if not lean_samples:
            self._neutral_lean_ratio = 0.3   # sensible default
            return
        self._neutral_lean_ratio = float(np.median(lean_samples))
        print(f"[PSS] Neutral lean ratio: {self._neutral_lean_ratio:.3f}")

    def reset(self):
        self.buffer.clear()
        self._neutral_cervical_offset  = None
        self._neutral_lean_ratio       = None
        self._neutral_torso_height     = None
        self._neutral_nose_drop        = None
        self._neutral_shoulder_y       = None
        self._neutral_shoulder_width   = None
