"""
intervention_policy.py
======================
Decides WHEN and HOW the cobot intervenes to correct posture.

Policy logic (intentionally simple and interpretable for the report):
  - PSS < threshold        -> no action
  - PSS >= threshold       -> arm a "monitoring" timer
  - Sustained >= 2 seconds -> trigger correction
  - After correction       -> 5-second cooldown
  - Hysteresis             -> PSS must drop below (threshold - 0.10) to reset

Correction strategy:
  - High trunk inclination  -> raise artifact (Z+)
  - High cervical disp.     -> tilt artifact toward conservator (rx+)
  - Both                    -> combined intervention

Latency tracking:
  - Records the timestamp at threshold crossing (t_arm)
  - Records the timestamp the robot completes its first move (t_act)
  - Latency = t_act - t_arm  (used for proposal's Spearman analysis)
"""
import time
import logging
import config
import numpy as np

logger = logging.getLogger(__name__)


class InterventionPolicy:
    def __init__(self, condition="experimental", threshold=None, hysteresis=None,
                 sustained_seconds=0.75, cooldown_seconds=1.5):
        assert condition in ("control", "experimental"), \
            "condition must be 'control' or 'experimental'"
        self.condition = condition
        self.threshold = threshold or config.PSS_THRESHOLD
        self.hysteresis = hysteresis or config.PSS_HYSTERESIS
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds
        self._last_rotation_at = float("-inf")

        self._above_threshold_since = None
        self._last_intervention_at = float("-inf")
        self._rotation_cooldown = 2.0
        self._last_pss_at_intervention = None
        self._intervention_count = 0
        self._in_corrected_state = False

    def reset(self):
        self._above_threshold_since = None
        self._last_intervention_at = float("-inf")
        self._last_pss_at_intervention = None
        self._intervention_count = 0
        self._in_corrected_state = False

    def evaluate(self, pss_components, robot, now=None):
        now = now or time.time()

        if not pss_components or "pss_smooth" not in pss_components:
            logger.error("[POLICY] Missing pss_smooth")
            return {"triggered": False, "reason": "error",
                    "interventions": [], "pss": 0.0}

        pss  = pss_components["pss_smooth"]
        gaze = pss_components.get("gaze_score", 0.0)
        result = {"triggered": False, "reason": "",
                "interventions": [], "pss": pss}

        if self.condition == "control":
            result["reason"] = "control_group"
            return result

        # ── ROTATION (independent of PSS, own cooldown) ──────────────────────
        rotation_triggered = False
        if abs(gaze) > 0.5 and pss >= (self.threshold * 0.6):
            # Only rotate if there's at least some postural strain
            # threshold * 0.6 = 0.15 — low bar but filters pure noise
            time_since_rotate = now - self._last_rotation_at
            if time_since_rotate >= self._rotation_cooldown:
                magnitude = -config.ROTATION_ADJUST_STEP * gaze
                robot.adjust_rotation(magnitude)
                self._last_rotation_at = now
                rotation_triggered = True
                logger.info(f"[POLICY] Rotation: gaze={gaze:+.2f} "
                            f"PSS={pss:.3f} mag={magnitude:+.3f}")

        # ── PSS-BASED POSTURAL CORRECTION (raise / tilt / forward / lateral) ─
        # Hysteresis recovery
        if (self._in_corrected_state
                and pss < (self.threshold - self.hysteresis)):
            self._in_corrected_state = False
            self._above_threshold_since = None
            logger.info(f"[POLICY] Recovered: PSS={pss:.3f}")

        # PSS cooldown
        if self._last_intervention_at != float("-inf"):
            time_since_last = now - self._last_intervention_at
            pss_improvement = (self._last_pss_at_intervention - pss
                            if self._last_pss_at_intervention else 0.0)

            if pss_improvement > 0.10:
                dynamic_cooldown = 0.5
            elif pss >= (self.threshold + 0.20):
                dynamic_cooldown = 0.8
            elif pss >= self.threshold:
                dynamic_cooldown = 1.0
            else:
                dynamic_cooldown = 0.3

            if time_since_last < dynamic_cooldown:
                reason = "rotation_only" if rotation_triggered else "cooldown"
                result["reason"] = reason
                result["triggered"] = rotation_triggered
                return result

        # Below PSS threshold
        if pss < self.threshold:
            self._above_threshold_since = None
            result["reason"] = "rotation_only" if rotation_triggered else "below_threshold"
            result["triggered"] = rotation_triggered
            return result

        # Arm sustained timer
        if self._above_threshold_since is None:
            self._above_threshold_since = now
            result["reason"] = "monitoring"
            result["triggered"] = rotation_triggered
            return result

        sustained = now - self._above_threshold_since
        if sustained < self.sustained_seconds:
            result["reason"] = f"sustained({sustained:.1f}s)"
            result["triggered"] = rotation_triggered
            return result

        # ── TRIGGER PSS INTERVENTION ─────────────────────────────────────────
        t_arm = self._above_threshold_since
        interventions = self._compute_postural_interventions(pss_components)

        dx = dy = dz = 0.0
        rotate_magnitude = 0.0

        for action, magnitude in interventions:
            if action == "raise":
                dz  += magnitude
            elif action == "rotate":
                rotate_magnitude += magnitude  # separate from tilt
            elif action == "forward":
                dy  += magnitude
            elif action == "lateral":
                dx  += magnitude

        dz  = max(min(dz,  config.Z_ADJUST_STEP),    -config.Z_ADJUST_STEP)
        dx  = max(min(dx,  config.X_ADJUST_STEP),    -config.X_ADJUST_STEP)
        dy  = max(min(dy,  config.Y_ADJUST_STEP),    -config.Y_ADJUST_STEP)

        # Execute linear moves (raise, forward, lateral)
        if any([dx, dy, dz]):
            ok = robot.move_relative(dx=dx, dy=dy, dz=dz, asynchronous=True)
            if not ok:
                logger.warning("[POLICY] Postural intervention blocked")

        # Execute wrist rotation separately
        if abs(rotate_magnitude) > 0.001:
            robot.adjust_rotation(rotate_magnitude)

        time.sleep(0.05)
        t_act = time.time()

        self._intervention_count += 1
        self._last_intervention_at        = now
        self._last_pss_at_intervention    = pss
        self._above_threshold_since       = None
        self._in_corrected_state          = True

        result.update({
            "triggered":        True,
            "reason":           "intervention",
            "interventions":    interventions,
            "intervention_id":  self._intervention_count,
            "t_arm":            t_arm,
            "t_act":            t_act,
            "latency_s":        t_act - t_arm,
        })
        logger.info(f"[POLICY] PSS intervention #{self._intervention_count}: "
                    f"PSS={pss:.3f} dz={dz:+.3f} rotate={rotate_magnitude:+.3f} "
                    f"dx={dx:+.3f} dy={dy:+.3f}")
        return result


    def _compute_postural_interventions(self, pss_components):
        actions  = []
        cervical    = pss_components.get("cervical_score", 0.0)
        cervical_cm = pss_components.get("cervical_cm",    0.0)

        # Forward bending (trunk inclination) → raise artifact to reduce reaching distance
        trunk_score = pss_components.get("trunk_score", 0.0)
        if trunk_score > 0.3:
            actions.append(("raise", config.Z_ADJUST_STEP * trunk_score))

        # Cervical directional rotation — compensate user's head tilt with wrist rotation
        # When user tilts head left (cervical_cm < 0), rotate robot RIGHT (positive)
        # When user tilts head right (cervical_cm > 0), rotate robot LEFT (negative)
        if cervical > 0.25:
            magnitude = config.ROTATION_ADJUST_STEP * cervical
            if cervical_cm > 0:  # head tilted right → rotate left (negative)
                magnitude = -magnitude
            actions.append(("rotate", magnitude))


        if not actions:
            actions.append(("raise", config.Z_ADJUST_STEP * 0.4))

        return actions