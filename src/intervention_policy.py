"""
intervention_policy.py
======================
Decides WHEN and HOW the cobot intervenes to correct posture.

Policy logic (intentionally simple and interpretable for the report):
  - PSS < threshold        -> no action
  - PSS >= threshold       -> arm a "monitoring" timer
  - Sustained >= 2 seconds -> trigger correction
  - After correction       -> 5-second cooldown (dynamic, PSS-dependent)
  - Hysteresis             -> PSS must drop below (threshold - 0.10) to reset

Correction strategy (NEW: Directional Tilt → Wrist Rotation):
  - High trunk inclination    -> raise artifact (Z+, for low-head posture)
  - Head left tilt            -> rotate wrist RIGHT (counterclockwise, positive drz)
  - Head right tilt           -> rotate wrist LEFT (clockwise, negative drz)
  - Forward lean              -> bring artifact toward participant (Y+)
  - Multiple issues           -> combined interventions

Rationale:
  Camera mounted on upper-left at ~15° elevation (low-angle, upward-looking).
  When conservator tilts head left/right, wrist rotates opposite direction
  to keep artifact in optimal viewing angle.

Robustness:
  - Visibility checks (0.6+ confidence) to handle arm occlusion
  - Dynamic cooldown based on PSS improvement trajectory
  - Fallback raise for unspecified high PSS

Latency tracking:
  - Records timestamp at threshold crossing (t_arm)
  - Records timestamp when robot completes motion (t_act)
  - Latency = t_act - t_arm
"""
import time
import logging
import numpy as np
import config

logger = logging.getLogger(__name__)


class InterventionPolicy:
    def __init__(self, condition="experimental", threshold=None, hysteresis=None,
                 sustained_seconds=2.0, cooldown_seconds=5.0):
        assert condition in ("control", "experimental"), \
            "condition must be 'control' or 'experimental'"
        self.condition = condition
        self.threshold = threshold or config.PSS_THRESHOLD
        self.hysteresis = hysteresis or config.PSS_HYSTERESIS
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds

        self._above_threshold_since = None
        self._last_intervention_at = float("-inf")
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
        """
        Decide and execute intervention if appropriate.
        For control group, only monitors PSS but never intervenes.
        """
        now = now or time.time()

        # Defensive checks for required fields
        if not pss_components or "pss_smooth" not in pss_components:
            logger.error("[POLICY] Missing 'pss_smooth' in pss_components")
            return {"triggered": False, "reason": "error", "interventions": [], "pss": 0.0}

        pss = pss_components["pss_smooth"]
        result = {"triggered": False, "reason": "",
                  "interventions": [], "pss": pss}

        # Control group: only log PSS, never intervene
        if self.condition == "control":
            result["reason"] = "control_group"
            return result

        # Hysteresis recovery
        if (self._in_corrected_state
                and pss < (self.threshold - self.hysteresis)):
            self._in_corrected_state = False
            self._above_threshold_since = None
            logger.info(f"[POLICY] Recovered: PSS={pss:.3f}")

        # Dynamic cooldown: adjust based on PSS improvement and current level
        if self._last_intervention_at != float("-inf"):
            time_since_last = now - self._last_intervention_at

            # Check if PSS has improved since last intervention
            pss_improvement = (self._last_pss_at_intervention - pss
                              if self._last_pss_at_intervention else 0.0)

            # Strategy: if PSS improved > 0.1, allow faster re-evaluation
            if pss_improvement > 0.10:
                # Participant is actively correcting, use very short cooldown (1s)
                dynamic_cooldown = 1.0
                cooldown_reason = "fast_recovery"
            # If PSS is much higher than threshold, need quick response (1.5s cooldown)
            elif pss >= (self.threshold + 0.20):
                dynamic_cooldown = 1.5
                cooldown_reason = "high_pss"
            # If PSS is moderately above threshold, use standard cooldown
            elif pss >= self.threshold:
                dynamic_cooldown = 3.0
                cooldown_reason = "standard"
            # If PSS dropped below threshold, minimal cooldown (0.5s)
            else:
                dynamic_cooldown = 0.5
                cooldown_reason = "recovering"

            if time_since_last < dynamic_cooldown:
                result["reason"] = f"cooldown({cooldown_reason}:{dynamic_cooldown:.1f}s)"
                return result

        # Below threshold
        if pss < self.threshold:
            self._above_threshold_since = None
            result["reason"] = "below_threshold"
            return result

        # Threshold exceeded - first time? Start arming timer.
        if self._above_threshold_since is None:
            self._above_threshold_since = now
            result["reason"] = "monitoring"
            return result

        # Sustained?
        sustained = now - self._above_threshold_since
        if sustained < self.sustained_seconds:
            result["reason"] = f"sustained({sustained:.1f}s)"
            return result

        # ----- TRIGGER -----
        t_arm = self._above_threshold_since
        interventions = self._compute_interventions(pss_components)

        # Execute interventions
        for action, magnitude in interventions:
            if action == "raise":
                ok = robot.adjust_height(magnitude)
            elif action == "rotate":
                ok = robot.adjust_rotate(magnitude)
            elif action == "tilt":
                ok = robot.adjust_tilt(magnitude)
            elif action == "forward":
                ok = robot.adjust_depth(magnitude)
            else:
                ok = False
            if not ok:
                logger.warning(f"[POLICY] Intervention {action} blocked")

        # Wait for robot motion to complete before recording t_act
        # This ensures latency = t_act - t_arm accurately reflects robot response time
        robot.wait_for_motion_complete(timeout_s=5.0)
        t_act = time.time()

        self._intervention_count += 1
        self._last_intervention_at = now
        self._last_pss_at_intervention = pss
        self._above_threshold_since = None
        self._in_corrected_state = True

        result.update({
            "triggered": True,
            "reason": "intervention",
            "interventions": interventions,
            "intervention_id": self._intervention_count,
            "t_arm": t_arm,
            "t_act": t_act,
            "latency_s": t_act - t_arm,
        })
        logger.info(f"[POLICY] Intervention #{self._intervention_count}: "
                    f"PSS={pss:.3f}, latency={t_act-t_arm:.2f}s, "
                    f"actions={interventions}")
        return result

    def _compute_interventions(self, pss_components):
        actions = []
        trunk        = pss_components.get("trunk_score",    0.0)
        tilt_score   = pss_components.get("tilt_score",     0.0)
        tilt_angle   = pss_components.get("tilt_angle_deg", 0.0)
        lean         = pss_components.get("lean_score",     0.0)

        logger.debug(f"[POLICY] _compute_interventions: "
                     f"trunk={trunk:.3f}, tilt_score={tilt_score:.3f}, "
                     f"tilt_angle={tilt_angle:.1f}°, lean={lean:.3f}")

        # Forward lean detected → bring artifact toward participant (Y axis)
        if lean > 0.3:
            actions.append(("forward", config.Y_ADJUST_STEP * lean))

        # Trunk inclined → raise artifact
        if trunk > 0.5:
            actions.append(("raise", config.Z_ADJUST_STEP * trunk))
            logger.info(f"[POLICY] Trunk intervention: score={trunk:.3f} → raise")

        # Head tilt (left/right) → wrist rotation
        # Positive angle = head tilted right → rotate wrist LEFT (negative drz)
        # Negative angle = head tilted left → rotate wrist RIGHT (positive drz)
        if tilt_score > 0.3:
            # Invert the sign: head right → wrist left
            magnitude = -config.ROTATE_ADJUST_STEP * tilt_score * np.sign(tilt_angle)
            actions.append(("rotate", magnitude))
            direction = "LEFT" if magnitude > 0 else "RIGHT"
            logger.info(f"[POLICY] Head tilt intervention: "
                        f"score={tilt_score:.3f}, angle={tilt_angle:.1f}° → rotate {direction}")

        if not actions:
            # Fallback: when PSS exceeds threshold but no specific intervention needed
            actions.append(("raise", config.Z_ADJUST_STEP * 0.3))
            logger.debug(f"[POLICY] Fallback raise (PSS high but no major issues)")

        return actions
