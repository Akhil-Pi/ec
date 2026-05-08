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

logger = logging.getLogger(__name__)


class InterventionPolicy:
    def __init__(self, threshold=None, hysteresis=None,
                 sustained_seconds=2.0, cooldown_seconds=5.0):
        self.threshold = threshold or config.PSS_THRESHOLD
        self.hysteresis = hysteresis or config.PSS_HYSTERESIS
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds

        self._above_threshold_since = None
        self._last_intervention_at = float("-inf")
        self._intervention_count = 0
        self._in_corrected_state = False

    def reset(self):
        self._above_threshold_since = None
        self._last_intervention_at = float("-inf")
        self._intervention_count = 0
        self._in_corrected_state = False

    def evaluate(self, pss_components, robot, now=None):
        """
        Decide and execute intervention if appropriate.
        """
        now = now or time.time()
        pss = pss_components["pss_smooth"]
        result = {"triggered": False, "reason": "",
                  "interventions": [], "pss": pss}

        # Hysteresis recovery
        if (self._in_corrected_state
                and pss < (self.threshold - self.hysteresis)):
            self._in_corrected_state = False
            self._above_threshold_since = None
            logger.info(f"[POLICY] Recovered: PSS={pss:.3f}")

        # Cooldown
        if (now - self._last_intervention_at) < self.cooldown_seconds:
            result["reason"] = "cooldown"
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
        for action, magnitude in interventions:
            if action == "raise":
                ok = robot.adjust_height(magnitude)
            elif action == "tilt":
                ok = robot.adjust_tilt(magnitude)
            else:
                ok = False
            if not ok:
                logger.warning(f"[POLICY] Intervention {action} blocked")
        t_act = now

        self._intervention_count += 1
        self._last_intervention_at = now
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
        """Decide which corrections to apply based on which sub-score dominates."""
        actions = []
        trunk = pss_components["trunk_score"]
        cervical = pss_components["cervical_score"]

        if trunk > 0.4:
            actions.append(("raise", config.Z_ADJUST_STEP * (0.5 + trunk)))
        if cervical > 0.4:
            actions.append(("tilt", config.TILT_ADJUST_STEP * (0.5 + cervical)))
        if not actions:
            # Generic fallback if PSS is high but neither sub-score dominates
            actions.append(("raise", config.Z_ADJUST_STEP * 0.5))
        return actions
