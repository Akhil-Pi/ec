"""
session_logger.py
=================
Records continuous PSS data, intervention events, and metadata for each
session. Outputs ready for the Wilcoxon / Spearman / Bland-Altman analyses
described in the proposal.

Files written per session:
  data/sessions/<participant>_<condition>_<timestamp>_frames.csv
  data/sessions/<participant>_<condition>_<timestamp>_events.csv
  data/sessions/<participant>_<condition>_<timestamp>_meta.txt
"""
import csv
import os
import time
from datetime import datetime
import config


class SessionLogger:
    def __init__(self, participant_id, condition, log_dir=None):
        assert condition in ("control", "experimental"), \
            "condition must be 'control' or 'experimental'"

        self.participant_id = participant_id
        self.condition = condition
        self.log_dir = log_dir or config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{participant_id}_{condition}_{ts}"
        self.frames_path = os.path.join(self.log_dir, f"{base}_frames.csv")
        self.events_path = os.path.join(self.log_dir, f"{base}_events.csv")
        self.meta_path = os.path.join(self.log_dir, f"{base}_meta.txt")

        self._frames_file = open(self.frames_path, "w", newline="")
        self._events_file = open(self.events_path, "w", newline="")
        self._frames_writer = csv.writer(self._frames_file)
        self._events_writer = csv.writer(self._events_file)

        self._frames_writer.writerow([
            "timestamp_s", "trunk_angle_deg", "trunk_score",
            "cervical_cm", "cervical_score", "pss_raw", "pss_smooth"
        ])
        self._events_writer.writerow([
            "timestamp_s", "event_type", "pss_at_event",
            "actions", "latency_s", "details"
        ])

        self.start_time = time.time()
        self._frames_logged = 0
        self._events_logged = 0
        self._last_frame_log_time = 0.0

    def log_frame(self, pss_components):
        """Throttled per-frame log (target: LOG_FREQUENCY_HZ)."""
        now = time.time()
        if (now - self._last_frame_log_time) < (1.0 / config.LOG_FREQUENCY_HZ):
            return
        self._last_frame_log_time = now

        self._frames_writer.writerow([
            f"{now - self.start_time:.3f}",
            f"{pss_components['trunk_angle']:.2f}",
            f"{pss_components['trunk_score']:.4f}",
            f"{pss_components['cervical_cm']:.2f}",
            f"{pss_components['cervical_score']:.4f}",
            f"{pss_components['pss_raw']:.4f}",
            f"{pss_components['pss_smooth']:.4f}",
        ])
        self._frames_logged += 1

    def log_event(self, event_type, pss_at_event,
                  actions=None, latency_s=None, details=""):
        """Discrete event logging (intervention, calibration, etc.)."""
        now = time.time()
        actions_str = "|".join([f"{a}:{m:.3f}" for a, m in (actions or [])])
        self._events_writer.writerow([
            f"{now - self.start_time:.3f}",
            event_type,
            f"{pss_at_event:.4f}",
            actions_str,
            f"{latency_s:.3f}" if latency_s is not None else "",
            details,
        ])
        self._events_logged += 1
        self._events_file.flush()

    def close(self, notes=""):
        duration = time.time() - self.start_time
        self._frames_file.close()
        self._events_file.close()

        with open(self.meta_path, "w") as f:
            f.write(f"participant_id: {self.participant_id}\n")
            f.write(f"condition: {self.condition}\n")
            f.write(f"start_time: "
                    f"{datetime.fromtimestamp(self.start_time).isoformat()}\n")
            f.write(f"duration_s: {duration:.1f}\n")
            f.write(f"frames_logged: {self._frames_logged}\n")
            f.write(f"events_logged: {self._events_logged}\n")
            f.write(f"pss_threshold: {config.PSS_THRESHOLD}\n")
            f.write(f"pss_hysteresis: {config.PSS_HYSTERESIS}\n")
            if notes:
                f.write(f"\nnotes:\n{notes}\n")

        print(f"[LOG] Session saved:\n  frames: {self.frames_path}\n  "
              f"events: {self.events_path}")
