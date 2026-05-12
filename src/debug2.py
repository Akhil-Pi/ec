"""
debug_signals.py
================
Shows all PSS signals live so you can verify each one independently.
Run this and perform each movement one at a time:
  1. Stand neutral
  2. Lean forward only
  3. Lean right only
  4. Look right (turn head only, don't move body)
  5. Bend down only

Watch which signals change for each movement.
"""
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import config
from pose_detector import PoseDetector
from pss_calculator import PSSCalculator

det  = PoseDetector()
calc = PSSCalculator(smoothing_window=1)  # no smoothing, instant values

# Fake calibration with neutral defaults so scores work immediately
calc._neutral_cervical_offset  = 0.0
calc._neutral_lean_ratio       = 0.05
calc._neutral_torso_height     = 0.35
calc._neutral_nose_drop        = 0.15
calc._neutral_shoulder_y       = 0.48
calc._neutral_shoulder_width   = 0.13

cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

print("Signal debug - press q to quit")
print("Columns: Trunk  Lean  Gaze  Cervical  PSS")
print("Perform each movement and watch which signal responds\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, lm = det.detect(frame)
    if lm:
        res = calc.compute(lm)

        trunk   = res["trunk_score"]
        lean    = res["lean_score"]
        gaze    = res["gaze_score"]
        cerv    = res["cervical_score"]
        pss     = res["pss_smooth"]
        cerv_cm = res["cervical_cm"]

        # Terminal output
        print(f"Trunk:{trunk:.2f}  "
              f"Lean:{lean:.2f}  "
              f"Gaze:{gaze:+.2f}  "
              f"Cerv:{cerv:.2f}({cerv_cm:+.1f}cm)  "
              f"PSS:{pss:.3f}",
              end="\r")

        # Visual bars on frame
        h, w = frame.shape[:2]
        bar_w = 120
        labels = [
            ("Trunk",  trunk,       (0, 200, 100)),
            ("Lean",   lean,        (0, 150, 255)),
            ("Cerv",   cerv,        (200, 100, 0)),
            ("PSS",    pss,         (0, 0, 255)),
        ]
        for i, (label, val, color) in enumerate(labels):
            y = 30 + i * 35
            cv2.rectangle(frame, (10, y), (10 + bar_w, y + 20),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (10, y),
                          (10 + int(bar_w * max(0, val)), y + 20),
                          color, -1)
            cv2.putText(frame, f"{label}:{val:.2f}",
                        (bar_w + 20, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # Gaze bar (signed, centered)
        gy = 30 + 4 * 35
        cx = 10 + bar_w // 2
        cv2.rectangle(frame, (10, gy), (10 + bar_w, gy + 20),
                      (50, 50, 50), -1)
        gaze_px = int(bar_w / 2 * max(-1, min(1, gaze)))
        if gaze_px >= 0:
            cv2.rectangle(frame, (cx, gy), (cx + gaze_px, gy + 20),
                          (255, 100, 0), -1)
        else:
            cv2.rectangle(frame, (cx + gaze_px, gy), (cx, gy + 20),
                          (100, 100, 255), -1)
        cv2.line(frame, (cx, gy), (cx, gy + 20), (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze:{gaze:+.2f}",
                    (bar_w + 20, gy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # Threshold line on PSS bar
        thresh_x = 10 + int(bar_w * config.PSS_THRESHOLD)
        pss_y = 30 + 3 * 35
        cv2.line(frame, (thresh_x, pss_y), (thresh_x, pss_y + 20),
                 (0, 0, 255), 2)

    cv2.imshow("Signal Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
det.close()
print()