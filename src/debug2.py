import cv2
import config
from pose_detector import PoseDetector
from pss_calculator import PSSCalculator

det  = PoseDetector()
calc = PSSCalculator(smoothing_window=1)  # no smoothing so values update instantly
cap  = cv2.VideoCapture(config.CAMERA_INDEX)

print("Bend forward as much as possible - watch Lean score")
print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, lm = det.detect(frame)
    if lm:
        res = calc.compute(lm)
        trunk    = res["trunk_angle"]
        cervical = res["cervical_cm"]
        lean     = res["lean_score"]
        pss      = res["pss_smooth"]

        # Raw ratio for debugging
        import numpy as np
        ls = lm["LEFT_SHOULDER"][:2]
        rs = lm["RIGHT_SHOULDER"][:2]
        lh = lm["LEFT_HIP"][:2]
        rh = lm["RIGHT_HIP"][:2]
        nose = lm["NOSE"][:2]
        sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        torso_h = abs(hip_mid[1] - sh_mid[1])
        nose_sh = abs(sh_mid[1] - nose[1])
        ratio   = nose_sh / torso_h if torso_h > 1e-6 else 0

        print(f"Trunk:{trunk:5.1f}deg  Cerv:{cervical:5.1f}cm  "
              f"Lean:{lean:.3f}  Ratio:{ratio:.3f}  PSS:{pss:.3f}", end="\r")

        cv2.putText(frame,
                    f"Trunk:{trunk:.0f} Cerv:{cervical:.1f} "
                    f"Lean:{lean:.2f} PSS:{pss:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame,
                    f"Ratio:{ratio:.3f} NeutralLean:{calc._neutral_lean_ratio}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("PSS Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
det.close()