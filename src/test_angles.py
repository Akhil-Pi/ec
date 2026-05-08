import cv2
import config
from pose_detector import PoseDetector
from pss_calculator import PSSCalculator

det  = PoseDetector()
calc = PSSCalculator(smoothing_window=1)
cap  = cv2.VideoCapture(config.CAMERA_INDEX)

print("Lean LEFT, RIGHT, FORWARD, BACK - watch values change")
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
        pss      = res["pss_smooth"]
        print(f"Trunk: {trunk:5.1f} deg   Cervical: {cervical:5.2f} cm   PSS: {pss:.3f}", end="\r")

        label = f"PSS:{pss:.2f}  Trunk:{trunk:.0f}deg  Cerv:{cervical:.1f}cm"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("PSS Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
det.close()