import argparse
import time
import logging
import cv2

import config
from pose_detector import PoseDetector
from pss_calculator import PSSCalculator
from intervention_policy import InterventionPolicy
from ur3_controller import UR3Controller
from session_logger import SessionLogger


# CALIBRATION PHASE
def calibration_phase(detector, pss_calc, cap, logger_obj,
                      duration_s=None):
    duration_s = duration_s or config.CALIBRATION_DURATION_S
    print(f"\n CALIBRATION PHASE ({duration_s}s)")
    print("Tell participant: 'Please sit upright in your natural neutral "
          "posture, looking straight ahead.'")
    print("Press SPACE to start...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "Press SPACE to start calibration",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.imshow("Empathetic Conservator", frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    cervical_samples       = []
    lean_samples           = []
    torso_samples          = []
    nose_samples           = []
    shoulder_y_samples     = []
    shoulder_width_samples = []

    start = time.time()
    while (time.time() - start) < duration_s:
        ret, frame = cap.read()
        if not ret:
            continue
        frame, landmarks = detector.detect(frame)
        if landmarks:
            # Cervical
            cm, _ = pss_calc.cervical_displacement_score(landmarks)
            cervical_samples.append(cm)

            ls   = landmarks["LEFT_SHOULDER"][:2]
            rs   = landmarks["RIGHT_SHOULDER"][:2]
            lh   = landmarks["LEFT_HIP"][:2]
            rh   = landmarks["RIGHT_HIP"][:2]
            nose = landmarks["NOSE"][:2]

            sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

            torso_samples.append(hip_mid[1] - sh_mid[1])
            nose_samples.append(sh_mid[1] - nose[1])
            shoulder_y_samples.append(sh_mid[1])
            shoulder_width_samples.append(abs(ls[0] - rs[0]))

            # Lean (arm spread)
            le = landmarks.get("LEFT_ELBOW")
            re = landmarks.get("RIGHT_ELBOW")
            lsv = landmarks.get("LEFT_SHOULDER")
            rsv = landmarks.get("RIGHT_SHOULDER")
            if all([le, re, lsv, rsv]) and all(
                    lm[3] > 0.6 for lm in [le, re, lsv, rsv]):
                left_spread  = abs(le[:2][0] - lsv[:2][0])
                right_spread = abs(re[:2][0] - rsv[:2][0])
                lean_samples.append((left_spread + right_spread) / 2.0)

        remaining = duration_s - (time.time() - start)
        cv2.putText(frame, f"Calibrating... {remaining:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Empathetic Conservator", frame)
        cv2.waitKey(1)

    pss_calc.calibrate_neutral(cervical_samples)
    pss_calc.calibrate_neutral_lean(lean_samples)
    pss_calc.calibrate_neutral_trunk(
        torso_samples,
        nose_samples,
        shoulder_y_samples,
        shoulder_width_samples
    )
    print("CALIBRATION COMPLETE \n")


# HUD OVERLAY
def render_overlay(frame, pss_components, action_info, elapsed_s, total_s):
    pss   = pss_components["pss_smooth"]
    color = ((0, 255, 0) if pss < 0.4
             else (0, 165, 255) if pss < 0.7
             else (0, 0, 255))
    h, w  = frame.shape[:2]

    # PSS bar
    cv2.rectangle(frame, (10, 60), (210, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 60),
                  (10 + int(200 * pss), 80), color, -1)
    cv2.putText(frame, f"PSS: {pss:.2f}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Sub-scores
    cv2.putText(frame,
                f"Trunk:{pss_components['trunk_angle']:.0f}deg  "
                f"Cerv:{pss_components['cervical_cm']:.1f}cm  "
                f"Lean:{pss_components.get('lean_score', 0):.2f}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Status
    if action_info.get("triggered"):
        cv2.putText(frame, "INTERVENTION",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame,
                    f"Status: {action_info.get('reason', '')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

    # Timer
    rem = max(0, total_s - elapsed_s)
    cv2.putText(frame,
                f"{int(rem // 60):02d}:{int(rem % 60):02d}",
                (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    return frame


# SESSION RUNNER
def run_session(participant_id, condition, simulate=False, duration_min=None):
    duration_s = (duration_min or config.SESSION_DURATION_MIN) * 60

    detector   = PoseDetector()
    pss_calc   = PSSCalculator()
    policy     = InterventionPolicy(condition=condition)
    logger_obj = SessionLogger(participant_id, condition)
    robot      = UR3Controller(simulate=simulate)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    try:
        # 1. Move to desired position
        if not simulate:
            ok = robot.move_to_desired()
            if not ok:
                print("ERROR: Could not reach DESIRED position.")
                return
        logger_obj.log_event("at_desired_pose", 0.0)

        # 2. Calibration
        print("\nTell participant: stand upright, look straight ahead.")
        calibration_phase(detector, pss_calc, cap, logger_obj)

        # 3. Task phase
        print(f"\n=== TASK PHASE ({duration_s/60:.0f} min) ===")
        print("Press 'q' to abort. Press 'i' to mark a manual event.")
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed >= duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame, landmarks = detector.detect(frame)

            if landmarks is None:
                cv2.putText(frame, "No pose - adjust camera",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Empathetic Conservator", frame)
                cv2.waitKey(1)
                continue

            pss_components = pss_calc.compute(landmarks)
            logger_obj.log_frame(pss_components)

            action_info = policy.evaluate(pss_components, robot)
            if action_info["triggered"]:
                logger_obj.log_event(
                    "intervention",
                    pss_components["pss_smooth"],
                    actions=action_info["interventions"],
                    latency_s=action_info.get("latency_s"),
                    details=f"id={action_info.get('intervention_id')}"
                )

            frame = render_overlay(frame, pss_components,
                                   action_info, elapsed, duration_s)
            cv2.imshow("Empathetic Conservator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger_obj.log_event("user_abort",
                                     pss_components["pss_smooth"])
                break
            elif key == ord("i"):
                logger_obj.log_event("manual_marker",
                                     pss_components["pss_smooth"])

        print("\n=== SESSION COMPLETE ===")
        print("Now run: python ../scripts/nasa_tlx_form.py")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        robot.disconnect()
        logger_obj.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run an Empathetic Conservator session")
    parser.add_argument("--participant", required=True,
                        help="Participant ID (e.g. P01)")
    parser.add_argument("--condition", required=True,
                        choices=["control", "experimental"])
    parser.add_argument("--simulate", action="store_true",
                        help="Run without UR3 (CV-only)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Override session duration (minutes)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    run_session(args.participant, args.condition,
                simulate=args.simulate, duration_min=args.duration)


if __name__ == "__main__":
    main()
