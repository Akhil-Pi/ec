"""
main.py
=======
Run a single experimental session for the Empathetic Conservator study.

Usage:
    python main.py --participant P01 --condition control
    python main.py --participant P01 --condition experimental
    python main.py --participant P01 --condition control --simulate
    python main.py --participant P01 --condition experimental --duration 30

Workflow:
  1. Calibration phase (30s):
     Participant sits in their natural neutral posture.
     We collect their baseline cervical displacement.
  2. Task phase (45 min):
     Participant performs facsimile fine brushwork (Task A or B
     from team's setup design).
     - Control: cobot is parked, artifact is on static platform
                (no robot intervention; vision still records PSS)
     - Experimental: cobot intervenes when PSS exceeds threshold
  3. NASA-TLX questionnaire (administered separately via nasa_tlx_form.py)
"""
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
    print(f"\n=== CALIBRATION PHASE ({duration_s}s) ===")
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

    samples = []
    start = time.time()
    while (time.time() - start) < duration_s:
        ret, frame = cap.read()
        if not ret:
            continue
        frame, landmarks = detector.detect(frame)
        if landmarks:
            cm, _ = pss_calc.cervical_displacement_score(landmarks)
            samples.append(cm)

        remaining = duration_s - (time.time() - start)
        cv2.putText(frame, f"Calibrating... {remaining:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Empathetic Conservator", frame)
        cv2.waitKey(1)

    pss_calc.calibrate_neutral(samples)
    logger_obj.log_event(
        "calibration_done", 0.0,
        details=(f"neutral_offset_cm="
                 f"{pss_calc._neutral_cervical_offset:.2f},"
                 f"samples={len(samples)}")
    )
    print("=== CALIBRATION COMPLETE ===\n")


# HUD OVERLAY
def render_overlay(frame, pss_components, action_info,
                   elapsed_s, total_s, transitioning=False):
    pss   = pss_components["pss_smooth"]
    color = ((0, 255, 0) if pss < 0.4
             else (0, 165, 255) if pss < 0.7
             else (0, 0, 255))
    h, w  = frame.shape[:2]

    # PSS bar
    cv2.rectangle(frame, (10, 60), (210, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 60), (10 + int(200 * pss), 80), color, -1)
    cv2.putText(frame, f"PSS: {pss:.2f}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Sub-scores
    cv2.putText(frame,
                f"Trunk: {pss_components['trunk_angle']:.0f} deg  "
                f"Cervical: {pss_components['cervical_cm']:.1f} cm",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Status
    if transitioning:
        cv2.putText(frame, "Robot moving to task position...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
    elif action_info.get("triggered"):
        cv2.putText(frame, "INTERVENTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Status: {action_info.get('reason', '')}",
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

    detector  = PoseDetector()
    pss_calc  = PSSCalculator()
    policy    = InterventionPolicy(condition=condition)
    logger_obj = SessionLogger(participant_id, condition)
    robot     = UR3Controller(simulate=simulate)
    if not simulate:
        print("\nChecking home position...")
        at_home = robot.ensure_at_home(auto_move=True)
        if not at_home:
            print("ERROR: Robot not at home. Aborting session.")
            robot.disconnect()
            return
        logger_obj.log_event("home_verified", 0.0)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Shared PSS state for the transition thread to read
    latest_pss = {"pss_smooth": 1.0}

    try:
        # 1. Go to home
        print("Moving to HOME_JOINTS via safe path...")
        # If robot is near DESIRED_POSE already, traverse waypoints in reverse
        # For now assume robot starts near HOME_JOINTS
        robot.go_home()
        logger_obj.log_event("robot_home", 0.0)

        # 2. Calibration (user stands upright, looks forward)
        print("\nTell participant: stand upright, look straight ahead.")
        calibration_phase(detector, pss_calc, cap, logger_obj)

        # 3. Start gradual transition to DESIRED_POSE in a background thread
        #    so the vision loop keeps running while the robot moves
        import threading

        def do_transition():
            if condition == "experimental":
                robot.gradual_transition(
                    pss_callback=lambda: latest_pss   # no path argument — uses JOINT_PATH
            )

        transition_thread = threading.Thread(target=do_transition, daemon=True)
        transition_thread.start()
        logger_obj.log_event("transition_started", 0.0)

        # 4. Task phase
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
                cv2.putText(frame, "No pose - adjust camera height",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Empathetic Conservator", frame)
                cv2.waitKey(1)
                continue

            pss_components = pss_calc.compute(landmarks)
            latest_pss.update(pss_components)   # share with transition thread
            logger_obj.log_frame(pss_components)

            action_info = {"triggered": False, "reason": ""}

            # For experimental: wait for robot to finish gradual transition
            # before enabling fine adjustments. For control: policy already
            # returns "control_group" reason.
            if condition == "experimental" and transition_thread.is_alive():
                action_info["reason"] = "transitioning"
            else:
                action_info = policy.evaluate(pss_components, robot)
                if action_info["triggered"]:
                    logger_obj.log_event(
                        "intervention",
                        pss_components["pss_smooth"],
                        actions=action_info["interventions"],
                        latency_s=action_info.get("latency_s"),
                        details=f"id={action_info.get('intervention_id')}"
                    )

            frame = render_overlay(frame, pss_components, action_info,
                                   elapsed, duration_s,
                                   transitioning=transition_thread.is_alive())
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
