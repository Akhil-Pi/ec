import sys
import time
import os

# Allow importing config from /src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("ERROR: ur_rtde not installed. Run: pip install ur-rtde")
    sys.exit(1)


def test_connection():
    print(f"Attempting to connect to UR3 at {config.UR3_IP} ...")
    try:
        rtde_r = RTDEReceiveInterface(config.UR3_IP)
    except Exception as e:
        print(f"FAIL: Could not connect.\n  Error: {e}")
        print("\nChecklist:")
        print("  - Is the robot powered on?")
        print("  - Is the IP correct? (config.py)")
        print(f"  - Try: ping {config.UR3_IP}")
        print("  - Is the robot in Remote Control mode?")
        return False

    pose = rtde_r.getActualTCPPose()
    joints = rtde_r.getActualQ()
    print("OK: Connected!")
    print(f"  TCP pose: {[f'{v:.3f}' for v in pose]}")
    print(f"  Joints:   {[f'{v:.3f}' for v in joints]}")
    return True


def test_safe_motion():
    print("\n" + "=" * 50)
    print("MOTION TEST - 3 cm up, then back")
    print("=" * 50)
    print("Confirm BEFORE pressing Enter:")
    print("  [ ] Workspace clear of obstacles")
    print("  [ ] Speed slider on teach pendant <= 20%")
    print("  [ ] One team member's hand is on E-stop")
    print("  [ ] An empty program is RUNNING on Polyscope")
    input("\nPress Enter to continue, Ctrl+C to abort... ")

    try:
        rtde_c = RTDEControlInterface(config.UR3_IP)
        rtde_r = RTDEReceiveInterface(config.UR3_IP)
    except Exception as e:
        print(f"FAIL connecting: {e}")
        return

    cur = rtde_r.getActualTCPPose()
    print(f"Starting pose: {cur}")
    target = list(cur)
    target[2] += 0.03  # +3 cm Z

    print("Moving up 3 cm in 3 seconds...")
    time.sleep(3)
    rtde_c.moveL(target, 0.05, 0.1)
    print("Pose now:", rtde_r.getActualTCPPose())

    print("Moving back...")
    rtde_c.moveL(cur, 0.05, 0.1)
    print("Pose now:", rtde_r.getActualTCPPose())

    rtde_c.stopScript()
    print("\nMotion test complete.")


if __name__ == "__main__":
    if test_connection():
        ans = input("\nRun motion test? (y/N): ").strip().lower()
        if ans == "y":
            test_safe_motion()
