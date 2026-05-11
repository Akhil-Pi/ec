import time
import logging
import config

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("ur_rtde not installed - running in SIMULATION mode")

logger = logging.getLogger(__name__)


class UR3Controller:

    def __init__(self, ip=None, simulate=False):
        self.ip = ip or config.UR3_IP
        self.simulate = simulate or not RTDE_AVAILABLE
        self.rtde_c = None
        self.rtde_r = None
        self._sim_joints = list(config.HOME_JOINTS)

        if not self.simulate:
            print(f"[UR3] Connecting to {self.ip} ...")
            self.rtde_c = RTDEControlInterface(self.ip)
            self.rtde_r = RTDEReceiveInterface(self.ip)
            print(f"[UR3] Connected. TCP pose: {self.get_pose()}")
        else:
            print("[UR3] SIMULATION mode")

    # ---------- STATE ----------

    def get_pose(self):
        """TCP pose [x,y,z,rx,ry,rz] in meters/radians."""
        if self.simulate:
            return [0.0] * 6
        return self.rtde_r.getActualTCPPose()

    def get_joint_positions(self):
        """Joint positions in radians."""
        if self.simulate:
            return list(self._sim_joints)
        return self.rtde_r.getActualQ()

    # ---------- SAFETY (for fine adjustments only) ----------

    @staticmethod
    def is_within_safe_bounds(pose):
        x, y, z = pose[0], pose[1], pose[2]
        b = config.SAFE_BOUNDS
        return (b["x"][0] <= x <= b["x"][1]
                and b["y"][0] <= y <= b["y"][1]
                and b["z"][0] <= z <= b["z"][1])

    # ---------- MOTION ----------

    def move_joints(self, joint_positions, speed=None, acceleration=None):
        """Move via joint space — exact path, matches pendant."""
        speed        = speed        or 0.05
        acceleration = acceleration or 0.05
        if self.simulate:
            self._sim_joints = list(joint_positions)
            time.sleep(0.05)
            return True
        logger.info(f"[UR3] moveJ: {[round(v,3) for v in joint_positions]}")
        return self.rtde_c.moveJ(joint_positions, speed, acceleration)

    def move_linear(self, target_pose, speed=None, acceleration=None):
        """
        Straight-line TCP move — used ONLY for small fine adjustments
        (±2cm nudges after transition is complete).
        NOT used for waypoint traversal.
        """
        speed        = speed        or config.MOVE_SPEED
        acceleration = acceleration or config.MOVE_ACCELERATION

        if not self.is_within_safe_bounds(target_pose):
            logger.warning(f"[UR3] BLOCKED: {target_pose[:3]} outside bounds")
            return False
        if self.simulate:
            time.sleep(0.05)
            return True
        return self.rtde_c.moveL(target_pose, speed, acceleration)

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0,
                      drx=0.0, dry=0.0, drz=0.0):
        """Relative TCP move — for fine adjustments only."""
        cur = self.get_pose()
        tgt = [cur[0]+dx, cur[1]+dy, cur[2]+dz,
               cur[3]+drx, cur[4]+dry, cur[5]+drz]
        return self.move_linear(tgt)

    def go_home(self):
        """Go to home using joint space — safe and predictable."""
        return self.move_joints(config.HOME_JOINTS)

    def stop(self, deceleration=2.0):
        if self.simulate:
            return
        self.rtde_c.stopL(deceleration)

    def wait_for_motion_complete(self, timeout_s=5.0):
        """
        Block until the robot stops moving (velocity near zero).
        Returns (success, time_waited_s).
        """
        if self.simulate:
            return True, 0.0
        start = time.time()
        while (time.time() - start) < timeout_s:
            q_dot = self.rtde_r.getActualQd()
            velocity = sum(v**2 for v in q_dot) ** 0.5
            if velocity < 0.01:
                waited = time.time() - start
                logger.info(f"[UR3] Motion complete (waited {waited:.2f}s)")
                return True, waited
            time.sleep(0.05)
        logger.warning(f"[UR3] Timeout waiting for motion ({timeout_s}s)")
        return False, timeout_s

    # ---------- POSTURAL FINE ADJUSTMENTS ----------

    def adjust_lateral(self, dx):
        dx = max(min(dx, config.X_ADJUST_STEP), -config.X_ADJUST_STEP)
        logger.info(f"[UR3] Lateral: {dx:+.3f} m")
        return self.move_relative(dx=dx)

    def adjust_depth(self, dy):
        dy = max(min(dy, config.Y_ADJUST_STEP), -config.Y_ADJUST_STEP)
        logger.info(f"[UR3] Depth: {dy:+.3f} m")
        return self.move_relative(dy=dy)

    def adjust_height(self, dz):
        dz = max(min(dz, config.Z_ADJUST_STEP), -config.Z_ADJUST_STEP)
        logger.info(f"[UR3] Height: {dz:+.3f} m")
        return self.move_relative(dz=dz)

    def adjust_tilt(self, drx):
        drx = max(min(drx, config.TILT_ADJUST_STEP), -config.TILT_ADJUST_STEP)
        logger.info(f"[UR3] Tilt: {drx:+.3f} rad")
        return self.move_relative(drx=drx)

    def adjust_rotate(self, drz):
        """
        Rotate wrist around Z-axis (tool frame).
        Positive drz = counterclockwise, negative = clockwise (when viewed from above).
        Maps to the 6th TCP component (tool rotation around z).
        """
        drz = max(min(drz, config.ROTATE_ADJUST_STEP), -config.ROTATE_ADJUST_STEP)
        logger.info(f"[UR3] Wrist rotate: {drz:+.3f} rad")
        return self.move_relative(drz=drz)

    # ---------- WAYPOINT TRAVERSAL ----------

    def move_through_joint_waypoints(self, joint_path, speed=None,
                                      acceleration=None, confirm_each=False):
        speed        = speed        or 0.05
        acceleration = acceleration or 0.05

        print(f"\n[UR3] Joint traversal: {len(joint_path)} waypoints "
              f"at {speed} rad/s")

        for i, joints in enumerate(joint_path):
            print(f"  Step {i+1}/{len(joint_path)}: "
                  f"{[round(v,3) for v in joints]}")

            if confirm_each:
                ans = input(f"  Move to step {i+1}? "
                            f"(Enter=yes, q=abort): ").strip().lower()
                if ans == "q":
                    print("  Aborted.")
                    return False

            try:
                ok = self.move_joints(joints, speed=speed,
                                      acceleration=acceleration)
            except Exception as e:
                print(f"\n  ERROR: {e}")
                print("  Unlock on pendant, press Play, then press Enter.")
                input("  Press Enter when ready: ")
                try:
                    ok = self.move_joints(joints, speed=speed,
                                          acceleration=acceleration)
                except Exception as e2:
                    logger.error(f"[UR3] Retry failed: {e2}")
                    return False

            if not ok:
                logger.warning(f"[UR3] Step {i+1} failed")
                return False

            time.sleep(0.3)
            print(f"  Step {i+1} done.")

        print("\n[UR3] Traversal complete.")
        return True


    # ---------- HOME CHECK ----------

    def is_at_home(self, tolerance_rad=0.05):
        """
        Compare JOINT positions (radians) to HOME_JOINTS.
        tolerance_rad=0.05 is about 3 degrees per joint.
        """
        current = self.get_joint_positions()
        home    = config.HOME_JOINTS
        dist    = sum((current[i] - home[i])**2 for i in range(6)) ** 0.5
        ok      = dist < tolerance_rad
        if not ok:
            logger.warning(
                f"[UR3] Not at home. Joint dist: {dist:.3f} rad\n"
                f"  Current: {[round(v,3) for v in current]}\n"
                f"  Home:    {[round(v,3) for v in home]}"
            )
        return ok, dist

    def ensure_at_home(self, auto_move=False):
        ok, dist = self.is_at_home()
        if ok:
            print(f"[UR3] Home verified (joint dist: {dist:.3f} rad)")
            return True

        print(f"\n[UR3] WARNING: {dist:.3f} rad from HOME_JOINTS")
        print(f"  Current: {[round(v,3) for v in self.get_joint_positions()]}")
        print(f"  Home:    {[round(v,3) for v in config.HOME_JOINTS]}")

        if auto_move:
            print("Will move directly to HOME_JOINTS.")
            print("Hand on E-stop.\n")
            ans = input("Proceed? (y/N): ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return False

            ok = self.move_joints(config.HOME_JOINTS, speed=0.05, acceleration=0.05)

            if ok:
                ok2, dist2 = self.is_at_home()
                print(f"Final check: {'OK' if ok2 else 'NOT AT HOME'} "
                      f"(dist: {dist2:.3f} rad)")
                return ok2
            return False
        else:
            input("Manually jog to HOME on pendant, then press Enter...")
            ok, dist = self.is_at_home()
            return ok

    # ---------- CLEANUP ----------

    def disconnect(self):
        if self.simulate:
            return
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass