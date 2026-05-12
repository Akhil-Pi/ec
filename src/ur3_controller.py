import time
import logging
import config

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("ur_rtde not installed - SIMULATION mode")

logger = logging.getLogger(__name__)


class UR3Controller:

    def __init__(self, ip=None, simulate=False):
        self.ip      = ip or config.UR3_IP
        self.simulate = simulate or not RTDE_AVAILABLE
        self.rtde_c  = None
        self.rtde_r  = None

        if not self.simulate:
            print(f"[UR3] Connecting to {self.ip} ...")
            self.rtde_c = RTDEControlInterface(self.ip)
            self.rtde_r = RTDEReceiveInterface(self.ip)
            print(f"[UR3] Connected. TCP: {self.get_pose()}")
        else:
            print("[UR3] SIMULATION mode")

    def get_pose(self):
        if self.simulate:
            return [0.0] * 6
        return self.rtde_r.getActualTCPPose()

    def get_joint_positions(self):
        if self.simulate:
            return [0.0] * 6
        return self.rtde_r.getActualQ()

    @staticmethod
    def is_within_safe_bounds(pose):
        x, y, z = pose[0], pose[1], pose[2]
        b = config.SAFE_BOUNDS
        return (b["x"][0] <= x <= b["x"][1]
                and b["y"][0] <= y <= b["y"][1]
                and b["z"][0] <= z <= b["z"][1])

    def move_joints(self, joint_positions, speed=None, acceleration=None):
        """Joint-space move — exact path."""
        speed        = speed        or 0.05
        acceleration = acceleration or 0.05
        if self.simulate:
            time.sleep(0.05)
            return True
        logger.info(f"[UR3] moveJ: {[round(v,3) for v in joint_positions]}")
        return self.rtde_c.moveJ(joint_positions, speed, acceleration)

    def move_to_desired(self):
        """Move directly to DESIRED_JOINTS."""
        print("[UR3] Moving to DESIRED position...")
        ok = self.move_joints(config.DESIRED_JOINTS, speed=0.05,
                              acceleration=0.05)
        if ok:
            print("[UR3] At DESIRED position.")
        return ok

    def move_linear(self, target_pose, speed=None, acceleration=None,
                    asynchronous=False):
        """Straight-line TCP move — for fine adjustments only."""
        speed        = speed        or config.MOVE_SPEED
        acceleration = acceleration or config.MOVE_ACCELERATION
        if not self.is_within_safe_bounds(target_pose):
            logger.warning(f"[UR3] BLOCKED: {target_pose[:3]} outside bounds")
            return False
        if self.simulate:
            time.sleep(0.05)
            return True
            #return self.rtde_c.
        return self.rtde_c.moveL(target_pose, speed, acceleration, asynchronous)

    def wait_for_motion_complete(self, timeout_s=5.0):
        """
        Wait until the robot stops moving.
        Used after async moveL to record accurate latency.
        """
        if self.simulate:
            return
        import time
        start = time.time()
        while time.time() - start < timeout_s:
            if not self.rtde_r.isRobotMoving():
                return
            time.sleep(0.02)
        logger.warning("[UR3] wait_for_motion_complete timed out")

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0,
                    drx=0.0, dry=0.0, drz=0.0,
                    asynchronous=False):
        """Relative TCP move — for fine adjustments only."""
        cur = self.get_pose()
        tgt = [cur[0]+dx, cur[1]+dy, cur[2]+dz,
            cur[3]+drx, cur[4]+dry, cur[5]+drz]
        return self.move_linear(tgt, asynchronous=asynchronous)

    def adjust_lateral(self, dx):
        dx = max(min(dx, config.X_ADJUST_STEP), -config.X_ADJUST_STEP)
        logger.info(f"[UR3] Lateral: {dx:+.3f} m")
        return self.move_relative(dx=dx, asynchronous=True)

    def adjust_depth(self, dy):
        dy = max(min(dy, config.Y_ADJUST_STEP), -config.Y_ADJUST_STEP)
        logger.info(f"[UR3] Depth: {dy:+.3f} m")
        return self.move_relative(dy=dy, asynchronous=True)

    def adjust_height(self, dz):
        dz = max(min(dz, config.Z_ADJUST_STEP), -config.Z_ADJUST_STEP)
        logger.info(f"[UR3] Height: {dz:+.3f} m")
        return self.move_relative(dz=dz, asynchronous=True)

    def adjust_tilt(self, drx):
        drx = max(min(drx, config.TILT_ADJUST_STEP), -config.TILT_ADJUST_STEP)
        logger.info(f"[UR3] Tilt: {drx:+.3f} rad")
        return self.move_relative(drx=drx, asynchronous=True)

    def adjust_rotation(self, drz):
        drz = max(min(drz, config.ROTATION_ADJUST_STEP), -config.ROTATION_ADJUST_STEP)
        logger.info(f"[UR3] Rotation: {drz:+.3f} rad")
        return self.move_relative(drz=drz, asynchronous=True)

    def disconnect(self):
        if self.simulate:
            return
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass