import time, logging, config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from ur3_controller import UR3Controller
robot = UR3Controller()

for name, joints in [
    ('HOME_JOINTS',    config.HOME_JOINTS),
    ('POINT_1_JOINTS', config.POINT_1_JOINTS),
    ('POINT_2_JOINTS', config.POINT_2_JOINTS),
    ('POINT_3_JOINTS', config.POINT_3_JOINTS),
    ('DESIRED_JOINTS', config.DESIRED_JOINTS),
]:
    input(f'Press Enter to move to {name}...')
    robot.move_joints(joints, speed=0.05, acceleration=0.05)
    print(f'{name} reached.')

robot.disconnect()