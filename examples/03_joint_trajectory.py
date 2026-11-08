import sys
from pathlib import Path
# __file__ 是脚本的位置，不受终端当前目录影响
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import math
import robotic_arm as ra
from tools.plot import Plot


## 测试-轨迹规划
theta_start = np.array([0, 0, 0, 0, 0, 0])
theta_end = np.array([90 * math.pi / 180, 0, 0, 180 * math.pi / 180, 0, 0])

ur5_model = ra.RobotModel()
interp_time = 0.001
robot = ra.Robot(ur5_model, interp_time)
dtheta_limit = ur5_model.dtheta_limit
ddtheta_limit = ur5_model.ddtheta_limit
t_list, theta_mat = robot.plan_joint(theta_start, theta_end, dtheta_limit, ddtheta_limit)
dt = t_list[1] - t_list[0]

plot = Plot()
# 关节1轨迹
plot.plot_joint_traj(theta_mat, 0, dt)
# 关节4轨迹
plot.plot_joint_traj(theta_mat, 3, dt)
