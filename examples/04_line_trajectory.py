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
pose_start = np.array([0, 0, 0, 0, 0, 0])
pose_end = np.array([0.5, 0, 0, 0, 0, 90 * math.pi / 180])

ur5_model = ra.RobotModel()
interp_time = 0.001
robot = ra.Robot(ur5_model, interp_time)
v_limit = ur5_model.v_limit
a_limit = ur5_model.a_limit
t_list, pose_mat = robot.plan_line(pose_start, pose_end, v_limit, a_limit)
dt = t_list[1] - t_list[0]

plot = Plot()
plot.plot_line_traj(pose_mat, dt)
