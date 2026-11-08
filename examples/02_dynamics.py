import sys
from pathlib import Path
# __file__ 是脚本的位置，不受终端当前目录影响
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import robotic_arm as ra


## 测试-逆动力学(和自身对比)
theta = np.ones(6)
dtheta = np.ones(6)
ddtheta = np.ones(6)

ur5_model = ra.RobotModel()
robot = ra.Robot(ur5_model)
tau1 = robot.inverse_dynamics(theta=theta, dtheta=dtheta, ddtheta=ddtheta)
Y = robot.dynamics_regress(theta=theta, dtheta=dtheta, ddtheta=ddtheta)
tau2 = np.dot(Y, ur5_model.min_param_set)
print("tau1 =", tau1)
print("tau2 =", tau2)
print("Y =", Y)
