import sys
from pathlib import Path
# __file__ 是脚本的位置，不受终端当前目录影响
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import math
import roboticstoolbox as rtb
import robotic_arm as ra


## 测试-雅克比矩阵(和roboticstoolbox对比)
q = np.array([90 * math.pi / 180, 0, 0, 180 * math.pi / 180, 0, 0])
ur5_model = ra.RobotModel()
robot = ra.Robot(ur5_model)
# 相对于基坐标系的雅可比矩阵(自定义)
J0 = robot.jacobian(q)

new_robot = rtb.models.DH.UR5()
# 相对于基坐标系的雅可比矩阵
new_J0 = new_robot.jacob0(q)
print("J0 =", J0)
print("new_J0 =", new_J0)
