"""定义机器人的运动学, 轨迹规划, 动力学参数."""

import math
import numpy as np

class RobotModel:
    """定义六轴机器人的运动学, 轨迹规划, 动力学参数.

    默认采用 UR5 的 MDH 几何参数.
    各关节的角度范围, 各关节的最大加加速度, 逆动力学使用的最小参数集, 
    各连杆的惯量, 各连杆的一阶质量矩, 各连杆的质量默认使用全 1 占位值.

    Attributes:
        a (np.ndarray): MDH 连杆长度, 形状为 (6,).
        alpha (np.ndarray): MDH 连杆扭转角, 形状为 (6,).
        d (np.ndarray): MDH 连杆偏距, 形状为 (6,).
        theta_limit (np.ndarray): 各关节的角度范围, 顺序为 [角度下限, 角度上限].
        dtheta_limit (np.ndarray): 各关节的最大速度, 形状为 (6,).
        ddtheta_limit (np.ndarray): 各关节的最大加速度, 形状为 (6,).
        dddtheta_limit (np.ndarray): 各关节的最大加加速度, 形状为 (6,).
        v_limit (np.ndarray): 笛卡尔空间的最大线速度和角速度, 形状为 (2,), 顺序为 [线速度, 角速度].
        a_limit (np.ndarray): 笛卡尔空间的最大线加速度和角加速度, 形状为 (2,), 顺序为 [线加速度, 角加速度].
        j_limit (np.ndarray): 笛卡尔空间的最大线加加速度和角加加速度, 形状为 (2,), 顺序为 [线加加速度, 角加加速度].
        gravity (np.ndarray): 基坐标系下的重力加速度, 形状为 (3,), 顺序为 [gx, gy, gz].
        min_param_set (np.ndarray): 逆动力学使用的最小参数集, 形状为 (52,).
        link_inertia (np.ndarray): 各连杆的惯量, 形状为 (6, 6).
        link_moment (np.ndarray): 各连杆的一阶质量矩, 即质量乘质心坐标, 形状为 (6, 3).
        mass (np.ndarray): 各连杆的质量, 形状为 (6,).
    """

    def __init__(self):
        """初始化 UR5 运动学, 轨迹规划, 动力学参数."""

        # 运动学参数
        self.a = np.array([0, 0, -0.425, -0.39225, 0, 0])
        self.alpha = np.array([0, math.pi/2, 0, 0, math.pi/2, -math.pi/2])
        self.d = np.array([0.089459, 0, 0, 0.10915, 0.09465, 0.0823])

        self.theta_limit = np.ones((6, 2))

        # 轨迹规划参数
        self.dtheta_limit = np.array([324 * math.pi / 180,
                                      489.96 * math.pi / 180,
                                      0.800,
                                      2000 * math.pi / 180,
                                      0,
                                      0])
        self.ddtheta_limit = np.array([1698  * math.pi / 180,
                                       9653 * math.pi / 180,
                                       27.9472,
                                       27947.3 * math.pi / 180,
                                       0,
                                       0])
        self.ddtheta_limit = np.ones(6)

        self.v_limit = np.array([0.3, 200*math.pi/180])
        self.a_limit = np.array([3, 2000*math.pi/180])
        self.j_limit = np.array([0, 0])

        # 动力学参数
        self.gravity = np.array([0, 0, -9.8])

        self.min_param_set = np.ones(52)

        self.link_inertia = np.ones((6, 6))
        self.link_moment = np.ones((6, 3))                
        self.mass = np.ones(6)
