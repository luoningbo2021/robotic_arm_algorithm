"""提供机器人运动学, 动力学, 轨迹规划的统一接口."""

import numpy as np
from .model import RobotModel
from .algs.kinematics import Kinematics
from .algs.dynamics import Dynamics
from .algs.planning.trajectory import TrajectoryPlanner

class Robot:
    """封装六轴机器人的模型与算法组件.

    Attributes:
        model (RobotModel): 机器人模型, 包含运动学, 轨迹规划, 动力学参数.
        kinematics (Kinematics): 基于机器人模型的运动学求解器.
        dynamics (Dynamics): 基于机器人模型的动力学求解器.
        planner (TrajectoryPlanner): 使用指定插补周期的轨迹规划器.
    """

    def __init__(self, model: RobotModel, interp_time: float = 0.001):
        """根据机器人模型初始化算法组件.

        Args:
            model: 机器人模型, 包含运动学和动力学参数.
            interp_time: 轨迹规划的插补周期.
        """
        self.model = model
        self.kinematics = Kinematics(model)
        self.dynamics = Dynamics(model)
        self.planner = TrajectoryPlanner(interp_time)

    def fk(self, theta: np.ndarray) -> np.ndarray:
        """计算正运动学.

        Args:
            theta: 关节角度, 形状为 (6,).

        Returns:
            末端位姿, 形状为 (6,).
        """
        return self.kinematics.forward_kinematics(theta)

    def ik(self, pose: np.ndarray, last_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算逆运动学, 并选择最接近参考关节角度的解.

        Args:
            pose: 末端位姿, 形状为 (6,).
            last_theta: 用于选择最近解的参考关节角度, 形状为 (6,).

        Returns:
            逆运动学结果.
                - theta: 最接近 last_theta 的关节角度, 形状为 (6,).
                - theta_mat: 全部候选解, 形状为 (8, 6).
        """
        return self.kinematics.inverse_kinematics(pose, last_theta)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """计算基坐标系下的雅可比矩阵.

        Args:
            theta: 关节角度, 形状为 (6,), 单位为 rad.

        Returns:
            雅可比矩阵, 形状为 (6, 6).
            前三行对应末端线速度, 后三行对应末端角速度.
        """
        return self.kinematics.jacobian(theta)

    def inverse_dynamics(self, theta: np.ndarray, dtheta: np.ndarray, ddtheta: np.ndarray) -> np.ndarray:
        """计算逆动力学.

        Args:
            theta: 关节角度, 形状为 (6,).
            dtheta: 关节速度, 形状为 (6,).
            ddtheta: 关节加速度, 形状为 (6,).

        Returns:
            关节力矩, 形状为 (6,).
        """
        return self.dynamics.inverse_dynamics(theta, dtheta, ddtheta)

    def forward_dynamics(self, theta: np.ndarray, dtheta: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """计算正动力学.

        Args:
            theta: 关节角度, 形状为 (6,).
            dtheta: 关节速度, 形状为 (6,).
            tau: 关节力矩, 形状为 (6,).

        Returns:
            关节加速度, 形状为 (6,).
        """
        return self.dynamics.forward_dynamics(theta, dtheta, tau)

    def dynamics_regress(self, theta: np.ndarray, dtheta: np.ndarray, ddtheta: np.ndarray) -> np.ndarray:
        """计算动力学最小参数集对应的回归矩阵.

        Args:
            theta: 关节角度, 形状为 (6,).
            dtheta: 关节速度, 形状为 (6,).
            ddtheta: 关节加速度, 形状为 (6,).

        Returns:
            动力学回归矩阵, 形状为 (6, 52).每行对应一个关节,
            各列与 model.min_param_set 中的参数顺序一致.
        """
        return self.dynamics.dynamics_regress(theta, dtheta, ddtheta)

    def plan_joint(self, theta_start: np.ndarray, theta_end: np.ndarray, dtheta_max: np.ndarray, ddtheata_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """使用梯形速度曲线规划关节轨迹.

        Args:
            theta_start: 起点关节角度, 形状为 (6,).
            theta_end: 终点关节角度, 形状为 (6,).
            dtheta_max: 各关节的最大速度, 形状为 (6,).
            ddtheata_max: 各关节的最大加速度, 形状为 (6,).

        Returns:
            关节轨迹规划结果.
                - t_array: 时间序列, 形状为 (N,), N 为采样点数.
                - theta_mat: 关节角度序列, 形状为 (N, 6).
        """
        return self.planner.plan_joint(theta_start, theta_end, dtheta_max, ddtheata_max)

    def plan_line(self, pose_start: np.ndarray, pose_end: np.ndarray, v_max: np.ndarray, a_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """使用梯形速度曲线规划笛卡尔空间中的直线轨迹.

        Args:
            pose_start: 起点位姿, 形状为 (6,).
            pose_end: 终点位姿, 形状为 (6,).
            v_max: 最大线速度和角速度, 形状为 (2,), 顺序为[线速度, 角速度].
            a_max: 最大线加速度和角加速度, 形状为 (2,), 顺序为[线加速度, 角加速度].

        Returns:
            直线轨迹规划结果.
                - t_array: 时间序列, 形状为 (N,), N 为采样点数.
                - pose_mat: 位姿序列, 形状为 (N, 6).
        """
        return self.planner.plan_line(pose_start, pose_end, v_max, a_max)
