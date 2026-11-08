"""提供关节空间和笛卡尔空间的梯形轨迹规划接口."""

import numpy as np
from ... import math as mtb
from .profile import TrapezoidProfile

class TrajectoryPlanner():
    """使用梯形速度曲线规划轨迹, 并按插补周期采样.

    Attributes:
        interp_time (float): 轨迹规划的插补周期, 单位为 s.
    """

    def __init__(self, interp_time: float = 0.001):
        """初始化轨迹规划器.

        Args:
            interp_time (float): 轨迹规划的插补周期, 应为正值,
                单位为 s, 默认为 0.001.
        """
        self.interp_time = interp_time

    def plan_joint(self, theta_start: np.ndarray, theta_end: np.ndarray, dtheta_max: np.ndarray, ddtheata_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """使用梯形速度曲线规划多关节同步轨迹.

        Args:
            theta_start: 起点关节角度, 形状为 (n,), n 为关节数量.
            theta_end: 终点关节角度, 形状为 (n,).
            dtheta_max: 各关节的最大速度, 形状为 (n,).
            ddtheata_max: 各关节的最大加速度, 形状为 (n,).

        Returns:
            关节轨迹规划结果.
                - t_array: 时间序列, 形状为 (N,), N 为采样点数,
                    采样区间为 [0, T), T 为轨迹总时长.
                - theta_mat: 关节角度序列, 形状为 (N, n).

        Raises:
            AssertionError: 运动距离不足以形成三段梯形速度曲线.
        """
        theta = theta_end - theta_start
        dis = np.linalg.norm(theta)
        dir = theta / dis
        mask = ~np.isclose(dir, 0)
        v_max = np.min(np.abs(dtheta_max[mask]/dir[mask]))
        a_max = np.min(np.abs(ddtheata_max[mask]/dir[mask]))

        # 根据路径速度和加速度限制计算各关节的同步运动时长
        profile = TrapezoidProfile(v_max=v_max, a_max=a_max, s=dis)
        T = profile.get_trapezoid_profile_time()

        # 按插补周期采样, 生成关节角度序列
        t_array = np.arange(0, T, self.interp_time)
        M = t_array.shape[0]
        N = theta_start.shape[0]
        theta_mat = np.zeros((M, N))
        j = 0
        for t in t_array:
            s = profile.get_trapezoid_profile_pos(t)
            theta_mat[j, :] = theta_start + dir * s
            j = j + 1
        return t_array, theta_mat
    
    def plan_line(self, pose_start: np.ndarray, pose_end: np.ndarray, v_max: np.ndarray, a_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """使用梯形速度曲线规划笛卡尔空间中的直线轨迹.

        沿直线插补位置, 同步插补起终点之间的相对轴角.
        起点与终点之间应存在非零位移或转角.

        Args:
            pose_start: 起点位姿, 形状为 (6,).
            pose_end: 终点位姿, 形状为 (6,).
            v_max: 最大线速度和角速度, 形状为 (2,), 顺序为 [线速度, 角速度].
            a_max: 最大线加速度和角加速度, 形状为 (2,), 顺序为 [线加速度, 角加速度].

        Returns:
            直线轨迹规划结果.
                - t_array: 时间序列, 形状为 (N,), N 为采样点数,
                    采样区间为 [0, T), T 为轨迹总时长.
                - pose_mat: 位姿序列, 形状为 (N, 6).

        Raises:
            AssertionError: 运动距离不足以形成三段梯形速度曲线.
        """
        trans_start = mtb.pose_to_transformation_matrix(pose_start)
        rot_start, p_start = mtb.transformation_matrix_to_RP(trans_start)
        trans_end = mtb.pose_to_transformation_matrix(pose_end)
        rot_end, p_end = mtb.transformation_matrix_to_RP(trans_end)
        rot_process = np.dot(rot_start.T, rot_end)
        n, theta = mtb.rotation_matrix_to_axis_angles(rot_process)
        
        p = p_end - p_start
        dis_p = np.linalg.norm(p)
        dis_r = theta
        dis = np.sqrt(np.square(dis_p) + np.square(dis_r))
        dir_p = p / dis
        dir_r = dis_r / dis

        if ((dis_p != 0) and (dis_r != 0)):
            temp_v_max = np.minimum(v_max[0]/np.linalg.norm(dir_p), np.abs(v_max[1]/dir_r))
            temp_a_max = np.minimum(np.abs(a_max[0]/np.linalg.norm(dir_p)), np.abs(a_max[1]/dir_r))
        elif ((dis_p == 0) and (dis_r != 0)):
            temp_v_max = np.abs(v_max[1]/dir_r)
            temp_a_max = np.abs(a_max[1]/dir_r)
        elif ((dis_p != 0) and (dis_r == 0)):
            temp_v_max = v_max[0]/np.linalg.norm(dir_p)
            temp_a_max = a_max[0]/np.linalg.norm(dir_p)

        # 根据路径速度和加速度限制计算位置与姿态的同步运动时长
        profile = TrapezoidProfile(v_max=temp_v_max, a_max=temp_a_max, s=dis)
        T = profile.get_trapezoid_profile_time()

        # 按插补周期采样, 生成位姿序列
        t_array = np.arange(0, T, self.interp_time)
        M = t_array.shape[0]
        N = pose_start.shape[0]
        pose_mat = np.zeros((M, N))
        j = 0
        for t in t_array:
            s = profile.get_trapezoid_profile_pos(t)

            s_p  = p_start + s * dir_p
            s_r = s * dir_r
            if dir_r != 0:
                temp_R_process = mtb.axis_angles_to_rotation_matrix(n, s_r)
                temp_R_end = np.dot(rot_start, temp_R_process)
            else:
                temp_R_end = rot_start
            euler = mtb.rotation_matrix_to_euler_angles(temp_R_end)
            pose_mat[j, :] = np.hstack((s_p, euler))
            j = j + 1
        return t_array, pose_mat
