"""提供六轴机器人的正逆运动学和雅可比矩阵计算接口."""

import math
import numpy as np
from .. import math as mtb
from ..model import RobotModel

class Kinematics:
    """基于机器人模型计算运动学.

    Attributes:
        model (RobotModel): 机器人模型, 包含 MDH 几何参数.
    """

    def __init__(self, model:RobotModel):
        """根据机器人模型初始化运动学求解器.

        Args:
            model: 机器人模型, 包含 MDH 几何参数.
        """
        self.model = model

    def _mdh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """根据 Craig MDH 参数计算相邻关节坐标系的齐次变换.

        变换顺序为 Rx(alpha) @ Tx(a) @ Rz(theta) @ Tz(d).

        Args:
            a: 连杆长度.
            alpha: 连杆扭转角, 单位为 rad.
            d: 连杆偏距, 单位与 a 一致.
            theta: 关节角度, 单位为 rad.

        Returns:
            齐次变换矩阵, 形状为 (4, 4).
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)

        return np.array([
            [   ct,      -st,     0,      a],
            [st*ca,    ct*ca,   -sa,  -d*sa],
            [st*sa,    ct*sa,    ca,   d*ca],
            [    0,        0,     0,      1]
        ])
    
    def _forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        """计算各关节坐标系相对于基坐标系的齐次变换.

        Args:
            theta: 关节角度, 形状为 (6,).

        Returns:
            按关节顺序排列的 6 个齐次变换矩阵, 每个矩阵的形状为 (4, 4).
        """
        T = np.eye(4)
        
        transforms = []
        for i in range(theta.shape[0]):
            A = self._mdh_transform(self.model.a[i], self.model.alpha[i], self.model.d[i], theta[i])
            T = T @ A
            transforms.append(T.copy())
        return transforms

    def _nearest_theta(self, theta_mat: np.ndarray, last_theta: np.ndarray)-> np.ndarray:
        """选择最接近参考关节角度的候选解.

        Args:
            theta_mat: 候选关节角度, 形状为 (M, 6), M 为候选解数量.
            last_theta: 参考关节角度, 形状为 (6,).

        Returns:
            候选解, 形状为 (6,).

            候选解为空时返回全 0 数组.
        """
        weights = np.array([1, 1, 1, 1, 1, 1])

        best_theta = np.zeros(6)
        best_theta_dist = np.inf
        for theta in theta_mat:
            theta_dist = np.sum(((theta - last_theta) * weights)**2)
            if theta_dist < best_theta_dist:
                best_theta_dist = theta_dist
                best_theta = theta
        return best_theta
    
    def forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        """计算正运动学.

        Args:
            theta: 关节角度, 形状为 (6,).

        Returns:
            末端位姿, 形状为 (6,).
        """
        T_list = self._forward_kinematics(theta)
        T = T_list[-1]
        pose = mtb.transformation_matrix_to_pose(T)
        return pose

    def inverse_kinematics(self, pose: np.ndarray, last_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算逆运动学, 并选择最接近参考关节角度的解.

        Args:
            pose:末端位姿, 形状为 (6,).
            last_theta (np.ndarray): 用于选择最近解的参考关节角度, 形状为 (6,).

        Returns:
            逆运动学结果.
                - theta: 最接近 last_theta 的关节角度, 形状为 (6,).
                - theta_mat: 全部候选解, 形状为 (8, 6).
        """
        theta_mat = np.zeros([8,6])

        T06 = mtb.pose_to_transformation_matrix(pose)

        # ---------- Theta 1 ----------
        P05 = (T06 @ np.array([0,0,-self.d[5], 1]))[0:3]
        phi1 = np.arctan2(P05[1],P05[0])
        phi2 = np.array([np.arccos(self.d[3]/np.linalg.norm(P05[0:2])), -np.arccos(self.d[3]/np.linalg.norm(P05[0:2]))])
        # np.linalg.norm(P05[0:2]) <= fabs(self.d[3])

        for i in range(4):
            theta_mat[i,0] = phi1 + phi2[0] + math.pi/2
            theta_mat[i+4,0] = phi1 + phi2[1] + math.pi/2

        for i in range(8):
            if theta_mat[i,0] <= math.pi:
                theta_mat[i,0] += 2*math.pi
            if theta_mat[i,0] > math.pi:
                theta_mat[i,0] -= 2*math.pi

        # ---------- Theta 5 ----------
        P06 = T06[0:3,3]
        for i in range(8):
            theta_mat[i,4] = np.arccos((P06[0]*np.sin(theta_mat[i,0])-P06[1]*np.cos(theta_mat[i,0])-self.d[3])/self.d[5])
            if np.isin(i, [2,3,6,7]):
                theta_mat[i,4] = -theta_mat[i,4]
        # fabs(P06[0]*np.sin(theta[i,0])-P06[1]*np.cos(theta[i,0])-self.d[3]) <= fabs(self.d[5])

        # ---------- Theta 6 ----------
        T60 = np.linalg.inv(T06)
        X60 = T60[0:3,0]
        Y60 = T60[0:3,1]

        for i in range(8):
            theta_mat[i,5] = np.arctan2((-X60[1]*np.sin(theta_mat[i,0])+Y60[1]*np.cos(theta_mat[i,0]))/np.sin(theta_mat[i,4]),
                                    ( X60[0]*np.sin(theta_mat[i,0])-Y60[0]*np.cos(theta_mat[i,0]))/np.sin(theta_mat[i,4]))

        # ------- Theta 3 and 2 -------
        for i in range(8):
            T01 = self.DHLink(self.alpha[0],self.a[0],self.d[0], theta_mat[i,0])
            T45 = self.DHLink(self.alpha[4],self.a[4],self.d[4], theta_mat[i,4])
            T56 = self.DHLink(self.alpha[5],self.a[5],self.d[5], theta_mat[i,5])

            T14 = np.linalg.inv(T01)@T06@np.linalg.inv(T45@T56)
            P14xz = np.array([T14[0,3], T14[2,3]])

            theta_mat[i,2] = np.arccos((np.linalg.norm(P14xz)**2-self.a[1]**2-self.a[2]**2)/(2*self.a[1]*self.a[2]))
            # fabs(self.a[1] - self.a[2]) <= fabs(P14xz) <= fabs(self.a[1] + self.a[2])

            if i % 2 != 0:
                theta_mat[i,2] = -theta_mat[i,2]

            theta_mat[i,1] = np.arctan2(-P14xz[1], -P14xz[0]) - np.arcsin(-self.a[2]*np.sin(theta_mat[i,2])/np.linalg.norm(P14xz))

        # ---------- Theta 4 ----------
        for i in range(8):
            T01 = self._mdh_transform(self.alpha[0],self.a[0],self.d[0], theta_mat[i,0])
            T12 = self._mdh_transform(self.alpha[1],self.a[1],self.d[1], theta_mat[i,1])
            T23 = self._mdh_transform(self.alpha[2],self.a[2],self.d[2], theta_mat[i,2])
            T45 = self._mdh_transform(self.alpha[4],self.a[4],self.d[4], theta_mat[i,4])
            T56 = self._mdh_transform(self.alpha[5],self.a[5],self.d[5], theta_mat[i,5])

            T34 = np.linalg.inv(T01@T12@T23)@T06@np.linalg.inv(T45@T56)

            theta_mat[i,3] = np.arctan2(T34[1,0], T34[0,0])

        theta = self._nearest_theta(theta_mat, last_theta)
        return theta, theta_mat
        
    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """计算基坐标系下的几何雅可比矩阵.

        Args:
            theta: 关节角度, 形状为 (6,).

        Returns:
            雅可比矩阵, 形状为 (6, 6).
            前三行对应末端线速度, 后三行对应末端角速度.
        """
        transforms = self._forward_kinematics(theta)
        # 末端参考点在基坐标系的位置
        p_e = transforms[-1][:3, 3]

        J = np.zeros((6, 6))
        for i in range(6):
            # MDH: 当前关节轴方向 z_i
            T_i = transforms[i]

            z_i = T_i[:3, 2]
            p_i = T_i[:3, 3]

            # 线速度部分
            J[:3, i] = np.cross(z_i, p_e - p_i)

            # 角速度部分
            J[3:, i] = z_i
        return J
