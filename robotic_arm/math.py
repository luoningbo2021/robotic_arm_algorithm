"""提供旋转矩阵, 欧拉角, 轴角, 齐次变换的转换函数."""

import math
import numpy as np

def euler_angles_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """将欧拉角转换为旋转矩阵.

    旋转约定为 R = Rz(rz) @ Ry(ry) @ Rx(rx).

    Args:
        euler: 欧拉角, 形状为 (3,), 顺序为 [rx, ry, rz].

    Returns:
        旋转矩阵, 形状为 (3, 3).
    """
    mat_x = np.array([[ 1, 0,                   0                 ],
                      [ 0, math.cos(euler[0]), -math.sin(euler[0])],
                      [ 0, math.sin(euler[0]),  math.cos(euler[0])]])

    mat_y = np.array([[ math.cos(euler[1]), 0, math.sin(euler[1])],
                      [ 0,                  1, 0                 ],
                      [-math.sin(euler[1]), 0, math.cos(euler[1])]])
    
    mat_z = np.array([[ math.cos(euler[2]), -math.sin(euler[2]), 0],
                      [ math.sin(euler[2]),  math.cos(euler[2]), 0],
                      [ 0,                   0,                  1]])
    return np.dot(mat_z, np.dot(mat_y, mat_x))

def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为欧拉角.

    旋转约定为 R = Rz(rz) @ Ry(ry) @ Rx(rx).姿态奇异时将 rz 设为 0.

    Args:
        R: 旋转矩阵, 形状为 (3, 3).

    Returns:
        欧拉角, 形状为 (3,), 顺序为 [rx, ry, rz].
    """
    if R[2, 0] != 1 and R[2, 0] != -1:  # 非奇异姿态
        ry = np.arctan2(-R[2, 0], (R[0, 0]**2+R[1, 0]**2)**0.5)
        rz = np.arctan2(R[1, 0], R[0, 0])
        rx = np.arctan2(R[2, 1], R[2, 2])
    else:  # 奇异姿态
        rz = 0
        if R[2, 0] == -1:
            ry = math.pi / 2
            rx = np.arctan2(R[0, 1], R[1, 1])
        else:
            ry = -math.pi / 2
            rx = -np.arctan2(R[0, 1], R[1, 1])

    return np.array([rx, ry, rz])

def axis_angles_to_rotation_matrix(n: np.ndarray, theta: float) -> np.ndarray:
    """使用罗德里格斯公式将轴角转换为旋转矩阵.

    Args:
        n: 非零旋转轴向量, 形状为 (3,), 函数内部进行归一化.
        theta: 绕旋转轴的旋转角度.

    Returns:
        旋转矩阵, 形状为 (3, 3).
    """
    # 将旋转轴归一化, 确保 n 是单位向量
    n = np.array(n)
    n = n / np.sqrt(np.dot(n, n))
    nx, ny, nz = n
    mat_k = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
    mat_i = np.eye(3)
    return mat_i + np.sin(theta) * mat_k + (1 - np.cos(theta)) * np.dot(mat_k, mat_k)

def rotation_matrix_to_axis_angles(R: np.ndarray)  -> tuple[np.ndarray, float]:
    """将旋转矩阵转换为轴角.

    当由矩阵迹计算的旋转角度为 0 或 pi 时, 返回零向量和零角度.

    Args:
        R: 旋转矩阵, 形状为 (3, 3).

    Returns:
        轴角转换结果.
            - n: 旋转轴向量, 形状为 (3,), 非退化情况下为单位向量.
            - theta: 旋转角度.
    """
    theta = np.arccos((R[0, 0]+R[1, 1]+R[2, 2]-1) / 2)
    if theta != 0 and theta != math.pi:
        n = (1/(2*np.sin(theta))) * np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    else:
        n = np.array([0, 0, 0])
        theta = 0
    return n, theta

def transformation_matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """将齐次变换矩阵转换为位姿.

    Args:
        T: 齐次变换矩阵, 形状为 (4, 4).

    Returns:
        位姿, 形状为 (6,).
    """
    p = np.array([T[0, 3], T[1, 3], T[2, 3]])
    R = T[0:4, 0:4]
    angle = rotation_matrix_to_euler_angles(R)
    return np.hstack((p, angle))

def pose_to_transformation_matrix(pose: np.ndarray) -> np.ndarray:
    """将位姿转换为齐次变换矩阵.

    Args:
        pose: 位姿, 形状为 (6,).

    Returns:
        齐次变换矩阵, 形状为 (4, 4).
    """
    pose = np.array(pose)
    p = np.array(pose[0:3]).T
    euler = pose[3:6]
    R = euler_angles_to_rotation_matrix(euler)
    trans = RP_to_transformation_matrix(R, p)
    return trans

def transformation_matrix_to_RP(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将齐次变换矩阵分解成旋转矩阵和位置向量.

    Args:
        T: 齐次变换矩阵, 形状为 (4, 4).

    Returns:
        旋转矩阵和位置向量.
            - R: 旋转矩阵, 形状为 (3, 3).
            - P: 位置向量, 形状为 (3,).
    """
    trans = np.array(T)
    return np.array(trans[0: 3, 0: 3]), np.array(trans[0: 3, 3])

def RP_to_transformation_matrix(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """将旋转矩阵和位置向量组合为齐次变换矩阵.

    Args:
        R: 旋转矩阵, 形状为 (3, 3).
        P: 位置向量, 形状为 (3,).

    Returns:
        齐次变换矩阵, 形状为 (4, 4).
    """
    return np.array(np.r_[np.c_[np.array(R), np.array([P[0], P[1], P[2]])], [[0, 0, 0, 1]]])
