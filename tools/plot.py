import numpy as np
import matplotlib.pyplot as plt


class Plot():
    def __init__(self):
        pass

    def plot_joint_traj(self, theta_mat, joint=0, dt=0.001, title = 'Plot of Trajectory of Joint'):
        '''绘制单轴轨迹：位置、速度、加速度
        '''
        M, N = np.array(theta_mat).shape[0], np.array(theta_mat).shape[1]
        # T = M * dt
        # time_stamp = np.linspace(0, T, M)
        time_stamp = np.arange(M) * dt

        dtheta_mat, ddtheta_mat = np.zeros((M, N)), np.zeros((M, N))
        for i in range(M - 1):
            dtheta_mat[i + 1, :] = (theta_mat[i + 1, :] - theta_mat[i, :]) / dt
            ddtheta_mat[i + 1, :] = (dtheta_mat[i + 1, :] - dtheta_mat[i, :]) / dt

        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(3, 1, 1)
        plt.grid('w')
        ax2 = fig.add_subplot(3, 1, 2)
        plt.grid('w')
        ax3 = fig.add_subplot(3, 1, 3)
        plt.grid('w')
        ax1.plot(time_stamp, theta_mat[:, joint],   color='r', linestyle='-', label='theta'  )
        ax2.plot(time_stamp, dtheta_mat[:, joint],  color='g', linestyle='-', label='dtheta' )
        ax3.plot(time_stamp, ddtheta_mat[:, joint], color='b', linestyle='-', label='ddtheta')
        ax1.set_xlabel("Time/(s)")
        ax1.set_ylabel("theta/(rad)")
        ax2.set_xlabel("Time/(s)")
        ax2.set_ylabel("dtheta/(rad/s)")
        ax3.set_xlabel("Time/(s)")
        ax3.set_ylabel("ddtheta/(rad/s^2)")
        ax1.set_title(f"Plot of Trajectory of Joint{joint+1}")

        # 自动调整布局
        fig.tight_layout()
        # fig.subplots_adjust(
        #     bottom=0.15,  # 底部留白占整张图的高度
        #     hspace=0.5    # 上下子图之间的间距
        # )
        # 显示窗口，并等待窗口关闭
        plt.show(block=True)

    def plot_line_traj(self, pose_mat, dt=0.001):
        '''绘制笛卡尔空间的梯形轨迹-时间曲线
        '''
        M =  pose_mat.shape[0]
        # T = step_num * dt
        # time_stamp = np.linspace(0, T, step_num)
        time_stamp = np.arange(M) * dt
        P_mat = pose_mat[:, 0:3]
        R_mat = pose_mat[:, 3:6]

        dP_mat, ddP_mat = np.zeros((M, 3)), np.zeros((M, 3))
        dR_mat, ddR_mat = np.zeros((M, 3)), np.zeros((M, 3))
        for i in range(M - 1):
            dP_mat[i + 1] = (P_mat[i + 1] - P_mat[i]) / dt
            ddP_mat[i + 1] = (dP_mat[i + 1] - dP_mat[i]) / dt
            dR_mat[i + 1] = (R_mat[i + 1] - R_mat[i]) / dt
            ddR_mat[i + 1] = (dR_mat[i + 1] - dR_mat[i]) / dt

        fig, axes = plt.subplots(3, 2, figsize=(16, 9))
        axis_names = ("x", "y", "z")
        colors = ("r", "g", "b")
        line_styles = ("-", "--", ":")

        # 每个子图对应的数据及名称
        plot_items = [
            (axes[0, 0], P_mat,   "P/(m)",         "P"),
            (axes[1, 0], dP_mat,  "dP/(m/s)",      "dP"),
            (axes[2, 0], ddP_mat, "ddP/(m/s^2)",   "ddP"),
            (axes[0, 1], R_mat,   "R/(rad)",       "R"),
            (axes[1, 1], dR_mat,  "dR/(rad/s)",    "dR"),
            (axes[2, 1], ddR_mat, "ddR/(rad/s^2)", "ddR"),
        ]
        for ax, data, name, legend in plot_items:
            for j in range(3):
                ax.plot(
                    time_stamp,
                    data[:, j],
                    color=colors[j],
                    linestyle=line_styles[j],
                    label=f"{legend}{axis_names[j]}",
                )

            ax.set_xlabel("Time/(s)")
            ax.set_ylabel(name)
            ax.grid(True)
            ax.legend()

        axes[0, 0].set_title("Plot of Trajectory of Position")
        axes[0, 1].set_title("Plot of Trajectory of Rotation")

        # 自动调整布局
        fig.tight_layout()
        # fig.subplots_adjust(
        #     bottom=0.15,  # 底部留白占整张图的高度
        #     hspace=0.5    # 上下子图之间的间距
        # )
        # 显示窗口，并等待窗口关闭
        plt.show(block=True)
