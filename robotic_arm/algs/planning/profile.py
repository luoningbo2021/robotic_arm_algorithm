"""提供一维三段梯形速度曲线的时间和位置计算接口."""

class TrapezoidProfile:
    """定义加速, 匀速, 减速三个阶段的梯形速度曲线.

    当前按起止速度均为 0 计算, v_start 和 v_end 仅保存, 未参与计算.
    v_max 和 a_max 应为正值, 路程应满足 s >= v_max**2 / a_max.

    Attributes:
        v_max (float): 最大速度.
        a_max (float): 最大加速度.
        s (float): 总路程, 单位应与速度和加速度的路程单位一致.
        v_start (float): 起点速度.
        v_end (float): 终点速度.
    """

    def __init__(self, v_max: float, a_max: float, s: float, v_start: float = 0, v_end: float = 0):
        """初始化梯形速度曲线参数.

        Args:
            v_max: 最大速度.
            a_max: 最大加速度.
            s: 总路程.
            v_start: 起点速度.
            v_end: 终点速度.
        """
        self.v_max = v_max
        self.a_max = a_max
        self.s = s
        self.v_start = v_start
        self.v_end = v_end

    def get_trapezoid_profile_time(self) -> float:
        """计算梯形速度曲线的总时长.

        Returns:
            总时长.

        Raises:
            AssertionError: 路程小于 v_max**2 / a_max, 无法形成三段梯形曲线.
        """
        # TODO: 处理路程不足时的三角形速度曲线
        assert(self.v_max**2 / self.a_max <= self.s), "必须满足 v**2 / a, 以确保三相梯形曲线"
        T = (self.a_max * self.s + self.v_max**2)/(self.a_max * self.v_max)
        return T

    def get_trapezoid_profile_pos(self, t: float) -> float:
        """计算指定时刻相对于起点的路程.

        Args:
            t: 采样时间, 应在 [0, T] 内, T 为轨迹总时长.

        Returns:
            相对于起点的路程.

        Raises:
            AssertionError: 路程小于 v_max**2 / a_max, 无法形成三段梯形曲线.
        """
        T = self.get_trapezoid_profile_time()

        # 根据采样时间选择加速段, 匀速段或减速段
        if t <= (self.v_max / self.a_max):
            st   = 0.5 * self.a_max * t**2
            # dst  = self.a_max * t
            # ddst = self.a_max
        elif (t > self.v_max / self.a_max) and (t <= T - self.v_max / self.a_max):
            st   = self.v_max * t - self.v_max**2 / (2 * self.a_max)
            # dst  = self.v_max
            # ddst = 0
        else:
            st  = (2 * self.a_max * self.v_max * T - 2 * self.v_max**2 - self.a_max**2 * (t - T)**2) / (2*self.a_max)
            # dst  = self.a_max * (T - t)
            # ddst = -self.a_max
        return st #, dst, ddst
