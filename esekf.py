import numpy as np
import numpy.linalg as la
import transformations as tr
import math


class ImuParameters:
    def __init__(self):
        """
        IMU参数初始化类
        用于存储和配置惯性测量单元(IMU)的各项参数
        """
        self.frequency = 200  # IMU采样频率，单位：Hz
        self.sigma_a_n = 0.0     # 加速度计噪声标准差，单位：m/(s*sqrt(s))，连续噪声标准差
        self.sigma_w_n = 0.0     # 陀螺仪噪声标准差，单位：rad/sqrt(s)，连续噪声标准差
        self.sigma_a_b = 0.0     # 加速度计零偏噪声标准差，单位：m/sqrt(s^5)，连续零偏噪声标准差
        self.sigma_w_b = 0.0     # 陀螺仪零偏噪声标准差，单位：rad/sqrt(s^3)，连续零偏噪声标准差


class ESEKF(object):
    def __init__(self, init_nominal_state: np.array, imu_parameters: ImuParameters):
        """
        扩展卡尔曼滤波器(ESEKF)初始化
        用于IMU姿态估计的误差状态卡尔曼滤波器
        
        :param init_nominal_state: 初始名义状态向量 [p, q, v, a_b, w_b, g]
                                   - p: 位置向量 (3维)
                                   - q: 四元数姿态 (4维)
                                   - v: 速度向量 (3维) 
                                   - a_b: 加速度计零偏 (3维)
                                   - w_b: 陀螺仪零偏 (3维)
                                   - g: 重力向量 (3维)
                                   总计19维向量 (3+4+3+3+3+3=19)
        :param imu_parameters: IMU参数对象，包含噪声和零偏参数
        """
        # 初始化名义状态
        self.nominal_state = init_nominal_state
        # 确保四元数的实部为正（四元数规范化要求）
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1    # 强制四元数实部为正
        
        # 存储IMU参数
        self.imu_parameters = imu_parameters

        # 初始化噪声协方差矩阵
        # 创建12x12的噪声协方差矩阵，对应4种噪声源，每种3维
        noise_covar = np.zeros((12, 12))
        # 假设噪声是各向同性的，可以预先计算并保存噪声协方差矩阵
        noise_covar[0:3, 0:3] = (imu_parameters.sigma_a_n**2) * np.eye(3)    # 加速度计噪声协方差
        noise_covar[3:6, 3:6] = (imu_parameters.sigma_w_n**2) * np.eye(3)    # 陀螺仪噪声协方差
        noise_covar[6:9, 6:9] = (imu_parameters.sigma_a_b**2) * np.eye(3)    # 加速度计零偏噪声协方差
        noise_covar[9:12, 9:12] = (imu_parameters.sigma_w_b**2) * np.eye(3)  # 陀螺仪零偏噪声协方差
        
        # 创建噪声传播矩阵G，将12维噪声映射到18维误差状态空间
        G = np.zeros((18, 12))
        G[3:6, 3:6] = -np.eye(3)    # 陀螺仪噪声影响角速度误差
        G[6:9, 0:3] = -np.eye(3)    # 加速度计噪声影响速度误差
        G[9:12, 6:9] = np.eye(3)    # 加速度计零偏噪声影响加速度零偏误差
        G[12:15, 9:12] = np.eye(3)  # 陀螺仪零偏噪声影响陀螺仪零偏误差
        
        # 计算完整的噪声协方差矩阵：G * noise_covar * G^T
        self.noise_covar = G @ noise_covar @ G.T

        # 初始化误差协方差矩阵
        # 使用噪声协方差矩阵的1%作为初始误差协方差
        self.error_covar = 0.01 * self.noise_covar

        # 记录上一次预测的时间戳，用于计算时间间隔
        self.last_predict_time = 0.0

    def predict(self, imu_measurement: np.array):
        """
        ESEKF预测步骤 - 使用IMU测量值进行状态预测
        基于IMU测量值更新滤波器状态和协方差矩阵
        
        :param imu_measurement: IMU测量值数组 [t, w_m, a_m]
                               - t: 时间戳 (秒)
                               - w_m: 陀螺仪测量值 (角速度，3维向量，单位：rad/s)
                               - a_m: 加速度计测量值 (加速度，3维向量，单位：m/s²)
        :return: 无返回值，直接更新滤波器内部状态
        """
        # 检查时间戳是否重复，避免重复预测
        if self.last_predict_time == imu_measurement[0]:
            return  # 如果时间戳相同，跳过本次预测
        
        # 先预测误差协方差矩阵，因为__predict_nominal_state会改变名义状态
        # 误差协方差预测需要基于当前状态，所以必须在名义状态更新之前进行
        self.__predict_error_covar(imu_measurement)
        
        # 预测名义状态（位置、姿态、速度等）
        self.__predict_nominal_state(imu_measurement)
        
        # 更新最后一次预测的时间戳
        self.last_predict_time = imu_measurement[0]  # 更新时间戳

    def __update_legacy(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        传统更新方法实现（已弃用）
        这是早期的更新过程实现，存在四元数直接减法的问题
        
        :param gt_measurement: 地面真值测量值 [p, q]，7维向量
                              - p: 位置向量 (3维)
                              - q: 四元数姿态 (4维)
        :param measurement_covar: 7x7测量噪声协方差矩阵
        :return: 无返回值，直接更新滤波器内部状态
        """
        """
        测量矩阵Hx = dh/dx = [[I, 0, 0, 0, 0, 0]
                             [0, I, 0, 0, 0, 0]]
        将19维名义状态映射到7维测量空间
        """
        Hx = np.zeros((7, 19))
        Hx[0:3, 0:3] = np.eye(3)    # 位置测量部分：测量位置误差
        Hx[3:7, 3:7] = np.eye(4)    # 姿态测量部分：直接测量四元数

        """
        雅可比矩阵X = dx/d(delta_x) = [[I_3, 0, 0],
                                     [0, Q_d_theta, 0],
                                     [0, 0, I_12]]
        将18维误差状态映射到19维名义状态空间
        """
        X = np.zeros((19, 18))
        q = self.nominal_state[3:7]  # 当前名义状态的四元数
        
        # 位置误差部分：直接映射
        X[0:3, 0:3] = np.eye(3)
        
        # 姿态误差部分：四元数关于姿态误差向量的雅可比矩阵
        # Q_d_theta = 0.5 * [[-q1, -q2, -q3],
        #                    [q0, -q3, q2],
        #                    [q3, q0, -q1],
        #                    [-q2, q1, q0]]
        X[3:7, 3:6] = 0.5 * np.array([[-q[1], -q[2], -q[3]],
                                      [q[0], -q[3], q[2]],
                                      [q[3], q[0], -q[1]],
                                      [-q[2], q[1], q[0]]])
        
        # 其余状态误差部分：直接映射
        X[7:19, 6:18] = np.eye(12)

        # 完整的测量矩阵H = Hx @ X，将18维误差状态映射到7维测量空间
        H = Hx @ X                      # 7x18
        
        # 计算PH^T矩阵，用于卡尔曼增益计算
        PHt = self.error_covar @ H.T    # 18x7
        
        # 计算卡尔曼增益K = PH^T * (HPH^T + R)^(-1)
        K = PHt @ la.inv(H @ PHt + measurement_covar)

        # 更新误差协方差矩阵：P = (I - KH)P
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        
        # 强制误差协方差矩阵为对称矩阵（数值稳定性）
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        """
        计算状态误差。
        限制测量和状态中的四元数具有正实部。
        这对于误差计算是必要的，因为我们直接进行四元数减法。
        """
        # 确保地面真值四元数的实部为正
        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        
        # 注意：直接进行四元数减法很棘手，这就是我们放弃这个实现的原因。
        # 计算测量残差：测量值 - 预测值
        errors = K @ (gt_measurement.reshape(-1, 1) - Hx @ self.nominal_state.reshape(-1, 1))

        # 将误差注入到名义状态中
        self.nominal_state[0:3] += errors[0:3, 0]  # 更新位置
        
        # 更新姿态：使用误差向量构建旋转四元数
        dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
        self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # 更新旋转
        
        # 四元数归一化
        self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
        
        # 确保四元数实部为正
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1
        
        # 更新其余状态变量（速度、零偏等）
        self.nominal_state[7:] += errors[6:, 0]  # 更新其余状态

        """
        重置误差为零并修改误差协方差矩阵。
        由于我们不保存误差，所以不对误差做任何操作。
        但我们需要根据 P = G P G^T 修改误差协方差矩阵。
        """
        # 构建误差重置矩阵G
        G = np.eye(18)
        # 姿态误差部分的雅可比矩阵：考虑误差注入后的协方差传播
        G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
        
        # 更新误差协方差矩阵以反映误差重置
        self.error_covar = G @ self.error_covar @ G.T

    def update(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        ESEKF更新步骤 - 使用地面真值测量值更新滤波器状态
        这是改进的更新方法，避免了直接四元数减法的问题
        
        :param gt_measurement: 地面真值测量值 [p, q]，7维向量
                              - p: 位置向量 (3维)
                              - q: 四元数姿态 (4维)
        :param measurement_covar: 6x6测量噪声协方差矩阵 = diag{sigma_p^2, sigma_theta^2}
                                 - sigma_p^2: 位置测量噪声方差
                                 - sigma_theta^2: 姿态测量噪声方差
        :return: 无返回值，直接更新滤波器内部状态
        """
        """
        我们模拟一个直接测量名义状态和地面真值状态之间误差的系统，
        这样可以避免直接进行四元数减法的问题。
        
        我们定义 q1 - q2 = conjugate(q2) x q1，这样 q2 x (q1 - q2) = q1。
        
        地面真值 - 名义状态 = delta = H @ 误差状态 + 噪声
        """
        
        # 构建测量矩阵H，将18维误差状态映射到6维测量空间
        H = np.zeros((6, 18))
        H[0:3, 0:3] = np.eye(3)    # 位置误差测量
        H[3:6, 3:6] = np.eye(3)    # 姿态误差测量
        
        # 计算PH^T矩阵，用于卡尔曼增益计算
        PHt = self.error_covar @ H.T  # 18x6
        
        # 计算卡尔曼增益K = PH^T * (HPH^T + R)^(-1)
        # 将误差协方差投影到测量空间并计算最优增益
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6

        # 更新误差协方差矩阵：P = (I - KH)P
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        
        # 强制误差协方差矩阵为对称矩阵（数值稳定性）
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        # 根据名义状态和地面真值状态计算测量值
        # 确保四元数的实部为正（规范化要求）
        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        
        # 提取地面真值的位置和姿态
        gt_p = gt_measurement[0:3]    # 地面真值位置
        gt_q = gt_measurement[3:7]    # 地面真值四元数
        q = self.nominal_state[3:7]   # 当前名义状态的四元数

        # 计算名义状态与地面真值之间的差异delta
        delta = np.zeros((6, 1))
        delta[0:3, 0] = gt_p - self.nominal_state[0:3]  # 位置差异
        
        # 计算四元数差异：delta_q = q_conj * gt_q
        delta_q = tr.quaternion_multiply(tr.quaternion_conjugate(q), gt_q)
        
        # 确保差异四元数的实部为正
        if delta_q[0] < 0:
            delta_q *= -1
        
        # 将四元数差异转换为轴角表示（3维姿态误差向量）
        angle = math.asin(la.norm(delta_q[1:4]))  # 旋转角度
        if math.isclose(angle, 0):
            axis = np.zeros(3,)  # 如果角度为0，轴向量为零
        else:
            axis = delta_q[1:4] / la.norm(delta_q[1:4])  # 旋转轴单位向量
        
        # 将轴角表示转换为3维姿态误差向量
        delta[3:6, 0] = angle * axis

        # 使用卡尔曼增益计算状态误差：误差 = K * delta
        errors = K @ delta

        # 将误差注入到名义状态中
        self.nominal_state[0:3] += errors[0:3, 0]  # 更新位置
        
        # 更新姿态：使用误差向量构建旋转四元数
        dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
        self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # 更新旋转
        
        # 四元数归一化
        self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
        
        # 确保四元数实部为正
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1
        
        # 更新其余状态变量（速度、零偏等）
        self.nominal_state[7:] += errors[6:, 0]  # 更新其余状态

        """
        重置误差为零并修改误差协方差矩阵。
        由于我们不保存误差，所以不对误差做任何操作。
        但我们需要根据 P = G P G^T 修改误差协方差矩阵。
        """
        # 构建误差重置矩阵G
        G = np.eye(18)
        # 姿态误差部分的雅可比矩阵：考虑误差注入后的协方差传播
        G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
        
        # 更新误差协方差矩阵以反映误差重置
        self.error_covar = G @ self.error_covar @ G.T

    def __predict_nominal_state(self, imu_measurement: np.array):
        """
        名义状态预测 - 使用IMU测量值预测系统状态
        基于IMU测量值更新位置、姿态、速度等名义状态变量
        
        :param imu_measurement: IMU测量值数组 [t, w_m, a_m]
        :return: 无返回值，直接更新名义状态
        """
        # 提取当前名义状态的各个分量
        p = self.nominal_state[:3].reshape(-1, 1)    # 位置向量 (3x1)
        q = self.nominal_state[3:7]                  # 四元数姿态 (4维)
        v = self.nominal_state[7:10].reshape(-1, 1)  # 速度向量 (3x1)
        a_b = self.nominal_state[10:13].reshape(-1, 1)  # 加速度计零偏 (3x1)
        w_b = self.nominal_state[13:16]              # 陀螺仪零偏 (3维)
        g = self.nominal_state[16:19].reshape(-1, 1) # 重力向量 (3x1)

        # 提取IMU测量值
        w_m = imu_measurement[1:4].copy()            # 陀螺仪测量值 (角速度)
        a_m = imu_measurement[4:7].reshape(-1, 1).copy()  # 加速度计测量值 (加速度)
        dt = imu_measurement[0] - self.last_predict_time  # 时间间隔

        """
        系统动态方程：
        dp/dt = v                          # 位置变化率等于速度
        dv/dt = R(a_m - a_b) + g           # 速度变化率等于旋转后的加速度加上重力
        dq/dt = 0.5 * q x (w_m - w_b)      # 四元数变化率等于四元数与角速度的乘积
        
        a_m 和 w_m 是IMU的测量值。
        a_b 和 w_b 分别是加速度计和陀螺仪的零偏。
        R = R{q}，是将点从局部坐标系转换到全局坐标系的旋转矩阵。
        """
        
        # 补偿零偏：从测量值中减去零偏
        w_m -= w_b    # 陀螺仪测量值减去零偏
        a_m -= a_b    # 加速度计测量值减去零偏

        # 使用零阶积分方法积分四元数
        # 下一时刻的四元数：q_{n+1} = q_n x q{(w_m - w_b) * dt}
        angle = la.norm(w_m)  # 旋转角度的大小
        axis = w_m / angle    # 旋转轴的单位向量
        
        # 计算半个时间步的旋转矩阵和四元数（用于RK4方法）
        R_w = tr.rotation_matrix(0.5 * dt * angle, axis)  # 半个时间步的旋转矩阵
        q_w = tr.quaternion_from_matrix(R_w, True)        # 对应的四元数
        q_half_next = tr.quaternion_multiply(q, q_w)      # 半个时间步后的四元数

        # 计算完整时间步的四元数
        R_w = tr.rotation_matrix(dt * angle, axis)        # 完整时间步的旋转矩阵
        q_w = tr.quaternion_from_matrix(R_w, True)        # 对应的四元数
        q_next = tr.quaternion_multiply(q, q_w)           # 下一时刻的四元数
        
        # 确保四元数的实部为正（规范化要求）
        if q_next[0] < 0:
            q_next *= -1

        # 使用RK4方法积分速度和位置
        # 先积分速度
        
        # 计算不同时间点的旋转矩阵
        R = tr.quaternion_matrix(q)[:3, :3]               # 当前时刻的旋转矩阵
        R_half_next = tr.quaternion_matrix(q_half_next)[:3, :3]  # 半个时间步后的旋转矩阵
        R_next = tr.quaternion_matrix(q_next)[:3, :3]     # 下一时刻的旋转矩阵
        
        # RK4方法的四个斜率计算
        v_k1 = R @ a_m + g           # k1：当前时刻的加速度
        v_k2 = R_half_next @ a_m + g # k2：半个时间步后的加速度
        v_k3 = v_k2                  # k3：与k2相同（因为旋转矩阵在半个时间步后不变）
        v_k4 = R_next @ a_m + g      # k4：下一时刻的加速度
        
        # 使用RK4公式计算下一时刻的速度
        v_next = v + dt * (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4) / 6

        # 积分位置
        p_k1 = v                     # k1：当前时刻的速度
        p_k2 = v + 0.5 * dt * v_k1   # k2：半个时间步后的速度（基于k1斜率）
        p_k3 = v + 0.5 * dt * v_k2   # k3：半个时间步后的速度（基于k2斜率）
        p_k4 = v + dt * v_k3         # k4：下一时刻的速度（基于k3斜率）
        
        # 使用RK4公式计算下一时刻的位置
        p_next = p + dt * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4) / 6

        # 更新名义状态
        self.nominal_state[:3] = p_next.reshape(3,)    # 更新位置
        self.nominal_state[3:7] = q_next              # 更新姿态
        self.nominal_state[7:10] = v_next.reshape(3,) # 更新速度
        # print(q_next)  # 调试输出（已注释）

    def __predict_error_covar(self, imu_measurement: np.array):
        """
        误差协方差预测 - 预测状态估计的不确定性如何随时间传播
        基于系统动态模型和噪声模型预测误差协方差矩阵
        
        :param imu_measurement: IMU测量值数组 [t, w_m, a_m]
        :return: 无返回值，直接更新误差协方差矩阵
        """
        # 提取IMU测量值
        w_m = imu_measurement[1:4]        # 陀螺仪测量值 (角速度)
        a_m = imu_measurement[4:7]        # 加速度计测量值 (加速度)
        
        # 提取当前名义状态的零偏和姿态
        a_b = self.nominal_state[9:12]    # 加速度计零偏
        w_b = self.nominal_state[12:15]   # 陀螺仪零偏
        q = self.nominal_state[3:7]       # 当前四元数姿态
        R = tr.quaternion_matrix(q)[:3, :3]  # 从四元数提取旋转矩阵

        # 构建系统动态矩阵F（状态转移矩阵的雅可比矩阵）
        F = np.zeros((18, 18))  # 18x18矩阵，对应18维误差状态
        
        # 位置误差动态：dp_error/dt = v_error
        F[0:3, 6:9] = np.eye(3)  # 位置误差对速度误差的导数
        
        # 姿态误差动态：dtheta/dt = -[w_m - w_b]× * theta - w_b_error
        F[3:6, 3:6] = -tr.skew_matrix(w_m - w_b)  # 姿态误差对自身的导数（角速度叉乘项）
        F[3:6, 12:15] = -np.eye(3)  # 姿态误差对陀螺仪零偏误差的导数
        
        # 速度误差动态：dv_error/dt = -R * [a_m - a_b]× * theta - R * a_b_error
        F[6:9, 3:6] = -R @ tr.skew_matrix(a_m - a_b)  # 速度误差对姿态误差的导数
        F[6:9, 9:12] = -R  # 速度误差对加速度计零偏误差的导数
        
        # 其余状态（零偏和重力）的动态方程为零，因为假设为随机游走过程

        # 使用三阶截断积分计算状态转移矩阵Phi
        dt = imu_measurement[0] - self.last_predict_time  # 时间间隔
        
        # 计算F矩阵的时间积分项
        Fdt = F * dt                    # 一阶项：F * dt
        Fdt2 = Fdt @ Fdt               # 二阶项：(F * dt)^2
        Fdt3 = Fdt2 @ Fdt              # 三阶项：(F * dt)^3
        
        # 计算状态转移矩阵Phi = exp(F * dt)的近似
        # 使用三阶截断：Phi ≈ I + Fdt + 0.5*Fdt^2 + (1/6)*Fdt^3
        Phi = np.eye(18) + Fdt + 0.5 * Fdt2 + (1. / 6.) * Fdt3

        """
        使用梯形积分方法积分噪声协方差：
          离散噪声协方差 Qd = 0.5 * dt * (Phi @ Qc @ Phi.T + Qc)
          误差协方差更新：P = Phi @ P @ Phi.T + Qd
          
        上述操作可以合并为下面的形式以提高效率。
        """
        # 计算连续噪声协方差的时间积分项
        Qc_dt = 0.5 * dt * self.noise_covar  # 0.5 * dt * Qc
        
        # 合并的误差协方差更新公式：
        # P = Phi @ (P + Qc_dt) @ Phi.T + Qc_dt
        # 这等价于：P = Phi @ P @ Phi.T + 0.5 * dt * (Phi @ Qc @ Phi.T + Qc)
        self.error_covar = Phi @ (self.error_covar + Qc_dt) @ Phi.T + Qc_dt