import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import yaml
import math
from esekf import *


def load_imu_parameters():
    """
    从YAML配置文件加载IMU参数，创建并初始化ImuParameters对象
    
    :return: ImuParameters对象，包含IMU的噪声参数和频率设置
    """
    # 打开并读取YAML配置文件
    f = open('./data/params.yaml', 'r')
    
    # 使用安全加载器解析YAML内容，避免潜在的安全风险
    yml = yaml.load(f.read(), Loader=yaml.SafeLoader)  # 添加 Loader 参数
    
    # 创建ImuParameters对象实例
    params = ImuParameters()
    
    # 设置IMU采样频率 (Hz)
    params.frequency = yml['IMU.frequency']
    
    # 设置加速度计白噪声标准差 (m/sqrt(s^3))
    # 表示加速度测量中的随机噪声强度
    params.sigma_a_n = yml['IMU.acc_noise_sigma']  # m/sqrt(s^3)
    
    # 设置陀螺仪白噪声标准差 (rad/sqrt(s))
    # 表示角速度测量中的随机噪声强度
    params.sigma_w_n = yml['IMU.gyro_noise_sigma']  # rad/sqrt(s)
    
    # 设置加速度计零偏随机游走标准差 (m/sqrt(s^5))
    # 表示加速度计零偏随时间变化的随机波动强度
    params.sigma_a_b = yml['IMU.acc_bias_sigma']  # m/sqrt(s^5)
    
    # 设置陀螺仪零偏随机游走标准差 (rad/sqrt(s^3))
    # 表示陀螺仪零偏随时间变化的随机波动强度
    params.sigma_w_b = yml['IMU.gyro_bias_sigma']  # rad/sqrt(s^3)
    
    # 关闭文件句柄，释放资源
    f.close()
    
    # 返回初始化完成的IMU参数对象
    return params



def main():
    """
    主函数 - 实现基于ESEKF的IMU姿态估计系统
    使用IMU数据和地面真值进行状态估计，并输出轨迹结果
    """
    # 加载IMU测量数据（包含噪声）
    imu_data = np.loadtxt('./data/imu_noise.txt')
    # 加载地面真值轨迹数据
    gt_data = np.loadtxt('./data/traj_gt.txt')

    # 从配置文件加载IMU参数（噪声特性、频率等）
    imu_parameters = load_imu_parameters()

    # 初始化19维名义状态向量
    init_nominal_state = np.zeros((19,))
    # 使用地面真值的初始位置、姿态和速度初始化前10个状态
    init_nominal_state[:10] = gt_data[0, 1:]                # init p, q, v
    # 初始化加速度计零偏为0
    init_nominal_state[10:13] = 0                           # init ba
    # 初始化陀螺仪零偏为0
    init_nominal_state[13:16] = 0                           # init bg
    # 初始化重力向量为[0, 0, -9.81] m/s²
    init_nominal_state[16:19] = np.array([0, 0, -9.81])     # init g
    
    # 创建误差状态扩展卡尔曼滤波器(ESEKF)实例
    estimator = ESEKF(init_nominal_state, imu_parameters)

    # 设置测试时间范围：0到61秒
    test_duration_s = [0., 61.]
    # 获取IMU数据的起始时间戳
    start_time = imu_data[0, 0]
    
    # 创建时间掩码，筛选指定时间范围内的IMU数据
    mask_imu = np.logical_and(imu_data[:, 0] <= start_time + test_duration_s[1],
                              imu_data[:, 0] >= start_time + test_duration_s[0])
    # 创建时间掩码，筛选指定时间范围内的地面真值数据
    mask_gt = np.logical_and(gt_data[:, 0] <= start_time + test_duration_s[1],
                             gt_data[:, 0] >= start_time + test_duration_s[0])

    # 应用时间掩码，截取指定时间范围内的数据
    imu_data = imu_data[mask_imu, :]
    gt_data = gt_data[mask_gt, :]

    # 初始化估计轨迹，以地面真值的初始帧作为起点
    traj_est = [gt_data[0, :8]]
    
    # 设置更新比率：每10个预测步骤执行1次更新
    update_ratio = 10    # control the frequency of ekf updating.
    
    # 设置位置测量噪声标准差：0.02米
    sigma_measurement_p = 0.02   # in meters
    # 设置姿态测量噪声标准差：0.015弧度
    sigma_measurement_q = 0.015  # in rad
    
    # 创建6x6测量噪声协方差矩阵（位置3维 + 姿态3维）
    sigma_measurement = np.eye(6)
    # 设置位置测量噪声协方差：对角元素为sigma_measurement_p²
    sigma_measurement[0:3, 0:3] *= sigma_measurement_p**2
    # 设置姿态测量噪声协方差：对角元素为sigma_measurement_q²
    sigma_measurement[3:6, 3:6] *= sigma_measurement_q**2
    
    # 主循环：遍历所有IMU数据点（从第2个点开始）
    for i in range(1, imu_data.shape[0]):
        # 获取当前时间戳
        timestamp = imu_data[i, 0]
        # 执行预测步骤：使用IMU测量值更新滤波器状态
        estimator.predict(imu_data[i, :])
        
        # 检查是否达到更新条件（每update_ratio个点更新一次）
        if i % update_ratio == 0:
            # 验证时间戳对齐（假设IMU和地面真值时间戳同步）
            assert math.isclose(gt_data[i, 0], timestamp)
            
            # 复制当前地面真值位姿 [位置(3), 姿态(4)]
            gt_pose = gt_data[i, 1:8].copy()  # gt_pose = [p, q]
            
            # 为位置测量添加高斯噪声
            gt_pose[:3] += np.random.randn(3,) * sigma_measurement_p
            
            # 为姿态测量添加旋转噪声
            # 生成随机旋转轴向量（3维高斯噪声）
            u = np.random.randn(3, ) * sigma_measurement_q
            # 创建绕随机轴旋转的四元数（旋转角度为向量模长）
            qn = tr.quaternion_about_axis(la.norm(u), u / la.norm(u))
            # 将旋转噪声应用到地面真值姿态
            gt_pose[3:] = tr.quaternion_multiply(gt_pose[3:], qn)
            
            # 执行更新步骤：使用带噪声的地面真值测量值更新滤波器
            estimator.update(gt_pose, sigma_measurement)

        # 打印当前时间戳和名义状态
        print('[%f]:' % timestamp, estimator.nominal_state)
        
        # 创建当前帧的位姿数据（TUM格式：时间戳 + 位置 + 姿态）
        frame_pose = np.zeros(8,)
        frame_pose[0] = timestamp
        frame_pose[1:] = estimator.nominal_state[:7]  # 提取位置和姿态
        
        # 将当前帧添加到估计轨迹中
        traj_est.append(frame_pose)

    # 保存轨迹到TUM格式文件
    traj_est = np.array(traj_est)
    # 保存地面真值轨迹（前8列：时间戳 + 位置 + 姿态）
    np.savetxt('./data/traj_gt_out.txt', gt_data[:, :8])
    # 保存估计轨迹
    np.savetxt('./data/traj_esekf_out.txt', traj_est)


if __name__ == '__main__':
    main()