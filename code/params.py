import numpy as np


g = 9.81                 # 重力加速度
v_missile = 300.0        # 导弹恒速
v_cloud = 3.0            # 云团下沉速度
v_min = 70.0             # 无人机最低速度
v_max = 140.0            # 无人机最高速度
R_effective = 10.0       # 有效遮蔽半径
t_effective = 20.0       # 有效遮蔽时间窗长度
interval = 1.0           # 最短投弹时间间隔

M1_0 = np.array([20000.0, 0.0, 2000.0])        # 导弹 M1
M2_0 = np.array([19000.0, 600.0, 2100.0])      # 导弹 M2
M3_0 = np.array([18000.0, -600.0, 1900.0])     # 导弹 M3
FY1_0 = np.array([17800.0, 0.0, 1800.0])       # 无人机 FY1
FY2_0 = np.array([12000.0, 1400.0, 1400.0])    # 无人机 FY2
FY3_0 = np.array([6000.0, -3000.0, 700.0])     # 无人机 FY3
FY4_0 = np.array([11000.0, 2000.0, 1800.0])    # 无人机 FY4
FY5_0 = np.array([13000.0, -2000.0, 1300.0])   # 无人机 FY5
fake = np.array([0.0, 0.0, 0.0])               # 假目标
real = np.array([0.0, 200.0, 0.0])             # 真目标圆柱底面中心

R = 7.0           # 圆柱半径
H = 10.0          # 圆柱高度

T_end = float(np.linalg.norm(fake - M1_0) / v_missile)      # 模拟终止时间（导弹到达假目标时刻）