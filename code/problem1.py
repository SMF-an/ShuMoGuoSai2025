from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import params as p
from evaluator import cover_intervals_cached_general, munition_pos_general, missile_pos_general, cloud_pos            
from funcs import unit, set_axes_equal_3d


# FY1的特化函数
def munition_pos(v: float, dir_xy: np.ndarray, t_drop: float, t: float) -> np.ndarray:
    return munition_pos_general(p.FY1_0, v, dir_xy, t_drop, t)


def cover_intervals_FY1(dt: float, v: float, direction: np.ndarray,
                        t_drop: float, t_burst: float, *, stage: int = 1, precheck: bool = False):
    dir_xy = unit(np.array([direction[0], direction[1], 0.0], float))
    return cover_intervals_cached_general(
        dt, v, dir_xy, t_drop, t_burst, p.FY1_0, p.M1_0, stage=stage, precheck=precheck
    )


def total_cover_length_FY1(dt: float, v: float, direction: np.ndarray,
                           t_drop: float, t_burst: float, *, stage: int = 1, precheck: bool = False) -> float:
    intervals, total = cover_intervals_FY1(dt, v, direction, t_drop, t_burst, stage=stage, precheck=precheck)
    return float(total)


def plot_problem1_3d(v_drone: float, direction: np.ndarray,
                     t_drop: float, delay: float, dt: float = 0.05) -> None:
    """
    画出从 t=0 到 t_burst 的 3D 轨迹，并标注云团（球体半径 = p.R_effective）。
    direction: 任意 3D 向量，仅取 XY 分量（自动单位化）
    save_path: 保存图片路径（None 则不保存）
    """
    # --- 基本时间轴 ---
    t_burst = float(t_drop + delay)
    t_vec0 = np.arange(0.0, t_burst + 1e-12, max(1e-3, dt))
    t_vec_bomb = np.arange(t_drop, t_burst + 1e-12, max(1e-3, dt))

    dir_xy = unit(np.array([direction[0], direction[1], 0.0], float))  # 只取平面分量

    # --- 轨迹计算 ---
    # 1) 无人机：等速直线到 t_burst
    FY_line = p.FY1_0[None, :3] + (v_drone * t_vec0[:, None]) * dir_xy[None, :]
    FY_line[:, 2] = p.FY1_0[2]

    # 2) 干扰弹：从 t_drop 到 t_burst 的抛物线
    bomb_traj = np.vstack([
        munition_pos_general(p.FY1_0, v_drone, dir_xy, t_drop, t)
        for t in t_vec_bomb
    ])

    # 3) 导弹 M1：从起始到 t_burst
    mis_traj = np.vstack([
        missile_pos_general(p.M1_0, t)
        for t in t_vec0
    ])

    # 起爆点与云团
    P_burst = munition_pos_general(p.FY1_0, v_drone, dir_xy, t_drop, t_burst)
    R_cloud = float(p.R_effective)

    # 作图
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 无人机轨迹（蓝）
    ax.plot(FY_line[:, 0], FY_line[:, 1], FY_line[:, 2], linewidth=2.0, color='blue', label="UAV path (0→t_burst)")

    # 干扰弹轨迹（橙）
    ax.plot(bomb_traj[:, 0], bomb_traj[:, 1], bomb_traj[:, 2], linewidth=2.0, color='orange', label="Bomb path (t_drop→t_burst)")

    # 导弹轨迹（绿）
    ax.plot(mis_traj[:, 0], mis_traj[:, 1], mis_traj[:, 2], linewidth=2.0, color='green', label="Missile path (0→t_burst)")

    # 起爆点
    ax.scatter([P_burst[0]], [P_burst[1]], [P_burst[2]], s=60, marker='o', color='orange', label="Burst point")

    # 云团球体（橙色 + 半透明）
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = P_burst[0] + R_cloud * np.outer(np.cos(u), np.sin(v))
    ys = P_burst[1] + R_cloud * np.outer(np.sin(u), np.sin(v))
    zs = P_burst[2] + R_cloud * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="orange", alpha=0.25, edgecolor="none")

    # 轴/标题
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Problem 1 - Scene Overview")
    ax.legend(loc="best")

    # 视窗与等比例
    xs_all = np.concatenate([FY_line[:,0], bomb_traj[:,0], mis_traj[:,0], [P_burst[0]]])
    ys_all = np.concatenate([FY_line[:,1], bomb_traj[:,1], mis_traj[:,1], [P_burst[1]]])
    zs_all = np.concatenate([FY_line[:,2], bomb_traj[:,2], mis_traj[:,2], [P_burst[2]]])
    ax.set_xlim(xs_all.min()-50, xs_all.max()+50)
    ax.set_ylim(ys_all.min()-50, ys_all.max()+50)
    ax.set_zlim(max(0.0, zs_all.min()-50), zs_all.max()+50)
    set_axes_equal_3d(ax)

    plt.tight_layout()

    plt.savefig("./figures/problem1_3d.png", dpi=220, bbox_inches="tight")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":

    v_drone = 120.0
    t_drop  = 1.5
    delay   = 3.6
    t_burst = t_drop + delay

    # 朝向“假目标”的单位方向（仅用 XY 分量）
    direction = unit(np.array([p.fake[0]-p.FY1_0[0], p.fake[1]-p.FY1_0[1], 0.0], float))

    # 保留原有的数值评估/打印（如不需要可注释）
    from evaluator import cover_intervals_cached_general
    ints, total = cover_intervals_cached_general(
        dt=0.01, v_drone=v_drone, direction=direction,
        t_drop=t_drop, t_burst=t_burst, FY0=p.FY1_0, M0=p.M1_0, stage=1, precheck=False
    )
    P_burst = munition_pos_general(p.FY1_0, v_drone, direction, t_drop, t_burst)
    print("Burst point (m): x=%.3f, y=%.3f, z=%.3f" % (P_burst[0], P_burst[1], P_burst[2]))
    print("[Total] dur=%.3f intervals=%s" % (total, ints))

    # 画 3D 轨迹图
    plot_problem1_3d(v_drone, direction, t_drop, delay, dt=0.05)
