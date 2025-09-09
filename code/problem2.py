from __future__ import annotations
import numpy as np
import math, time
import matplotlib.pyplot as plt

import params as p
import problem1 as P1
from ga import GeneticAlgorithm
from evaluator import missile_pos_general, munition_pos_general, cover_intervals_cached_general
from funcs import set_axes_equal_3d


# 基因: [v, theta(rad), t_drop, delay]
LB = np.array([p.v_min, -math.pi, 0.0, 0.0], dtype=float)
UB = np.array([p.v_max,  math.pi, 30.0, p.t_effective], dtype=float)


def decode(x: np.ndarray):
    """
    解码基因为可读参数
    """
    v = float(x[0])
    theta = float(x[1])
    dir_xy = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    n = np.linalg.norm(dir_xy); dir_xy = dir_xy / n if n > 0 else np.array([1.0, 0.0, 0.0])
    t_drop = float(x[2]); delay = float(x[3]); t_burst = t_drop + delay
    T_end = getattr(p, "T_end", float("inf"))
    if t_burst > T_end:
        delay = max(0.0, T_end - t_drop)
        t_burst = t_drop + delay
    return v, dir_xy, t_drop, delay, t_burst


def eval_total(x: np.ndarray, dt: float, stage: int, *, precheck: bool) -> float:
    v, dir_xy, t_drop, delay, t_burst = decode(x)
    return P1.total_cover_length_FY1(dt, v, dir_xy, t_drop, t_burst, stage=stage, precheck=precheck)


def plot_problem2_3d(FY0, v_drone, dir_xy, t_drop, delay, save_path, dt=0.05):
    """
    绘制三维场景图
    """

    # 时间轴
    t_burst = float(t_drop + delay)
    t_uav_mis = np.arange(0.0, t_burst + 1e-12, max(1e-3, dt))
    t_bomb    = np.arange(t_drop, t_burst + 1e-12, max(1e-3, dt))

    # UAV：等速直线（仅 XY，Z 恒定）
    FY_line = FY0[None, :3] + (v_drone * t_uav_mis[:, None]) * dir_xy[None, :]
    FY_line[:, 2] = FY0[2]

    # Bomb：抛物线
    bomb_traj = np.vstack([
        munition_pos_general(FY0, v_drone, dir_xy, t_drop, t) for t in t_bomb
    ])

    # Missile：直线
    mis1 = np.vstack([
        missile_pos_general(p.M1_0, t) for t in t_uav_mis
    ])

    # 起爆点 & 云团
    P_burst = munition_pos_general(FY0, v_drone, dir_xy, t_drop, t_burst)
    R_cloud = float(getattr(p, "R_effective", 10.0))

    # 作图
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 轨迹
    ax.plot(mis1[:,0], mis1[:,1], mis1[:,2], color="green", linewidth=2.0, label="Missile path(0->t_burst)")
    ax.plot(FY_line[:,0], FY_line[:,1], FY_line[:,2], color="blue",  linewidth=2.0, label="UAV path(0->t_burst)")
    ax.plot(bomb_traj[:,0], bomb_traj[:,1], bomb_traj[:,2], color="orange", linewidth=2.0, label="Bomb path(t_drop->t_burst)")

    # 起爆点
    ax.scatter([P_burst[0]], [P_burst[1]], [P_burst[2]], s=60, marker='o', color="orange", label="Burst point")

    # 云团（橙色半透明）
    u = np.linspace(0, 2*np.pi, 40); v = np.linspace(0, np.pi, 20)
    xs = P_burst[0] + R_cloud * np.outer(np.cos(u), np.sin(v))
    ys = P_burst[1] + R_cloud * np.outer(np.sin(u), np.sin(v))
    zs = P_burst[2] + R_cloud * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="orange", alpha=0.25, edgecolor="none")

    # 轴/标题
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Problem 2 — Scene Overview")
    ax.legend(loc="best")

    # 视窗范围（只基于三条轨迹和起爆点）
    xs_all = np.concatenate([mis1[:,0], FY_line[:,0], bomb_traj[:,0], [P_burst[0]]])
    ys_all = np.concatenate([mis1[:,1], FY_line[:,1], bomb_traj[:,1], [P_burst[1]]])
    zs_all = np.concatenate([mis1[:,2], FY_line[:,2], bomb_traj[:,2], [P_burst[2]]])
    ax.set_xlim(xs_all.min()-50, xs_all.max()+50)
    ax.set_ylim(ys_all.min()-50, ys_all.max()+50)
    ax.set_zlim(max(0.0, zs_all.min()-50), zs_all.max()+50)
    set_axes_equal_3d(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_problem2(cfg):

    ga = GeneticAlgorithm(
        lb=LB, ub=UB,
        pop_size=cfg["pop"], n_gen=cfg["gen"], seed=cfg["seed"],
        elite_frac=cfg["elite_frac"], tournament_k=3,
        sbx_eta=12.0, sbx_prob=0.9, mut_rate=0.20, mut_scale=0.10,
        objective="max",
    )

    # 退火变异幅度
    def sigma_fn(gen: int, n_gen: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        anneal = max(0.45, 1.0 - gen / n_gen)
        return np.array([
            (p.v_max - p.v_min)*0.05,  # v
            0.06*math.pi,              # theta
            0.6*anneal,                # t_drop
            0.5*anneal,                # delay
        ], dtype=float)

    fitness_fn = lambda x: eval_total(x, cfg["ga_dt"], stage=0, precheck=True)
    reeval_fn  = lambda x: eval_total(x, cfg["final_dt"], stage=1, precheck=False)

    hist_gen, hist_best, hist_mean = [], [], [] 

    def stats_hook(gen: int, best_s: float, mean_s: float):
        hist_gen.append(int(gen))
        hist_best.append(float(best_s))
        hist_mean.append(float(mean_s))

    def log_fn(gen: int, best_score: float, best_x: np.ndarray):
        if gen % 4 == 0:
            print(f"Gen {gen:02d}: best≈{best_score:.3f}s")

    seeds = [
        np.array([80.0, math.radians(  5.0), 0.60, 1.00], float),
        np.array([75.0, math.radians( 10.0), 1.20, 0.80], float),
        np.array([86.0, math.radians(-40.0), 0.40, 1.20], float),
        np.array([92.0, math.radians( 25.0), 1.60, 0.60], float),
    ]

    t0 = time.time()
    x_best, _ = ga.evolve(
        fitness_fn=fitness_fn,
        reeval_fn=reeval_fn, reeval_frac=cfg["reeval_frac"],
        verbose_every=4, log_fn=log_fn,
        seed_population=seeds, sigma_fn=sigma_fn,
        # ★ 关键：把"均值"也回调出来
        stats_hook=stats_hook,
    )
    t1 = time.time()

    length = eval_total(x_best, cfg["final_dt"], stage=1, precheck=False)
    v, dir_xy, t_drop, delay, t_burst = decode(x_best)
    theta_deg = math.degrees(math.atan2(dir_xy[1], dir_xy[0]))
    try:
        burst_pos = P1.munition_pos(v, dir_xy, t_drop, t_burst)
    except Exception:
        burst_pos = (float("nan"),)*3

    print("\n=== Problem 2 — GA Best Plan (final, total-length objective) ===")
    print(f"Total cover length: {length:.3f} s")
    print(f"Speed v = {v:.3f} m/s, Heading θ = {theta_deg:.3f}°")
    print(f"Drop time: {t_drop:.3f} s, Delay: {delay:.3f} s, Burst: {t_burst:.3f} s")
    print(f"Burst pos: ({burst_pos[0]:.3f}, {burst_pos[1]:.3f}, {burst_pos[2]:.3f}) m")
    print("Elapsed: %.2f s" % (t1 - t0))
    
    intervals, total_length = cover_intervals_cached_general(
        dt=cfg["final_dt"], 
        v_drone=v, 
        direction=dir_xy,
        t_drop=t_drop, 
        t_burst=t_burst, 
        FY0=p.FY1_0, 
        M0=p.M1_0, 
        stage=1, 
        precheck=False
    )
    
    if intervals:
        print("\nEffective cover intervals:")
        for i, (start, end) in enumerate(intervals, 1):
            print(f"  Interval {i}: {start:.3f} s to {end:.3f} s (duration: {end-start:.3f} s)")
    else:
        print("\nNo effective cover intervals found.")

    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(hist_gen, hist_best, 'r-', linewidth=1.2, label="Best")
    ax.plot(hist_gen, hist_mean, 'b-', linewidth=1.2, label="Population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Total cover length (s)")
    ax.set_title("GA progress (Problem 2)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/problem2_ga_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_problem2_3d(
        FY0=p.FY1_0,                
        v_drone=v,
        dir_xy=dir_xy,
        t_drop=t_drop,
        delay=delay,
        save_path="../figures/problem2_3d.png",
        dt=0.05      
    )


# 遗传算法配置
CFG = dict(
        pop=48, gen=36, seed=202509,
        elite_frac=0.16, reeval_frac=0.22,
        ga_dt=0.06, final_dt=0.01
    )


if __name__ == "__main__":

    run_problem2(CFG)