from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import math, time

import params as p
from evaluator import cover_intervals_cached_general, missile_pos_general, munition_pos_general
from ga import GeneticAlgorithm
from funcs import set_axes_equal_3d


T_UP = float(getattr(p, "T_end", 60.0))
# [v, theta, t_drop, delay]
LB = np.array([p.v_min, -math.pi, 0.0, 0.0], dtype=float)              
UB = np.array([p.v_max,  math.pi, T_UP, p.t_effective], dtype=float)


def _decode(x: np.ndarray):
    """
    解码基因为可读参数
    """
    v = float(x[0])
    theta = float(x[1])
    dir_xy = np.array([math.cos(theta), math.sin(theta), 0.0], float)
    n = np.linalg.norm(dir_xy); dir_xy = dir_xy/n if n>0 else np.array([1.0,0.0,0.0], float)
    t_drop = float(x[2])
    delay  = float(min(max(0.0, x[3]), float(p.t_effective)))
    t_burst = t_drop + delay
    if t_burst > T_UP:
        delay = max(0.0, T_UP - t_drop)
        t_burst = t_drop + delay
    return v, dir_xy, t_drop, delay, t_burst


def _cover_FY3(dt: float, v: float, dir_xy: np.ndarray,
               t_drop: float, t_burst: float,
               *, stage: int, precheck: bool):
    try:
        return cover_intervals_cached_general(dt, v, dir_xy, t_drop, t_burst,
                                              p.FY3_0, p.M1_0, stage=stage, precheck=precheck)
    except TypeError:
        return cover_intervals_cached_general(dt, v, dir_xy, t_drop, t_burst,
                                              p.FY3_0, p.M1_0, stage=stage)


def _eval_total(x: np.ndarray, dt: float, stage: int, precheck: bool) -> float:
    v, dir_xy, t_drop, delay, t_burst = _decode(x)
    _, total = _cover_FY3(dt, v, dir_xy, t_drop, t_burst, stage=stage, precheck=precheck)
    return float(total)


def _smart_seeds_for_FY3_nominal() -> List[np.ndarray]:
    """
    基于"指定近似最优点"的智能种子（最多32个）
    """
    
    NOM_TH_DEG = 90.0
    NOM_V      = 110.0
    NOM_TDROP  = 23.8
    NOM_DELAY  = 4.2

    th0 = math.radians(NOM_TH_DEG)
    def _clip_v(v):  return float(max(p.v_min, min(p.v_max, v)))
    def _clip_td(td): return float(max(0.0, min(T_UP, td)))
    def _make(v, th, td, dl): return np.array([_clip_v(v), th, _clip_td(td), float(dl)], float)

    td_set = [NOM_TDROP + d for d in (-3.0, -1.5, 0.0, +1.2, +3.0)]
    dl_set = [NOM_DELAY + d for d in (-1.0, -0.4, 0.0, +0.4, +1.0)]
    v_set  = [NOM_V, NOM_V-6, NOM_V+6, NOM_V-12, NOM_V+12]

    seeds: List[np.ndarray] = []
    for th in [th0,
               th0 + math.radians(7), th0 - math.radians(7),
               th0 + math.radians(12), th0 - math.radians(12),
               th0 + math.radians(18), th0 - math.radians(18)]:
        seeds.append(_make(NOM_V, th, NOM_TDROP, NOM_DELAY))
    for td in td_set:
        for dl in [NOM_DELAY-0.4, NOM_DELAY, NOM_DELAY+0.4]:
            seeds.append(_make(NOM_V, th0, td, dl))
    for v in v_set:
        seeds.append(_make(v, th0, NOM_TDROP, NOM_DELAY))
    for th in [th0 + math.radians(24), th0 - math.radians(24),
               th0 + math.radians(30), th0 - math.radians(30)]:
        seeds.append(_make(NOM_V, th, NOM_TDROP + 3.5, NOM_DELAY + 0.8))
    for th in [th0 + math.radians(12), th0 - math.radians(12)]:
        for v in [NOM_V-12, NOM_V+12]:
            seeds.append(_make(v, th, NOM_TDROP + 1.2, NOM_DELAY))

    uniq: List[np.ndarray] = []
    for s in seeds:
        if not any(np.allclose(s, t) for t in uniq):
            uniq.append(s)
    return uniq[:32]


def plot_fy3_3d(FY0, v_drone, dir_xy, t_drop, delay, save_path, dt=0.05):
    """
    绘制三维场景图
    """

    # 时间轴
    t_burst = float(t_drop + delay)
    t_uav_mis = np.arange(0.0, t_burst + 1e-12, max(1e-3, dt))
    t_bomb    = np.arange(t_drop, t_burst + 1e-12, max(1e-3, dt))

    # UAV：等速直线（XY 按航向，Z 恒定）
    FY_line = FY0[None, :3] + (v_drone * t_uav_mis[:, None]) * dir_xy[None, :]
    FY_line[:, 2] = FY0[2]

    # Bomb：抛物线（投放到起爆）
    bomb_traj = np.vstack([munition_pos_general(FY0, v_drone, dir_xy, t_drop, t) for t in t_bomb])

    # Missile1：直线（朝原点）
    mis1 = np.vstack([missile_pos_general(p.M1_0, t) for t in t_uav_mis])

    # 起爆点 & 云团
    P_burst = munition_pos_general(FY0, v_drone, dir_xy, t_drop, t_burst)
    R_cloud = float(getattr(p, "R_effective", 10.0))

    # 绘图
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 三条轨迹
    ax.plot(mis1[:,0],  mis1[:,1],  mis1[:,2],  color="green",  linewidth=1.8, label="Missile 1")
    ax.plot(FY_line[:,0],FY_line[:,1],FY_line[:,2],color="blue",   linewidth=2.2, label="UAV (FY3)")
    ax.plot(bomb_traj[:,0], bomb_traj[:,1], bomb_traj[:,2], color="orange", linewidth=2.2, label="Bomb")

    # 起爆点
    ax.scatter([P_burst[0]], [P_burst[1]], [P_burst[2]], s=60, marker='o', color="orange", label="Burst point")

    # 云团（橙色半透明）
    u = np.linspace(0, 2*np.pi, 40); v_ = np.linspace(0, np.pi, 20)
    xs = P_burst[0] + R_cloud * np.outer(np.cos(u), np.sin(v_))
    ys = P_burst[1] + R_cloud * np.outer(np.sin(u), np.sin(v_))
    zs = P_burst[2] + R_cloud * np.outer(np.ones_like(u), np.cos(v_))
    ax.plot_surface(xs, ys, zs, color="orange", alpha=0.25, edgecolor="none")

    # 坐标轴与范围
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Problem 4 (FY3) — Scene Overview")
    ax.legend(loc="best")

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


def run_problem4_FY3(cfg: Dict[str, Any]):

    ga = GeneticAlgorithm(
        lb=LB, ub=UB,
        pop_size=cfg["pop"], n_gen=cfg["gen"], seed=cfg["seed"],
        elite_frac=cfg["elite_frac"], tournament_k=3,
        sbx_eta=12.0, sbx_prob=0.90,
        mut_rate=0.20, mut_scale=0.10,
        objective="max",
    )

    # 自适应扰动幅度（随迭代进程逐渐减小）
    def sigma_fn(g: int, G: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        phase = g / G
        anneal = max(0.40, 1.0 - phase)
        ang_scale = (0.12 if phase < 0.5 else 0.08) * math.pi  # 早大后小
        return np.array([
            (p.v_max - p.v_min)*0.05,  # v
            ang_scale,                 # theta
            0.10*T_UP*anneal,          # t_drop
            0.60*anneal,               # delay
        ], float)

    fitness_fn = lambda x: _eval_total(x, cfg["ga_dt"],   stage=0, precheck=cfg["precheck_stage0"])
    reeval_fn  = lambda x: _eval_total(x, cfg["final_dt"], stage=1, precheck=False)

    seeds = _smart_seeds_for_FY3_nominal()

    hist_gen, hist_best, hist_mean = [], [], []
    def stats_hook(g: int, best: float, mean: float):
        hist_gen.append(int(g)); hist_best.append(float(best)); hist_mean.append(float(mean))
    def _log(g, best, x):
        if g % 4 == 0:
            print(f"[FY3] Gen {g:02d} best≈{best:.3f}s")

    t0 = time.time()
    x_best, _ = ga.evolve(fitness_fn=fitness_fn, reeval_fn=reeval_fn, reeval_frac=cfg["reeval_frac"],
                          verbose_every=4, log_fn=_log,
                          seed_population=seeds, sigma_fn=sigma_fn,
                          stats_hook=stats_hook)
    t1 = time.time()

    val = _eval_total(x_best, cfg["final_dt"], stage=1, precheck=False)
    v, dir_xy, t_drop, delay, t_burst = _decode(x_best)
    theta_deg = math.degrees(math.atan2(dir_xy[1], dir_xy[0]))

    print("\n=== FY3 — Single Bomb (Total-Length Objective) ===")
    print(f"Total cover length: {val:.3f} s")
    print(f"Speed v = {v:.3f} m/s, Heading θ = {theta_deg:.3f}°")
    print(f"Drop: {t_drop:.3f}s, Delay: {delay:.3f}s, Burst: {t_burst:.3f}s")
    print("Elapsed: %.2f s" % (t1 - t0))
    
    intervals, total_length = _cover_FY3(
        dt=cfg["final_dt"], 
        v=v, 
        dir_xy=dir_xy,
        t_drop=t_drop, 
        t_burst=t_burst, 
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
    ax.set_title("GA progress (Problem4-FY3)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/fy3_ga_progress.png", dpi=180, bbox_inches="tight")
    plt.show()

    # 3D 可视化
    plot_fy3_3d(
        FY0=p.FY3_0,
        v_drone=v,
        dir_xy=dir_xy,
        t_drop=t_drop,
        delay=delay,
        dt=0.05,
        save_path="../figures/problem4_fy3_3d.png"
    )


# 遗传算法配置
CFG = dict(
        pop=60, gen=44, seed=202509,
        elite_frac=0.18, reeval_frac=0.32,
        ga_dt=0.06, final_dt=0.01, precheck_stage0=True
    )


if __name__ == "__main__":

    run_problem4_FY3(CFG)