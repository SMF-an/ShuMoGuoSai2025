from __future__ import annotations
import math
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import params as p
from funcs import unit, seg_point_min_dist

# 开关：为避免误杀，这里关闭快速充分判据预筛（需要速度再打开）
USE_QUICK_PRECHECK = False


def sample_cylinder_surface(R=p.R, H=p.H, n_theta=160, n_z=25, n_r=8) -> np.ndarray:
    """
    采样圆柱体表面点
    """
    pts = []
    angle_1, angle_2 = -0.75*np.pi, 0.75*np.pi   # 只采前 270° 贴题意，同时减少计算量
    
    # 前侧面
    for i in range(n_theta):
        th = angle_1 + (angle_2 - angle_1) * i / n_theta
        x = p.real[0] + R * np.cos(th)
        y = p.real[1] + R * np.sin(th)
        for j in range(n_z + 1):
            z = p.real[2] + H * j / n_z
            pts.append([x, y, z])

    # 上端面
    z = p.real[2] + H
    for ir in range(n_r + 1):
        r = R * ir / n_r
        for i in range(n_theta):
            th = 2 * np.pi * i / n_theta
            x = p.real[0] + r * np.cos(th)
            y = p.real[1] + r * np.sin(th)
            pts.append([x, y, z])

    return np.asarray(pts, float)


SURF = sample_cylinder_surface()


"""
基本运动学函数
"""
def missile_dir(M0: np.ndarray) -> np.ndarray:
    return unit(p.fake - M0)

def missile_pos_general(M0: np.ndarray, t: float) -> np.ndarray:
    return M0 + p.v_missile * t * missile_dir(M0)

def munition_pos_general(FY0: np.ndarray, v: float, dir_xy: np.ndarray, t_drop: float, t: float) -> np.ndarray:
    if t < t_drop:
        raise ValueError("t 必须 ≥ t_drop")
    FY_drop = FY0 + v * t_drop * dir_xy
    dt = t - t_drop
    pos = FY_drop + v * dt * dir_xy
    pos = pos.copy()
    pos[2] = FY_drop[2] - 0.5 * p.g * dt**2
    return pos

def cloud_pos(t: float, t_burst: float, P_burst: np.ndarray) -> np.ndarray:
    if t < t_burst:
        raise ValueError("t 必须 ≥ t_burst")
    return np.array([P_burst[0], P_burst[1], P_burst[2] - p.v_cloud * (t - t_burst)], float)


def covered_strict_at_time(t: float, t_burst: float, P_burst: np.ndarray,
                            M0: np.ndarray, Psurf: np.ndarray) -> bool:
    """
    严格判据：t 时刻导弹位置 Mt 是否被云雾覆盖
    """
    Mt = missile_pos_general(M0, t)
    Ct = cloud_pos(t, t_burst, P_burst)
    d = seg_point_min_dist(Ct, Mt, Psurf)
    return np.max(d - p.R_effective) <= 1e-6


Tb = np.array([p.real[0], p.real[1], p.real[2] + p.H/2.0], float) # 圆柱体中心点
rb = float(np.sqrt(p.R**2 + (p.H/2.0)**2)) # 圆柱体外接球半径


def quick_sufficient_cover(Mt: np.ndarray, Ct: np.ndarray) -> bool:
    """
    以圆柱体外接球代替圆柱体的快速充分判据
    仅当云雾中心 Ct 在导弹 Mt 与圆柱体外接球连线的“导弹侧”时，才可能遮蔽
    可能误杀（即判定不遮蔽但实际上遮蔽了），但不会误判
    """
    v1, v2 = Tb - Mt, Ct - Mt
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)

    if d1 <= rb: 
        return False
    if d2 <= p.R_effective: 
        return True
    
    alpha = math.asin(min(1.0, rb / d1))
    beta  = math.asin(min(1.0, p.R_effective / d2))
    cosd = float(np.dot(v1, v2) / (d1 * d2)); cosd = max(-1.0, min(1.0, cosd))
    delta = math.acos(cosd)
    return beta >= alpha + delta


def quick_has_any(M0: np.ndarray, P_burst: np.ndarray, t_burst: float,
                   dt: float, t_cap: float) -> bool:
    """
    只要有一个采样点被遮蔽就返回 True
    """
    dt = float(dt if dt > 0 else 1e-6)
    for t in np.arange(t_burst, min(t_burst + p.t_effective, t_cap) + 1e-12, dt):
        Ct = cloud_pos(t, t_burst, P_burst)
        Mt = missile_pos_general(M0, t)
        if quick_sufficient_cover(Mt, Ct):
            return True
    return False


def _q(x: float, step: float) -> float:
    return float(round(x / step) * step)

def _pos_dt(x: float) -> float:
    return float(x if x > 0 else 1e-6)


@lru_cache(maxsize=120000)
def _core_intervals_q(
    dt_q: float, v_q: float, theta_q: float, t_drop_q: float, t_burst_q: float,
    FYx: float, FYy: float, FYz: float, Mx: float, My: float, Mz: float
) -> Tuple[Tuple[Tuple[float,float], ...], float]:
    dt_q = _pos_dt(dt_q)
    FY0 = np.array([FYx, FYy, FYz], float)
    M0  = np.array([Mx,  My,  Mz ], float)
    dir_xy = unit(np.array([math.cos(theta_q), math.sin(theta_q), 0.0], float))

    P_burst = munition_pos_general(FY0, v_q, dir_xy, t_drop_q, t_burst_q)
    T_cap = min(getattr(p, "T_end", float("inf")), float(np.linalg.norm(p.fake - M0) / p.v_missile))
    t0, t1 = t_burst_q, min(t_burst_q + p.t_effective, T_cap)
    if t1 <= t0:
        return tuple(), 0.0

    if USE_QUICK_PRECHECK:
        if not quick_has_any(M0, P_burst, t_burst_q, max(0.12, 3.0*dt_q), T_cap):
            return tuple(), 0.0

    times = np.arange(t0, t1 + 1e-12, dt_q)
    flags = [covered_strict_at_time(tt, t_burst_q, P_burst, M0, SURF) for tt in times]

    ints: List[Tuple[float,float]] = []
    in_run, s = False, None
    for i, ok in enumerate(flags):
        if ok and not in_run:
            in_run, s = True, times[i]
        elif (not ok) and in_run:
            ints.append((float(s), float(times[i-1])))
            in_run = False
    if in_run:
        ints.append((float(s), float(times[-1])))

    total = float(sum(e - s for s, e in ints))
    return tuple(ints), total


def cover_intervals_cached_general(
    dt: float, v_drone: float, direction: np.ndarray,
    t_drop: float, t_burst: float, FY0: np.ndarray, M0: np.ndarray,
    *, stage: int = 0, precheck: bool = False
) -> Tuple[List[Tuple[float,float]], float]:
    """
    量化参数后调用核心函数计算覆盖区间
    """
    # 分阶段量化（精评更细）
    theta_step = math.radians(0.25 if stage == 1 else 0.5)
    v_step     = 0.25 if stage == 1 else 0.5
    t_step     = 0.01 if stage == 1 else 0.02
    dt_step    = 0.005 if stage == 1 else 0.01

    dir_xy = unit(np.array([direction[0], direction[1], 0.0], float))
    theta = math.atan2(dir_xy[1], dir_xy[0])

    dt_q    = _pos_dt(_q(dt, dt_step))
    v_q     = _q(v_drone, v_step)
    theta_q = _q(theta, theta_step)
    t_drop_q  = _q(t_drop, t_step)
    t_burst_q = _q(t_burst, t_step)

    # 快速充分判据预筛
    if precheck and stage == 0:
        P_burst = munition_pos_general(FY0, v_q, unit(np.array([math.cos(theta_q), math.sin(theta_q), 0.0], float)),
                                       t_drop_q, t_burst_q)
        T_cap = min(getattr(p, "T_end", float("inf")),
                    float(np.linalg.norm(p.fake - M0) / p.v_missile))
        if not quick_has_any(M0, P_burst, t_burst_q, max(0.12, 3.0*dt_q), T_cap):
            return [], 0.0

    ints_t, total = _core_intervals_q(
        dt_q, v_q, theta_q, t_drop_q, t_burst_q,
        float(FY0[0]), float(FY0[1]), float(FY0[2]),
        float(M0[0]),  float(M0[1]),  float(M0[2]),
    )
    return list(ints_t), float(total)


# 兼容：FY1/M1
def cover_intervals(dt: float, v_drone: float, direction: np.ndarray,
                    t_drop: float, t_burst: float) -> Tuple[List[Tuple[float,float]], float]:
    return cover_intervals_cached_general(dt, v_drone, direction, t_drop, t_burst, p.FY1_0, p.M1_0, stage=1)
