from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import math, os, time
import pandas as pd

import params as p
from funcs import unit, union_length
from ga import GeneticAlgorithm
from evaluator import cover_intervals_cached_general, munition_pos_general


# 赛题常量
_YAXIS   = float(getattr(p, "y_axis", 200.0))
_R_CYL   = float(getattr(p, "R_cyl", 7.0))
_H_CYL   = float(getattr(p, "H_cyl", 10.0))
_R_SMOKE = float(getattr(p, "R_smoke", 10.0))
_SINK    = float(getattr(p, "SINK", 3.0))
_g       = float(getattr(p, "g", 9.8))
t_effective = float(getattr(p, "cloud_lifetime", 20.0))
V_MIN, V_MAX = float(getattr(p, "v_min", 70.0)), float(getattr(p, "v_max", 140.0))
T_END = float(getattr(p, "T_end", 60.0))

# 运动学
def _arrival_time(M0: np.ndarray) -> float:
    return float(np.linalg.norm(p.fake - M0) / p.v_missile)

def _heading_vec(theta_rad: float) -> np.ndarray:
    return unit(np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0], float))

def _missile_pos(M0: np.ndarray, t: float) -> np.ndarray:
    u = unit(p.fake - M0)
    return M0 + u * (p.v_missile * float(t))

def _uav_xy(FY0: np.ndarray, v: float, theta: float, t: float) -> np.ndarray:
    dir_xy = _heading_vec(theta)
    return FY0[:2] + dir_xy[:2] * (v * t)

def _burst_xyz(FY0: np.ndarray, v: float, theta: float, t_drop: float, delay: float) -> np.ndarray:
    xy = _uav_xy(FY0, v, theta, t_drop + delay)
    z  = FY0[2] - 0.5*_g*(delay**2) if delay > 0 else FY0[2]
    return np.array([xy[0], xy[1], z], float)


# 快速充分判据（预筛）
TB = np.array([0.0, _YAXIS, _H_CYL/2.0], float)
RB = float((_R_CYL**2 + (_H_CYL/2.0)**2) ** 0.5)

def quick_sufficient_cover(Mt: np.ndarray, Ct: np.ndarray) -> bool:
    v1, v2 = TB - Mt, Ct - Mt
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if d1 <= RB: return False
    if d2 <= _R_SMOKE: return True
    alpha = math.asin(min(1.0, RB/d1))
    beta  = math.asin(min(1.0, _R_SMOKE/d2))
    cosd  = float(np.dot(v1, v2)/(d1*d2)); cosd = max(-1.0, min(1.0, cosd))
    delta = math.acos(cosd)
    return beta >= alpha + delta

def quick_has_any(FY0: np.ndarray, v: float, theta: float,
                   t_drop: float, delay: float,
                   M0: np.ndarray, T_cap: float) -> bool:
    t_burst = t_drop + delay
    if t_burst <= 0.0 or t_burst > T_cap: return False
    t_end = min(t_burst + t_effective, _arrival_time(M0), T_cap)
    if t_end <= t_burst: return False

    Ct0 = _burst_xyz(FY0, v, theta, t_drop, delay)
    span = float(t_end - t_burst)
    steps = max(5, min(10, int(round(span / 0.7))))
    step  = float(max(0.5, min(1.0, span / steps)))

    t = t_burst
    while t <= t_end + 1e-9:
        Mt = _missile_pos(M0, t)
        Ct = Ct0.copy(); Ct[2] = Ct0[2] - _SINK * (t - t_burst)
        if quick_sufficient_cover(Mt, Ct): return True
        t += step
    return False

# 缓存 + 上界剪枝
class FitnessCache:
    def __init__(self, quant: float = 1e-3): 
        self.q=quant
        self._map: Dict[Tuple, float]={}
    def key(self, arr: List[float], dt: float, stage: int) -> Tuple:
        return tuple(int(round(float(v)/self.q)) for v in (arr+[dt,stage]))
    def get(self, arr: List[float], dt: float, stage: int):
        return self._map.get(self.key(arr, dt, stage))
    def put(self, arr: List[float], dt: float, stage: int, val: float):
        self._map[self.key(arr, dt, stage)] = float(val)

class SegmentCache:
    def __init__(self, quant: float = 2e-3): self.q=quant; self._map: Dict[Tuple, List[Tuple[float,float]]]={}
    def key(self, FY0, M0, v, th, tr, delay, dt, stage):
        arr = list(FY0)+list(M0)+[v, th, tr, delay, dt, stage]
        return tuple(int(round(float(a)/self.q)) for a in arr)
    def get(self, FY0, M0, v, th, tr, delay, dt, stage):
        return self._map.get(self.key(FY0, M0, v, th, tr, delay, dt, stage))
    def put(self, FY0, M0, v, th, tr, delay, dt, stage, segs):
        self._map[self.key(FY0, M0, v, th, tr, delay, dt, stage)] = segs

FIT_CACHE = FitnessCache(1e-3)
SEG_CACHE = SegmentCache(2e-3)

def _max_possible_sum(M_list: List[np.ndarray], tdet_list: List[float], T_cap: float) -> float:
    total = 0.0
    for tdet in tdet_list:
        if tdet <= 0.0 or tdet > T_cap: continue
        spans = [max(0.0, min(t_effective, _arrival_time(M0) - tdet, T_cap - tdet)) for M0 in M_list]
        if spans: total += 3.0 * (sum(spans) / len(spans))
    return float(total)

# 基因编码：tr2=tr1+(1+Δ2), tr3=tr2+(1+Δ3)，Δ>0（严格 >1s）
_GAP_EPS = 1e-6

def gene_from_seed_radians(theta_rad: float, v: float,
                           rel: List[float], delay: List[float]) -> np.ndarray:
    tr1, tr2, tr3 = map(float, rel)
    delay1, delay2, delay3 = map(float, delay)
    d1  = max(0.0, tr1)
    d2e = max(_GAP_EPS, (tr2 - tr1) - 1.0)
    d3e = max(_GAP_EPS, (tr3 - tr2) - 1.0)
    return np.array([theta_rad, float(v), d1, delay1, d2e, delay2, d3e, delay3], float)

def decode_gene(x: np.ndarray, T_cap: float, delay_min: float, delay_max: float):
    th = float(x[0])
    v  = float(np.clip(x[1], V_MIN, V_MAX))
    d1 = max(0.0, float(x[2]))
    d2 = max(_GAP_EPS, float(x[4]))
    d3 = max(_GAP_EPS, float(x[6]))
    tr1 = d1
    tr2 = tr1 + 1.0 + d2
    tr3 = tr2 + 1.0 + d3
    delay1 = float(np.clip(x[3], delay_min, delay_max))
    delay2 = float(np.clip(x[5], delay_min, delay_max))
    delay3 = float(np.clip(x[7], delay_min, delay_max))

    # 裁剪起爆到窗口内
    for (tr, delayv_name) in [(tr1, "delay1"), (tr2, "delay2"), (tr3, "delay3")]:
        tdet = tr + locals()[delayv_name]
        if tdet > T_cap:
            locals()[delayv_name] = max(0.0, min(delay_max, T_cap - tr)) if tr < T_cap else 0.0
    
    t1 = min(tr1 + delay1, T_cap); delay1 = max(delay_min, min(delay_max, t1 - tr1))
    t2 = min(tr2 + delay2, T_cap); delay2 = max(delay_min, min(delay_max, t2 - tr2))
    t3 = min(tr3 + delay3, T_cap); delay3 = max(delay_min, min(delay_max, t3 - tr3))
    return th, v, tr1, delay1, tr2, delay2, tr3, delay3


# 单 UAV × 三导弹
def eval_uav_three(
    FY0: np.ndarray, M_list: List[np.ndarray],
    theta: float, v: float,
    tr1: float, delay1: float, tr2: float, delay2: float, tr3: float, delay3: float,
    *, dt: float, stage: int, precheck: bool, use_quick: bool, T_cap: float,
    aspiration: float = -1.0, margin: float = 0.08
) -> Tuple[float, Dict[str, Any]]:
    
    tdet_list = [tr1+delay1, tr2+delay2, tr3+delay3]
    ub = _max_possible_sum(M_list, tdet_list, T_cap)
    if aspiration > 0.0 and ub + 1e-9 <= aspiration - margin:
        return 0.0, dict(per_missile_len=[0.0,0.0,0.0], per_missile_merged={0:[],1:[],2:[]})

    dir_vec = _heading_vec(theta)
    by_missile: Dict[int, List[Tuple[float,float]]] = {0:[],1:[],2:[]}

    for t_drop, delay in [(tr1,delay1),(tr2,delay2),(tr3,delay3)]:
        t_burst = t_drop + delay
        if t_burst <= 0.0 or t_burst > T_cap: continue
        z0 = FY0[2] - 0.5*_g*(delay**2) if delay>0 else FY0[2]
        if z0 < -1e-6: continue

        for mi, M0 in enumerate(M_list):
            if use_quick and not quick_has_any(FY0, v, theta, t_drop, delay, M0, T_cap): 
                continue
            segs = SEG_CACHE.get(FY0, M0, v, theta, t_drop, delay, dt, stage)
            if segs is None:
                try:
                    segs, _ = cover_intervals_cached_general(dt, v, dir_vec, t_drop, t_burst, FY0, M0, stage=stage, precheck=precheck)
                except TypeError:
                    segs, _ = cover_intervals_cached_general(dt, v, dir_vec, t_drop, t_burst, FY0, M0, stage=stage)
                SEG_CACHE.put(FY0, M0, v, theta, t_drop, delay, dt, stage, segs)
            if segs: by_missile[mi].extend(segs)

    per_len, per_merged = [], {}
    for mi in range(3):
        L, merged = union_length(by_missile[mi])
        per_len.append(float(L)); per_merged[mi] = merged
    return float(sum(per_len)), dict(per_missile_len=per_len, per_missile_merged=per_merged)

# 种子
UAV_PARAMS = {
    "FY1": dict(direction_angle=5.01, speed=140.00,
                release_time=[0.00, 1.00, 10.10],
                delay_time=[0.00, 0.00, 1.21]),
    "FY2": dict(direction_angle=5.15, speed=127.02,
                release_time=[7.91, 13.14, 25.33],
                delay_time=[0.61, 2.68, 7.32]),
    "FY3": dict(direction_angle=91.40, speed=110.00,
                release_time=[23.80, 27.20, 28.40],
                delay_time=[4.20, 2.30, 1.80]),
    "FY4": dict(direction_angle=0.00, speed=70.00,
                release_time=[0.92, 1.92, 2.92],
                delay_time=[0.50, 1.34, 0.50]),
    "FY5": dict(direction_angle=2.10, speed=104.78,
                release_time=[16.04, 17.59, 21.57],
                delay_time=[1.65, 4.69, 1.89]),
}

FY0_MAP = {"FY1": p.FY1_0, "FY2": p.FY2_0, "FY3": p.FY3_0, "FY4": p.FY4_0, "FY5": p.FY5_0}

def build_seed_from_params(name: str) -> np.ndarray:
    prm = UAV_PARAMS[name]
    return gene_from_seed_radians(float(prm["direction_angle"]),
                                  float(prm["speed"]),
                                  list(map(float, prm["release_time"])),
                                  list(map(float, prm["delay_time"])))

def add_gaussian_jitter(x: np.ndarray, rng: np.random.Generator,
                        th_sigma_deg=6.0, v_sigma=8.0, t_sigma=1.2, delay_sigma=0.8):
    j = np.array([
        math.radians(th_sigma_deg)*rng.standard_normal(),
        v_sigma*rng.standard_normal(),
        t_sigma*rng.standard_normal(),  delay_sigma*rng.standard_normal(),
        max(0.15, abs(t_sigma*rng.standard_normal())), delay_sigma*rng.standard_normal(),
        max(0.20, abs(t_sigma*rng.standard_normal())), delay_sigma*rng.standard_normal(),
    ], float)
    xj = x + j
    xj[4] = max(_GAP_EPS, xj[4]); xj[6] = max(_GAP_EPS, xj[6])
    return xj


# 单 UAV 子问题
def solve_uav_subproblem(
    name: str, FY0: np.ndarray, M_list: List[np.ndarray], *,
    seed: int, seed_gene: np.ndarray,
    theta_center: float, theta_jitter_deg: float,
    v_center: float | None, v_jitter: float,
    d1_min: float, delay_min: float, delay_max: float,
    d_extra_max: float = 16.0,
    pop_size: int = 44, n_gen: int = 22,
    elite_frac: float = 0.28, reeval_frac: float = 0.18,
    dt_fit: float = 0.18, dt_ref: float = 0.010,
    use_quick: bool = True
) -> Dict[str, Any]:

    T_cap = min(T_END, max(_arrival_time(M0) for M0 in M_list))

    # 盒约束
    th_lb, th_ub = theta_center - math.radians(theta_jitter_deg), theta_center + math.radians(theta_jitter_deg)
    if v_center is None:
        v_lb, v_ub = V_MIN, V_MAX
    else:
        v_lb = max(V_MIN, v_center - v_jitter)
        v_ub = min(V_MAX, v_center + v_jitter)

    LB = np.array([th_lb, v_lb, max(0.0, d1_min), delay_min, _GAP_EPS, delay_min, _GAP_EPS, delay_min], float)
    UB = np.array([th_ub, v_ub, d1_min+25.0, delay_max, d_extra_max, delay_max, d_extra_max, delay_max], float)

    ga = GeneticAlgorithm(lb=LB, ub=UB, pop_size=pop_size, n_gen=n_gen, seed=seed,
                          elite_frac=elite_frac, tournament_k=4, sbx_eta=12.0, sbx_prob=0.90,
                          mut_rate=0.22, mut_scale=0.12, objective="max")

    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    seeds = [np.minimum(UB, np.maximum(LB, seed_gene.copy()))]
    for _ in range(5):
        seeds.append(np.minimum(UB, np.maximum(LB, add_gaussian_jitter(seed_gene, rng))))

    for de in [0.3, 1.0, 2.0, 4.0, 6.0, 10.0]:
        base = seeds[0].copy(); base[4] = de; base[6] = de
        seeds.append(np.minimum(UB, np.maximum(LB, base)))
        seeds.append(np.minimum(UB, np.maximum(LB, add_gaussian_jitter(base, rng, th_sigma_deg=8.0, v_sigma=10.0, t_sigma=1.6, delay_sigma=0.9))))

    ASP = {"best": 0.0}
    def core(dt: float, stage: int, precheck: bool, use_quick_: bool, x: np.ndarray) -> float:
        th, v, tr1, delay1, tr2, delay2, tr3, delay3 = decode_gene(x, T_cap, delay_min, delay_max)
        key = [th, v, tr1, delay1, tr2, delay2, tr3, delay3]
        cached = FIT_CACHE.get(key, dt, stage)
        if cached is not None:
            if cached > ASP["best"]: ASP["best"] = cached
            return cached
        val, _ = eval_uav_three(FY0, M_list, th, v, tr1, delay1, tr2, delay2, tr3, delay3,
                                dt=dt, stage=stage, precheck=precheck, use_quick=use_quick_, T_cap=T_cap,
                                aspiration=ASP["best"], margin=0.08)
        FIT_CACHE.put(key, dt, stage, val)
        if val > ASP["best"]: ASP["best"] = val
        return val

    fitness_fn = lambda x: core(dt_fit, stage=0, precheck=True,  use_quick_=use_quick, x=x)
    reeval_fn  = lambda x: core(dt_ref, stage=1, precheck=False, use_quick_=False,    x=x)

    def sigma_fn(g: int, G: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        phase = g/max(1,G)
        anneal = max(0.45, 1.0 - phase)
        th_scale = (ub[0]-lb[0]) * 0.26
        v_scale  = 0.0 if (v_center is not None) else (ub[1]-lb[1]) * 0.28
        return np.array([th_scale*anneal, v_scale*anneal,
                         1.5*anneal, 0.70*anneal, 2.0*anneal, 0.70*anneal, 2.6*anneal, 0.70*anneal], float)

    def log_fn(g: int, best: float, x: np.ndarray):
        if g % 2 == 0:
            print(f"  [{name}] Gen {g:02d} | best≈{best:.3f}s  (fit_cache={len(FIT_CACHE._map)}, seg_cache={len(SEG_CACHE._map)})")

    x_best, _ = ga.evolve(fitness_fn=fitness_fn, reeval_fn=reeval_fn, reeval_frac=reeval_frac,
                          verbose_every=2, log_fn=log_fn,
                          seed_population=seeds, sigma_fn=sigma_fn)

    th, v, tr1, delay1, tr2, delay2, tr3, delay3 = decode_gene(x_best, T_cap, delay_min, delay_max)
    best_total, info = eval_uav_three(FY0, M_list, th, v, tr1, delay1, tr2, delay2, tr3, delay3,
                                      dt=dt_ref, stage=1, precheck=False, use_quick=False, T_cap=T_cap)
    return dict(
        UAV=name,
        theta_deg=float((math.degrees(th) % 360.0)),
        speed_mps=float(v),
        bombs=[dict(t_drop=tr1, delay=delay1), dict(t_drop=tr2, delay=delay2), dict(t_drop=tr3, delay=delay3)],
        per_missile_len=info["per_missile_len"],
        best_total=float(best_total),
    )

# 主流程：五架机全部求解 + 写 Excel
def run_problem5(out_path: str = "result3.xlsx",
                 pop_size: int = 44, n_gen: int = 22,
                 dt_fit: float = 0.18, dt_ref: float = 0.01):
    M_list = [p.M1_0, p.M2_0, p.M3_0]
    rng = np.random.default_rng(20250907)

    UAVS = []
    for idx, name in enumerate(["FY1","FY2","FY3","FY4","FY5"], 1):
        prm = UAV_PARAMS[name]
        seed_gene = build_seed_from_params(name)
        UAVS.append(dict(
            name=name, FY0=FY0_MAP[name], seed=(20250907 ^ (idx<<12)),
            seed_gene=seed_gene,
            theta_center=float(prm["direction_angle"]),         # 弧度中心
            theta_jitter_deg=14.0 if name!="FY1" else 12.0,     # FY1 略窄
            v_center=float(prm["speed"]), v_jitter=12.0,
            d1_min=0.0 if name in ["FY1","FY4"] else 3.0,
            delay_min=0.0, delay_max=8.0 if name in ["FY4","FY5"] else 6.5,
        ))

    details_rows, summary_rows, total_sum = [], [], 0.0
    t0 = time.time()

    for cfg in UAVS:
        print(f"\n=== 子问题：{cfg['name']}（严格判据·加速） ===")
        res = solve_uav_subproblem(M_list=M_list, dt_fit=dt_fit, dt_ref=dt_ref,
                                   pop_size=pop_size, n_gen=n_gen, **cfg)
        total_sum += res["best_total"]
        summary_rows.append(dict(UAV=cfg["name"], total_cover_sum_s=res["best_total"]))

        theta = res["theta_deg"]; v = res["speed_mps"]
        L1, L2, L3 = res["per_missile_len"]
        b1,b2,b3 = res["bombs"]
        print(f" -> {cfg['name']}: best sum(M1,M2,M3) = {res['best_total']:.3f}s | per-missile = [{L1:.3f}, {L2:.3f}, {L3:.3f}]")
        print("    params: θ=%.3f°  v=%.3f m/s" % (theta, v))
        print("    bombs : (t_drop=%.3f, delay=%.3f) | (t_drop=%.3f, delay=%.3f) | (t_drop=%.3f, delay=%.3f)"
              % (b1['t_drop'], b1['delay'], b2['t_drop'], b2['delay'], b3['t_drop'], b3['delay']))

        for idx_b, b in enumerate(res["bombs"], 1):
            details_rows.append({
                "无人机": cfg["name"],
                "航向角(度)": theta,
                "速度(m/s)": v,
                "弹号": idx_b,
                "投放时刻t_drop(s)": float(b["t_drop"]),
                "起爆延迟delay(s)":  float(b["delay"]),
                "对M1并集时长(s)": float(L1),
                "对M2并集时长(s)": float(L2),
                "对M3并集时长(s)": float(L3),
                "三导弹合计(s)":   float(L1 + L2 + L3),
            })

    t1 = time.time()
    print("\n=== 问题5 汇总 ===")
    for r in summary_rows:
        print(f"  {r['UAV']}: {r['total_cover_sum_s']:.3f}s")
    print(f"  合计：{total_sum:.3f}s   耗时：{(t1 - t0):.2f}s")

    df_detail  = pd.DataFrame(details_rows)
    df_summary = pd.DataFrame(summary_rows + [dict(UAV="TOTAL", total_cover_sum_s=total_sum)])
    with pd.ExcelWriter(out_path) as writer:
        df_detail.to_excel(writer, sheet_name="Details", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    return dict(total=total_sum, detail_path=os.path.abspath(out_path), elapsed=t1-t0)


if __name__ == "__main__":

    run_problem5(out_path="result3.xlsx", pop_size=44, n_gen=22, dt_fit=0.18, dt_ref=0.01)
