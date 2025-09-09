import numpy as np, math, time, random
import matplotlib.pyplot as plt
import pandas as pd
from funcs import set_axes_equal_3d


# 场景参数
FY1_0 = np.array([17800.0, 0.0, 1800.0])  
g = 9.8
R_cyl, H_cyl = 7.0, 10.0
Y_axis = 200.0
R_effective = 10.0
SINK = 3.0
CLOUD_LIFETIME = 20.0
vM = 300.0
M0 = np.array([20000.0, 0.0, 2000.0])
O  = np.array([0.0, 0.0, 0.0])
uM = (O - M0) / np.linalg.norm(O - M0)
t_arrival = float(np.linalg.norm(O - M0) / vM)

def M(t): 
    return M0 + uM * (vM * t)


# 圆柱闭合表面采样（侧面+上下底）
def sample_cylinder(n_theta, n_z, cap_radial_steps):
    
    pts=[]
    thetas=np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    zs=np.linspace(0.0, H_cyl, n_z)

    # 侧面
    for z in zs:
        for th in thetas:
            pts.append([R_cyl*math.cos(th), Y_axis + R_cyl*math.sin(th), z])

    # 上下底（圆盘内部）
    for z in [0.0, H_cyl]:
        for r in np.linspace(0, R_cyl, cap_radial_steps+1):
            for th in thetas:
                pts.append([r*math.cos(th), Y_axis + r*math.sin(th), z])
    return np.asarray(pts, float)


# 评估档位：GA 粗评 / 最终高精
P_fit = sample_cylinder(36, 9, 5)     # 更稀，配合大 dt_fit，加速
P_ref = sample_cylinder(160, 25, 8)   # 高精用于精英复评 & 最终


def seg_point_min_dist(c, m, P):
    pm=P-m[None,:]
    d2=np.einsum("ij,ij->i", pm, pm); d2=np.where(d2==0.0,1e-12,d2)
    s =np.einsum("j,ij->i", c-m, pm)/d2; s=np.clip(s,0.0,1.0)
    closest = m[None,:] + pm*s[:,None]
    return np.linalg.norm(c[None,:]-closest,axis=1)

def fully_occluded_strict(t, P_burst, t_burst, P, eps=1e-6):
    if not (t_burst <= t <= t_burst + CLOUD_LIFETIME): 
        return False
    m = M(t)
    c = P_burst + np.array([0.0,0.0,-SINK*(t - t_burst)])
    return np.max(seg_point_min_dist(c, m, P) - R_effective) <= eps

def intervals_strict(P_burst, t_burst, P, dt=0.02, tmax=12.0, eps=1e-6):
    t0, t1 = t_burst, min(t_burst + CLOUD_LIFETIME, t_burst + tmax, t_arrival)
    if t1 <= t0: 
        return [], 0.0
    times = np.arange(t0, t1 + 1e-12, max(dt, 1e-6))
    flags=[ fully_occluded_strict(t, P_burst, t_burst, P, eps=eps) for t in times ]
    intervals=[]; in_seg=False; start=None
    for i, ok in enumerate(flags):
        if ok and not in_seg: 
            in_seg=True; start=times[i]
        elif not ok and in_seg: 
            intervals.append((start, times[i-1])); in_seg=False
    if in_seg: 
        intervals.append((start, times[-1]))
    return intervals, sum(b-a for a,b in intervals)


USE_QUICK = True
T_b = np.array([0.0, 200.0, H_cyl/2.0])
r_b = math.sqrt(R_cyl**2 + (H_cyl/2.0)**2)
def quick_sufficient_cover(Mt, Ct):
    v1, v2 = T_b - Mt, Ct - Mt
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if d1 <= r_b: return False
    if d2 <= R_effective: return True
    alpha = math.asin(min(1.0, r_b/d1))
    beta  = math.asin(min(1.0, R_effective/d2))
    cosd  = float(np.dot(v1, v2)/(d1*d2)); cosd = max(-1.0, min(1.0, cosd))
    delta = math.acos(cosd)
    return beta >= alpha + delta

def quick_has_any(P_burst, t_burst, dt=0.12):
    t0, t1 = t_burst, min(t_burst + CLOUD_LIFETIME, t_arrival)
    for t in np.arange(t0, t1 + 1e-12, max(dt,1e-6)):
        Ct = P_burst + np.array([0,0,-SINK*(t - t_burst)])
        if quick_sufficient_cover(M(t), Ct): 
            return True
    return False


def uav_pos(FY0, t, v, heading_deg):
    rad=math.radians(heading_deg)
    return FY0 + np.array([math.cos(rad), math.sin(rad), 0.0])*v*t

def detonation_point(FY0, t_burst, v, heading_deg, t_drop):
    xy = uav_pos(FY0, t_burst, v, heading_deg)[:2]
    if t_drop > t_burst:
        z = FY0[2]
    else:
        z = FY0[2] - 0.5*g*(t_burst - t_drop)**2
    return np.array([xy[0], xy[1], z])


def drop_point(FY0, t_drop, v, heading_deg):
    xy = uav_pos(FY0, t_drop, v, heading_deg)[:2]
    return np.array([xy[0], xy[1], FY0[2]], float)


def union_length(intervals):
    if not intervals: return 0.0, []
    intervals = sorted(intervals)
    merged=[]; cs,ce=intervals[0]
    for s,e in intervals[1:]:
        if s<=ce: ce=max(ce,e)
        else: merged.append((cs,ce)); cs,ce=s,e
    merged.append((cs,ce))
    return sum(e-s for s,e in merged), merged


# 染色体: [theta_deg, v, d1, tau1, d2, tau2, d3, tau3]
def wrap_deg(th): 
    return th % 360.0

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def decode_intervals(ch):
    theta, v, d1, tau1, d2, tau2, d3, tau3 = ch
    tr1 = max(0.0, d1)
    tr2 = tr1 + 1.0 + max(0.0, d2)
    tr3 = tr2 + 1.0 + max(0.0, d3)
    tau1 = clamp(tau1, 0.0, 6.0)
    tau2 = clamp(tau2, 0.0, 6.0)
    tau3 = clamp(tau3, 0.0, 6.0)
    return theta, v, tr1, tau1, tr2, tau2, tr3, tau3

def repair_gene(ch):
    theta, v, tr1, tau1, tr2, tau2, tr3, tau3 = decode_intervals(ch)
    theta = wrap_deg(theta)
    v = clamp(v, 70.0, 140.0)
    tdet1 = min(tr1 + tau1, t_arrival)
    tdet2 = min(tr2 + tau2, t_arrival)
    tdet3 = min(tr3 + tau3, t_arrival)
    tau1 = max(0.0, tdet1 - tr1)
    tau2 = max(0.0, tdet2 - tr2)
    tau3 = max(0.0, tdet3 - tr3)
    return [theta, v, tr1, tau1, tr2, tau2, tr3, tau3]


def eval_three(theta, v, tr1, tau1, tr2, tau2, tr3, tau3, P, dt=0.03, use_quick=True):
    """
    返回：三团“所有片段”的并集覆盖时间（严格判据）
    """
    inters_all=[]; per=[]
    for tr, tau in [(tr1,tau1),(tr2,tau2),(tr3,tau3)]:
        tdet = tr + tau
        P_burst = detonation_point(FY1_0, tdet, v, theta, tr)
        if P_burst[2] < 0.0 or tdet > t_arrival:
            inters, L = [], 0.0
        else:
            if use_quick and (not quick_has_any(P_burst, tdet, dt=0.12)):
                inters, L = [], 0.0
            else:
                inters, L = intervals_strict(P_burst, tdet, P, dt=dt, tmax=12.0)
        per.append((inters, L, tdet, P_burst))
        inters_all += inters
    L_union, merged = union_length(inters_all)
    return L_union, merged, per

def fitness(ch, P, dt_fit=0.04, use_quick=True):
    theta, v, tr1, tau1, tr2, tau2, tr3, tau3 = repair_gene(ch)
    L_union, merged, per = eval_three(theta, v, tr1, tau1, tr2, tau2, tr3, tau3, P, dt=dt_fit, use_quick=use_quick)
    return L_union, merged, per, [theta, v, tr1, tau1, tr2, tau2, tr3, tau3]

def tournament_select(pop, fits, k=3):
    n=len(pop); idxs=np.random.randint(0, n, size=k)
    best_i = max(idxs, key=lambda i: fits[i])
    return pop[best_i].copy()

def sbx_crossover(p1, p2, eta=10.0, prob=0.9):
    if random.random() > prob: return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for j in range(len(p1)):
        u = random.random()
        if u <= 0.5: beta = (2*u)**(1.0/(eta+1.0))
        else:        beta = (1/(2*(1-u)))**(1.0/(eta+1.0))
        c1[j] = 0.5*((1+beta)*p1[j] + (1-beta)*p2[j])
        c2[j] = 0.5*((1-beta)*p1[j] + (1+beta)*p2[j])
    return c1, c2

def gaussian_mutation(c, sigmas, prob=0.25):
    for j in range(len(c)):
        if random.random() < prob:
            c[j] += random.gauss(0.0, sigmas[j])
    return c


def plot_problem3_3d(theta_deg, v, bombs, *, dt=0.05, save_path):
    """
    只绘制：Missile1(0→t_max)、UAV(0→t_max)、三颗干扰弹(t_drop→t_burst) 的 3D 轨迹；
    三个起爆点分别绘制橙色半透明云团（半径 = R_effective）。
      - 导弹：绿色
      - 无人机：蓝色
      - 三颗干扰弹：橙色
      - 云团：橙色 + alpha=0.25
    bombs: [(t_drop1, tau1), (t_drop2, tau2), (t_drop3, tau3)]
    """

    # 统一时间轴：到三枚弹的最晚起爆时刻
    t_bursts = [tr+tau for (tr, tau) in bombs]
    t_max = max(1e-6, max(t_bursts))
    t_uav_mis = np.arange(0.0, t_max + 1e-12, max(1e-3, dt))

    # UAV 航向向量（由航向角度）
    rad = math.radians(float(theta_deg))
    dir_xy = np.array([math.cos(rad), math.sin(rad), 0.0], float)

    # UAV：等速直线
    uav = FY1_0[None, :3] + (v * t_uav_mis[:, None]) * dir_xy[None, :]
    uav[:, 2] = FY1_0[2]

    # Missile1：直线（指向原点）
    mis = np.vstack([M(t) for t in t_uav_mis])

    # Bombs：每枚弹从 t_drop 到 t_burst 的抛物线（XY 随 UAV 匀速，Z 做自由落体）
    bomb_trajs = []
    burst_points = []
    for (tr, tau) in bombs:
        t_b = float(tr + tau)
        if t_b <= 0.0:  # 不可行的直接跳过
            bomb_trajs.append(None)
            burst_points.append(None)
            continue
        t_bomb = np.arange(tr, t_b + 1e-12, max(1e-3, dt))
        xy = FY1_0[:2][None, :] + (v * t_bomb[:, None]) * dir_xy[None, :2]
        z  = FY1_0[2] - 0.5 * g * (t_bomb - tr)**2
        bomb_trajs.append(np.column_stack([xy[:,0], xy[:,1], z]))
        burst_points.append(np.array([xy[-1,0], xy[-1,1], z[-1]], float))

    # 绘图
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 轨迹（颜色规范）
    ax.plot(mis[:,0], mis[:,1], mis[:,2], color="green",  linewidth=2.0, label="Missile path(0->t_burst)")
    ax.plot(uav[:,0], uav[:,1], uav[:,2], color="blue",   linewidth=2.0, label="UAV path(0->t_burst)")

    # 修改 bomb 轨迹的标签设置
    for i, traj in enumerate(bomb_trajs, 1):
        if traj is None: continue
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color="orange", linewidth=2.0, label="Bomb path" if i==1 else None)  # 所有 bomb 使用同一个标签

    # 修改起爆点的标签设置
    for i, P_burst in enumerate(burst_points, 1):
        if P_burst is None: continue
        ax.scatter([P_burst[0]], [P_burst[1]], [P_burst[2]], s=60, marker='o', color="orange", label="Burst point" if i==1 else None)  # 所有起爆点使用同一个标签

    # 轴与视窗
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Problem 3 — Scene Overview")
    ax.legend(loc="best")

    # 自适应范围
    xs = [mis[:,0], uav[:,0]] + [t[:,0] for t in bomb_trajs if t is not None] + [np.array([p[0]]) for p in burst_points if p is not None]
    ys = [mis[:,1], uav[:,1]] + [t[:,1] for t in bomb_trajs if t is not None] + [np.array([p[1]]) for p in burst_points if p is not None]
    zs = [mis[:,2], uav[:,2]] + [t[:,2] for t in bomb_trajs if t is not None] + [np.array([p[2]]) for p in burst_points if p is not None]
    xs_all = np.concatenate(xs); ys_all = np.concatenate(ys); zs_all = np.concatenate(zs)
    ax.set_xlim(xs_all.min()-50, xs_all.max()+50)
    ax.set_ylim(ys_all.min()-50, ys_all.max()+50)
    ax.set_zlim(max(0.0, zs_all.min()-50), zs_all.max()+50)
    set_axes_equal_3d(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_problem3(seed=314159, pop_size=90, n_gen=40,
                         elite_frac=0.12, elite_eval_frac=0.20,
                         dt_fit=0.04, dt_ref=0.01, use_quick=USE_QUICK):
    
    np.random.seed(seed); random.seed(seed)

    # 初始种群（含启发式解）
    seeds = []
    seeds.append([5.0, 70.0, 0.20, 1.20, 1.00, 0.00, 1.00, 0.20])
    seeds.append([3.8, 80.0, 0.60, 1.00, 1.00, 0.00, 1.20, 0.20])
    seeds.append([300.0, 85.0, 0.50, 0.40, 1.50, 0.30, 1.50, 0.60])
    seeds.append([25.0,  90.0, 1.20, 0.20, 0.80, 0.40, 1.20, 0.30])

    pop=[s.copy() for s in seeds]
    while len(pop) < pop_size:
        theta = np.random.uniform(0.0, 360.0)
        v = np.random.uniform(70.0, 140.0)
        d1 = np.random.uniform(0.0, 3.0)
        d2 = np.random.uniform(0.0, 2.0)
        d3 = np.random.uniform(0.0, 2.0)
        tau1 = np.random.uniform(0.0, 1.6)
        tau2 = np.random.uniform(0.0, 1.6)
        tau3 = np.random.uniform(0.0, 1.6)
        pop.append([theta, v, d1, tau1, d2, tau2, d3, tau3])

    best=None; best_pack=None
    elite_k = max(2, int(pop_size*elite_frac))

    hist_gen, hist_best, hist_mean = [], [], []

    for gen in range(1, n_gen+1):
        anneal = max(0.3, 1.0 - gen/n_gen)
        sigmas = [ 360.0*0.05, (140-70)*0.05, 0.5*anneal, 0.3*anneal,
                   0.6*anneal, 0.3*anneal, 0.8*anneal, 0.3*anneal ]

        fits=[]; infos=[]
        for ind in pop:
            L, merged, per, repaired = fitness(ind, P_fit, dt_fit=dt_fit, use_quick=use_quick)
            fits.append(L); infos.append((merged, per, repaired))

        order_tmp = np.argsort(fits)[::-1]
        top_eval = order_tmp[:max(2, int(pop_size*elite_eval_frac))]
        for i in top_eval:
            rep_ch = infos[i][2]
            theta, v, tr1, tau1, tr2, tau2, tr3, tau3 = rep_ch
            L2, merged2, per2 = eval_three(theta, v, tr1, tau1, tr2, tau2, tr3, tau3,
                                           P_ref, dt=0.015, use_quick=False)
            if L2 > fits[i] + 1e-6:
                fits[i]   = L2
                infos[i]  = (merged2, per2, rep_ch)

        g_best = int(np.argmax(fits))
        if (best is None) or (fits[g_best] > best + 1e-9):
            best = fits[g_best]; best_pack = (pop[g_best].copy(), infos[g_best])

        hist_gen.append(gen)
        hist_best.append(float(fits[g_best]))
        hist_mean.append(float(np.mean(fits)))

        print(f"Gen {gen:02d}: best≈{fits[g_best]:.3f} s, avg={np.mean(fits):.3f} s")
 
        order = np.argsort(fits)[::-1]
        new_pop = [ pop[i].copy() for i in order[:elite_k] ]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits, k=3)
            p2 = tournament_select(pop, fits, k=3)
            c1, c2 = sbx_crossover(p1, p2, eta=12.0, prob=0.9)
            c1 = gaussian_mutation(c1, sigmas, prob=0.25)
            c2 = gaussian_mutation(c2, sigmas, prob=0.25)
            new_pop.append(c1)
            if len(new_pop) < pop_size: new_pop.append(c2)
        pop = new_pop

    chrom_best, (merged_fit, per_fit, repaired) = best_pack
    theta, v, tr1, tau1, tr2, tau2, tr3, tau3 = repaired

    details=[]; inters_all=[]
    bombs = [(tr1,tau1),(tr2,tau2),(tr3,tau3)]
    for (tr,tau) in bombs:
        td = tr + tau
        P_burst = detonation_point(FY1_0, td, v, theta, tr)
        inters, L = intervals_strict(P_burst, td, P_ref, dt=dt_ref, tmax=12.0)
        details.append((tr, td, tau, P_burst, L, inters))
        inters_all += inters
    L_union_ref, merged_ref = union_length(inters_all)

    print("\n=== 遗传算法 · 问题3 · 最优解（高精评估） ===")
    print(f"三团并集（严格）= {L_union_ref:.3f} s")
    print("并集区间 =", merged_ref)
    print(f"航向={theta:.2f}°，速度={v:.1f} m/s")
    for k,(tr,td,tau,P_burst,L,inters) in enumerate(details,1):
        print(f"  弹#{k}: t_drop={tr:.2f} s, t_burst={td:.2f} s, τ={tau:.2f} s | 单窗={L:.2f} s")
        print(f"        P_burst=({P_burst[0]:.2f}, {P_burst[1]:.2f}, {P_burst[2]:.2f}) | 区间={inters}")


    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(hist_gen, hist_best, 'r-', linewidth=1.2, label="Best")
    ax.plot(hist_gen, hist_mean, 'b-', linewidth=1.2, label="Population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Union cover length (s)")
    ax.set_title("GA progress (Problem 3)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/problem3_ga_progress.png", dpi=180, bbox_inches="tight")
    plt.show()

    # Excel 导出
    cols = [
            "无人机运动方向",
            "无人机运动速度 (m/s)",
            "烟幕干扰弹编号",
            "烟幕干扰弹投放点的x坐标 (m)",
            "烟幕干扰弹投放点的y坐标 (m)",
            "烟幕干扰弹投放点的z坐标 (m)",
            "烟幕干扰弹起爆点的x坐标 (m)",
            "烟幕干扰弹起爆点的y坐标 (m)",
            "烟幕干扰弹起爆点的z坐标 (m)",
            "有效干扰时长 (s)",
        ]
    rows = []
    for idx, (tr, td, tau, P_burst, L, _) in enumerate(details, 1):
        P_drop = drop_point(FY1_0, tr, v, theta)
        rows.append({
                "无人机运动方向": float(theta % 360.0),
                "无人机运动速度 (m/s)": float(v),
                "烟幕干扰弹编号": int(idx),
                "烟幕干扰弹投放点的x坐标 (m)": float(P_drop[0]),
                "烟幕干扰弹投放点的y坐标 (m)": float(P_drop[1]),
                "烟幕干扰弹投放点的z坐标 (m)": float(P_drop[2]),
                "烟幕干扰弹起爆点的x坐标 (m)": float(P_burst[0]),
                "烟幕干扰弹起爆点的y坐标 (m)": float(P_burst[1]),
                "烟幕干扰弹起爆点的z坐标 (m)": float(P_burst[2]),
                "有效干扰时长 (s)": float(L),
            })
        
    # 追加空行与注释行
    blank = {c: np.nan for c in cols}
    note  = {c: np.nan for c in cols}
    note["无人机运动方向"] = "注：以x轴为正向，逆时针方向为正，取值0~360（度）。"
    df_out = pd.DataFrame(rows, columns=cols)
    df_out = pd.concat([df_out, pd.DataFrame([blank, note])], ignore_index=True)

    with pd.ExcelWriter("../result/result1.xlsx") as w:
        df_out.to_excel(w, index=False, sheet_name="Sheet1")

    # 3D 可视化
    plot_problem3_3d(
        theta_deg=theta,
        v=v,
        bombs=[(tr1, tau1), (tr2, tau2), (tr3, tau3)],
        dt=0.05,
        save_path="../figures/problem3_3d.png"
    )


if __name__ == "__main__":

    res = run_problem3(
        seed=314159, pop_size=90, n_gen=40,
        elite_frac=0.12, elite_eval_frac=0.20,
        dt_fit=0.04, dt_ref=0.01, use_quick=True
    )
