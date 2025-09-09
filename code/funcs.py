import numpy as np


def wrap_deg_0_360(deg: float) -> float:
    """
    将角度归一化到 [0, 360)
    """
    x = deg % 360.0
    return x + 360.0 if x < 0 else x


def unit(vec: np.ndarray) -> np.ndarray:
    """
    返回单位向量
    """
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def seg_point_min_dist(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算点 c 到线段 ab 的最短距离（向量化）
    """
    ab = b - a[None, :]
    d2 = np.einsum("ij,ij->i", ab, ab)
    d2 = np.where(d2 == 0.0, 1e-12, d2)
    s = np.einsum("j,ij->i", c - a, ab) / d2
    s = np.clip(s, 0.0, 1.0)
    closest = a[None, :] + ab * s[:, None] # 投影点
    return np.linalg.norm(c[None, :] - closest, axis=1)


def union_length(intervals):
    """
    计算多个区间的并集长度
    """
    valid = [(s, e) for (s, e) in intervals if (s is not None and e is not None and e > s)]
    if not valid:
        return 0.0, []
    valid.sort()
    merged = []
    cs, ce = valid[0]
    for s, e in valid[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return sum(e - s for s, e in merged), merged


def set_axes_equal_3d(ax):
    """
    让三轴缩放一致，球体/圆柱不被拉伸
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])