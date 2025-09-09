from __future__ import annotations
import io, re, importlib, math
from contextlib import redirect_stdout
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import params as p 
from funcs import wrap_deg_0_360

# 需要汇总的模块与主函数、以及各自 FY 初始点在 params 中的变量名
ITEMS = [
    ("FY1", "problem2",      "run_problem2",      "FY1_0"),
    ("FY2", "problem4_fy2",  "run_problem4_FY2",  "FY2_0"),
    ("FY3", "problem4_fy3",  "run_problem4_FY3",  "FY3_0"),
]


def safe_import(modname: str):
    """
    安全导入模块；失败时返回异常对象。
    """
    try:
        return importlib.import_module(modname)
    except Exception as e:
        return e


def call_entry(mod, func_name: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    调用模块的入口函数：
      - 优先调用无参；若报 TypeError，再尝试传入默认 CFG；
      - 捕获 stdout 便于后续解析；
    返回 (ret, raw_log, error)
      - ret: 函数返回（可能是 dict，也可能是 None）
      - raw_log: 捕获的标准输出文本
      - error: 若出错，返回错误信息字符串
    """
    raw_log = ""
    try:
        f = getattr(mod, func_name)
    except AttributeError:
        return None, raw_log, f"Function {func_name} not found in {getattr(mod,'__name__',str(mod))}"

    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            # 首先尝试无参数调用
            ret = f()
        raw_log = buf.getvalue()
        return ret, raw_log, None
    except TypeError as e:
        # 如果函数需要参数，尝试获取参数并调用
        try:
            # 检查函数签名，看看需要什么参数
            import inspect
            sig = inspect.signature(f)
            params = sig.parameters
            
            # 准备参数
            kwargs = {}
            for name, param in params.items():
                if name == 'cfg':
                    # 尝试从模块获取 CFG 配置
                    cfg = getattr(mod, "CFG", None)
                    if cfg is not None:
                        kwargs[name] = cfg
                # 可以添加其他参数的处理逻辑
            
            buf = io.StringIO()
            with redirect_stdout(buf):
                ret = f(**kwargs)
            raw_log = buf.getvalue()
            return ret, raw_log, None
        except Exception as inner_e:
            return None, raw_log, f"Execution error with parameters: {repr(inner_e)}"
    except Exception as e:
        return None, raw_log, f"Execution error: {repr(e)}"


def parse_stdout_metrics(raw_log: str) -> Dict[str, float]:
    """
    从打印文本中解析关键数值指标，返回字典。
    Total cover length, Speed v, Heading θ, Drop/Delay/Burst
    """
    m: Dict[str, float] = {}
    # Total cover length: 1.234 s
    r = re.search(r"Total cover length:\s*([0-9.+-eE]+)\s*s", raw_log)
    if r:
        m["total_cover_s"] = float(r.group(1))
    # Speed v = 80.000 m/s, Heading θ = 12.345°
    r = re.search(r"Speed v\s*=\s*([0-9.+-eE]+)\s*m/s,\s*Heading\s*.*?=\s*([0-9.+-eE]+)°", raw_log)
    if r:
        m["speed_mps"] = float(r.group(1))
        m["heading_deg"] = float(r.group(2))
    # Drop time: ... / 或 Drop: ...
    r = re.search(r"(?:Drop time|Drop):\s*([0-9.+-eE]+)\s*s,\s*Delay:\s*([0-9.+-eE]+)\s*s,\s*Burst:\s*([0-9.+-eE]+)\s*s", raw_log)
    if r:
        m["t_drop_s"]  = float(r.group(1))
        m["delay_s"]   = float(r.group(2))
        m["t_burst_s"] = float(r.group(3))
    return m


def xy_at(FY0: np.ndarray, v: float, theta_deg: float, t: float) -> Tuple[float, float]:
    """
    给定 FY 初始点、速度、航向角（度）、时刻 t，计算 XY 坐标。
    """
    th = math.radians(theta_deg)
    x = float(FY0[0] + math.cos(th) * v * t)
    y = float(FY0[1] + math.sin(th) * v * t)
    return x, y

def drop_point(FY0: np.ndarray, v: float, theta_deg: float, t_drop: float) -> Tuple[float, float, float]:
    x, y = xy_at(FY0, v, theta_deg, t_drop)
    z = float(FY0[2])
    return x, y, z

def burst_point(FY0: np.ndarray, v: float, theta_deg: float, t_drop: float, t_burst: float) -> Tuple[float, float, float]:
    x, y = xy_at(FY0, v, theta_deg, t_burst)
    dt = max(0.0, float(t_burst - t_drop))
    z = float(FY0[2] - 0.5 * p.g * dt * dt)
    return x, y, z

def collect_one(label: str, modname: str, func_name: str, fy_attr: str) -> Dict[str, Any]:
    """
    对单个 UAV 运行模块、采集并计算输出行。
    """
    row: Dict[str, Any] = {
        "无人机编号": label,
        "无人机运动方向": None,
        "无人机运动速度 (m/s)": None,
        "烟幕干扰弹投放点的x坐标 (m)": None,
        "烟幕干扰弹投放点的y坐标 (m)": None,
        "烟幕干扰弹投放点的z坐标 (m)": None,
        "烟幕干扰弹起爆点的x坐标 (m)": None,
        "烟幕干扰弹起爆点的y坐标 (m)": None,
        "烟幕干扰弹起爆点的z坐标 (m)": None,
        "有效干扰时长 (s)": None,
        "error": None,
    }

    mod = safe_import(modname)
    if isinstance(mod, Exception):
        row["error"] = f"Import error: {repr(mod)}"
        return row

    ret, log, err = call_entry(mod, func_name)
    if err:
        row["error"] = err
        return row

    # 1) 优先读取返回的 dict
    speed = heading = t_drop = delay = t_burst = total = None
    if isinstance(ret, dict):
        total = ret.get("total_cover_s", total)
        plan = ret.get("plan", {})
        if isinstance(plan, dict):
            speed   = plan.get("speed_mps", speed)
            heading = plan.get("heading_deg", heading)
            t_drop  = plan.get("t_drop_s", t_drop)
            delay   = plan.get("delay_s", delay)
            t_burst = plan.get("t_burst_s", t_burst)

    # 2) stdout 兜底解析
    parsed = parse_stdout_metrics(log or "")
    total   = float(parsed.get("total_cover_s", total)) if parsed.get("total_cover_s") is not None else total
    speed   = float(parsed.get("speed_mps", speed))      if parsed.get("speed_mps")    is not None else speed
    heading = float(parsed.get("heading_deg", heading))  if parsed.get("heading_deg")  is not None else heading
    t_drop  = float(parsed.get("t_drop_s", t_drop))      if parsed.get("t_drop_s")     is not None else t_drop
    delay   = float(parsed.get("delay_s", delay))        if parsed.get("delay_s")      is not None else delay
    t_burst = float(parsed.get("t_burst_s", t_burst))    if parsed.get("t_burst_s")    is not None else t_burst

    # 3) 计算坐标并填充行
    row["有效干扰时长 (s)"]       = float(total)   if total   is not None else None
    row["无人机运动速度 (m/s)"]    = float(speed)   if speed   is not None else None
    if heading is not None:
        heading = wrap_deg_0_360(float(heading))
        row["无人机运动方向"] = heading

    if all(v is not None for v in [speed, heading, t_drop, t_burst]):
        FY0 = getattr(p, fy_attr)  # FY 初始点
        x1, y1, z1 = drop_point(FY0, float(speed), heading, float(t_drop))
        x2, y2, z2 = burst_point(FY0, float(speed), heading, float(t_drop), float(t_burst))
        row["烟幕干扰弹投放点的x坐标 (m)"] = x1
        row["烟幕干扰弹投放点的y坐标 (m)"] = y1
        row["烟幕干扰弹投放点的z坐标 (m)"] = z1
        row["烟幕干扰弹起爆点的x坐标 (m)"] = x2
        row["烟幕干扰弹起爆点的y坐标 (m)"] = y2
        row["烟幕干扰弹起爆点的z坐标 (m)"] = z2

    return row


if __name__ == "__main__":
    """
    执行汇总并输出到 Excel，返回输出路径。
    """
    rows = []
    for label, modname, func_name, fy_attr in ITEMS:
        rows.append(collect_one(label, modname, func_name, fy_attr))

    # 构建 DataFrame，维持列顺序
    cols = [
        "无人机编号",
        "无人机运动方向",
        "无人机运动速度 (m/s)",
        "烟幕干扰弹投放点的x坐标 (m)",
        "烟幕干扰弹投放点的y坐标 (m)",
        "烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)",
        "烟幕干扰弹起爆点的y坐标 (m)",
        "烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)",
        "error",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # 追加：空行 + 注释行
    blank = {c: np.nan for c in cols}
    note  = {c: np.nan for c in cols}
    note["无人机运动方向"] = "注：以x轴为正向，逆时针方向为正，取值0~360（度）。"
    df_out = pd.concat([df, pd.DataFrame([blank, note])], ignore_index=True)

    with pd.ExcelWriter("../result/result2.xlsx") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Sheet1")
