"""
wicaliper_core.py
=================
模块一：正向物理仿真引擎  [原论文 §III 公式(13)~(17)]
模块二：多视角传递矩阵构建 [原论文 §IV-A 公式(20)]

设计原则
--------
- 严格贴合论文公式，不做任何近似简化
- 支持三种物体形状：cylinder / cuboid / tube
- 通过 Radon 变换在视角坐标系中计算厚度函数 d(x)
- rho_0 强制由实际几何参数派生，禁止硬编码
- 传递矩阵 F 完整实现 f(p) 公式(5)，含 K_θ 和 1/(r_T·r_R)
"""

import numpy as np
from scipy.integrate import quad

# ──────────────────────────────────────────────────────────────
# 物理常量（对应原论文 §V 实验配置：5.32 GHz 中心频率）
# ──────────────────────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8          # m/s
FREQ           = 5.32e9       # Hz  原论文 §VI-A
WAVELENGTH     = SPEED_OF_LIGHT / FREQ
K0             = 2 * np.pi / WAVELENGTH   # 自由空间波数 k


# ══════════════════════════════════════════════════════════════
# 模块一 · 子函数：材质电磁参数
# ══════════════════════════════════════════════════════════════

def calc_alpha_beta(epsilon_r):
    """
    严格按原论文公式 (2) 推导衰减常数 α 和相位常数 β。

    α = k · sqrt(ε'_r · (η-1) / 2)
    β = k · sqrt(ε'_r · (η+1) / 2)
    η = sqrt(1 + (ε''_r / ε'_r)²)

    Parameters
    ----------
    epsilon_r : complex  复相对介电常数，如 80.0 - 5.0j
    Returns
    -------
    alpha, beta : float
    """
    eps_real = np.real(epsilon_r)
    eps_imag = np.abs(np.imag(epsilon_r))        # ε'' 恒取正值
    eta      = np.sqrt(1.0 + (eps_imag / eps_real) ** 2)
    alpha    = K0 * np.sqrt(eps_real * (eta - 1.0) / 2.0)
    beta     = K0 * np.sqrt(eps_real * (eta + 1.0) / 2.0)
    return alpha, beta


def calc_transmission_coef(epsilon_r):
    """
    计算空气→介质→空气双界面合并透射系数。
    Tio = 4n / (n+1)²，对应原论文公式 (3)。

    Parameters
    ----------
    epsilon_r : complex
    Returns
    -------
    T_io : float
    """
    n    = np.sqrt(epsilon_r)
    n_re = np.real(n)
    return (4.0 * n_re) / (n_re + 1.0) ** 2


def calc_rho0(tx_pos, rx_y):
    """
    由实际几何参数派生 ρ₀ = 1/(2·r_T0) + 1/(2·r_R0)。
    r_T0：Tx 到物体中心（原点）的距离；r_R0：Rx 到原点的距离。
    [框架约束] 禁止硬编码，必须由 tx_pos 和 rx_y 自动计算。

    Parameters
    ----------
    tx_pos : (float, float)  Tx 坐标 (X, Y)
    rx_y   : float           Rx 阵列中心的 Y 坐标
    """
    r_T0  = np.sqrt(tx_pos[0] ** 2 + tx_pos[1] ** 2)
    r_R0  = abs(rx_y)
    return 1.0 / (2.0 * r_T0) + 1.0 / (2.0 * r_R0)


# ══════════════════════════════════════════════════════════════
# 模块一 · 子函数：厚度函数与 Radon 变换
# ══════════════════════════════════════════════════════════════

def _inside_shape(xg, yg, shape_type, shape_params):
    """
    判断坐标 (xg, yg) 是否在物体截面内（向量化）。
    供 radon_thickness 内部调用。
    """
    if shape_type == 'cylinder':
        R = shape_params['R']
        return (xg ** 2 + yg ** 2) <= R ** 2

    elif shape_type == 'cuboid':
        a = shape_params['a']   # 沿 Y 轴方向的厚度（穿透方向）
        w = shape_params['w']   # 沿 X 轴方向的宽度
        return (np.abs(xg) <= w / 2.0) & (np.abs(yg) <= a / 2.0)

    elif shape_type == 'tube':
        R_out = shape_params['R_out']
        R_in  = shape_params['R_out'] - shape_params['t_wall']
        r2 = xg ** 2 + yg ** 2
        return (r2 <= R_out ** 2) & (r2 >= R_in ** 2)

    else:
        raise ValueError(f"未知形状类型：{shape_type}，支持：cylinder / cuboid / tube")


def radon_thickness(x_proj, psi, shape_type, shape_params, n_ray=600):
    """
    通过 Radon 投影计算视角 ψ 下的厚度函数 d(x)。

    物理含义：信号以角度 ψ 穿过物体截面时，沿投影方向的路径长度。
    对应关系：d(x) = R{C(x,y), ψ}，见框架文档 §1 模块二。

    实现：沿视线方向（与 Y 轴夹角 ψ）数值积分截面指示函数。
    对于圆柱体，利用几何对称性直接解析计算（精确且高效）。

    Parameters
    ----------
    x_proj      : (N,) 投影坐标数组（视角坐标系的 x 轴）
    psi         : float 视角角度（弧度，相对于 Y 轴）
    shape_type  : str   'cylinder' / 'cuboid' / 'tube'
    shape_params: dict  形状参数（见下方各形状说明）
    n_ray       : int   数值积分分辨率（圆柱除外）

    形状参数
    ---------
    cylinder : {'R': 半径(m)}
    cuboid   : {'a': Y方向厚度(m), 'w': X方向宽度(m)}
    tube     : {'R_out': 外径(m), 't_wall': 壁厚(m)}

    Returns
    -------
    d : (N,) float  各投影坐标处的穿透路径长度
    """
    # ── 圆柱：利用对称性解析计算，无需数值积分 ──────────────
    if shape_type == 'cylinder':
        R = shape_params['R']
        d = np.zeros_like(x_proj, dtype=np.float64)
        mask = np.abs(x_proj) <= R
        d[mask] = 2.0 * np.sqrt(R ** 2 - x_proj[mask] ** 2)
        return d    # 圆柱对 ψ 具有旋转对称性，结果与视角无关

    # ── 长方体 / 圆管：沿视线方向数值积分 ───────────────────
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # 射线参数范围：覆盖物体最大尺寸的 1.5 倍以保证完整穿越
    if shape_type == 'cuboid':
        half_diag = 0.5 * np.sqrt(shape_params['a'] ** 2 + shape_params['w'] ** 2)
    else:
        half_diag = shape_params['R_out']
    t_max = half_diag * 1.5
    t_vals = np.linspace(-t_max, t_max, n_ray)
    dt     = t_vals[1] - t_vals[0]

    d = np.zeros(len(x_proj), dtype=np.float64)
    for i, xp in enumerate(x_proj):
        # 视角坐标 xp 处的射线：全局坐标 (x_g, y_g) 随 t 变化
        x_g = xp * cos_psi - t_vals * sin_psi
        y_g = xp * sin_psi + t_vals * cos_psi
        inside = _inside_shape(x_g, y_g, shape_type, shape_params)
        d[i] = np.sum(inside) * dt

    return d


def compute_view_angle(tx_pos, rx_center_x, rx_y):
    """
    计算 Tx → Rx 中心方向与 Y 轴的夹角 ψ（弧度）。
    对应论文中 Radon 变换的投影角度。

    Parameters
    ----------
    tx_pos      : (float, float)  Tx 坐标 (X, Y)
    rx_center_x : float           Rx 阵列中心的 X 坐标
    rx_y        : float           Rx 的 Y 坐标

    Returns
    -------
    psi : float  以 Y 轴为基准的视角（弧度）
    """
    dx = rx_center_x - tx_pos[0]
    dy = rx_y - tx_pos[1]
    return np.arctan2(dx, dy)   # arctan2(Δx, Δy) = 与 Y 轴夹角


# ══════════════════════════════════════════════════════════════
# 模块一 · 主函数：属性函数 P(x)
# ══════════════════════════════════════════════════════════════

def generate_property_function_P(x_points, shape_type, shape_params,
                                  h, epsilon_r,
                                  tx_pos=(0.0, -1.0), rx_y=1.0,
                                  rx_center_x=0.0, z0=0.0):
    """
    计算属性函数 P(x) = Pₚ(x) · Pd(w, h)，对应原论文公式 (16)。

    Parameters
    ----------
    x_points    : (N,)  物体横坐标离散点（视角坐标系，Δx = 2mm）
    shape_type  : str   'cylinder' / 'cuboid' / 'tube'
    shape_params: dict  形状参数（传入 radon_thickness）
    h           : float 物体高度 (m)
    epsilon_r   : complex 复相对介电常数
    tx_pos      : (float, float) Tx 坐标
    rx_y        : float Rx 的 Y 坐标
    rx_center_x : float Rx 阵列中心的 X 坐标（用于计算视角 ψ）
    z0          : float 物体底端 z 坐标

    Returns
    -------
    P_x : (N,) complex  属性函数向量
    """
    # ── 穿透因子 Pₚ(x)，公式 (13) ────────────────────────────
    alpha, beta = calc_alpha_beta(epsilon_r)
    T_io        = calc_transmission_coef(epsilon_r)

    # 在视角 ψ 下计算厚度函数 d(x)（Radon 投影）
    psi = compute_view_angle(tx_pos, rx_center_x, rx_y)
    d_x = radon_thickness(x_points, psi, shape_type, shape_params)

    P_p    = np.zeros(len(x_points), dtype=np.complex128)
    nonzero = d_x > 0
    P_p[nonzero] = (
        T_io
        * np.exp(-alpha * d_x[nonzero])
        * np.exp(-1j * (beta - K0) * d_x[nonzero])
        - 1.0
    )
    # d_x = 0 处（物体边界外）：P_p = 0，已由初始化保证

    # ── 衍射因子 Pd(w, h)，公式 (17) ─────────────────────────
    rho_0 = calc_rho0(tx_pos, rx_y)

    I_real, _ = quad(lambda z: np.cos(-K0 * rho_0 * z ** 2), z0, z0 + h)
    I_imag, _ = quad(lambda z: np.sin(-K0 * rho_0 * z ** 2), z0, z0 + h)
    P_d = I_real + 1j * I_imag

    return P_p * P_d


# ══════════════════════════════════════════════════════════════
# 模块二：多视角传递矩阵构建
# ══════════════════════════════════════════════════════════════

def build_transfer_matrix_F(x_points, object_positions,
                             tx_pos=(0.0, -1.0),
                             rx_pos=(0.0, 1.0)):
    """
    构建单视角 M×N 传递矩阵 F，严格实现原论文公式 (5) + (20)。

    物理场景：Tx 和 Rx 均固定，物体沿 X 轴移动。
    第 i 次测量时物体中心在 (object_positions[i], 0)，
    物体上第 j 个局部坐标 x_points[j] 的全局 X 坐标为
    object_positions[i] + x_points[j]。

    Parameters
    ----------
    x_points         : (N,)   物体局部横坐标离散点，Δx = 0.002 m
    object_positions : (M,)   物体中心沿 X 轴的移动位置序列
    tx_pos           : (X, Y) 发射天线坐标（固定），Y < 0
    rx_pos           : (X, Y) 接收天线坐标（固定），Y > 0

    Returns
    -------
    F : (M, N) complex
    """
    M  = len(object_positions)
    N  = len(x_points)
    F  = np.zeros((M, N), dtype=np.complex128)
    dx = x_points[1] - x_points[0]

    for i, X_i in enumerate(object_positions):
        # 物体上各点在全局坐标系中的 X 坐标
        global_x = X_i + x_points   # (N,)

        # Tx 到各点的距离（物体面固定在 y=0）
        d_tx = np.sqrt((global_x - tx_pos[0]) ** 2 + tx_pos[1] ** 2)

        # Rx 到各点的距离
        d_rx = np.sqrt((global_x - rx_pos[0]) ** 2 + rx_pos[1] ** 2)

        # 斜率因子 Kθ(p) = (cos θ_T + cos θ_R) / 2
        cos_theta_tx = abs(tx_pos[1]) / d_tx
        cos_theta_rx = abs(rx_pos[1]) / d_rx
        K_theta = (cos_theta_tx + cos_theta_rx) / 2.0

        # f(p) · Δx
        A_factor = 1.0 / (1j * WAVELENGTH)
        f_p = A_factor * K_theta * np.exp(-1j * K0 * (d_tx + d_rx)) / (d_tx * d_rx)
        F[i, :] = f_p * dx

    return F


def build_multiview_matrices(view_configs, x_points_list):
    """
    为每个视角分别构建传递矩阵，返回矩阵列表。

    Parameters
    ----------
    view_configs   : List[dict]  每个视角的几何配置
                     每个 dict 包含：
                       tx_pos           : (X, Y) 发射天线坐标（固定）
                       rx_pos           : (X, Y) 接收天线坐标（固定）
                       object_positions : (M,)   物体中心移动位置序列
    x_points_list  : List[(N,)]  各视角的横坐标离散点

    Returns
    -------
    F_list : List[(M_i, N)]  各视角传递矩阵
    """
    F_list = []
    for cfg, x_pts in zip(view_configs, x_points_list):
        F = build_transfer_matrix_F(
            x_pts,
            cfg['object_positions'],
            tx_pos=cfg['tx_pos'],
            rx_pos=cfg['rx_pos'],
        )
        F_list.append(F)
    return F_list