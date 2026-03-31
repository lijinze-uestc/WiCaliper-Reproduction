"""
wicaliper_module4_bayes.py
==========================
模块四：贝叶斯联合参数寻优
对应原论文 §IV-B Algorithm 1：Joint Optimization of Object Parameters

框架要求
--------
- 输入：多视角 {P̂_ψ}、视角角度集合 {ψ}、形状类型 i
- 目标函数 L(ε_r, a, w, h)：多视角联合损失
- 搜索空间：ε_r ∈ [1, 81]，尺寸参数 ∈ [1cm, 20cm]
- 默认迭代次数 K = 100

形状类型 i 与参数的对应关系（原论文 Fig.9）
------------------------------------------
  i=0 'cylinder' : 参数 = (ε_r, R, h)        R = a = w（底面直径）
  i=1 'cuboid'   : 参数 = (ε_r, a, w, h)     a=厚度，w=宽度，h=高度
  i=2 'tube'     : 参数 = (ε_r, R_out, t_wall, h)

依赖：pip install bayesian-optimization
"""

import numpy as np
from bayes_opt import BayesianOptimization

from wicaliper_core import generate_property_function_P


# ──────────────────────────────────────────────────────────────
# 形状类型映射（对应论文 Algorithm 1 中的 shape type i）
# ──────────────────────────────────────────────────────────────
SHAPE_TYPES = {
    0: 'cylinder',
    1: 'cuboid',
    2: 'tube',
}

# 各形状的默认搜索边界（对应论文 §VI-A：尺寸 ∈ [1, 20] cm）
_DEFAULT_BOUNDS = {
    'cylinder': {
        'epsilon_r': (1.0, 81.0),
        'R':         (0.005, 0.10),   # 半径 0.5~10 cm
        'h':         (0.01, 0.30),    # 高度 1~20 cm
    },
    'cuboid': {
        'epsilon_r': (1.0, 81.0),
        'a':         (0.005, 0.10),   # 厚度 0.5~10 cm
        'w':         (0.005, 0.20),   # 宽度 0.5~20 cm
        'h':         (0.01, 0.30),
    },
    'tube': {
        'epsilon_r': (1.0, 81.0),
        'R_out':     (0.005, 0.10),   # 外径 0.5~10 cm
        't_wall':    (0.001, 0.03),   # 壁厚 0.1~3 cm
        'h':         (0.01, 0.30),
    },
}


def _params_to_shape_dict(shape_type, **kwargs):
    """
    将优化器输出的参数 dict 转化为 radon_thickness/generate_property_function_P
    所需的 shape_params 格式。
    """
    if shape_type == 'cylinder':
        return {'R': kwargs['R']}
    elif shape_type == 'cuboid':
        return {'a': kwargs['a'], 'w': kwargs['w']}
    elif shape_type == 'tube':
        return {'R_out': kwargs['R_out'], 't_wall': kwargs['t_wall']}


def _multiview_loss(shape_type, h, epsilon_r, shape_kwargs,
                    P_hat_list, view_configs, x_points_list, z0=0.0):
    """
    多视角联合损失 L = Σ_ψ ‖P̃_ψ - P̂_ψ‖。
    对应 Algorithm 1 第 5~6 行。

    Parameters
    ----------
    shape_type   : str   'cylinder' / 'cuboid' / 'tube'
    h            : float 物体高度
    epsilon_r    : float 介电常数（实部，虚部暂设为 0.1·实部）
    shape_kwargs : dict  形状尺寸参数
    P_hat_list   : List[(N,)]  模块三估计出的各视角属性函数
    view_configs : List[dict]  各视角几何配置
    x_points_list: List[(N,)]  各视角离散坐标
    z0           : float       物体底端 z 坐标
    """
    shape_params = _params_to_shape_dict(shape_type, **shape_kwargs)
    # 虚部设为 5% 的实部（简化处理，真实场景应一并优化）
    eps_complex  = complex(epsilon_r, -0.05 * epsilon_r)

    total_loss = 0.0
    for cfg, x_pts, P_hat in zip(view_configs, x_points_list, P_hat_list):
        rx_center_x = cfg['rx_pos'][0]      # 改：从 rx_pos 元组取
        rx_y        = cfg['rx_pos'][1]       # 改：从 rx_pos 元组取
        P_sim = generate_property_function_P(
            x_pts, shape_type, shape_params, h, eps_complex,
            tx_pos=cfg['tx_pos'], rx_y=rx_y,
            rx_center_x=rx_center_x, z0=z0
        )

        # 归一化：消除 Pd 的绝对幅度影响，只比较形状
        norm_sim = np.linalg.norm(P_sim)
        norm_hat = np.linalg.norm(P_hat)
        if norm_sim > 1e-12 and norm_hat > 1e-12:
            P_sim_n = P_sim / norm_sim
            P_hat_n = P_hat / norm_hat
            total_loss += np.linalg.norm(P_sim_n - P_hat_n)
        else:
            total_loss += 1e6  # 退化情况，给大惩罚

    return total_loss


def bayesian_joint_optimization(shape_type_i, P_hat_list,
                                 view_configs, x_points_list,
                                 search_bounds=None,
                                 K_iter=100, n_init=10,
                                 z0=0.0, verbose=True):
    """
    贝叶斯优化联合估计物体材质与 3D 尺寸参数。
    对应原论文 Algorithm 1。

    Parameters
    ----------
    shape_type_i  : int  形状类型编号（框架要求的参数 i）
                         0=cylinder, 1=cuboid, 2=tube
    P_hat_list    : List[(N,)]  模块三输出的多视角属性函数估计
    view_configs  : List[dict]  各视角几何配置（含 tx_pos, rx_y, rx_positions）
    x_points_list : List[(N,)]  各视角横坐标离散点
    search_bounds : dict or None  自定义搜索边界（None 使用论文默认值）
    K_iter        : int  贝叶斯迭代次数（论文默认 K=100）
    n_init        : int  随机初始化探索次数
    z0            : float  物体底端 z 坐标
    verbose       : bool  是否打印优化进度

    Returns
    -------
    result : dict  最优参数估计，包含：
             'shape_type', 'epsilon_r', 尺寸参数, 'loss'
    """
    if shape_type_i not in SHAPE_TYPES:
        raise ValueError(f"shape_type_i 须为 0/1/2，当前值={shape_type_i}")

    shape_type = SHAPE_TYPES[shape_type_i]
    bounds     = search_bounds or _DEFAULT_BOUNDS[shape_type]

    if verbose:
        print(f"\n[模块四] 形状类型 i={shape_type_i} ({shape_type})")
        print(f"  搜索空间：{bounds}")
        print(f"  迭代次数：K={K_iter}（含 {n_init} 次随机初始化）\n")

    # ── 包装目标函数（BayesianOptimization 执行最大化）────────
    def objective(**kwargs):
        h         = kwargs.pop('h')
        epsilon_r = kwargs.pop('epsilon_r')
        # kwargs 中剩余的是形状尺寸参数
        loss = _multiview_loss(
            shape_type, h, epsilon_r, kwargs,
            P_hat_list, view_configs, x_points_list, z0
        )
        return -loss    # 最大化 -L ≡ 最小化 L

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=bounds,
        random_state=42,
        verbose=2 if verbose else 0,
    )
    optimizer.maximize(init_points=n_init, n_iter=K_iter)

    # ── 提取最优结果 ──────────────────────────────────────────
    best = optimizer.max
    result = {'shape_type': shape_type, 'loss': -best['target']}
    result.update(best['params'])

    if verbose:
        print("\n" + "=" * 55)
        print(f"[模块四] 寻优完成（形状={shape_type}，迭代={K_iter} 次）")
        for k, v in result.items():
            if k in ('shape_type', 'loss'):
                continue
            unit = 'cm' if k != 'epsilon_r' else ''
            val  = v * 100 if k != 'epsilon_r' else v
            print(f"  {k:12s} = {val:.4f} {unit}")
        print(f"  {'loss':12s} = {result['loss']:.6f}")
        print("=" * 55)

    return result
