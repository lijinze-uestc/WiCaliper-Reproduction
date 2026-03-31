"""
wicaliper_module3_svd.py
========================
模块三：截断奇异值分解（Truncated SVD）属性函数恢复
对应原论文 §IV-A 公式 (21)：P̂ = Ṽ · Σ̃⁺ · Ũ* · H

设计说明
--------
阈值策略参数化为两种模式，解决框架文档约束（阈值=10）
与仿真环境（F 矩阵量级≈10⁻³）之间的系统性矛盾：

  mode='hardware'  : 绝对阈值 = 10，对应原论文 §VI-A 设定，
                     适用于真实 WARP 硬件采集的 CSI 数据
  mode='simulation': 相对阈值 = rel_thr × σ_max，
                     适用于仿真生成的 F 矩阵
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# 阈值常量（对应框架约束清单）
# ──────────────────────────────────────────────────────────────
SVD_THRESHOLD_HARDWARE   = 10.0   # 原论文 §VI-A 绝对阈值（真实硬件）
SVD_THRESHOLD_REL_SIM    = 0.01   # 仿真相对阈值：保留 > 1% × σ_max 的奇异值


def truncated_svd_solve(F, H_ob, mode='simulation',
                         abs_threshold=SVD_THRESHOLD_HARDWARE,
                         rel_threshold=SVD_THRESHOLD_REL_SIM,
                         verbose=False):
    """
    截断 SVD 稳健求解属性函数 P̂。

    核心步骤
    --------
    1. 全量 SVD：F = U · Σ · V*
    2. 阈值截断：将 Σ 中低于阈值的奇异值置零
    3. 闭式解：P̂ = Ṽ · Σ̃⁺ · Ũ* · H

    Parameters
    ----------
    F            : (M, N) complex  传递矩阵，M ≥ N
    H_ob         : (M,) complex    含噪目标差分 CSI 向量
    mode         : str
        'hardware'   → 使用绝对阈值 abs_threshold（论文默认 10）
        'simulation' → 使用相对阈值 rel_threshold × σ_max
    abs_threshold: float  硬件模式绝对截断阈值（默认 10）
    rel_threshold: float  仿真模式相对截断阈值（默认 0.01）
    verbose      : bool   是否打印奇异值诊断信息

    Returns
    -------
    P_hat   : (N,) complex  恢复出的属性函数估计
    n_kept  : int           保留的奇异值数量
    s_vals  : (K,) float    全量奇异值（诊断用）
    """
    if F.shape[0] < F.shape[1]:
        raise ValueError(
            f"传递矩阵需满足 M ≥ N（框架约束），"
            f"当前 M={F.shape[0]}, N={F.shape[1]}。"
        )

    # ── 步骤 1：全量 SVD 分解 ────────────────────────────────
    U, s, Vh = np.linalg.svd(F, full_matrices=False)
    # U: (M, K)，s: (K,)，Vh: (K, N)，K = min(M, N)

    # ── 步骤 2：确定截断阈值 ─────────────────────────────────
    if mode == 'hardware':
        threshold = abs_threshold
    elif mode == 'simulation':
        threshold = rel_threshold * s.max()
    else:
        raise ValueError(f"未知 mode：'{mode}'，支持：'hardware' / 'simulation'")

    if verbose:
        print(f"  [SVD] 模式={mode}，σ_max={s.max():.4e}，"
              f"σ_min={s.min():.4e}，阈值={threshold:.4e}")
        print(f"  [SVD] 有效奇异值：{(s > threshold).sum()}/{len(s)}")

    mask   = s > threshold
    n_kept = int(mask.sum())

    if n_kept == 0:
        raise ValueError(
            f"所有 {len(s)} 个奇异值均低于阈值 {threshold:.4e}。\n"
            f"  建议：降低 rel_threshold（仿真）或 abs_threshold（硬件）。"
        )

    # ── 步骤 3：构建截断伪逆并求解 ──────────────────────────
    s_inv_trunc = np.where(mask, 1.0 / s, 0.0)   # Σ̃⁺ 的对角元素
    # P̂ = V* · diag(s_inv_trunc) · U* · H
    P_hat = (Vh.conj().T * s_inv_trunc) @ (U.conj().T @ H_ob)

    return P_hat, n_kept, s


def multiview_svd_solve(F_list, H_ob_list, mode='simulation',
                         abs_threshold=SVD_THRESHOLD_HARDWARE,
                         rel_threshold=SVD_THRESHOLD_REL_SIM,
                         verbose=False):
    """
    对多个视角分别执行截断 SVD，返回各视角属性函数估计列表。
    对应框架文档 §1 模块三的多视角循环。

    Parameters
    ----------
    F_list    : List[(M_i, N)]  各视角传递矩阵
    H_ob_list : List[(M_i,)]   各视角差分 CSI 观测向量
    mode      : str             'hardware' / 'simulation'
    ...       : 其余参数同 truncated_svd_solve

    Returns
    -------
    P_hat_list : List[(N,) complex]  各视角属性函数估计
    """
    P_hat_list = []
    for vi, (F, H_ob) in enumerate(zip(F_list, H_ob_list)):
        if verbose:
            print(f"\n  [模块三] 视角 {vi+1}/{len(F_list)} SVD 求解...")
        P_hat, n_kept, s = truncated_svd_solve(
            F, H_ob, mode=mode,
            abs_threshold=abs_threshold,
            rel_threshold=rel_threshold,
            verbose=verbose
        )
        if verbose:
            print(f"  [模块三] 保留 {n_kept} 个奇异值，"
                  f"|P̂| ∈ [{np.abs(P_hat).min():.4e}, "
                  f"{np.abs(P_hat).max():.4e}]")
        P_hat_list.append(P_hat)
    return P_hat_list


def evaluate_recovery_quality(P_true, P_hat):
    """
    仿真闭环验证：计算 NMSE 和相位相关系数。
    真实场景下 P_true 未知，此函数仅用于仿真验证。

    Returns
    -------
    nmse       : float  归一化均方误差（越小越好）
    phase_corr : float  相位相关系数 ∈ [-1, 1]（越大越好）
    """
    nmse = (np.mean(np.abs(P_hat - P_true) ** 2)
            / (np.mean(np.abs(P_true) ** 2) + 1e-12))
    phase_corr = float(np.mean(
        np.cos(np.angle(P_hat) - np.angle(P_true))
    ))
    return nmse, phase_corr
