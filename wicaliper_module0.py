"""
wicaliper_module0.py
====================
模块零：背景信号估计与 CSI 预处理  [原论文 §V Signal Processing]

框架要求
--------
1. 采集无物体时的 CSI 基线序列
2. 计算背景均值 H_bg = mean(H_baseline)
3. 输出目标差分信号 H_ob = H_rx - H_bg

设计说明
--------
本模块提供可独立调用的函数接口，使整条管线
在真实硬件数据接入时无需修改下游模块。
仿真场景下 H_clean 即为差分信号，可直接调用
simulate_background_subtraction() 跳过真实采集。
"""

import numpy as np


def estimate_background(csi_baseline_list):
    """
    从多次无物体采集的 CSI 序列中估计背景信号 H_bg。
    对应原论文 §V：H_bg = mean({H_rx_i}，无物体时)

    Parameters
    ----------
    csi_baseline_list : List[(M,) complex]
        无物体时采集的 M 次 CSI 测量值列表。
        M = 接收天线采样位置数。
        列表长度 ≥ 1，建议 ≥ 10 次以降低噪声。

    Returns
    -------
    H_bg : (M,) complex  背景信号均值估计
    """
    if len(csi_baseline_list) == 0:
        raise ValueError("基线序列为空，至少需要 1 次无物体测量。")

    stack = np.stack(csi_baseline_list, axis=0)   # (n_baseline, M)
    H_bg  = np.mean(stack, axis=0)                # (M,)
    return H_bg


def subtract_background(H_rx, H_bg):
    """
    从含物体的 CSI 测量值中减去背景，得到目标差分信号。
    对应原论文 §V：H_ob = H_rx - H_bg

    Parameters
    ----------
    H_rx : (M,) complex  含物体时的 CSI 测量值
    H_bg : (M,) complex  背景信号估计（来自 estimate_background）

    Returns
    -------
    H_ob : (M,) complex  目标引起的 CSI 变化量
    """
    if H_rx.shape != H_bg.shape:
        raise ValueError(
            f"H_rx 形状 {H_rx.shape} 与 H_bg 形状 {H_bg.shape} 不匹配。"
        )
    return H_rx - H_bg


def preprocess_csi(H_rx_sequence, csi_baseline_list, rescale=True):
    """
    完整预处理流程：基线估计 → 差分 → 可选幅度归一化。
    对应原论文 §V "Static Removal & Rescaling" 子模块。

    Parameters
    ----------
    H_rx_sequence    : List[(M,) complex]
        含物体时在各位置采集的 CSI 序列，长度 = 物体位置数。
    csi_baseline_list: List[(M,) complex]
        无物体时采集的基线序列（同一 Rx 采样配置）。
    rescale          : bool
        是否对差分信号进行 L2 归一化（默认开启，
        提升后续 SVD 求解的数值稳定性）。

    Returns
    -------
    H_ob_sequence : List[(M,) complex]
        各物体位置对应的差分 CSI 信号列表。
    H_bg          : (M,) complex
        估计出的背景信号（供诊断使用）。
    """
    H_bg = estimate_background(csi_baseline_list)

    H_ob_sequence = []
    for H_rx in H_rx_sequence:
        H_ob = subtract_background(H_rx, H_bg)
        if rescale:
            norm = np.linalg.norm(H_ob)
            if norm > 1e-12:
                H_ob = H_ob / norm
        H_ob_sequence.append(H_ob)

    return H_ob_sequence, H_bg


# ──────────────────────────────────────────────────────────────
# 仿真专用工具
# ──────────────────────────────────────────────────────────────

def simulate_background_subtraction(H_clean, snr_db=20.0, n_baseline=20,
                                     seed=None):
    """
    仿真场景下的模块零替代方案。
    生成含背景噪声的 H_rx，再用模拟的基线均值还原 H_ob，
    验证预处理流程的完整性。

    Parameters
    ----------
    H_clean   : (M,) complex  无噪差分 CSI（来自模块一正向仿真）
    snr_db    : float         信噪比（dB）
    n_baseline: int           模拟基线采集次数
    seed      : int or None   随机种子

    Returns
    -------
    H_ob        : (M,) complex  经预处理后的差分信号
    H_bg_est    : (M,) complex  估计出的背景信号
    noise_power : float         实际注入的噪声功率
    """
    rng = np.random.default_rng(seed)

    sig_power   = np.mean(np.abs(H_clean) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    sigma       = np.sqrt(noise_power / 2.0)

    def _awgn(shape):
        return sigma * (rng.standard_normal(shape)
                        + 1j * rng.standard_normal(shape))

    # 真实 H_rx = H_bg_true + H_ob_true + 噪声
    # 仿真中：H_bg_true = 0（已是差分信号），加噪后得 H_rx
    H_rx = H_clean + _awgn(len(H_clean))

    # 模拟 n_baseline 次无物体采集（纯背景 = 0 + 噪声）
    baseline_list = [_awgn(len(H_clean)) for _ in range(n_baseline)]
    H_bg_est      = estimate_background(baseline_list)

    # 差分：H_ob = H_rx - H_bg_est
    H_ob = subtract_background(H_rx, H_bg_est)

    return H_ob, H_bg_est, noise_power
