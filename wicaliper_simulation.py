"""
wicaliper_simulation.py
=======================
WiCaliper 端到端仿真主程序
对应框架文档 §2 的数据流与闭环验证流程

关键参数（严格对应框架约束清单）
--------------------------------
Δx = 2 mm     → 原论文 §V
M ≥ N         → 原论文 §IV-A
SVD 阈值 = 10 → 原论文 §VI-A（仿真中使用相对阈值，见模块三说明）
K = 100 次迭代 → 原论文 §VI-E
ε_r ∈ [1,81]  → 原论文 §VI-A
至少 2 个视角  → 原论文 Algorithm 1
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



from wicaliper_core import (
    generate_property_function_P,
    build_transfer_matrix_F,
    #build_multiview_matrices,
    compute_view_angle,
)
from wicaliper_module0 import simulate_background_subtraction
from wicaliper_module3_svd import (
    #multiview_svd_solve,
    evaluate_recovery_quality,
    SVD_THRESHOLD_HARDWARE,
)

# ──────────────────────────────────────────────────────────────
# 仿真全局参数
# ──────────────────────────────────────────────────────────────

# [框架约束] Δx = 2 mm
DELTA_X   = 0.002                      # m
X_MIN     = -0.15                      # m  物体横向扫描范围
X_MAX     =  0.15                      # m
X_POINTS  = np.arange(X_MIN, X_MAX, DELTA_X)   # N ≈ 150 个点
N_POINTS  = len(X_POINTS)

# [框架约束] M ≥ N
M_POINTS  = 200                        # Rx 采样位置数（M > N = 150）

SNR_DB    = 100                       # 信噪比

# ── 多视角几何配置（至少 2 个独立视角）──────────────────────
# 视角一：Tx 正对物体，Rx 阵列沿 X 扫描
# 视角二：Tx 偏置 0.4 m，使 Tx→Rx 方向与 Y 轴形成约 11° 夹角
# 物体在传送带上移动的 M 个位置（沿 X 轴）
OBJECT_POSITIONS = np.linspace(-0.5, 0.5, M_POINTS)

# 视角配置
VIEW_CONFIGS = [
    {
        'label'           : '视角 1（Rx1，正面）',
        'tx_pos'          : (0.0, -0.30),      # Tx 固定
        'rx_pos'          : (0.0,  0.30),       # Rx1 固定，正对 Tx
        'object_positions': OBJECT_POSITIONS,   # 物体移动位置
    },
    {
        'label'           : '视角 2（Rx2，偏置 20cm）',
        'tx_pos'          : (0.0, -0.30),       # 同一个 Tx
        'rx_pos'          : (0.20, 0.30),        # Rx2 偏置 20cm
        'object_positions': OBJECT_POSITIONS,
    },
]

# ── 目标物体真值参数（模拟水瓶：圆柱体）────────────────────
TRUE_SHAPE_TYPE   = 'cylinder'
TRUE_SHAPE_PARAMS = {'R': 0.05}        # 半径 5 cm
TRUE_H            = 0.20               # 高度 20 cm
TRUE_EPSILON_R    = 80.0 - 4.0j        # 水的介电常数（5.32 GHz）


# ══════════════════════════════════════════════════════════════
# 各模块执行函数
# ══════════════════════════════════════════════════════════════

def run_view(cfg, shape_type, shape_params, h, epsilon_r):
    tx_pos           = cfg['tx_pos']
    rx_pos           = cfg['rx_pos']
    object_positions = cfg['object_positions']
    rx_center_x      = rx_pos[0]

    """
    针对单个视角执行模块零 → 模块一 → 模块二 → 模块三。

    Returns
    -------
    dict  包含该视角所有中间结果和最终 P̂
    """

     # 模块一：正向仿真，生成 P_true
    P_true = generate_property_function_P(
        X_POINTS, shape_type, shape_params, h, epsilon_r,
        tx_pos=tx_pos, rx_y=rx_pos[1],
        rx_center_x=rx_center_x
    )

    # 模块二：构建传递矩阵 F（修改后的接口）
    F = build_transfer_matrix_F(X_POINTS, object_positions,
                                 tx_pos=tx_pos, rx_pos=rx_pos)

    # 正向映射 H = F · P
    H_clean = F @ P_true

    # 模块零：背景减除 + 噪声注入
    H_ob, H_bg_est, noise_power = simulate_background_subtraction(
        H_clean, snr_db=SNR_DB, n_baseline=20
    )

    # 模块三：截断 SVD
    from wicaliper_module3_svd import truncated_svd_solve
    P_hat, n_kept, s_vals = truncated_svd_solve(
        F, H_ob, mode='simulation', verbose=True
    )

    psi_deg = np.degrees(compute_view_angle(tx_pos, rx_center_x, rx_pos[1]))
    return {
        'label'           : cfg['label'],
        'psi_deg'         : psi_deg,
        'x_points'        : X_POINTS,
        'object_positions': object_positions,
        'F'               : F,
        'P_true'          : P_true,
        'P_hat'           : P_hat,
        'H_clean'         : H_clean,
        'H_ob'            : H_ob,
        'n_kept'          : n_kept,
        's_vals'          : s_vals,
        'noise_power'     : noise_power,
    }


# ══════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════

def plot_all_results(view_results, bay_result=None):
    """
    生成综合可视化：三列（P(x) 对比 / CSI 幅度 / 奇异值谱）× n_views 行。
    """
    n_views = len(view_results)
    fig, axes = plt.subplots(n_views, 3, figsize=(18, 5 * n_views))
    if n_views == 1:
        axes = [axes]

    for vi, res in enumerate(view_results):
        ax_p, ax_csi, ax_sv = axes[vi]
        x_cm  = res['x_points'] * 100
        rx_cm = res['object_positions'] * 100
        R_cm  = TRUE_SHAPE_PARAMS['R'] * 100
        nmse, phase_corr = evaluate_recovery_quality(res['P_true'], res['P_hat'])

        # ── 子图 1：属性函数 P(x) 闭环对比 ──────────────────
        ax_p.plot(x_cm, np.abs(res['P_true']), 'b-',  lw=2,
                  label='|P_true(x)|（Ground Truth）')
        ax_p.plot(x_cm, np.abs(res['P_hat']),  'r--', lw=2,
                  label=fr"$|\hat{{P}}(x)|$（NMSE={nmse:.4f}，相位相关={phase_corr:.3f}）")
        ax_p.axvline(x=-R_cm, color='gray', ls=':', lw=1.2, alpha=0.7,
                     label='物理边界 ±R')
        ax_p.axvline(x= R_cm, color='gray', ls=':', lw=1.2, alpha=0.7)
        ax_p.set_title(f"{res['label']}（ψ={res['psi_deg']:.1f}°）— 属性函数恢复\n"
                       f"保留奇异值 {res['n_kept']} / {len(res['s_vals'])} 个")
        ax_p.set_xlabel('X 坐标 (cm)')
        ax_p.set_ylabel('幅度')
        ax_p.legend(fontsize=8)
        ax_p.grid(True, alpha=0.3)

        # ── 子图 2：CSI 幅度 ─────────────────────────────────
        ax_csi.plot(rx_cm, np.abs(res['H_clean']), 'g-', lw=2,
                    label='干净 CSI（H_clean）')
        ax_csi.plot(rx_cm, np.abs(res['H_ob']),    'r.', ms=2.5,
                    alpha=0.6, label=f'预处理后 H_ob（含 {SNR_DB}dB AWGN）')
        ax_csi.set_title(f"{res['label']} — CSI 幅度观测")
        ax_csi.set_xlabel('物体中心位置 (cm)')
        ax_csi.set_ylabel('CSI 幅度')
        ax_csi.legend(fontsize=8)
        ax_csi.grid(True, alpha=0.3)

        # ── 子图 3：奇异值谱 ─────────────────────────────────
        s = res['s_vals']
        ax_sv.semilogy(np.arange(1, len(s)+1), s, 'bo-', ms=3,
                       lw=1.5, label='奇异值')
        ax_sv.axhline(y=SVD_THRESHOLD_HARDWARE, color='darkred',
                      ls='--', lw=1.5,
                      label=f'原论文阈值（硬件）= {SVD_THRESHOLD_HARDWARE}')
        sim_thr = 0.01 * s.max()
        ax_sv.axhline(y=sim_thr, color='orange', ls='--', lw=1.5,
                      label=f'仿真相对阈值 = {sim_thr:.2e}')
        if res['n_kept'] > 0:
            ax_sv.fill_between(
                np.arange(1, res['n_kept']+1),
                s[:res['n_kept']], sim_thr,
                alpha=0.15, color='green', label='保留区域'
            )
        ax_sv.set_title(f"{res['label']} — 奇异值谱（截断 SVD）\n"
                        f"[框架约束] Δx={DELTA_X*1000:.0f}mm，"
                        f"M={M_POINTS}，N={N_POINTS}")
        ax_sv.set_xlabel('奇异值序号')
        ax_sv.set_ylabel('奇异值（对数轴）')
        ax_sv.legend(fontsize=7)
        ax_sv.grid(True, alpha=0.3)

    # 标注贝叶斯结果
    if bay_result:
        st = bay_result['shape_type']
        er = bay_result['epsilon_r']
        if st == 'cylinder':
            size_str = f"R*={bay_result['R']*100:.1f}cm"
        elif st == 'cuboid':
            size_str = (f"a*={bay_result['a']*100:.1f}cm，"
                        f"w*={bay_result['w']*100:.1f}cm")
        else:
            size_str = (f"R_out*={bay_result['R_out']*100:.1f}cm，"
                        f"t*={bay_result['t_wall']*100:.1f}cm")
        fig.suptitle(
            f"模块四 贝叶斯寻优结果 — "
            f"ε_r*={er:.1f}，{size_str}，"
            f"h*={bay_result['h']*100:.1f}cm",
            fontsize=12, fontweight='bold', y=1.01
        )

    plt.tight_layout()
    out_path = 'wicaliper_results_v2.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存：{out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(42)

    print('=' * 65)
    print('WiCaliper 四模块端到端仿真（严格对应框架约束清单）')
    print('=' * 65)
    print(f'[框架约束] Δx = {DELTA_X*1000:.0f} mm  →  N = {N_POINTS} 个离散点')
    print(f'[框架约束] M = {M_POINTS} ≥ N = {N_POINTS}  ✓')
    print(f'[框架约束] 视角数 = {len(VIEW_CONFIGS)} ≥ 2  ✓')
    print(f'目标：{TRUE_SHAPE_TYPE}，R={TRUE_SHAPE_PARAMS["R"]*100}cm，'
          f'h={TRUE_H*100}cm，ε_r={TRUE_EPSILON_R}')

    # ──────────────────────────────────────────────────────────
    # 模块零 + 一 + 二 + 三：逐视角处理
    # ──────────────────────────────────────────────────────────
    view_results  = []
    P_hat_list    = []
    view_cfg_list = []
    x_pts_list    = []

    for vi, cfg in enumerate(VIEW_CONFIGS):
        print(f'\n{"─"*55}')
        print(f'[模块零~三] 处理 {cfg["label"]}...')
        res = run_view(cfg, TRUE_SHAPE_TYPE, TRUE_SHAPE_PARAMS,
                       TRUE_H, TRUE_EPSILON_R)
        view_results.append(res)

        nmse, phase_corr = evaluate_recovery_quality(res['P_true'],
                                                      res['P_hat'])
        print(f'  视角角度 ψ = {res["psi_deg"]:.2f}°')
        print(f'  NMSE = {nmse:.6f}，相位相关 = {phase_corr:.4f}')
        print(f'  噪声功率 = {res["noise_power"]:.4e}')

        P_hat_list.append(res['P_hat'])
        view_cfg_list.append(cfg)
        x_pts_list.append(res['x_points'])

    # ──────────────────────────────────────────────────────────
    # 模块四：贝叶斯联合参数寻优
    # ──────────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('[模块四] 贝叶斯联合参数寻优...')

    bay_result = None
    try:
        from wicaliper_module4_bayes import bayesian_joint_optimization
        # [框架约束] shape_type_i = 0 对应 cylinder
        # [框架约束] K_iter = 100，搜索范围 ε_r ∈ [1,81]，尺寸 ∈ [1,20]cm
        bay_result = bayesian_joint_optimization(
            shape_type_i=0,
            P_hat_list=P_hat_list,
            view_configs=view_cfg_list,
            x_points_list=x_pts_list,
            K_iter=100, n_init=10, verbose=True
        )

        print(f'\n{"─"*55}')
        print('[结果对比] 真值 vs 估计：')
        print(f'  ε_r  真值={TRUE_EPSILON_R.real:.1f}，'
              f'估计={bay_result["epsilon_r"]:.2f}，'
              f'误差={abs(bay_result["epsilon_r"]-TRUE_EPSILON_R.real):.2f}')
        print(f'  R    真值={TRUE_SHAPE_PARAMS["R"]*100:.1f}cm，'
              f'估计={bay_result["R"]*100:.2f}cm，'
              f'误差={abs(bay_result["R"]-TRUE_SHAPE_PARAMS["R"])*100:.2f}cm')
        print(f'  h    真值={TRUE_H*100:.1f}cm，'
              f'估计={bay_result["h"]*100:.2f}cm，'
              f'误差={abs(bay_result["h"]-TRUE_H)*100:.2f}cm')

    except ImportError:
        print('  [跳过模块四] bayesian-optimization 未安装。')
        print('  安装：pip install bayesian-optimization --break-system-packages')

    # ──────────────────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('[可视化] 生成综合结果图...')
    plot_all_results(view_results, bay_result)
    print('\n✅ 全流程完成。')
