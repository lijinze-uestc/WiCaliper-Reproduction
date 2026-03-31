# WiCaliper-Reproduction

北京大学张大庆团队 WiCaliper 系统（IEEE JSAC 2026）的理论复现与仿真验证。

## 项目简介

本项目基于 WiCaliper 论文提出的 DP-CSI（Diffraction-Penetration CSI）联合感知模型，
实现了从正向物理仿真到逆向参数估计的完整四模块管线：

- **模块零**：CSI 背景信号估计与预处理
- **模块一**：正向物理仿真引擎（穿透因子 + 衍射因子 → 属性函数）
- **模块二**：多视角传递矩阵构建
- **模块三**：截断 SVD 属性函数恢复
- **模块四**：贝叶斯联合参数寻优

## 运行方式
```bash
pip install numpy scipy matplotlib bayesian-optimization
python wicaliper_simulation.py
```

## 依赖环境

- Python 3.9+
- NumPy, SciPy, Matplotlib
- bayesian-optimization

## 参考文献

Z. Yao et al., "WiCaliper: Simultaneous Material and 3D Size Sensing for Everyday Objects Using WiFi," IEEE JSAC, vol. 44, 2026.
