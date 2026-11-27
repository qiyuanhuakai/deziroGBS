# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 111 轮进展

四个有损单元共享一个 1 弧度相移时，显式保留跨单元干涉，使完整 16 位点击分布的 TV 误差从 **12.270% 降至 0.04884%**。单 CPU 线程为 **11.91 微秒/条**；同输入 Frugal K4 为 **20.47 微秒/条、TV 2.018%**。扩到 131,072 端口后，两条路线都生成了完整记录，点击与光子数的整记录 TV 误差预算分别为 **0.620% / 0.686%**。[光路、样本与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/research/iteration111_checkpoint_v1.md)

实际九章 3.0 数据上，用前半段记录校准空间端口后，直接点击 / 光子数两条路线在后半段 10,296 个端口对上的平均经验 TV 降至 **0.540% / 0.530%**，每条完整 1,152 位记录耗时 **1.130 / 0.437 微秒**。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/results/iteration111_release_v1/comparison.png) · [精选核验](results/selected/111/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
