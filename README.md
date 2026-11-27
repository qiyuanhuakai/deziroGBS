# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 120 轮进展

对九章 3.0 公开的高功率光学标定，两种读出的 **1,152 端口完整联合分布 TV 上界均为 0.4815%**。该界适用于理想算术采样，可从原始光学输入在 CPU 上复核。浮点程序另用新随机流各生成 256 条候选，交付 70 条光子数记录、69 条直接点击记录，分别耗时 9.18 和 11.93 秒/条。[预印本第 8 版](https://github.com/qiyuanhuakai/deziroGBS/blob/4fcb0614cc41cacf1877ceb17e66808f628c48fb/papers/number-tilted-gbs/manuscript.pdf) · [复核步骤](https://github.com/qiyuanhuakai/deziroGBS/blob/4fcb0614cc41cacf1877ceb17e66808f628c48fb/papers/number-tilted-gbs/README.md)

在 147.6 万条留出硬件记录上，三组 16 端口的双组点击重数各有 81 种结果。加入泊松背景后的采样律到硬件平均条件分布的同时 99% TV 上界分别为 **0.954%、0.879%、1.043%**。完整硬件记录的距离仍待确定。[精选结果](results/selected/120/full-record.json) · [固定归档](https://github.com/qiyuanhuakai/deziroGBS/blob/4fcb0614cc41cacf1877ceb17e66808f628c48fb/results/iteration120_contractions/manifest.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
