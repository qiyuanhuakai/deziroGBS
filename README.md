# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 121 轮进展

在九章 3.0 公开的高功率光学标定上，本方法对 **1,152 个端口完整记录**的理想算术 TV 上界为 **0.4815%**。实测交付光子数记录耗时 **9.18 秒/条**，直接点击 **11.93 秒/条**。同输入的 Goodman 正 P 方法独立复现，在百万轨迹双精度训练后分别为 **51.54、51.25 微秒/条**。每条路线用 2,097,152 条完整记录检验，整机 TV 同时 99% 下界分别为 **0.785%、0.497%**。[第 9 版预印本](https://github.com/qiyuanhuakai/deziroGBS/blob/d1654c41d24bf417bd4c6200f49fb302520c1086/papers/number-tilted-gbs/manuscript.pdf) · [速度、精度与全部八项检验](results/selected/121/full-width-comparison.json)

上述精度比较针对固定光学输入。硬件留出数据的三组 16 端口点击重数检验、浮点实现的验证范围和两种训练配置均可复核。[复现步骤](https://github.com/qiyuanhuakai/deziroGBS/blob/d1654c41d24bf417bd4c6200f49fb302520c1086/papers/number-tilted-gbs/README.md) · [硬件结果](results/selected/120/full-record.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
