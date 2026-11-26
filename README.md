# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 98 轮进展

| 输入 | 速度与精度 |
|---|---|
| 1,152 端口闭环，平衡分束器 | 16 线程构表后，直接点击 0.365 微秒/条，光子数 0.295 微秒/条。两条路线的完整分布 TV 均低于 0.0002460%。[分布误差与七批计时](results/selected/98/ring.json) |
| 同一 16 端口闭环，直接点击 | 本方法完整 TV ≤ 0.663%，Frugal K2 为 7.150%，K4 为 1.217%。单线程每批 2,048 条，本方法构表后 0.0364 微秒/条，首批计入构表约 121 微秒/条；Frugal 完整调用分别为 3.61、13.20 微秒/条。[同输入基线](results/selected/98/baselines.json) |

闭环数据来自无损成对光源。九章有损双谱光学参考中，约 50% 的记录具有奇数总光子数，当前闭环模型生成此类记录的概率为零。[推导、适用范围与运行示例](https://github.com/qiyuanhuakai/deziroGBS/blob/a0f7a455fe024fe271b5cb1bfe0ca430cff90ca9/research/iteration98_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
