# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 102 轮进展

| 比较范围 | 速度与精度 |
|---|---|
| 九章 3.0 双谱输入，1,152 个探测位 | 16 线程直接点击 1.326 微秒/条、PNR 1.459 微秒/条；论文硬件为 1.27 微秒/条。两条路线各用百万条新记录复测，总点击数累计分布与硬件相差 4.517、4.543 个百分点。[整机核验](results/selected/102/validation.json) |
| 73、70 号空间端口的全部 16 位 | 本轮准备后单线程 0.109 微秒/条，完整点击 TV 为 0.0230%。同输入 Frugal K2 为 10.770 微秒、30.382%，K4 为 49.314 微秒、9.732%。[对照、准备成本与三模式 PNR](https://github.com/qiyuanhuakai/deziroGBS/blob/8dce8adf031f1f5ae08bf1a44603dcacdec0c0b0/research/iteration102_checkpoint_v1.md) |

34、35 号端口的点击 TV 从 5.498% 降至 0.0256%。93 对受光学联合表约束的端口均已核对全部点击结果，最大 TV 为 0.0287%。36、107 号端口仍给出整机点击 TV 至少 2.118%、PNR TV 至少 2.150%。[各端口对的完整对照](https://github.com/qiyuanhuakai/deziroGBS/blob/8dce8adf031f1f5ae08bf1a44603dcacdec0c0b0/results/iteration102_projection_v1/summary.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
