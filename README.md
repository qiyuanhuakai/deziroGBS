# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 101 轮进展

| 比较范围 | 速度与精度 |
|---|---|
| 九章 3.0 输入，1,152 个探测位 | 16 线程直接点击 1.894 微秒/条、PNR 2.207 微秒/条，分别为硬件每条用时的 1.49、1.74 倍。总点击数累计分布与硬件相差 4.400、4.586 个百分点。[整机复核](results/selected/101/validation.json) |
| 73、70 号空间端口的全部 16 位 | 准备后单线程 0.167 微秒/条，完整点击 TV 为 3.453%。同输入 Frugal K2 为 12.125 微秒、30.382%，K4 为 46.120 微秒、9.732%。[对照和准备成本](https://github.com/qiyuanhuakai/deziroGBS/blob/92ccb2a4c413a2a4f270e8d6a1953ebfea6b0c35/research/iteration101_checkpoint_v1.md) |

新模型同时匹配 10,296 对空间端口的光子数协方差。34、35 号端口的完整分布仍显示整机点击 TV 至少 5.498%，PNR TV 至少 5.552%。[端口对扫描](https://github.com/qiyuanhuakai/deziroGBS/blob/92ccb2a4c413a2a4f270e8d6a1953ebfea6b0c35/results/iteration101_vacuum_v1/summary.json) · [第 100 轮更快的空间树](results/selected/100/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
