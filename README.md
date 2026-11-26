# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 103 轮进展

在九章 3.0 的 1,152 个探测位上，RTX 5080 Laptop 批量生成并传回一条点击记录需 **0.354 微秒**，一条光子数记录需 **0.379 微秒**。九章实验报告每条 1.27 微秒。两条模拟路线各复测 1,048,576 条记录，总点击数累计分布与硬件仍相差 **4.69 个百分点**。[整机精选核验](results/selected/103/validation.json)

对 36、107 号空间端口的全部 65,536 种点击结果，本轮与光学参考的总变差距离（TV）从 2.119% 降至 **0.01635%**。同样 16 个探测位上，Frugal K2、K4 的 TV 分别为 11.620% 和 6.007%。[同宽速度、精度与复现命令](https://github.com/qiyuanhuakai/deziroGBS/blob/efcf413ca93a283ff9269e4ae470c5f429d8ee7a/research/iteration103_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
