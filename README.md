# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 117 轮进展

在九章 3.0 三套公开功率标定上，我们用同一算法生成完整 1,152 通道光子数和直接点击。高功率 16 通道压力面板的完整 PNR 误差为 **0.4378%–0.5797%**，完整点击误差为 **0.4323%**。每种读出各跑 4,096 条，八线程 CPU 分别用 **21.93、22.38 毫秒/条**。整机 TV 的数值上界仍为 **52.05%**。[第 5 版预印本](https://github.com/qiyuanhuakai/deziroGBS/blob/adce0699a5fe5f4b6d162317d31165cfaaeae652/papers/number-tilted-gbs/manuscript.pdf) · [速度与精度](results/selected/117/correlated-readout.json)

恢复各端口之间的一阶相干矩阵后，三档功率合计 432 个单组点击分布中有 **418 个**改善，高功率压力面板的完整点击误差则升至 **0.5055%**。[完整记录与复现步骤](https://github.com/qiyuanhuakai/deziroGBS/blob/adce0699a5fe5f4b6d162317d31165cfaaeae652/papers/number-tilted-gbs/README.md) · [第 116 轮快速采样器](results/selected/116/spatial-filter.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
