# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 118 轮进展

对于九章 3.0 高功率标定的 **1,152 端口完整联合分布**，光子数采样和直接点击的 TV 数值上界均为 **3.0914%**，上一版为 **52.05%**。独立的环境积分给出 **2.330%** 的单侧 99% 置信上限。固定标定所预测的平均总光子数为 116.29。[第 6 版预印本](https://github.com/qiyuanhuakai/deziroGBS/blob/ce56b3677fc1ec90ab2c581240a2f8d6c0dc8d45/papers/number-tilted-gbs/manuscript.pdf) · [精选结果](results/selected/118/full-record.json)

在原 16 端口压力面板上，完整点击 TV 为 **0.003639%**，完整光子数 TV 为 **0.00368%–0.1490%**。高精度档每条路线实跑 8 条，四个 CPU 工作进程的批次耗时折合 **57.23 秒/条光子数记录、36.62 秒/条直接点击记录**。微秒级结果属于另一种近似模型。[逐条记录与复核](https://github.com/qiyuanhuakai/deziroGBS/blob/ce56b3677fc1ec90ab2c581240a2f8d6c0dc8d45/papers/number-tilted-gbs/README.md) · [上一版 4,096 条对照](results/selected/117/correlated-readout.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
