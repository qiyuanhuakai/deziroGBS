# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 108 轮进展

16 个端口、6 对源、平均发射 1.2 对光子时，相干法生成完整点击记录为 **1.99 微秒/条，完整分布 TV 为 0.457%**。同机 Frugal K4 为 **22.03 微秒/条、0.616%**。实际九章 3.0 输入上，GPU 的直接点击 / 逐位光子数为 0.384 / 0.402 微秒/条。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/a09ccbe38c5ad7023fbbb005d5d8dac27012312b/results/iteration108_figure_v1/accuracy.png) · [精选核验](results/selected/108/validation.json)

4,096 个端口、1,024 个独立谱的稠密输入上，平均探测到 156 个光子，完整分布的理想算术 TV 上界为 0.000506%。相干法在实际九章双谱输入上的前八位模型 TV 为 12.72%，报告给出了强光谱的差距与分解结果。[完整报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/a09ccbe38c5ad7023fbbb005d5d8dac27012312b/research/iteration108_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
