# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 99 轮进展

| 输入 | 速度与精度 |
|---|---|
| 九章 3.0 双谱光学输入，1,152 个探测位 | 16 线程准备后，直接点击 0.946 微秒/条，光子数 0.808 微秒/条。对硬件的总点击数 CDF 差分别为 4.68、4.78 个百分点。[九章对照](results/selected/99/actual.json) |
| 同一有损 16 端口闭环 | 单线程准备后，点击 0.123、光子数 0.136 微秒/条，完整 TV 均低于 7.354 × 10⁻¹⁰。Frugal K2、K4 完整调用为 7.86、24.48 微秒/条，点击 TV 为 1.1861%、0.1926%。[准备成本与全部基线](https://github.com/qiyuanhuakai/deziroGBS/blob/3b64f2301fb4247542778efb6bdf4f03a8c48a33/research/iteration99_checkpoint_v1.md) |

在九章输入上，总光子数概率表与光学参考的 TV 差为 2.26 × 10⁻¹⁴ 以内。另用 1,048,576 条记录检验未做联合校正的端口，1,152 位完整分布 TV 的联合 99% 置信下限为 **20.896%**。[独立留出核验](results/selected/99/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
