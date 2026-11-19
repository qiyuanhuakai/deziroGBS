# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 96 轮进展

| 路线 | 同输入速度与精度 |
|---|---|
| 直接点击 | 九章约化 16 位，本方法、Frugal K2、K4 准备后分别用 0.0485、8.85、14.25 微秒/条，完整 TV 分别为 2.85 × 10⁻¹²、0.000942、0.00000990。首批 2,048 条含准备分别用 31.47、9.27、29.95 微秒/条。[完整点击对照](results/selected/96/click-validation.json) |
| 先生成光子数 | 16 模式成对光源加干涉，单线程 0.306 微秒/条，photonic-engines 为 653.85。两者完整 PNR TV 的名义 99% 区间分别为 0–4.93%、0–4.34%。[同输入光子数对照](results/selected/96/pnr-comparison.json) |

实际 1,152 个时间仓上，新方法校正后直接点击和光子数转点击的总数与九章实测分别相差 4.209、4.775 个百分点。[第 96 轮报告与运行示例](https://github.com/qiyuanhuakai/deziroGBS/blob/393d3592b8e8013be76844cca88605f428666d39/research/iteration96_checkpoint_v1.md) · [第 95 轮光纤环方法](docs/reports/95.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
