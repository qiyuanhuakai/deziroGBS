# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 97 轮进展

| 输入 | 速度与精度 |
|---|---|
| 1,152 端口连通链，单元间耦合角 0.08 弧度 | 准备后 16 线程，直接点击 0.297 微秒/条，光子数 0.238 微秒/条；两条路线的完整分布 TV 均低于 0.0002460%。[整串误差与计时](results/selected/97/forest.json) |
| 九章约化 16 位点击 | 同一 CPU 单线程，首批 2,048 条含准备，本方法 4.16 微秒/条，Frugal K2、K4 分别为 9.03、32.41。完整 TV 分别为 2.98 × 10⁻¹¹、0.000942、0.00000990。[首批复测](results/selected/97/validation.json) |

九章真实 1,152 位输入上，本轮位移场模型的总点击数与硬件实测相差 4.23 个百分点。完整连通链的误差界依赖单元间的连接结构。[第 97 轮报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/d99c25e2cdb0f4f0c24dcc53f7a6efb7963fde9f/research/iteration97_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
