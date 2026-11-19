# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 95 轮进展

| 路线 | 1152 端口上的结果 |
|---|---|
| 直接点击 | 16 线程批量均摊 0.446 微秒/条，单条调用 26.6 微秒。总点击数对九章实测的累计分布最大差为 4.209 个百分点，144 环中最大的八位点击 TV 对光学参考为 0.0179%。[精度与计时](results/selected/95/sampling.json) |
| 先生成光子数，再转点击 | 16 线程批量均摊 0.446 微秒/条，单条调用 27.3 微秒。转点击后总数对九章实测相差 4.612 个百分点。独立的四、八模式完整光子数分布 TV 分别约为 0.4093%、0.0846%。[完整分布核验](results/selected/95/optical-validation.json) |

[144 位同输入 Frugal 计时与精度](results/selected/95/frugal144.json) · [硬件计数复核](results/selected/95/frugal144-validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
