# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 95 轮进展

| 路线 | 1152 端口上的结果 |
|---|---|
| 直接点击 | 16 线程批量均摊 0.490 微秒/条，单条调用 8.8 微秒。总点击数对九章实测的累计分布最大差为 4.224 个百分点；三组八端口的完整局部 TV 为 0.073–0.096%。[精度与计时](https://github.com/qiyuanhuakai/deziroGBS/blob/cc7f15865d89444bb15a12fec9776cdaeafc5218/research/iteration95_checkpoint_v6.md) |
| 先生成光子数，再转点击 | 16 线程批量均摊 1.025 微秒/条，单条调用 15.3 微秒。总光子数对光学参考的累计分布最大差为 0.01105 个百分点；转成点击后，总点击数对九章实测相差 4.756 个百分点。[计数校正与硬件对照](https://github.com/qiyuanhuakai/deziroGBS/blob/cc7f15865d89444bb15a12fec9776cdaeafc5218/research/iteration95_checkpoint_v6.md) |

[Frugal 与 photonic-engines 同输入比较](https://github.com/qiyuanhuakai/deziroGBS/blob/cc7f15865d89444bb15a12fec9776cdaeafc5218/research/iteration95_checkpoint_v6.md) · [采样与校正的 C++ 实现](https://github.com/qiyuanhuakai/deziroGBS/blob/cc7f15865d89444bb15a12fec9776cdaeafc5218/native/gaussian_count_filter95_v1.cpp) · [144 个光纤环的误差分解](https://github.com/qiyuanhuakai/deziroGBS/blob/cc7f15865d89444bb15a12fec9776cdaeafc5218/results/iteration95_spatial_optical_v1/summary.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
