# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 95 轮进展

| 路线 | 1152 端口上的结果 |
|---|---|
| 直接点击 | 三阶窗口模型与九章 3.0 实测总点击数的累计分布最大差为 4.21 个百分点。16 线程批量均摊 0.436 微秒/条，单条调用 16.5 微秒。[精度与计时](https://github.com/qiyuanhuakai/deziroGBS/blob/905a652075d1ae0382d1ae086071c58cc364aba9/research/iteration95_checkpoint_v4.md) |
| 先生成光子数，再转点击 | 全秩高斯 copula 单线程批量生成光子数耗时 70.46 微秒/条。总光子数累计分布对光学参考的最大差为 1.04 个百分点；转成点击后，总点击数对九章实测的最大差为 4.79 个百分点。[样本、误差区间与两端口检验](https://github.com/qiyuanhuakai/deziroGBS/blob/9972467c5c19d39b77760273bd8c09b3f243667a/research/iteration95_checkpoint_v5.md) |

[Frugal 同输入速度与精度比较](https://github.com/qiyuanhuakai/deziroGBS/blob/905a652075d1ae0382d1ae086071c58cc364aba9/research/iteration95_checkpoint_v4.md) · [PNR 的 C++ 实现](native/gaussian_mps_core94_v2.cpp) · [直接点击的 C++ 实现](native/gaussian_click_branch94_v1.cpp)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
