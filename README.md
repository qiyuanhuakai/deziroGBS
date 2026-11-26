# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 105 轮进展

九章 3.0 高功率输入、1,152 位完整点击、同机单线程 CPU，每批 4 条。计入初始化，本轮模型约 **0.0796 秒/条**，Frugal K2 实测 **1,513.67 秒/条**。对照完整前 16 位的联合点击分布，两者相对光学参考的总变差距离（TV）分别为 **10.598% 和 10.771%**。[速度、精度与运行条件](results/selected/105/validation.json)

保留每对光源内部干涉后，总光子数分布的 TV 为 **0.079%**。计算显示，每组最多 8 个光源联合干涉时，前 16 位点击 TV 降至 **5.325%**。相应多源组采样器是下一步实现目标。[本轮研究与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/57fdc29f6b5abe7c75058d5d546eba45dc4482f0/research/iteration105_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
