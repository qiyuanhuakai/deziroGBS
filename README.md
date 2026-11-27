# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 109 轮进展

实际九章 3.0 输入上，每批 16,384 条，GPU 生成完整 1,152 位点击 / 光子数记录分别用 **1.106 / 0.348 微秒/条**。同卡同批返回前八位时，直接法为 **1.335 微秒、经验 TV 0.633%**，Frugal K4 为 **4.112 微秒、1.317%**。GPU 光子数模型的整机总光子数分布 TV 为 **1.961%**。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/3a8651b49554a1d87ef371caa105918b5f0ca77c/results/iteration109_release_v1/accuracy.png) · [精选核验](results/selected/109/validation.json)

保留位移干涉后，前八位的模型 TV 从 1.1913% 降到 0.00478%。在另一组可解析的纠缠双光束输入上，两条路线扩到 131,073 个端口，平均每条记录含 93,804 个光子。[完整报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/3a8651b49554a1d87ef371caa105918b5f0ca77c/research/iteration109_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
