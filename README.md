# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 115 轮进展

九章 3.0 原功率的 1,152 通道光路上，两种读出使用同一模型。总光子数分布的 TV 误差上界从 **2.74083% 降至 0.0008379%**。强相关的两个空间端口仍有 **2.1221%** 的触发数联合分布误差。[方法与精度对照](https://github.com/qiyuanhuakai/deziroGBS/blob/6bee5fc91af41fc0fb4a84abf0e4d983d7e3a6b1/research/iteration115_checkpoint_v1.md)

每批 16,384 条时，GPU 直接生成点击耗时 **1.056 微秒/条**，逐通道生成光子数耗时 **0.346 微秒/条**。与九章两份实测文件相比，直接采样的总点击数直方图 TV 为 **4.11%、4.02%**。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/6bee5fc91af41fc0fb4a84abf0e4d983d7e3a6b1/results/iteration115_release_v1/comparison.png) · [精选核验](results/selected/115/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
