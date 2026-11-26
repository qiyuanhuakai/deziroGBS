# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 106 轮进展

九章 3.0 高功率输入、1,152 位完整输出，在 RTX 5080 Laptop 上每批 16,384 条，**直接点击 0.285 微秒/条，逐位光子数 0.360 微秒/条**。前 16 位联合点击的总变差距离（TV）从逐源模型的 10.598% 降至 **0.235%**，实测面板中最大为 **1.220%**。[速度与精度对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/f20a5019741376f56eda206d7bfd2ecb684aed6b/results/iteration106_figure_v1/speed_accuracy.png) · [精选核验](results/selected/106/validation.json)

Frugal K4 在三组随机分散面板上的 TV 为 **0.00075–0.00326%**，本轮光场模型为 0.1115–0.2246%。另一条保留量子态的路线已在两个双光束源上测到 **131,072 端口**。[完整报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/f20a5019741376f56eda206d7bfd2ecb684aed6b/research/iteration106_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
