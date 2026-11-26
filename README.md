# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 104 轮进展

在九章 3.0 的 1,152 个探测位上，按光学输入校正后，RTX 5080 Laptop 每条点击用时 **0.437 微秒**，每条光子数记录用时 **0.462 微秒**；总光子数累计分布误差降至 **0.08–0.10 个百分点**。按硬件校准后，两条路线分别为 0.956、1.002 微秒/条，对留出硬件记录的总点击数误差为 0.21–0.23 个百分点。[速度、精度与测量条件](results/selected/104/validation.json)

名义光学输入与硬件总体的总点击数分布至少相差 **3.16 个百分点**（99% 置信，假设硬件记录独立同分布）。另一个每谱仅占用一个受压缩空间模式的算法已测到 16,384 端口。[研究报告与复现命令](https://github.com/qiyuanhuakai/deziroGBS/blob/0ec773c88e9efe621db5e455c8b95eb67d6806b6/research/iteration104_checkpoint_v1.md) · [同 GPU 的 Frugal 速度与精度](https://github.com/qiyuanhuakai/deziroGBS/blob/762e3cacdd917d2c0ce59bf756ed79af523cdfa6/research/iteration104_device_comparison_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
