# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 112 轮进展

131,072 端口、1 弧度共享相移的光路上，直接点击 / 逐端口光子数的**整记录 TV 误差预算为 0.04249% / 0.32043%**，GPU 生成并回传完整记录耗时 **16.66 / 21.43 微秒**。对 458,789 个非均匀端口、两个不对易相移，也得到 0.14317% 的完整点击误差预算。[计算方法与对照](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/research/iteration112_checkpoint_v1.md)

九章 3.0 两份独立高功率文件共 **3,186,388 条**记录。仅用文件 1 训练后，在文件 10、11 上，10,296 个空间端口对的平均经验 TV 分别降至 **0.295% / 0.296%**（直接点击）与 **0.278% / 0.282%**（先生成光子数）。完整 1,152 位记录耗时 **1.145 / 0.433 微秒**。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/results/iteration112_release_v1/comparison.png) · [精选核验](results/selected/112/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
