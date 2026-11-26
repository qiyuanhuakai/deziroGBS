# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 100 轮进展

| 输入 | 速度与精度 |
|---|---|
| 九章 3.0 双谱输入，1,152 个探测位 | 校正指定 16 位后，16 线程直接点击 0.988 微秒/条，光子数 1.198 微秒/条。总点击数 CDF 对硬件分别相差 4.082、4.084 个百分点。[百万条独立核验](results/selected/100/validation.json) |
| 同一对空间端口的全部 16 个探测位 | 单线程准备后，本方法 0.107 微秒/条，完整点击 TV 为 0.00630%。Frugal K2、K4 各为 14.60、47.12 微秒/条，TV 为 20.557%、4.186%。[三组同输入对照及准备成本](https://github.com/qiyuanhuakai/deziroGBS/blob/fc1500936471304e44476dc1469980eded219bcc/research/iteration100_checkpoint_v1.md) |

按空间端口模拟聚束后，基础模型在最亮留出组的冻结事件概率差从 21.026 降至 0.077 个百分点。另两个空间端口仍揭示整机点击、PNR 分布的 TV 分别至少为 **5.404%、5.575%**（校正模型，联合 99%）。[剩余误差定位](results/selected/100/residual.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
