# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 119 轮进展

对九章 3.0 已公开的高功率光学标定，光子数和直接点击两条路线的 **1,152 端口完整联合分布 TV 数值上界均为 0.4824%**。每条路线输入 64 条独立候选，分别接受 16 条和 17 条。两个阶段的实测耗时合计折合 60.11 秒/条；光子数和点击的分路线小批计时见精选结果。[第 7 版预印本](https://github.com/qiyuanhuakai/deziroGBS/blob/d467fa25f2e5277e78bfe1cd1a9d2d14f925f343/papers/number-tilted-gbs/manuscript.pdf) · [精选结果](results/selected/119/full-record.json)

在预先固定的八组探测器事件上，两份留出硬件文件到同一标定的完整点击 TV 下界分别为 3.8996% 和 3.9478%。同输入 FrugalGBS 的单线程 CPU、K=2 配置，其完整点击 TV 下界为 10.7710%。[硬件检验与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/d467fa25f2e5277e78bfe1cd1a9d2d14f925f343/papers/number-tilted-gbs/README.md) · [归档校验清单](https://github.com/qiyuanhuakai/deziroGBS/blob/d467fa25f2e5277e78bfe1cd1a9d2d14f925f343/results/iteration119_weights/manifest.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
