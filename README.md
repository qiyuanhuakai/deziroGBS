# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 116 轮进展

在九章 3.0 原功率光学模型上，任意调整旧模型的总光子数权重，完整 PNR 记录的 TV 误差仍至少约 **2.0153%**。加入八组探测器的空、非空信息后，新模型在「总光子数和八组占用」联合分布上的 TV 为 **0.2741%–0.2747%**。完整 1,152 通道分布的精度仍待确定。[修订论文 PDF](https://github.com/qiyuanhuakai/deziroGBS/blob/c679bf9a5f1bcdf06347d9d9b31bb48141d989f5/papers/number-tilted-gbs/manuscript.pdf) · [速度与精度](results/selected/116/spatial-filter.json)

每批 4,096 条候选记录时，新模型生成逐端口光子数和直接点击分别耗时 **1.133、2.078 微秒/条接受记录**。16 通道压力面板的完整点击 TV 从 2.1221% 降至 1.1310%，两个未参与拟合的面板一好一差。[复现步骤](https://github.com/qiyuanhuakai/deziroGBS/blob/c679bf9a5f1bcdf06347d9d9b31bb48141d989f5/papers/number-tilted-gbs/README.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
