# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 107 轮进展

九章 3.0 高功率输入、1,152 位完整输出，每批 16,384 条时，**直接点击 0.267 微秒/条，逐位光子数 0.329 微秒/条**。同一 GPU 上比较前 8 位，直接点击与 Frugal K4 分别为 0.392、4.112 微秒/条，等样本经验 TV 为 0.599%、1.317%。[速度与精度核验](results/selected/107/validation.json)

4,096 端口、64 对源的跨块八位模型 TV 从 4.711% 降至 **0.00990%–0.01764%**。在 131,072 端口、两个源对上，全零事件仍给出完整点击 TV 至少 8.12% 的统计下界；同输入的量子态路线已有 0.0352% 的完整 TV 上界。[报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/3050dc1482526215675d247b3df11bcb7eca34c5/research/iteration107_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
