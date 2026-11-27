# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 122 轮进展

九章 3.0 的三档公开光学输入都完成了两种读出的整机误差验证。低、中、高功率的完整记录 TV 上界分别为 **0.2276%、0.3628%、0.4815%**。高功率光子数采样与直接点击实测为 **9.18、11.93 秒/条**。[三档速度与精度](results/selected/122/loss-families.json) · [同输入正 P 对照](results/selected/121/full-width-comparison.json)

第 10 版给出了适用于任意输出宽度和有限压缩强度的损耗定理。双谱各 50 个压缩输入、最大强度透射率至多 0.51 时，每谱保留 15 对残余光子，完整记录 TV 至多 **0.312211%**。这条理论界与三档实测采用不同算法。论文另附强损耗合成电路的两组 4,096 条双路线实验。[预印本 PDF](https://github.com/qiyuanhuakai/deziroGBS/blob/be9e847ece311034827295e493239e7bfb685fe4/papers/number-tilted-gbs/manuscript.pdf) · [证明、数据与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/be9e847ece311034827295e493239e7bfb685fe4/papers/number-tilted-gbs/README.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
