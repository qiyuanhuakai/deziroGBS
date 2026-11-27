# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 110 轮进展

实际九章 3.0 双谱输入上，校正整机总光子数与八位点击字后，262,144 条光子数记录的总数 CDF 最大误差为 **0.124 个百分点**。另用前半段硬件数据校准点击重数，后半段留出检验中，直接点击 / 光子数路线的总点击数 CDF 最大差为 **0.162 / 0.229 个百分点**，每条完整记录耗时 **1.132 / 0.402 微秒**。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/dc28b1df9803c431009e64b0ad2abfbec8402d0d/results/iteration110_release_v1/comparison.png) · [精选核验](results/selected/110/validation.json)

四模干涉单元接 16 位探测器时，同卡完整点击采样为 **0.213 微秒/条**，Frugal K4 为 **7.626 微秒/条**；各抽 262,144 条的完整分布经验 TV 为 **3.476% / 4.458%**。报告另给出 131,072 端口的完整误差预算，以及强跨单元干涉下与 Frugal 的差距。[完整报告与复现](https://github.com/qiyuanhuakai/deziroGBS/blob/dc28b1df9803c431009e64b0ad2abfbec8402d0d/research/iteration110_checkpoint_v1.md)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
