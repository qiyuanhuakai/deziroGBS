# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 114 轮进展

1,048,576 个输出端口、两个受激光源的稠密有损光路上，8 线程 CPU 生成整条光子数记录需 **35.77 微秒**，整记录 TV 截断上界为 **1.265 × 10⁻⁹**。直接点击的高精度配置需 **92.50 微秒**，整记录 TV 投影上界为 **0.002764%**。[算法、速度与精度对照](https://github.com/qiyuanhuakai/deziroGBS/blob/bc0c7333f5fbcc2720370bc8a596a480d1ee0966/research/iteration114_checkpoint_v1.md)

九章 3.0 的高功率光学模型预测，每条记录中因同一通道多光子而合并的光子数平均为 **12.39**，标准差为 **6.35**。两份实测文件共 3,186,388 条，可核对全部 10,296 个空间端口对的触发数协方差。[对照图](https://github.com/qiyuanhuakai/deziroGBS/blob/bc0c7333f5fbcc2720370bc8a596a480d1ee0966/results/iteration114_release_v1/comparison.png) · [精选核验](results/selected/114/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
