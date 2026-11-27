# deziroGBS

用经典计算机模拟高斯玻色采样（GBS）。一条路线生成每个端口的光子数（PNR），另一条直接生成点击。C++ 执行采样与矩阵计算，Python 编排实验和复核结果。

[运行四端口示例](docs/quickstart.md) · [采样核心与实验入口](docs/quickstart.md#采样核心与实验入口) · [历轮研究报告](docs/reports/README.md) · [精选结果](results/README.md) · [取回原始数据](docs/archive.md)

## 第 113 轮进展

同一条 13 端口有损光路上，整条光子数记录的 TV 截断上界从 **12.5895% 降到 0.69823%**。准备需 **9.876 秒**，之后每条耗时 **0.0740 微秒**。直接点击覆盖全部 8,192 种结果，与独立光学计算的 TV 差为 **5.95 × 10⁻¹²**。[速度、精度与对照方法](https://github.com/qiyuanhuakai/deziroGBS/blob/e9d19cb0501e2ddb03cde06ea7fb293fbc50cf6c/research/iteration113_checkpoint_v1.md)

九章 3.0 的每个空间端口用八路点击估计光子数。高功率光学模型预测，每条记录平均有 **116.286 个光子到达、103.896 路触发**。两份共 3,186,388 条实测记录平均触发 **106.101 / 106.044 路**。[144 个空间端口的核验](results/selected/113/validation.json)

**统计勘误**　第 83 轮旧 PNR 样本的随机流发生重叠，原有置信保证已撤回。第 84 轮使用分离的随机流重新采样。[随机流勘误](docs/corrections/83-rng.md) · [第 86 轮直接点击舍入勘误](docs/corrections/86-rounding.md)
