# deziroGBS

研究高斯玻色采样（GBS）的两条经典计算路线：逐端口生成光子数（PNR），以及直接生成点击。采样、矩阵分解和压缩由 C++ 执行，Python 负责实验编排与独立核验。

**[最新报告 · 第 94 轮](docs/reports/76-94/94.md)**　·　[全部 94 轮](docs/reports/README.md)　·　[精选结果](results/README.md)　·　[原始数据与复现](docs/archive.md)

## 目前做到哪里

| 任务 | 同输入的观测结果 |
|---|---|
| 四模式，生成光子数 | PNR 与 photonic-engines 的完整分布 TV 区间分别为 0–3.4055% 和 0–3.3247%。同一 CPU 单线程下，PNR 用时是其 6.81 倍。 |
| 四模式，直接生成点击 | 新方法 K64 的完整点击 TV 上界为 1.3737%，Frugal K2 下界为 3.6693%，Frugal K2 快 114.93 倍。Frugal K4 的区间与新方法重叠，快 111.49 倍。 |
| 1152 输出，双谱模型 | PNR 完成 53/64 次，每次尝试平均 9.14 秒。直接点击 K8 的 64 次尝试全部超出误差预算。 |

TV 为总变差距离。表中区间属于预定的名义联合 99% 置信族，条件是样本独立同分布且采用固定数值参考。九章 3.0 论文报告每条记录 1.27 微秒，与上表 PNR 的原始耗时相差约 719 万倍，两者的硬件和输出质量不同。[完整比较](docs/reports/76-94/94.md)

## 阅读与运行

- **找算法**　[PNR 的 C++ 核心](native/gaussian_mps_core94_v2.cpp) · [直接点击的 C++ 核心](native/gaussian_click_branch94_v1.cpp) · [Python 包](src/gbslab/)
- **运行入门实验**　使用 Python 3.10 或更新版本，在仓库根目录执行以下命令。

```sh
python -m pip install -e .
python -m gbslab.experiment --help
```

- **复核结论**　[第 94 轮核验](results/iteration94_verification_v1/validation.json) · [取得原始记录](docs/archive.md) · [下一轮研究方向](research/iteration95_plan.md)
- **查阅勘误**　第 83 轮旧 PNR 样本曾复用随机流，相关置信保证已撤回，第 84 轮已重新采样。第 86 轮另有直接点击舍入更正。[随机流勘误](docs/corrections/83-rng.md) · [舍入勘误](docs/corrections/86-rounding.md)
