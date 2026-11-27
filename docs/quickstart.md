# 生成一批四端口点击记录

使用 Python 3.10 或更新版本。下面的命令只下载主分支的最新快照。

```sh
git clone --depth 1 --branch main https://github.com/qiyuanhuakai/deziroGBS.git
cd deziroGBS
```

在仓库根目录运行 `python -m pip install -e .`，再用下面的 Python 示例生成 32 条记录。每行对应一次采样，每列对应一个端口，1 表示探测到点击。

```python
import numpy as np
from gbslab.gaussian import haar_unitary, covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.zero_memory import ZeroMemorySampler

rng = np.random.default_rng(42)
optics = haar_unitary(4, rng)
covariance = covariance_from_interferometer(optics, 0.6, loss=0.5)
model = ThresholdModel(covariance, max_exact_modes=4)
sampler = ZeroMemorySampler(model, max_clicks=2, selection="pairwise")
clicks = sampler.sample(32, rng)
print(clicks.shape)  # (32, 4)
print(clicks[:4])
```

这个示例调用 Python 包中的阈值采样器。复现第 94 轮 C++ 核心的性能与精度比较，需要该轮的完整输入、构建产物和原始样本。[第 94 轮报告](reports/76-94/94.md) · [取回实验环境与数据](archive.md)

## 采样核心与实验入口

| 想看什么 | 实现与实验 |
|---|---|
| 如何从协方差直接求点击和光子数，复核九章的八路读出？ | [C++ 计算与采样](https://github.com/qiyuanhuakai/deziroGBS/blob/e9d19cb0501e2ddb03cde06ea7fb293fbc50cf6c/native/sector_support113_v1.cpp) · [光子数与点击实验](https://github.com/qiyuanhuakai/deziroGBS/blob/e9d19cb0501e2ddb03cde06ea7fb293fbc50cf6c/experiments/iteration113_support_v1.py) · [九章空间端口](https://github.com/qiyuanhuakai/deziroGBS/blob/e9d19cb0501e2ddb03cde06ea7fb293fbc50cf6c/experiments/iteration113_device_v1.py) |
| 如何从相关单元直接抽点击或逐端口光子数？ | [C++ 条件读出](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/native/coarse_readout112_v1.cpp) · [相干分支拟合](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/experiments/iteration112_positive_v1.py) |
| 如何在 CPU、GPU 上生成几十万端口的完整记录并核对误差？ | [整记录采样](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/experiments/iteration112_emit_v1.py) · [整记录误差计算](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/experiments/iteration112_marginals_v1.py) |
| 九章独立高功率文件如何取回并核验？ | [提取与校验](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/experiments/iteration112_hardware_files_v1.py) · [跨文件对照](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/experiments/iteration112_hardware_v1.py) |

复现完整实验时，按报告取回固定快照和相应的外部输入。[第 113 轮运行方法](https://github.com/qiyuanhuakai/deziroGBS/blob/e9d19cb0501e2ddb03cde06ea7fb293fbc50cf6c/research/iteration113_checkpoint_v1.md) · [第 112 轮运行方法](https://github.com/qiyuanhuakai/deziroGBS/blob/cad142bb09f9348730613eee94446b057bcc7467/research/iteration112_checkpoint_v1.md) · [归档目录](archive.md)
