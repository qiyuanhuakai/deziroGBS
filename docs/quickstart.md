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

| 想看什么 | 第 111 轮的实现 |
|---|---|
| 如何保留跨单元相位干涉并返回各端口的点击或光子数？ | [C++ 双读出采样](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/native/shared_measure111_v1.cpp) · [光学输入与相干分支](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/experiments/iteration111_memory_v1.py) |
| 如何按九章空间端口的实测分布校准记录？ | [CUDA 空间窗口校正](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/native/spatial_windows111_v1.cu) |
| 如何在同一组输入上测速度和精度？ | [同样本量对照](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/experiments/iteration111_matched_v1.py) · [核验程序](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/experiments/verify_iteration111_v1.py) |

复现实验时，使用同轮归档中的计算核心、脚本和输入文件。[第 111 轮运行方法](https://github.com/qiyuanhuakai/deziroGBS/blob/c0c82e81283460ece8147c6360c0a38c61325e74/research/iteration111_checkpoint_v1.md) · [按轮取回数据](archive.md)
