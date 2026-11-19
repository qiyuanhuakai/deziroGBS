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
