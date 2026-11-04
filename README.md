# deziroGBS / GBS Classical Lab

量子项目的可复现经典 Gaussian Boson Sampling 研究。最近完成：**第六轮，2026-11-04**。

**第六轮改变条件采样律，检验遗漏点击的正事件补全。** 18 个固定输入上的 126 个完整分布评估、42 项独立表事件/原生交叉检查全部完成。相同 k 下，逐点补全 18/18 项降低 TV；两两补全 k4 为 16/18 项降低 TV。但与直接多保留两个点击相比，pair-k4 只在 3/18 项胜出，且两个低亮度输入有约 0.0011 的数值失败质量，已计入完整 TV。事后完整表诊断消除了这两项失败，但仍未超过 retained-k6。

本轮推导了精确 Gaussian 阈值概率下的支持保持性质及条件 logit 残差恒等式；尚无便宜的普适残差界，也没有固定全局 TV 的多项式复杂度结论。Bayes/森林分解属于已知方法思想，未宣称原创或超越完整九章。下一轮已转向补入跨分块相关性的条件依赖树。

先读 [第六轮研究报告](RESEARCH_REPORT_ITERATION6.md)、[完整结果表](results/iteration6_analysis/summary_tables.md)、[独立理论审查](research/iteration6_theory_audit.md) 和 [文献审查](research/iteration6_literature.md)。本轮模块的 18 项定向测试及两项枚举器测试通过；新的全套回归尚未在本轮报告中宣称完成。

![第六轮：补全信息与更强基线的完整TV比较](results/iteration6_analysis/completion_tv_ratios.png)

## 第五轮（历史）

**本轮新增保持候选采样律的有界前缀缓存，并得到一个大模式质量反例。** 216次正式方法运行全部完成；32组缓存开/关配对中24组更快，中位加速1.075倍。8/12模式的8192样本批次加速5.534–25.998倍；32–256模式的中位数只有1.010倍。缓存默认关闭，不把小系统收益外推为大规模普适加速。

预定四个小模式Frugal输入的补充完整分布枚举中，候选TV均较小，三个8192样本批次还同时有更低的原生准备加采样成本；12模式高亮度长批次仍由Frugal更快。固定质量批次的主D1为2组有条件接受、12组证据不足。保持原事件、样本和alpha的事后Clopper–Pearson精化，在256模式高亮度输入给出TV下界 **0.106672 > 0.1**，因此该输入不满足本轮目标。原主分析与D2报告没有改写。

九章一号作者Gaussian模型的官方Frugal K5新批次已完整保存1024条样本：准备848.715秒、生成149.121秒；平均点击41.153，目标41.042，候选39.864。两者仍未建立完整分布质量接受；第四轮的额外logQ审计超时记录保留。本轮最终 **203项测试通过，0失败、0错误、0跳过**。

先读 [第五轮研究报告](RESEARCH_REPORT_ITERATION5.md)、[完整结果表](results/iteration5_analysis/summary_tables.md) 和 [独立审查](research/iteration5_independent_audit.md)。当前方法是零均值Gaussian态的近似阈值采样器；不是完整PNR实现，没有战胜完整大型九章实验，原创性也尚未建立。

![第五轮：同一候选采样律的缓存速度](results/iteration5_analysis/prefix_cache_speed.png)

## 第五轮接口与复现

核心C++实现 `experiments/click_sampler.cpp` 新增 `--prefix-cache-entries N`，Python包装层对应 `prefix_cache_entries`；默认0。正式实验使用65536项、3 MiB前缀向量载荷，分配和所有未命中计入采样时间。该数字不是进程总内存。

性能配置为 `configs/iteration5_prefix_benchmark.json`，质量配置为 `configs/iteration5_quality.json`。入口分别是 `experiments/iteration5_benchmark.py` 与 `experiments/iteration5_quality.py`。质量抽样和评估分开；CP只读取原始产物并写入独立目录。旧结果和失败记录不覆盖。

构建、独立旧版参考、二进制门禁与复现命令见 [第五轮报告第7节](RESEARCH_REPORT_ITERATION5.md#7-验证源码与恢复)。重新编译后必须生成绑定新构建的验证记录；不能改写旧门禁中的散列来绕过检查。原生集成测试缺失对应程序时会跳过，归档的零跳过结果只属于实际验证过的环境。

## 第四轮（历史）

第四轮在8–256模式32个输入上完成192次配对性能运行，product核中位加速1.095倍。32组质量批次为10组有条件接受、22组证据不足；大型高亮度质量问题未解决。两个判别器分别检查指定Gaussian阈值分布，以及相对真实九章实例的输出、质量、噪声、协议和资源差距。

第四轮九章一号Frugal批次的额外logQ审计在总预算1800秒下超时，完整样本丢失；随后增加生成完成先保存checkpoint的v2接口。第五轮独立新批次实际使用这一接口，没有替换旧失败。详见 [第四轮报告](RESEARCH_REPORT_ITERATION4.md)、[历史结果表](results/iteration4_analysis/summary_tables.md)、[Frugal声明审计](research/iteration4_frugal_claims.md)。第四轮153项测试是当时的历史数字，不与第五轮203项相加。

## 第三轮原生C++对赛（历史）

两个 retained-click 选择器、Gaussian状态更新、概率内核、随机抽样和样本存储在第三轮迁入C++20。统一原生编译条件的官方Frugal对赛共有366次方法运行，0次失败，当时106项测试通过。修复了互信息舍入影响对照选集的问题；原始失败记录与旧结果保留。

第三轮k4选择器在27个10/12模式输入上，完整TV均低于官方Frugal K5；预处理加128样本短批在26/27组更快。不过8192样本小规模持续抽样中，候选抽样耗时约为Frugal的7–9倍。16/32/64模式只测成本，没有完整质量结论。这些历史输入此前参与探索，不能当作独立确认组。

详见 [原生C++报告](RESEARCH_REPORT_CPP.md)、[历史结果表](results/cpp_analysis/summary_tables.md) 和 [独立审查](research/cpp_independent_audit.md)。

## C++ 复现

构建、独立 CLI 采样和统一对赛命令见 [原生报告第 7 节](RESEARCH_REPORT_CPP.md#7-复现入口)。核心文件是 `experiments/click_sampler.cpp`，Python 包装层是 `src/gbslab/cpp_click.py`。本机版采用 `-O3 -march=native -ffp-contract=off`，未使用 fast-math；生成代码已含 SIMD，本轮没有手写汇编。移植使用浮点保护而非自动重试来处理不可靠条件概率。

当前 pairwise 采用修复后的稳定分数。要逐版本恢复旧 pairwise 结果，请使用历史源码快照或对应 Git 提交；直接用最新代码运行旧配置会使用新公式。归档保留首次原生试验的 13 个跨语言分布失败例，以及修复后的 27 个完整对照。

## 历史里程碑

第二轮完成了 63 个新输入上的 879 次完整分布方法评估，接入官方 FrugalGBS，增加条件点击影响选择器、Schur 内核、解析失败家族和无需完整枚举的 TV 统计验证。其报告 [RESEARCH_REPORT_PHASE2.md](RESEARCH_REPORT_PHASE2.md) 与 [结果表](results/phase2_analysis/summary_tables.md) 保留当时的 Python/C++ 实现差异；当时的性能结论已由本轮统一原生对赛推进。旧 pairwise 对照中的舍入敏感问题详见本轮报告。

第一轮报告 [RESEARCH_REPORT.md](RESEARCH_REPORT.md) 与数据保持原样，第一轮的 2,124 次模型评估、216 个输入和 60 次成本探测不计入第二轮数字。

## 第二轮复现

Python 近似采样器只需下文依赖；官方 FrugalGBS 对赛另需 C++20、OpenMP、Eigen。已附带固定提交的 FrugalGBS CPU 源码与 xsimd，原许可证一并保留。构建与验证方法见 [接入说明](research/frugal_integration.md)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m gbslab.phase2 --config configs/phase2_holdout.json --output results/new_holdout
python experiments/build_frugal.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m gbslab.frugal --validate --output .cache/frugal-build/validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/benchmark_frugal.py --output results/new_frugal_comparison
python experiments/analyze_phase2.py
```

对赛默认读取已归档的三组原始输入；要改变输入组，传 `--suites`。分析脚本默认总结已归档的正式结果。所有重复实验应使用新输出目录。

第二轮扩展模块包括 `conditional_memory.py`、`schur_memory.py`、`paired_stress.py`、`tv_validation.py` 和 `frugal.py`。全局统计上界明确依赖概率计算准确，不是经过向外舍入的数值证书。64 模式成本实验不提供完整分布质量结论。

## 快速运行

Python 3.10+；只依赖 NumPy、SciPy、Matplotlib。以下为 Bash/Linux 命令；本轮没有使用 GPU。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m gbslab.experiment --config configs/heldout.json --output results/my_heldout_run
python experiments/analyze.py
```

`experiments/analyze.py` 默认汇总仓库中已经归档的正式运行。要分析新运行，可以调整脚本中的输入目录；它不会把新结果覆盖为旧结果。

如果暂时不安装项目，可以在命令前设置 `PYTHONPATH=src`。标准库 `unittest` 足够，不依赖 pytest。

## 一个采样器示例

```python
import numpy as np
from gbslab.gaussian import haar_unitary, covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.zero_memory import ZeroMemorySampler
from gbslab.photon_tail import photon_number_tail

rng = np.random.default_rng(42)
m, r, eta, k = 32, 0.6, 0.25, 2
U = haar_unitary(m, rng)
V = covariance_from_interferometer(U, r, loss=eta)
oracle = ThresholdModel(V, max_exact_modes=3, vacuum_cache_size=8192)
sampler = ZeroMemorySampler(oracle, max_clicks=k, selection="pairwise")
samples = sampler.sample(1000, rng)  # shape (1000, 32), each entry 0/1
bound = photon_number_tail(np.full(m, r), eta, k)
print(samples.shape, bound, sampler.cache_info())
```

`bound` 是均匀损耗假设下、在精确算术中成立的保守 TV 上界的浮点估计，不是经过区间舍入认证的数值证书。它可能接近 1。能生成样本不代表已经证明样本质量。

## 第一轮完整复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python experiments/reproduce.py --output-root results/reproduction
```

脚本按顺序运行第一轮全部配置。输出目录若已有 `metrics.csv` 会拒绝覆盖；换一个目录即可。预处理和采样时间会因硬件、系统负载与缓存而变化；在同一数值环境中，概率误差指标应可复现。

小规模完整分布运行保存了当时的完整 `gbslab` 源码快照、源代码 SHA-256、参数、种子、NumPy/SciPy 版本、输入矩阵、精确参考分布、逐方法记录及失败记录。较大规模成本和 TV 统计验证运行保存实际使用的输入、样本和诊断，不含未计算的完整目标分布。也可直接通过 `PYTHONPATH=results/heldout_v1/source_snapshot` 运行冻结版本。

## 方法与范围

| 模块 | 用途 | 主要限制 |
|---|---|---|
| `exact.py` | 从 Gaussian 真空边缘经 Möbius 反演得到小规模完整阈值分布 | 随模式数指数增长；float64 行列式 |
| `fock.py` | 独立振幅递推、损耗与阈值测量校验 | 总光子截断；明确记录遗漏质量 |
| `surrogates.py` | 有界父集贝叶斯网络，按最近位置、互信息或条件互信息选取父模式 | 已知图模型基线，不是新算法 |
| `calibration.py` | 精确校准第一个点击的位置和无点击类别 | 保证前向 KL 不增加，不保证 TV 单调 |
| `zero_memory.py` | 保留全部零观测、保留 k 个点击约束 | 固定 k 仍有大矩阵行列式成本；选择策略不是普适最优 |
| `photon_tail.py` | 均匀损耗下可计算的光子数尾界 | 高计数时界很松；不支持一般非均匀输出损耗 |
| `cache.py` | 以配置和引擎代码哈希缓存精确参考 | 单写者；不允许多个进程同时写同一缓存目录 |

本轮研究的是**零均值 Gaussian 态的阈值点击分布**。不是完整 PNR 采样实现；没有模拟暗计数、模式失配、真实探测器时间复用或实验校准误差。`loss` 参数表示透过率 eta，不是丢失比例。

协方差约定为 hbar=2、真空 V=I、xxpp 顺序。分布数组下标是 little-endian bitmask，bit=1 表示对应模式至少一个光子到达。

## 目录与归档

- `src/gbslab/`：可运行实现。
- `tests/`：解析验证、独立 Fock 校验、支持与缓存检查。
- `configs/`：正式实验参数；预留种子与探索种子分开。
- `results/survey_v2/`、`adaptive_v2/`、`heldout_v1/`：正式完整分布比较。
- `results/scaling_v1/`：16/32/64 模式的成本探测，不包含完整 TV。
- `results/survey_v1/`、`adaptive_v1/`：初始探索记录，保留研究历史；不用于最终数值结论。
- `results/analysis/`：图、摘要和所有反例索引。
- `research/`：方法审计、查新记录和下一轮问题。
- `.cache/targets/`：可重建的精确参考缓存；不跟踪到 Git。历史独立归档可能保留缓存；第五轮恢复包只包含Git跟踪的研究成果。

未下载或复制论文全文。`third_party/FrugalGBS`（MIT）和 `third_party/xsimd`（BSD-3-Clause）保存用于可复现对赛的上游源码子集，文件均核对固定提交的 Git blob；参见 [上游说明](third_party/README.md)。本项目自有新增代码尚未选择对外开放源码许可证。

## GitHub 与历史恢复状态

目标远程地址为 **[qiyuanhuakai/deziroGBS](https://github.com/qiyuanhuakai/deziroGBS)**。本轮恢复了原四轮本地研究历史，并将 `37518d5` 与已知远端初始化提交 `52e38ae` 合并为本地 `ce73c2e`；第五轮继续在其后提交。Python包名仍为 `gbslab`。

**本轮尚未公开发布。** 自动审批审查拒绝了将具体研究源码、实验结果和历史推送至公开仓库的操作，要求用户明确批准这批内容和目的地。没有用其他接口绕过；因此不能把远程克隆描述为已经包含完整研究。第五轮交付包内的Git bundle可恢复全部历史及最终源码，确切提交和恢复命令见包内清单。

编译产物、计算缓存和作者原始协方差不包含在该恢复包或拟公开源码树中。九章模型复现需按来源说明独立取得协方差并校验散列。上游FrugalGBS和xsimd许可证保留；项目自有新增代码尚未选择对外开放源码许可证。
