# 第二轮定向查新与 FrugalGBS 对赛入口

核查日期：2026-11-04。范围是与本项目最近的精确顺序采样、条件 Gaussian 近似、低阶边缘近似及 elision；不是 GBS 文献的完整综述。所有文献结论以原文适用条件为准。

## 1. 已经存在的关键组成部分

| 一手来源 | 与本项目直接相关的已有内容 | 对新颖性的约束 |
|---|---|---|
| [Quesada, Arrazola, Killoran, *Gaussian Boson Sampling using threshold detectors* (2018)](https://arxiv.org/html/1807.01639v2) | 精确阈值顺序采样；未点击的 Gaussian 更新；点击使 Gaussian 分支数增长 | 不能把“未点击便宜、点击昂贵”或 Gaussian 零条件化本身当成新发现 |
| [Quesada & Arrazola, *Exact simulation of Gaussian Boson Sampling in polynomial space and exponential time* (2020)](https://arxiv.org/abs/1908.08068) | 用边缘和链式条件概率精确采样；覆盖多种 GBS 测量场景；提供 The Walrus 实现 | “链式采样＋部分边缘”不是新算法框架 |
| [Villalonga et al., *Efficient approximation of experimental Gaussian boson sampling*](https://arxiv.org/abs/2109.11525) | 利用低阶边缘构造经典替代分布，并研究实验检验表现 | 必须超出简单低阶匹配和已知代理模型，才能主张方法增量 |
| [Dodd et al., *A fast and frugal Gaussian Boson Sampling emulator*](https://arxiv.org/html/2511.14923v1) | 低阶累积量预计算、顺序近似边缘、double-elision；有官方 CPU/GPU 实现 | 这是当前最直接的实用近似对手，不能用我们自己的弱基线代替它 |
| [Lim, *Classical Simulation of Lossy Gaussian Boson Sampling beyond Square-Root Regime* (2026-10-05 预印本)](https://arxiv.org/html/2610.06526v1) | 精确演化未探测过程，在光子探测后用协方差匹配 Gaussian 态替换；论文给出规定假设下 O(Nη³) 的 TV 界 | “保留 Gaussian 条件态、近似非 Gaussian 事件”已有非常接近的研究；其 PNR 演化和误差证明不直接适用于我们的阈值约束舍弃 |

## 2. 本项目的区别应表述得多窄

本项目目前的具体规则是：在下一模式的条件概率中保留所有已出现的零约束，精确保留选中的至多 k 个点击约束，其余点击变量边缘化。选择可以按时间最近、无条件成对互信息，或者所有零条件化后的成对互信息。

FrugalGBS 的 elision 也使用缺失变量边缘，但其近似来自有限阶累积量及递推构造边缘。本项目则对选定部分事件作精确 Gaussian 真空包含排除，近似来自不再施加一部分已知点击约束。二者不能只凭“都遗漏变量”就认作完全相同，也不能只因名字不同就认作独创。

Lim 的方法保持探测演化的量子态描述并在探测后作 Gaussian 替换；本项目没有保持那个状态表示，也没有复现其误差分析。尤其不能把论文中的 N^(2/3) 可模拟规模或 O(Nη³) 界标到本项目上。

定向检索覆盖了 “Gaussian boson sampling” 与 conditional、truncated、finite memory、no-click、elision、marginals 等组合。此次没有定位到完全一致的“所有零约束＋自适应选择 k 个旧点击”算法，但这仅表示**未找到直接对应文献**，不构成原创性确认。下一步查新应围绕具体公式、Gaussian signed-mixture 剪枝及近似推断文献推进，不能只查算法名称。

独立双模压缩对的反例家族已有文献使用。`phase2_theory.md` 中对 recent-k 规则得到的 TV 公式，应标为对本项目规则的解析审计，而非新物理家族或 GBS 复杂度下界。

## 3. FrugalGBS：无需实验文件即可接入合成协方差

官方仓库：[tomdodd4598/FrugalGBS](https://github.com/tomdodd4598/FrugalGBS)。本节基于官方 README、构建配置、头文件及测试的直接阅读；本节作者未执行上游构建。团队随后进行的编译、适配与对赛，以独立构建日志和实验归档为准。

检查的核心文件包括：

- `include/fgbs/data.hpp`、`layout.hpp`、`cumulants.hpp`、`sampler.hpp`；
- `include/fgbs/core/correlator_utils.hpp`、`matrix_utils.hpp`；
- `include/fgbs/global.hpp.in`、根 `CMakeLists.txt`；
- `tests/test_vacuum.cpp`、`tests/test_sampler_frontend.cpp`。

### 3.1 约定和数值精度

| 项目 | 官方代码中确认的事实 | 对赛处理 |
|---|---|---|
| 协方差 | 真空测试使用 2N×2N 单位矩阵 | 与本项目 hbar=2、真空 I 一致 |
| 坐标排序 | correlator 提取模式 j 与 j+N | 与本项目 xxpp 一致 |
| 输入 | `generate_sampling_target` 直接接收 Eigen 协方差矩阵 | 可以生成新的合成输入，不必先取得九章数据 |
| 精度 | `FGBS_REAL_PRECISION=1` 默认单精度；2 为双精度 | 与 float64 参考对赛时明确使用 2 |
| 模式数 | N 是 C++ 模板参数 | 适配器需实例化实际比较的模式数 |
| 输出 | 上游 bitstring 容器 | 必须独立检查其位顺序与本项目 little-endian bitmask 的转换 |

接口链条如下；这是从官方 API 摘要得到的调用轮廓，不是已通过编译的完整程序：

```cpp
constexpr fgbs::usize N = /* modes */;
auto bits_result = fgbs::generate_bitstring_layout<N>(K);
auto cumulants_result = fgbs::generate_cumulant_layout<N>(K);
// Check .ok(), then keep both layout values alive.
auto target_result = fgbs::generate_sampling_target<N>(
    bits_result.value(), cumulants_result.value(), covariance);
// Check .ok() before dereferencing.
auto samples_result = fgbs::generate_samples<N>(
    bits_result.value(), target_result.value(), sample_count, seed);
```

这些函数返回带 `.ok()`、`.value()`、`.error()` 的结果类型。`test_vacuum.cpp` 使用 N=100、K=1、32 个样本验证整条链；一个最小合成试验可以先验证真空、独立热态与已知双模相关的输出，之后再比较随机干涉器。

### 3.2 构建和可复现记录

官方构建要求 CMake≥3.18、C++20、Eigen≥3.4、OpenMP；README 还列出 xsimd。`FGBS_ENABLE_CUDA=OFF` 可使用 CPU 路线；`BUILD_TESTING=OFF` 可避免 Catch2 测试依赖。CPU CMake 目标为 `FrugalGBS::FrugalGBS_CPU`。默认 `FGBS_MAX_CUTOFF_ORDER` 为 5。

本地常规 `/usr/include/eigen3` 位置缺失并不代表环境没有 Eigen，团队已发现其他软件包附带的副本。缺少 CMake 也不自动阻止直接编译一个 CPU 适配器。需要归档实际选用的头文件版本、编译器、优化参数、OpenMP 线程数、精度宏和运行时版本，而不能仅记录 README 的推荐配置。

引用和再分发须保留上游 `LICENSE.md`：官方仓库当前使用 MIT 许可证。适配器若加入追踪每个条件概率的 instrumentation，应归档精确补丁并分别验证：原始采样路径输出、追踪路径输出、概率归一化和异常处理。不能把追踪修改后的程序无说明地称为原样官方二进制。

同条件比较至少分别列出预处理时间、采样时间、内存、全分布 TV（小规模）、支持问题及失败数。FrugalGBS 论文的实验基准表现不等于对所有输入的 TV 保证；我们的小规模 TV 优势也不等于在大型实验采样速率上获胜。

## 4. 本轮审计后的判断

当前最有研究价值的不是重复一般“量子优势可被经典挑战”的论点，而是三件具体工作：与官方 FrugalGBS 同输入对赛；检验选择器与测量排序的失效区域；尝试把约束舍弃的误差诊断做成有用的、可运行的保守界。

我们已有一个能解释为什么简单最近点击策略失败的严格公式，也有与其对应的物理测试。它提高了研究的可证伪性，但不意味着已经超过 FrugalGBS 或量子实验。算法原创性和性能领先仍是待验证问题。
