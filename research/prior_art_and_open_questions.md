# GBS 探索：相关方法、事件记忆变体与待验证问题

检索与审计日期：2026-11-03。这里记录的是有针对性的先行工作检索，不是穷尽式查新。数值结果应以 `results/` 中的配置、种子和记录为准；本文不宣称已经战胜大型量子实验。

## 1. 当前原型在整个领域中的位置

| 路线 | 已有基础与主要来源 | 本仓库状态 | 值得继续追问的问题 |
| --- | --- | --- | --- |
| 低阶边缘分布、累积量与归一化代理 | Villalonga 等 [R2] 已研究低阶边缘匹配；FrugalGBS [R3] 用累积量截断和 double-elision 动态边缘表 | `surrogates.py` 实现有界父集贝叶斯网络；`calibration.py` 实现首个点击类别校准；尚未实现或公平对比 FrugalGBS | 在同一误差与预处理成本下，怎样分配高阶条件信息，比固定阶数更划算？ |
| 保留无点击历史，限制点击历史 | 精确阈值采样 [R1] 已明确无点击保持 Gaussian、点击使表示扩张；链式条件采样 [R4] 也是已知基础 | `zero_memory.py` 是本轮事件记忆原型；本文给出其适用条件和初步查新判断 | 保留哪些点击比“最近 k 个”更有效？能否获得比总点击尾界明显更紧、又可计算的实例误差界？ |
| 噪声分解与张量网络 | Oh 等 [R5] 将输出分为较小的纯 Gaussian 核心与经典位移噪声；GSVD–PCO [R6] 改善低纠缠 Gaussian 到 MPS 的转换 | 文献路线，未在本仓库实现 | 误差如何经过位移噪声和平滑测量传播？能否在相同精度下减少完整构造和采样的总成本？ |
| 检测后的 Gaussian 重置 | Lim 的 2026 年预印本 [R7] 在检测间精确演化、检测后匹配协方差，给出高损耗误差与复杂度分析 | 未实现；理论需独立核查，不把预印本结论视为本项目证明 | 重置误差的实例依赖控制、非 Gaussian 残差保留，以及位移噪声变体的严谨界 |
| 非负图上的 Glauber/MCMC | Zhang 等 [R8] 对一类稠密图给出多项式混合分析；加权转移规则已经写在补充材料中 | 未实现；是不同于随机复干涉矩阵的结构化目标 | 给出新的、非平凡加权图类的混合保证，或识别并解决现有链的具体瓶颈 |
| 相空间近似与样本校准 | Goodman 等 2026 年预印本 [R9] 研究大规模近似输出与统计基准 | 未实现；应作为后续大规模比较对象 | 哪些误差只体现在已测的统计量之外？怎样把基准表现连接到更强的分布误差判断？ |
| 精确概率、低秩/稀疏 hafnian | [R4] 及 The Walrus 的算法说明 [R10] 已有精确链式采样、低秩与稀疏算法 | 当前仅有小规模阈值行列式基准和独立 Fock 基准；没有新 hafnian 算法 | 真实损耗后的有效秩、特定结构和重复行列能否在完整采样中带来优势，而不仅是单个概率计算？ |

这些路线的误差对象不同。改善少数统计量、改善小子系统 TV、逼近完整阈值分布、逼近 PNR 分布、加速张量构造和加速完整采样，都应分别陈述。

## 2. 事件记忆原型：精确定义

按一个固定模式顺序生成点击位。已经生成的无点击位置组成 Z；从已出现的点击位置中保留至多 k 个，组成 C。被遗忘的旧点击变量在下一步被边缘化。原型支持最近 k 个点击，或者与当前模式无条件阈值互信息最大的 k 个点击。后者每一步可以重新选择；这不保证优于最近点击，也可能在严格支持受限的态上遇到无定义条件事件，程序会明确报错。

定义精确的部分事件概率

\[
w(Z,C)=P(X_Z=0,X_C=1)
      =\sum_{T\subseteq C}(-1)^{|T|}q(Z\cup T),
\]

其中 \(q(S)=P(X_S=0)\) 是 Gaussian 真空边缘概率。下一模式 i 的条件概率使用

\[
Q(X_i=0\mid\text{生成历史})=
\frac{w(Z\cup\{i\},C)}{w(Z,C)}.
\]

这是一个有效条件概率，来自一个明确的、部分已观测事件。它不等于实际完整历史下的条件概率，除非没有遗忘任何点击，或者被遗忘变量在该实例中不影响后续分布。

每次概率计算的容斥项数由保留的点击数控制，为至多 2^k；全部无点击位置仍保留在 Gaussian 行列式或条件协方差中。直接每步从头计算大行列式的朴素实现仍有较大的模式数多项式成本。利用 Gaussian 投影的 Schur 补更新可以改善这一部分，但这类更新本身已经是先行工作的基础技术。跨样本缓存也必须有内存预算，不能把任意多历史对应的大表视为免费资源。

### 与已知方法的关系

[R1] 第 V 节的精确阈值算法已利用同一种物理不对称：无点击后的条件态仍为 Gaussian，而点击会增加非 Gaussian 表示的复杂度。因此，“无点击容易、点击困难”不是这里的新发现。

[R3] 的 double-elision 用截断累积量和动态规划近似含缺失变量的边缘分布。这里的原型直接计算保留事件的精确 Gaussian 概率，然后遗忘一部分曾经点击的约束。两者都近似链式条件概率，但近似对象和数据结构不同。

本轮检索在 [R1]、[R3]、[R4] 及低阶代理文献 [R2] 中，未定位到与“保留所有无点击，只保留最近 k 个点击”完全一致的规则。**这只支持把它作为待查新的探索变体，不能支持“首创算法”的判断。** 后续查新应覆盖有界记忆过滤、signed-Gaussian 表示截断、检测历史截断和相关软件实现；仅凭名称搜索不足以排除已有等价方法。

## 3. 一个可核查、但通常较松的误差界

以下论证是通用的截断/耦合原理在该原型上的应用，不作为新的复杂性定理。

假设所有可达保留事件的条件概率都有定义并精确计算，算法在累计点击数不超过 k 时从不丢弃点击历史。对于任意总点击数至多 k 的完整位串 x，沿它的整条生成路径都没有遗忘点击，因此每步条件概率与目标一致：

\[
C(x)\le k\quad\Longrightarrow\quad Q(x)=P(x).
\]

于是 P 与 Q 在其余位串集合上也有相同的总质量，直接得到

\[
\operatorname{TV}(P,Q)
\le P(C\ge k+1).
\]

也可以把两个采样器耦合到第一次产生第 k+1 个点击为止，得到同一个上界。它不依赖“最近”这一选择方式，只要求在不超过预算时没有提前遗忘。

对于高点击数实例，右端可能几乎为 1，因而不能证明近似有用。小规模精确计算可以提供真实点击尾概率；在大规模实例上，不能先用完整枚举取得该尾概率，再声称误差认证可扩展。浮点计算的误差也不包含在上述精确算术论证中，必须另外诊断或界定。

此外，每个 k 都精确保留**首个点击的位置分布**，包括完全无点击的类别，因为首次点击之前的所有条件概率都是精确的。这个性质并不意味着完整分布精确。

### 用光子数尾概率取得可计算的保守界

阈值点击数 C 不超过探测到的总光子数 N，故

\[
P(C\ge k+1)\le P(N\ge k+1).
\]

若输入为相互独立的单模压缩真空、干涉仪被动、所有输出模式具有相同透射率 eta，均匀损耗可交换到干涉仪之前，总光子数的概率生成函数为

\[
G(z)=\prod_j
\frac{\operatorname{sech}(r_j)}{
\sqrt{1-\tanh^2(r_j)(1-\eta+\eta z)^2}}.
\]

这是单模压缩真空光子分布的生成函数，经独立二项损耗替换 z -> 1-eta+eta*z，再对独立输入取乘积所得。它不适用于任意非均匀输出损耗；不能把不同输出 eta 简单逐项放入输入 r 的乘积。

只需取得 G 的 0 到 k 次系数，就可计算 \(1-\sum_{n=0}^k[z^n]G(z)\)。单模系数可以用非负递推求得。令

\[
t=\tanh r,\quad
a=1-t^2(1-\eta)^2,\quad
b=2t^2\eta(1-\eta),\quad c=t^2\eta^2.
\]

对 \(g(z)=\operatorname{sech}(r)(a-bz-cz^2)^{-1/2}\)，有

\[
g_{-1}=0,\qquad g_0=\operatorname{sech}(r)/\sqrt a,
\]

\[
g_{n+1}=\frac{b(n+\tfrac12)g_n+c n g_{n-1}}{a(n+1)}.
\]

将各单模多项式卷积，并在每一步只保留 0 到 k 次项即可。这个界的好处是与干涉仪具体 U 无关、可用少量系数计算；代价是忽略碰撞，可能比真实点击尾界松得多。有限精度的 `1 - partial_sum` 仍须注意消去误差；严格数值上界需要区间或定向舍入等额外措施。

## 4. 首个点击类别校准的保证与限制

对任意代理分布 Q，令 F 表示按采样顺序的首个点击位置，另设无点击类别。其目标质量可以从连续前缀的真空概率之差精确求出，不需要完整枚举。设 Q_F 在 P_F 为正的类别上也为正，定义

\[
Q^*(x)=P_F(F(x))\,Q(x\mid F(x)).
\]

这是已知的固定分区质量校准/信息投影构造，参见一般信息投影背景 [R11]；此处的闭式公式也可直接用 KL 链式分解验证：

\[
D_{\rm KL}(P\|Q^*)=
D_{\rm KL}(P\|Q)-D_{\rm KL}(P_F\|Q_F).
\]

因此它保证前向 KL 不增加，但**不保证 TV 单调下降**。例如两个类别各含两个原子，P=(0.4,0.1,0.1,0.4)、Q=(0.4,0.05,0.1,0.45)，类别质量校准后 TV 从 0.05 上升到约 0.05354。任何测试实例上的 TV 改善都应当作为实测结果，而非上述公式的推论。

若 F 的定义顺序与有向代理的生成顺序相同，给定 F 后前缀完全固定，后缀仍可沿原条件表生成，因而可以直接采样 Q*。若两个顺序不一致，条件后缀采样可能不再简单。

## 5. 本轮之后最有信息量的实验

1. 对同一物理实例，比较有界父集代理、首个点击校准、事件记忆变体；报告完整小规模 TV/KL，同时计入冷缓存构造和采样时间。
2. 用未参与选择的种子检查改善能否泛化；将 Haar、局部网络和非均匀压缩分开汇报，避免只挑改善最大的结构。
3. 搜索事件记忆的失败实例，尤其是远距离强相关和多个弱相关共同决定后续检测的情形；比较最近点击、结构得分和混合策略。
4. 同时绘制真实 TV、真实点击尾界和可计算光子尾界，找出“误差小但证明松”的参数区间，再决定是否值得研究更紧的界。
5. 在扩大模式数前，把真正相关的已发布实现纳入相同输入、输出和精度约定下的基线。当前代码尚不能证明优于 FrugalGBS、MPS 或相空间方法。

## 参考资料与查新记录

- **[R1]** Quesada, Arrazola, Killoran (2018), *Gaussian Boson Sampling using threshold detectors*. <https://arxiv.org/html/1807.01639v2>；重点检查第 V 节与条件 Gaussian 更新。
- **[R2]** Villalonga 等 (2021/2022), *Efficient approximation of experimental Gaussian boson sampling*. <https://arxiv.org/abs/2109.11525>；全文 <https://arxiv.org/pdf/2109.11525>。低阶边缘、Boltzmann 与 greedy 代理不是新方向。
- **[R3]** Dodd 等, *A fast and frugal Gaussian boson sampling emulator*. 预印本 <https://arxiv.org/html/2511.14923v1>；期刊全文 <https://pmc.ncbi.nlm.nih.gov/articles/PMC13665930/>；代码 <https://github.com/tomdodd4598/FrugalGBS>。重点检查附录 C 的 elision 规则。本文没有运行其实现。
- **[R4]** Quesada, Arrazola (2020), *Exact simulation of Gaussian boson sampling in polynomial space and exponential time*. <https://arxiv.org/abs/1908.08068>；<https://doi.org/10.1103/PhysRevResearch.2.023005>。
- **[R5]** Oh 等 (2024), *Classical algorithm for simulating experimental Gaussian boson sampling*. <https://arxiv.org/html/2306.03709v2>；<https://doi.org/10.1038/s41567-024-02535-8>。
- **[R6]** *Efficient simulation of low-entanglement bosonic Gaussian states in polynomial time* (2026). <https://arxiv.org/html/2512.10643v2>；<https://doi.org/10.1038/s41534-026-01254-7>；代码/数据 <https://zenodo.org/records/18045305>。改善低纠缠 MPS 转换不等于完整实验采样已被超越。
- **[R7]** Lim (2026-10-05), *Classical Simulation of Lossy Gaussian Boson Sampling beyond Square-Root Regime*, 预印本。<https://arxiv.org/html/2610.06526v1>。中心化重置的误差界不能直接移植到其位移噪声变体。
- **[R8]** Zhang 等 (2025), *Efficient Classical Sampling from Gaussian Boson Sampling Distributions on Unweighted Graphs*. <https://arxiv.org/html/2505.02445v2>；加权版本见 Supplementary Note 4，混合时间仍有开放问题。
- **[R9]** Goodman, Dellios, Reid, Drummond (2026-04-14), *Gaussian boson sampling: Benchmarking quantum advantage*, 预印本。<https://arxiv.org/html/2604.12330v1>；代码 <https://github.com/peterddrummond/xqsim>。已报统计测试的表现不自动构成完整 TV 证明。
- **[R10]** The Walrus 官方算法文档，low-rank、sparse/banded 与精确算法。<https://the-walrus.readthedocs.io/en/stable/algorithms.html>；Björklund, Gupt, Quesada, *A faster hafnian formula for complex matrices and its benchmarking on a supercomputer*, <https://arxiv.org/abs/1805.12498>。
- **[R11]** Csiszár (1975), *I-Divergence Geometry of Probability Distributions and Minimization Problems*. <https://doi.org/10.1214/aop/1176996454>。此处的类别校准等式由文中展示的链式分解直接得到，不依赖查新主张。

检索关键词覆盖 `GBS elision`、`truncated conditional click history`、`no-click approximate conditional sampling`、`Bayesian GBS approximation` 与已知精确/近似算法的引用。没有在这些检索结果中看到完全相同的规则，是有限检索结果；它不排除名称不同、近期尚未收录或未公开的软件先行工作。
