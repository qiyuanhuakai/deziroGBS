# 第五轮：冻结见证的事后二项式置信界精化

本分析是在看到主要Hoeffding结果之后提出的**次级统计分析**，没有预注册Clopper–Pearson方法。它使用相同14组比较、相同样本、相同pilot/held-out分割、相同事件及方向、相同目标概率和相同alpha预算；原始报告保持不变，没有重抽。

结果单独保存在 `results/iteration5_binomial_refinement/`，没有写回原D1组合报告或D2差距报告。

## 1. 为什么这次方法替换不额外消耗alpha

给定pilot后，某个窗口的事件A已经固定。设held-out部分有n次独立尝试，成功次数为k，真实事件概率为q=Q(A)。失败符号不属于A。原来的单侧Hoeffding下界是

\[
H(k)=\max\left(0,\frac{k}{n}-\sqrt{\frac{\log(1/a)}{2n}}\right),
\]

其中a是这个事件已分配的错误概率，已经包含窗口及比较组的多重比较调整。

对k>0，记

\[
T_k(p)=\Pr_{K\sim\mathrm{Bin}(n,p)}(K\ge k).
\]

它在0<p<1上连续严格递增。单侧Clopper–Pearson下端点C满足

\[
T_k(C)=a,
\qquad C=\operatorname{BetaQuantile}(a;k,n-k+1).
\]

k=0时取C=0；k=n时C=a^(1/n)。这里用a，没有因为两侧区间而再除以2。

若H>0，将原Hoeffding尾界用于参数p=H，有

\[
T_k(H)\le
\exp\!\left[-2n\left(\frac{k}{n}-H\right)^2\right]=a=T_k(C),
\]

所以H≤C。若H=0，同样成立。**CP在每一种可能计数上都支配原Hoeffding下界**，不是只在这批观察值上更大。

目标P(A)不变，因此

\[
L_H=(H-P(A))_+\le L_C=(C-P(A))_+.
\]

对固定窗口族，取各下界最大值后，这个逐点关系仍成立。即使看到结果后再在这两种方法之间选择，最终报告值也不会超过CP族下界。任何错误声称下界超过真实TV的事件，都包含于CP族覆盖失败事件之中，仍由原先的Bonferroni预算控制。因此不需为**这两个逐点嵌套、同事件同数据同alpha的规则**再乘一个“方法数2”。

这项论证不允许事后修改A、翻转方向、选择更多窗口、改变held-out样本数、移用未使用的alpha，或任意搜索互不嵌套的估计方法。它也不能把此前其他轮次的独立置信族自动合并成一次全研究同时覆盖声明。

## 2. 实际预算

冻结配置把总族错误概率0.05分为原D1通道0.025和新增边缘见证通道0.025。后者分给14组比较，每组再分给4个预定窗口：

\[
\alpha_{\rm comparison}=0.025/14,
\qquad a=0.025/(14\cdot4)
=0.00044642857142857147.
\]

某个窗口或比较没有完成也不转移其预算。pilot用于选择事件，held-out用于统计检验；对每个pilot实现，CP覆盖性质成立，再对pilot取平均即可保持无条件覆盖。样本失败后的整批中止沿用原规则，不作条件于成功的覆盖声明。

## 3. 实现与绑定检查

`src/gbslab/binomial_refinement.py`只读取保存证据。它核对完整样本、pilot和held-out的SHA-256，重算每个窗口的计数，确认事件确实由原pilot频率严格大于原目标概率产生，并核对事件SHA-256、成功数k、n及a。

原目标协方差与采样attempt、native完成记录及原报告相互绑定。保存的小边缘概率向量另用同一协方差重新计算作一致性检查；精化时仍使用原报告保存的P(A)标量，不用新估计替换它。成功窗口与数值失败窗口必须恰好覆盖预定窗口编号，失败窗口不生成替代事件。

量化使用SciPy `beta.ppf(a,k,n-k+1)`；使用 `binom.sf(k-1,n,C)`检查尾反演残差，避免用`1-cdf`相减。非有限值、k>0却返回0或1、异常反演残差及未支配原下界均拒绝。这个检查不是向外舍入的区间算术证明，“exact”指统计构造。

真实归档按固定计划统一处理14组，没有只处理恰好接近阈值的那一组；处理前后所有读取的primary文件散列不变。输出目录要求全新，且必须位于primary目录外。

## 4. 实际结果

14组全部完成精化，零缺失、零绑定失败。CP边缘通道给出1组拒绝和13组证据不足；这只是次级边缘通道结论，不覆盖小模式完整TV已经给出的接受证据。

| 比较 | 原Hoeffding下界 | 次级CP下界 | CP边缘通道结论，epsilon=0.1 |
|---|---:|---:|---|
| m100_bright | 0.002629714881 | 0.013121773579 | 证据不足 |
| m144_bright | 0.021053214375 | 0.031386444616 | 证据不足 |
| m256_bright | 0.096167007655 | **0.106671852353** | **拒绝TV≤0.1** |
| 其余11组，包括JZ1候选与Frugal | 0 | 0 | 证据不足 |

256模式高亮度的最强事件来自固定末段窗口248…255。pilot大小256；独立held-out大小768，其中448条落入冻结事件A：

\[
\widehat Q(A)=448/768=0.5833333333333334,
\quad P(A)=0.41629823764468765,
\]

\[
C=0.5229700899978389,
\quad C-P(A)=0.10667185235315124.
\]

四个窗口的目标概率向量重算差异均为0，末段CP反演的相对尾残差约1.08e−14。原主要结果0.09616700765464792及其“证据不足”结论仍留在原目录；不会回写成“预注册检验已经拒绝”。

这项结果表明：在原先限定的Gaussian阈值目标、固定采样协议、独立尝试模型及浮点概率计算可靠的条件下，这个256模式高亮度候选的全分布TV超过0.1有统计支持。它没有说明Frugal通过同一TV容差，也没有说明九章硬件实验被复现。

## 5. 验证与复现

`tests/test_binomial_refinement.py`共10项测试通过，涵盖k=0/n解析边界、独立精确二项式反演、逐计数CP支配、直接枚举二项式覆盖失败概率、输入和反演错误、原证据不可修改、样本/目标/事件篡改拒绝、失败批次拒绝、失败窗口保留预算，以及archive不覆盖和缺失比较不挪预算。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_binomial_refinement.py -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m gbslab.binomial_refinement \
    --quality-directory results/iteration5_quality \
    --output results/new_binomial_refinement
```

统计方法出处为[Clopper–Pearson方法的SciPy官方说明与原文引注](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)及[Hoeffding 1963原论文](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)。上面的逐点支配与方法选择覆盖论证在本节完整给出，未将它包装为新的GBS算法原理。
