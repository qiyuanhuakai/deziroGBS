# 第八轮独立数值与状态审查

日期：2026-11-05。基线：第七轮封存提交 108b732。本文记录独立审查者的静态复核、解析数值微核查和新增数值门禁；不替代本轮最终二进制绑定的总门禁或正式质量、性能面板。

## 1. 当前结论与绑定范围

**首次独立数值门禁通过：10 项测试，2.529 秒，0 failure、0 error、0 skip。** 未发现 shared-Schur 公式、LD→quad 状态机、解析零、合并森林分数或规范状态重建方面的阻断问题。

此次实际加载的是 portable 共享库，而非之后准备正式计时的最终 --native 构建。其 SHA256 为：

    21f7c167e5b8ec8a4f2ba00f3d25f4bdd01d52cd977479b389e174d08d99d93b

测试显式传入 .cache/batched-forest-native/libbatched_forest.so，没有隐式编译，也没有把缺库时的默认 skip 当成成功。首次运行时的相关源码如下。

| 文件 | SHA256 |
|---|---|
| experiments/batched_forest_kernel.cpp | 37f19f146db72b87b9d46d549a640dc13ef746cf96d1b57dd9f1c660772dc984 |
| experiments/batched_forest_kernel.h | ab19c485daae4525a6fa20274ed2db3c4f97a56313df7a50b5d65dfc3ae9cf87 |
| experiments/build_batched_forest.py | 78df04c106f9f6e0e8b4849500a79553866435ecff49de7328131209e7388064 |
| src/gbslab/batched_forest.py | 1afa8002ade83b137a73542a5a63a581053fc5c398b7bc12b0aeaae090c83ed6 |
| tests/test_batched_forest_numerics.py | 27c65c76e047eb178bef4a55deff873d85960ac9ea26935fbb76e7a6b0d91cb5 |

共享库 build.json 的 SHA256 为 772db8ea60c71b0f80991d8cd84977c22c51f4b8927d980f73e8ca24caaa3f53。构建记录绑定上述 C++、头文件和 builder，并记录 libquadmath 依赖。测试代码另在结束时复核实际库文件的哈希未改变。

复现命令（从仓库根目录执行）：

~~~sh
env PYTHONPATH=src \
  GBSLAB_BATCHED_FOREST_LIBRARY=.cache/batched-forest-native/libbatched_forest.so \
  python -m unittest discover -s tests -p test_batched_forest_numerics.py -v
~~~

首次测试后，实现方最终只澄清了 Python 的 memory_scope 说明；C++ 数学、误差传播和预算没有修改。冻结后的 Python 模块 SHA256 为 2b93fcb732c93979189d86f7c743efd2abf6eb8be6d97b0751703e45f8747638，最终 --native 库位于 .cache/iteration8-batched-native/libbatched_forest.so，SHA256 为 8c4217c57559c741be9a65adcd390e92fa52ed8daacb68136452eab04b282263。它们不能自动继承本节的 portable 二进制绑定；主流程总门禁须重新执行并绑定最终源码、共享库和 CLI。本文保留首次审查的准确对象，最终总门禁完成后另列结果。

## 2. 固定数值策略及其依据

令全部历史零为 Z，保留点击集合为 R，当前位为 i，节点或边集合为 G。事件按统一尺度计算：

\[
\overline M_{a,G}
=P(X_R=1,X_i=a,X_G=1\mid X_Z=0)
=\sum_{S\subseteq R}(-1)^{|S|}\beta_S h_{a,G}^{(S)}.
\]

其中 \(\beta_S=P(X_S=0\mid X_Z=0)\)，局部 \(h\) 最多涉及三个模式。共同去掉 P(Z=0) 合法，且森林分数对全部事件的总幂次恰为 1，故两个分支共用的尺度在归一化时抵消。baseline、节点、边和分支一致性检查也使用这个尺度。完整推导见 [第七轮批量事件内核推导](iteration7_batched_event_kernel.md)。

独立静态复核确认：

- 初始 binary64 V 先精确拓宽到相应精度，再在该精度内形成 \(\mathcal Q=(V+I)/2\)。没有先形成 binary64 Q 再转换。
- xxpp 输入到内部交错坐标的转换、rank-two 真空投影、每个 S 的 Cholesky、共享 \(L^{-1}Q_{SD}\) 求解和稀疏 Schur 块索引一致。
- 一个、两个和三个模式的真空行列式组合与两分支的包含排除公式一致。点击不被当作一次 Gaussian 投影。
- LD 和 quad 各自维护原始输入出发的零条件状态；quad 不使用 LD Schur 中间结果。
- AUTO 对固定的 R、E、前缀和 guard 最多计算一次 LD 和一次 quad。只捕获数值异常；参数、内存和程序错误不被转为 refinement。
- 每个事件的相对指标与最终合并 log-score 的阈值仍为 \(10^{-6}\)。没有 epsilon、负值裁剪、成功律重归一化或降低保护系数。

每个标量携带值与绝对指标 \((v,e)\)，以固定 \(\mu=8\epsilon_{\rm format}\) 跟踪加减、乘除和平方根，因此原始 Q、零投影、Cholesky、三角求解、Schur 相减和包含排除都进入同一传播链。

分母的 \(|v|-e\le0\)、平方根输入的 \(v-e\le0\)、非有限值或指标、下溢或 subnormal 会显式失败。quad→LD 输出另计数值转换误差和 LD 舍入项，并检查指标转换未消失。合并 log-score 的指标使用事件系数的绝对值，负指数不会抵消不确定性。

**这些指标仍为 heuristic。** 误差指标自身的算术和数学库不是有向舍入区间实现，固定 \(\mu\) 也不是本项目已经证明的全算术严格误差常数。通过 guard 不构成严格事件概率、总变差或统计覆盖证书。两种精度间一致也不提供这样的证书。

## 3. 解析热态夹具与真实 refinement

夹具固定为

\[
V=(1+2^{-b})I,\qquad
p_{\rm click}=\frac{1}{2^{b+1}+1}.
\]

这是独立热态；协方差数值是精确 binary64，下一位条件概率与任何历史、保留集和树结构无关。选取 b=8 与 b=40 来分别检验可分辨与超出固定精度预算的相消尺度，未用正式面板或已见样本拟合参数。

### 3.1 b=8：真实的 LD→quad 回算

m=8、k=4、长度 7 的全点击前缀使最宽事件含七个点击，质量约为 \(2^{-63}\)，而真下一位概率为 \(1/513\)。

新增 Python 门禁用 recent 选择器隔离事件求值：LD 显式失败；quad 和 AUTO 成功且返回完全相同的 binary64 概率。该单个前缀的 AUTO 只有一次 refinement，R 和 E 各选择一次。重复请求命中条件缓存，不再次执行 refinement。数值与 \(1/513\) 的绝对差小于测试预先给定的 \(4\times10^{-13}\) 容差。

另以已编 native CLI 作四次单样本微核查，使用 pairwise 选择器和全零 uniforms 选择全点击路径。该检查只核对数值策略，不属于正式质量或性能面板。

| 热态指数与策略 | 完成样本数 | 已消费 uniform | refinement | 观察结果 |
|---|---:|---:|---:|---|
| b=8，longdouble | 0 | 3 | 0 | 事件相对指标 \(7.67\times10^{-6}\) 超过 guard |
| b=8，quad | 1 | 8 | 0 | 最后条件概率 0.0019493177387914231 |
| b=8，auto | 1 | 8 | 5 次，全部成功 | 最后条件概率同 quad |
| b=40，auto | 0 | 2 | 3 次，2 成功、1 失败 | 最终相对指标约 \(9.18\times10^5\)，保留失败 |

真值 \(1/513\) 的 binary64 表示为 0.001949317738791423。微核查 CLI 的 SHA256 为 770aea52bd8a0e39e52b77a389b00a5bc7f0d6e5c8f5fe9476993b9bf2a76f00，其构建记录绑定的内核源码与第 1 节相同。

### 3.2 b=40：预算不足不能制造支持证书

此时七点击质量约为 \(2^{-287}\)，而 quad 的有效精度为 113 位。当前带符号展开不能因此可靠分辨这个事件。

独立门禁要求 quad 和 AUTO 对该固定前缀继续抛数值异常；两次相同请求各自执行完整的逐前缀预算，不存在“前若干次失败耗尽全批次 refinement 配额”的策略。原事件严格为正；没有把求值零当作解析零。

真实采样门禁用全零 uniforms 走向困难前缀。失败异常保留已经完成的样本和失败前缀；该前缀失败时不再消费 uniform，也不补抽请求中的剩余样本。

### 3.3 最终表示范围仍是独立边界

对单模式 \(V=2^{60}I\)，两分支严格为正，但下一位点击概率舍入为 binary64 的 1。三种 precision 策略都保持失败；quad 的更准确事件求值不能修复最终 Bernoulli API 的表示范围。这个端点没有被改写为解析确定分支。

## 4. 状态、缓存、完整律和异常门禁

新增测试文件的 10 项检查如下。

| 检查 | 要求及首次结果 |
|---|---|
| 真实热态 refinement | LD 失败；quad/AUTO 同值；固定 R/E；一次整前缀回算，通过 |
| quad 预算不足 | 严格正事件继续失败；重复请求不受全局配额影响，通过 |
| 真实采样失败 | 失败前不多取 uniform；无 refill；保留已完成样本，通过 |
| 失败扩展律 | 失败前缀整块质量记入失败符号；成功叶子不重归一化，通过 |
| 最终 binary64 端点 | 两正分支舍入端点仍失败，通过 |
| 状态与缓存历史 | 108 次条件记录逐值比较全部一致 |
| 同一 uniform 流 | 三种缓存容量的样本与全部实际条件概率轨迹逐值相同 |
| Gaussian/table 同 R/E | 378 个条件比较与 6 组完整扩展律比较通过 |
| 非数值异常 | RuntimeError、ValueError、MemoryError 在 conditional/sample/enumerate 三入口均原样传播 |
| 注入数值异常 | 根前缀失败质量为 1、全部成功叶子为 0、uniform 消费为 0，通过 |

### 4.1 108 次逐值状态比较的具体范围

固定非自然测量顺序为 (6,2,0,7,3,5,1,4)。三个目标前缀包含不同零集合及全点击历史。先走一组前缀，再反向访问，使状态经历追加、回退、分支变化和缓存命中/驱逐。

比较组合为三种 precision × 两种 forest × 三种缓存容量 (0,1,32) × 六个目标查询，共 108 次。逐值比较包含 R、F、E、条件概率及返回的事件/分数指标，不仅比较抽到的 bit。参考值来自新建的冷状态采样器。

这验证了固定测试族的路径不变性，不是穷尽所有协方差或所有测量顺序的形式证明。静态依据是：每种精度仅允许旧规范零序列为新序列前缀时追加，否则从原始输入按同序列重放。Python selector 也使用独立的 CanonicalVacuumConditioner 适配器，从而不再依赖旧缓存恰好保留哪个子集父状态。

### 4.2 同 R/E 的独立 Gaussian/table 核对

输入为预先固定 seed=180707 的 m=6 有损 Gaussian，测量顺序 (4,0,5,2,1,3)。比较两种 forest 与 k=0/2/4，共 6 组完整律；每组访问全部 63 个非终止前缀，共 378 个条件。

native 每个前缀选出的 R/E 直接提供给独立 TableEventOracle 的森林公式，因此比较中没有混入不同选择器结构。完整 P 只用于这个指数级测试参考；native 执行期间对 model.full_distribution 的访问被禁止。单步差通过 \(5\times10^{-9}\) 容差，完整扩展律 TV 通过 \(2\times10^{-11}\) 容差，双方失败质量均为零。这里记录的是测试通过的上限，不伪称已保存未输出的最大实测差。

### 4.3 失败质量没有被隐藏

m=6、b=8 的 longdouble 热态枚举有非零失败质量。门禁逐项检查失败前缀覆盖的成功输出均为零、失败条目质量之和等于失败符号质量、整个扩展律归一化。全零成功输出的质量还与解析 \((1-p)^6\) 比较，以排除对成功叶子统一重归一化。带失败质量调用 full_distribution 必须抛异常。

## 5. 完整 native CLI 的独立只读复核

另读了完整 CLI、builder 和 wrapper。未发现以下链条的数学或计时阻断：

- 原选择器计算一次 R，静态 MI 的 Prim 排序与 Python 使用相同的权重排序和物理索引平权规则；对 R 排序仅规范事件展开顺序，另保留 selection_order。
- 统一调用 bf_probability；refinement 内部没有 RNG。实际返回的 binary64 p 同时用于抽样及 log/log1p 的 logQ 累计。
- 条件失败在抽取下一 uniform 前停止；旧 selector state 的零更新若在抽到零之后失败，枚举将该 child 质量记入失败，而不是把 parent 全部质量记入失败。
- 最后一个输出不再做无后续用途的零状态更新。完整枚举不重归一化成功叶子。
- setup、选择、构树、状态维护、失败的首次计算、refinement、RNG、样本存储和请求的 logQ 都进入完整 sampling 计时。内核子计时不是端到端速度。

审查提出并已落实一项 wrapper 补强：失败批次也在抛出 CppForestError 前核对 completed/partial 前缀与精确 uniform 消费数、已完成输出的统计计数、注入 uniform 的阈值关系、条件轨迹和 completed logQ。未请求展开 raw 时，使用失败 checkpoint 的 packed words 解码复核。

更新后 wrapper SHA256 为 3988149b0f731e1027e56c2e1973c35f31164011950a2e0b5b4e08940a33f999。这部分属于静态审查；CLI 的最终动态门禁由主流程绑定。

## 6. 必须保留的解释边界

1. **数值 refinement 没有消除森林近似的结构误差。** 同理，局部最大三个模式不意味着整次采样常数成本；外层仍有 \(2^{|R|}\)，全部历史零状态、选择器和构树也要计费。
2. **selector 仍用既有 float64 数学和 guard。** 新事件内核的 quad refinement 不修复在选择 R 之前发生的 selector 数值失败，也不重新排序 R。
3. **Python 与 CLI 的 selector 状态求值时机不同。** Python 适配器在选择器需要状态时规范追加或重放；完整 CLI 为保留旧 selector 流程，在非最终零输出后立即更新其 double State。若这部分更新本身失败，两后端可以在不同时刻触发失败。不能仅凭共用事件内核宣称任意输入下两者有限精度扩展律完全相同。
4. **规范路径不等于旧路径逐比特重现。** 旧 Python conditioner 根据缓存选择父状态，规范适配器修复这个历史依赖；它并不承诺复制每个旧缓存访问路径的舍入。
5. **解析零证书范围没有扩张。** 仅使用原输入中恰好隔离的真空结构。条件 Schur 看起来像真空、TMSV 中出现小残差或高精度结果为零，都不能据此扩大支持证书。
6. **预算逐前缀确定。** 运行整体超时或资源不足是未完成任务，不应通过耗尽全局 refinement 配额后静默切换求值规则来继续报告 IID 样本。
7. **诊断计数须按其定义解释。** raw query_successes 表示事件批次完成，之后分数仍可失败；返回的最大指标只覆盖通过相应 guard 的量。不能把这些字段直接当作成功条件次数或所有失败事件的最大值。
8. **最终报告以总门禁为准。** 本文的 portable 库验证和解析小夹具不提供正式面板结论，不支持与 Frugal 或旧 C++ 的速度比值，也不构成任意 Gaussian 输入的精度保证。

## 7. 冻结安排

独立测试文件在首次 10/10 成功后冻结；未为通过测试调节 precision、guard、\(\mu\)、热态指数、结构选择或测试容差。本文完成后不追加可选实验。最终源码和 --native 二进制由主流程的完整门禁重新绑定，正式结果到齐后再审查对速度、失败质量和结构误差的解释。

## 8. 最终 native 总门禁绑定

主流程随后完成了 [最终门禁](../results/iteration8_validation/validation.json)：**268 项测试全部通过，0 failure、0 error、0 skip、0 expected failure、0 unexpected success**，用时 36.4624 秒。记录创建时间为 2026-11-05T01:40:48.870333+00:00；源码和三种二进制在执行期间稳定。测试使用显式 native 路径，测试本身不编译程序。

最终 validation.json 的 SHA256 为：

    aa4d55c84602ad3e3f54955b73d470085a67e491d96f5e9652feef253a3ca666

| 绑定对象 | SHA256 |
|---|---|
| shared-Schur native 共享库 | 8c4217c57559c741be9a65adcd390e92fa52ed8daacb68136452eab04b282263 |
| retained-click native CLI | a8d5a830e7167aab20f829bee74bc404736f2551ff42e6e48491caf301d0fc3f |
| forest native CLI | 770aea52bd8a0e39e52b77a389b00a5bc7f0d6e5c8f5fe9476993b9bf2a76f00 |

独立审查者在只读收尾中重算上述三文件 SHA256，均与门禁一致；没有重跑测试或正式面板。本节明确把最终 native 构建与第 1 节的首次 portable 构建分开记录。

总门禁保存了新增数值测试的具体现象：真实 LD→quad 热态回算通过；quad 预算不足仍失败且不认证零；两正分支舍入 binary64 端点仍失败；108 次规范状态/测量顺序/缓存条件比较逐值相同；相同 uniform 流的三次采样缓存对照消费数均为 24。Gaussian/table 同 R/E 的 378 个条件比较最大实测差为 \(5.767608612927688\times10^{-14}\)，6 组完整扩展律的最大 TV 为 \(5.2617415632894016\times10^{-15}\)。这些是最终门禁保存的实测数，不是对第 4.2 节首次运行未输出最大值的追溯补写。

真实枚举失败夹具保留 \(7.38545354345751\times10^{-8}\) 失败质量、10 个失败条目，成功律未重归一化；注入数值失败则使失败符号质量为 1，uniform 消费为 0。RuntimeError、ValueError、MemoryError 在三入口没有被吞掉。完整 CLI 的实际概率/logQ、注入 uniform 和失败批次 partial checkpoint 也属于该总门禁的范围。

268 项通过解决了本轮最终构建的回归门禁要求。它仍不把 heuristic 指标变成严格区间，不保证任意输入的两后端失败律一致，也不提供大模式总体 TV 或速度结论；正式质量、回归和计时面板须按各自冻结协议单独解释。
