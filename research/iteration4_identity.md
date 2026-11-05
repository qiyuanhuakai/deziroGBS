# 判别器一：是否接近指定的 GBS 物理目标

日期：2026-11-04。实现入口为 `src/gbslab/gbs_identity.py`，命令行入口为 `experiments/validate_gbs_identity.py`。

## 1. 判别对象必须先说清楚

GBS 的“Gaussian”指输入量子态与 Gaussian 光学变换，不表示输出点击直方图应呈高斯钟形。阈值探测输出是离散、通常有相关性的二元分布。

本判别器回答的是：**在给定协方差、零位移和阈值探测模型下，候选输出是否足够接近相应目标分布 P？** 它不回答“是否存在某个 Gaussian 态可以拟合这些数据”。例如独立热态的阈值输出本身也是一种容易模拟的 GBS；它可以匹配指定压缩态实验的单模均值，却不匹配其相关分布。我们用它作为错误目标的对照，不能将其泛称为所有意义上的“非 GBS”。

当前支持零均值 Gaussian 态、hbar=2、真空协方差 I、xxpp 坐标和阈值探测。非零位移、PNR 计数及未经明确建模的探测器响应返回 `out_of_scope`。这不表示这些物理模型不合法。默认协方差物理性检查资源上限为 256 模式，完整分布枚举上限为 14 模式。

协方差物理性通过不意味着采样输出正确。它仅表示所给矩阵通过了 `V+iΩ >= 0` 等数值检查，能够定义本判别器范围内的目标态。实际光学实验的校准与模型参数是否可信，是另一个证据问题。

## 2. 证据等级与输出含义

| 证据 | 能支持的结论 | 不能支持的结论 |
|---|---|---|
| 完整小规模 P、Q | 数值计算全部输出的 TV，比较预设 ε | 单凭一份自报 Q 证明外部采样程序诚实 |
| 固定 iid 验证批次、无条件路径 Q(x)、目标点概率 | 在记录的假设下给全分布 TV 统计上界 | 条件于“终于抽到一次成功批次”的置信保证 |
| 预先指定的无点击事件与总点击数均值 | 反驳与目标的 ε 接近性，或发现错误 | 通过少数见证就证明全分布接近 |
| 仅物理协方差 | 定义目标、检查输入物理性 | 验证输出样本或宣布量子优势已被模拟 |

决策采用多值状态，避免强行二分类：

- `accepted_within_tolerance_conditional`：完整数值 TV 或全局统计上界不超过预设 ε；概率算术、实现绑定和统计假设仍明确记录。
- `rejected`：完整数值 TV 超过 ε，或预设见证给出的统计 TV 下界超过 ε。
- `insufficient_evidence`：上界过松、只有未拒绝的见证，或缺少必要的统计协议。
- `invalid_evidence`：概率不归一、重复样本的 Q 密度冲突、完整 Q 与路径密度不一致，或样本与所声明完整 Q 的证据冲突。
- `evaluation_failed`：固定采样批次失败或不完整；没有重试或筛除失败后接受的路径。
- `invalid_model` / `out_of_scope`：分别表示指定模型检查失败和当前验证范围不支持。

接受 ε=0.1 是一项预先定义的近似质量结论，不是声称精确采样。所有默认 Gaussian 概率计算都是带保护的浮点估计，`strict_floating_point_certificate` 始终为 false。

## 3. 无需完整枚举的全局上界

对归一化目标 P 和输出过程 Q，有

\[
\operatorname{TV}(P,Q)
=\mathbb E_{X\sim Q}\left[\left(1-\frac{P(X)}{Q(X)}\right)_+\right].
\]

这来自 `TV=sum_x(Q(x)-P(x))_+`，即使 P 在 Q 的支持之外有质量，恒等式仍成立。它避免为了估计全部 TV 而必须枚举 P 的所有输出。

在每个成功样本上，原生程序记录实际采样所用条件概率的对数乘积。若结果为 1，用 `log(p_one)`；若为 0，用 `log1p(-p_one)`。它与真正抽样共用条件核，记录开销计入采样时间。包装层在同时给出完整 Q 或条件轨迹时检查这些 log Q 是否一致；默认关闭记录，以免污染普通性能对赛。

设单样本得分为 Y。超过预设点击截断、目标概率计算失败或无法可靠求值时，Y 取 1，并保留在样本总数 N 内。其余样本使用 `(1-P(x)/Q(x))_+`；若目标使用可信下界，所得分数同样保守。在概率数值正确的条件下，

\[
\operatorname{TV}\le
\min\left(1,\bar Y+\sqrt{\frac{\log(1/\alpha)}{2N}}\right)
\]

具有相应的单侧统计保证。上界大于 ε 只说明尚不能接受，不说明真实 TV 必定超过 ε。

缓存或计算预算可能使哪些点被可靠求值依赖运行历史；只要未解析点的处理逐点不小于真实得分，保守得分仍支配真正的 iid 得分，不能因此挑掉困难样本。

## 4. 数值拒绝作为输出的一部分

不能仅因为一批样本都生成成功，就假定其他历史不会触发数值保护。为此，把单次采样的输出空间扩展为全部二进制串和失败符号 `⊥`。

程序在每个已解析前缀用计算出的概率分支；若首次触发保护，该子树的剩余概率归入 `⊥`。于是定义了一个归一化的 `Q_ext`。目标扩展为 `P_ext(⊥)=0`。成功位串的无条件概率仍为路径概率乘积；不需要除以“成功概率”。

若成功位串质量和为 `1-f`，则

\[
\operatorname{TV}(P_\mathrm{ext},Q_\mathrm{ext})
=\frac12\left[\sum_x|P(x)-q(x)|+f\right]
=\mathbb E_{Q_\mathrm{ext}}\left[\left(1-\frac{P_\mathrm{ext}(X)}{Q_\mathrm{ext}(X)}\right)_+\right].
\]

失败得分为 1。真实批次一旦失败，本验证实现直接中止质量接受，不自动生成替代批次。若预先固定的 N 次尝试全部成功，这一数据仍来自无条件实验的一个事件；错误接受事件包含在通常的 Hoeffding 尾事件中。因此无需假定未观测失败质量 f=0。

这里的置信保证是对预先规定的完整实验无条件成立，**不是条件于一个成功批次**成立。先反复运行直到成功、删除失败批次或从多个种子中挑最好结果，会破坏协议。

这个扩展距离也控制成功条件下的输出误差：由于 `q(x) <= q(x)/(1-f)`，目标与未归一化成功质量的重叠不大于与条件成功分布的重叠，故

\[
\operatorname{TV}(P,Q(\cdot\mid\mathrm{success}))
\le\operatorname{TV}(P_\mathrm{ext},Q_\mathrm{ext}).
\]

它同时惩罚失败质量，并不靠后选择掩盖失败。

## 5. 大模式见证与多重比较

默认见证选择只依赖模式数，抽样前固定：全部单模无点击、相邻对、前后半区对应模式对、互不重叠的最多四模式无点击块，以及总点击数除以模式数。具体子集列表与哈希随结果保存，也可由协议提前提供自定义子集。

无点击事件的目标概率只需对应 Gaussian 子矩阵行列式。所有见证函数 f 取值在 [0,1]，并在失败符号处定义为 0，因此 `|E_Q f-E_P f| <= TV(P_ext,Q_ext)`。对 J 个预设见证采用同步 Hoeffding 半径

\[
r=\sqrt{\frac{\log(2J/\alpha_w)}{2N}},\qquad
L=\max_j\max(0,|\widehat{E_Q f_j}-E_P f_j|-r).
\]

若 L>ε，可在分配的置信水平下拒绝目标接近性。若 L<=ε，只是未拒绝。总点击数均值也在这一预算内，并没有单独做未校正的额外检验。

`ConfidencePlan` 预先列出全部比较 ID。整个家族错误预算平均分配给这些比较；每个比较再固定分成两半，一半给全局上界，一半给所有见证。共享样本批次不破坏家族并集界，但不能在看结果后增加比较、重用未花预算或选择最好结果而仍沿用原置信度。

## 6. 目标、样本和实现绑定

结果记录 target ID、规范化协方差哈希、探测器模型、位移/坐标范围、样本哈希，以及 native 可执行文件哈希、选择器、k、测量顺序和概率内核。

在把质量结论转移到性能运行时，必须匹配目标协方差和采样规则。独立验证批次的随机种子和样本数可以与计时批次不同；这本来就是避免在同一批数据上选择和检验的做法。更换可执行文件或概率内核，需对应的等价验证依据，不能仅因名称相同就转移结论。

元数据与哈希提供可追溯绑定，不是密码学意义上对任意外部程序诚实性的证明。实际采样与所声明 Q 共用概率核、注入随机数轨迹核验、完整小规模 Q 核验及独立物理参考仍是实现证据的一部分。若样本见证与所声明完整 Q 冲突，整体判定变为 `invalid_evidence`，不会让一份看起来正确的 Q 表覆盖实际样本的问题。

## 7. 调用与复现

原生包装器新增 `record_log_probabilities` 和 `vacuum_kernel` 参数。默认 log 内核兼容原先冻结的程序；新功能需要支持这些选项的原生二进制。

```python
from gbslab.cpp_click import run_cpp_click
from gbslab.gbs_identity import ConfidencePlan, fixed_batch_protocol, assess_native_samples

# 在抽样之前写入计划文件，并固定输入、算法、容差和随机种子。
plan = ConfidencePlan(("case_a", "case_b"), familywise_alpha=0.05)
out = run_cpp_click(V, selection="click_influence", max_clicks=4,
                    samples=1024, seed=123, raw_samples=True,
                    record_log_probabilities=True, vacuum_kernel="product",
                    executable=".cache/click-iteration4-native/click_sampler")
evidence = assess_native_samples(
    V, out, plan=plan, comparison_id="case_a", epsilon=0.1,
    max_exact_clicks=6,
    protocol=fixed_batch_protocol(1024, source="prespecified_experiment"))
```

`fixed_batch_protocol` 是对真实实验流程的明确声明，不能用于把重试或后选择伪装成合规协议。实际运行失败时应调用失败路径或保存失败结果，不能再抽一批后忽略旧失败。

命令行读取一个包含 `covariance` 及可选 `distribution`、`samples`、`sample_log_probabilities` 的 NPZ；也可单独提供原生结果 JSON：

```bash
PYTHONPATH=src python experiments/validate_gbs_identity.py \
  --input case_a.npz --native-result case_a_native.json \
  --plan confidence_plan.json --protocol sampling_protocol.json \
  --comparison-id case_a --epsilon 0.1 --max-exact-clicks 6 \
  --output results/new_identity_assessment
```

小规模正反对照使用固定种子、提前保存的计划，包含正确 TMSV、相同单模均值的错误相关结构、仅见证证据、故意无法求值的目标、范围外 PNR、独立 Fock 截断参考和完整条件化的原生采样器：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/validate_gbs_identity.py --demo \
  --executable .cache/click-iteration4-native/click_sampler \
  --vacuum-kernel log --output results/new_identity_demo
```

所有输出目录必须新建或为空，保留原始数组、决策、源代码快照、参数和哈希。演示是判别逻辑的校准，不是大型 GBS 质量结果。更大规模的真实结论应读取根任务实际运行的独立质量结果，不能从这些小对照外推。

## 8. 本轮已执行的判别逻辑校准

`results/iteration4_identity/demo_v1/` 保存了实际执行的 8 个对照；全部按预期给出不同层级的结论。置信家族在采样前包含这 8 个 ID，α 总预算为 0.05，没有因部分对照不使用统计预算而重新分配。

| 对照 | 已计算证据 | 决策 |
|---|---|---|
| 完整双模压缩目标 | 完整数值 TV 为 0 | 在 ε 内，数值资格保留 |
| 正确 TMSV 的 2048 次 iid 抽样 | 全局 TV 上界 0.0375271 | 在 ε=0.1 内，协议/数值条件保留 |
| 单模理论均值相同的独立热点击 | 见证 TV 下界 0.207693，超过 ε=0.1 | 拒绝这个指定相关目标 |
| 正确 TMSV，仅提供见证数据 | 未拒绝见证，没有全局上界 | 证据不足 |
| 目标点概率故意无法解析 | 所有相应得分记 1，全局上界 1 | 证据不足 |
| PNR 探测声明 | 当前判别器只支持阈值探测 | 范围外 |
| 独立 Fock 递推/损耗/阈值参考 | 归一化截断分布 TV 为 7.35570e-8，在显式遗漏质量允许范围内 | 在预设 ε=1e-5 内 |
| 完整条件化的原生三模式采样 | 全部 8 个输出的数值 TV 为 3.39138e-16 | 在预设 ε=0.01 内 |

最后一个原生对照只有 256 个验证样本，其统计上界为 0.106143，单独看并不足以接受 ε=0.01；完整小规模分布提供了更强证据。这是证据层级的实际区别，不把松上界误读成真实误差很大。

独立热点击对照的完整理论 TV 为 0.5，但该条判别调用没有把完整 Q 交给评估器；拒绝由预先指定的观测见证给出，说明单模均值匹配不能掩盖相关结构错误。

`cli_replay_v1/` 另外验证了从保存的 NPZ、原生 JSON、计划和协议文件重新读取的实际 CLI 路径。针对性测试共 16 项通过，其中包括支持之外的目标质量、失败符号的恒等式、错误完整 Q/路径密度、重复密度不一致、归一化错误、全局上界为 1 时仍阻止虚假 Q 覆盖不相容样本，以及拒绝对重试协议给统计接受。根任务负责在全部模块冻结后运行完整项目测试。
