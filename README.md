# deziroGBS / GBS Classical Lab

量子项目的可复现经典 Gaussian Boson Sampling 研究。最近完成：**第31轮实际原生森林核的无损标量接口，2026-11-09**。历轮代码、理论、原始成功和失败结果均按各自证据范围保留。

## 第31轮进展

新wire接口已接通实际森林事件、评分、精炼和完整采样，在Windows上显式可用。24字节dyadic编码以Fraction无损解码，计算期间设置并恢复宿主精度；科学核与旧ABI保护不变。481项独立进程/不同宿主环境对照全部通过，4个预设拒绝案例保留原错误语义。

16个小系统与独立同规则事件表的完整分布最大TV差4.19e−15，2,048条同随机数样本一致；森林代理相对Gaussian目标的TV最高仍约0.012697，另行报告。四种144模式配置共32条输出完成。19项新模块测试及653项产物审核通过，全库466运行、437通过、0失败、0错误，其它原生依赖跳过保留。

详见[第31轮报告](RESEARCH_REPORT_ITERATION31.md)、[接口范围](research/iteration31_wire_native_interface.md)、[下一轮方向](research/iteration32_plan.md)。

## 第30轮进展

实现指定有限MPS经过经典Gaussian位移后的完整阈值字符串直接采样。用正的秩一测量分支沿MPS逐站抽样，合并隐标签后恢复二元点击律，不增加位移后输出光子截断；这仍不证明有限态接近Gaussian目标。

22项正式工作生成1,073,152条输出，其中24,576条为144模式。小系统全部隐分支合并与独立完整分布最大差4.44e−16；1,471项独立核验通过。12项新测试通过，全库447运行、418通过、0失败、0错误，原生依赖跳过保留。

详见[第30轮报告](RESEARCH_REPORT_ITERATION30.md)、[正分支与测量推导](research/iteration30_positive_threshold_trajectories.md)、[下一轮计划](research/iteration31_plan.md)。

## 第29轮进展

将六种144模式有限MPS与第28轮Gaussian核心送入相同位移噪声通道。144个单模式和24个预选相邻对的直接阈值边缘分布，给出秩200完整输出TV约0.00462045的数值下界诊断；它不是完整TV或上界，也不能与此前纯态下界相减来估计误差下降。

另建立稀有真空事件的重要性积分，每个大系统固定4,096点，秩200真空概率约2.534e−19，相对经验标准误差约1.09%。22项正式工作、4,153项独立核验和11项新测试通过；全库435运行、406通过、0失败、0错误，原生依赖跳过保留。

详见[第29轮报告](RESEARCH_REPORT_ITERATION29.md)、[同通道推导](research/iteration29_same_channel.md)、[下一轮计划](research/iteration30_plan.md)。

## 第28轮进展

给出完整144模式纯态B和实噪声因子，主核心约4.964618个光子，活动噪声最小特征值+8.70e−7；明确构造的邻近输入与原矩阵相差约7.22e−14。16项正式工作完成，15接受、1拒绝；568项独立核验通过，未把浮点结果称为原输入精确分解或实验误差证书。

隔离标量编码发现Python宿主x87仅53位运算问题，调用内建立并恢复精度后301/301检查通过。原生采样接口尚未启用。全库424运行、395通过、0失败、0错误，依赖跳过保留。

详见[第28轮报告](RESEARCH_REPORT_ITERATION28.md)、[分解与精度诊断](research/iteration28_resource_and_precision.md)、[下一轮计划](research/iteration29_plan.md)。

## 第27轮进展

实现零均值混合Gaussian的全部0/1/2点击事件，44项正式工作、369项产物核验、8项新增测试通过；与小系统完整阈值分布最大绝对差4.18e−16。全库416运行、387通过、0失败、0错误，依赖跳过保留。

找到作者新混合/优化协方差，确认其与旧核链候选不同；混合减作者纯态的最小特征值约−0.011805，混合减候选约−0.00040687，独立特征值与80位二次型均确认负值。现有文件不能直接支持合法经典位移分解，也不能将第26轮纯态阈值差转移为实验输出误差。本机可移植接口目前只有迁移设计，尚未启用。

详见[第27轮报告](RESEARCH_REPORT_ITERATION27.md)、[公式推导](research/iteration27_mixed_low_clicks.md)、[来源审计](research/iteration27_source_audit.md)、[下一轮计划](research/iteration28_plan.md)。

## 第26轮进展

直接计算0/1/2点击的10441个完整模式事件，给出指定144模式Gaussian候选态与秩200有限参考之间约0.2237的阈值TV下界估计。它直接针对指定零位移输出，不是由纯态距离下界推断，也不认定候选态为原实验输入。24个小系统完整分布对照与7个大系统诊断全部通过。

原4失败与1导入错误逐项处理：全库408运行、379通过、0失败、0错误，仍有29案例和7类依赖跳过。11项新增测试、1347项主审核和118项回归审核通过；16组旧新森林分布最大TV差约1.52×10⁻¹⁶，512条配对样本相同。本机C++编译成功，但16字节扩展精度与Python的8字节格式不兼容，适配器正确拒绝，尚未解决原生依赖门禁。

详见[第26轮报告](RESEARCH_REPORT_ITERATION26.md)、[阈值事件与数值推导](research/iteration26_low_clicks_and_numerics.md)、[全部结果](results/iteration26_analysis/summary_tables.md)、[后续方向](research/iteration27_plan.md)。

## 第二十五轮进展（历史）

扫描144模式核链Gaussian候选态的全部143个物理切口。相同站点顺序下，任意秩200纯MPS到该候选态的距离下界约0.46335，秩2048仍约0.28624；这不是输出TV下界，也未识别原实验态。d4/d6/d8共21组充分上界均为1。

实际重建四种秩并保存全部张量，均未达到0.1预算；另16项非均匀压缩小系统的实际距离全部满足上界，13项达到预算。新秩200与旧PCO/递推有限参考的完整距离约7.78×10⁻⁹，有限态之间接近不代表其接近Gaussian目标。首次路径错误的17项失败及完整重跑记录一并保留。

新增5项测试及2,079项产物审核通过；全库385运行、351通过，原4失败、1错误及依赖跳过保留。详见[第二十五轮报告](RESEARCH_REPORT_ITERATION25.md)、[物理谱与必要秩推导](research/iteration25_physical_spectra.md)、[完整结果](results/iteration25_analysis/summary_tables.md)、[产物核验](results/iteration25_validation/validation.json)。

## 第二十四轮进展（历史）

57项Gaussian核链重建全部通过，外部144模式的正反向及整体稀疏消元结果一致。到该核链Gaussian候选态，有限PCO/hafnian参考的纯态距离下界约0.2304，PNR粗粒化TV下界约0.2220；两套有限参考彼此接近并不证明它们接近该候选态。它尚未被识别为原实验态，下界不能直接转给实验混合输出。

几何检查共301项（含校准及源码检查），其中145项外部上界前提失败：局部等距条件及物理/虚拟Schmidt谱不匹配，故不套用第21轮上界。新增8项测试及2,073项产物核验通过；全库380运行、346通过，原4失败、1错误和依赖跳过保留。

详见[第二十四轮报告](RESEARCH_REPORT_ITERATION24.md)、[推导与证据范围](research/iteration24_kernel_contraction.md)、[完整结果](results/iteration24_analysis/summary_tables.md)。

## 第二十三轮进展（历史）

递推接入Gaussian基MPS实际构造。8个32/128模式深层局域输入，两后端各3次、共48次隔离运行全部达到0.1构造预算；包含输入、构造、位移和16条阈值样本的完整流程，中位加速1.37–1.58倍。两后端大系统纯态最大距离9.72×10⁻¹⁵，固定种子配对样本全部相同。

四组外部144模式精度对照显示，测试的末端量化和乘法变化约1.7–2.0×10⁻⁷，未解释约2.10×10⁻⁵的全部参考差异。新增5项测试及3,278项产物检查通过；全库372运行、338通过，既有4失败、1错误和依赖跳过不变。尚未建立完整有限精度TV证书或对强经典基线的同质量成本优势。

详见[第二十三轮报告](RESEARCH_REPORT_ITERATION23.md)、[全部结果](results/iteration23_analysis/summary_tables.md)、[来源审计](research/iteration23_source_audit.md)、[文献复核](research/iteration23_literature_update.md)及[后续计划](research/iteration24_plan.md)。

## 第二十二轮进展（历史）

全部144个局部张量通过作者PCO参考交叉检查，最大系数差6.55×10⁻¹⁵；本地递推与作者PCO的完整纯态距离约2.20×10⁻¹⁴。作者PCO与hafnian有限MPS之间仍有约2.10×10⁻⁵距离，误差来源尚未完全隔离，不能直接解释为原实验输出TV。

5站点、两算法各3次串行复核全部通过。三个内部站点的递推中位耗时比本地PCO快29.7–40.2倍，仅限局部系数计算，未证明端到端GBS加速。新增11项测试及2,590项产物审计通过；全库原有4失败、1错误及依赖跳过不变。

详见[第二十二轮报告](RESEARCH_REPORT_ITERATION22.md)、[递推与稳定距离推导](research/iteration22_recurrence_overlap.md)、[完整结果](results/iteration22_analysis/summary_tables.md)、[下一轮计划](research/iteration23_plan.md)。

## 第二十一轮进展（历史）

实现Gaussian前缀酉分解与有限占据集合的直接张量生成，基于已有GSVD/PCO路线。原4个32/128模式深层局域输入在d10、χ128下均实际构造成功，上界0.01788–0.05386；4个局域确认输入也达标。正式58项工作中30项接受、27项预算超标、1项128模式Haar构造超时；该Haar输入另有约0.97564的纯态秩下界。

48个小系统实际态距离均满足新界，独立递推复算最大幅度差约1.17×10⁻¹⁵。新增8项测试、6,629项产物检查通过；遗留全库失败与跳过未变。本轮解决的是原4个输入的纯核心构造预算，未建立完整有限精度TV证书或同质量速度优势。详见[第二十一轮报告](RESEARCH_REPORT_ITERATION21.md)、[原态投影推导](research/iteration21_gaussian_projectors.md)、[完整结果](results/iteration21_analysis_release/summary_tables.md)。

## 第二十轮进展（历史）

完成60项直接前沿构造工作：10项达到0.1纯态预算、46项未达到、4项被预设资源守卫拒绝；56个状态和1,792条样本全部保留。原4个深层输入的24个窄带候选下界均超过0.58，不能达到纯态目标。250条带宽扫描表明深度12输入在字面删边规则下需保留宽度11–15，当前前沿表示的指数成本仍未解决。

新增7项测试及4,324项产物检查通过，遗留全库失败与跳过未变。1个32模式浅层构造达标，不能据此声称深层问题或完整输出TV已解决。详见[第二十轮报告](RESEARCH_REPORT_ITERATION20.md)、[转移构造推导](research/iteration20_banded_frontier.md)、[完整结果表](results/iteration20_analysis/summary_tables.md)。

## 第十九轮进展（历史）

完成980条切口谱、700条有限维MPS上下界记录。原4个局域构造失败输入在d10、χ128下的最终态存在性上界估计为0.018–0.054，4个新局域输入也低于0.1；Haar对照仍存在明显秩障碍。大系统MPS尚未实际构造，旧失败继续保留。新界依赖最终盒的一次TT-SVD，不能直接替换旧逐层预算。

72个小系统实际状态均通过完整Gaussian态距离核对，24对局域同维数/秩比较中直接法距离全部更小，但盒构造成本指数增长。新增8项测试及24,233项产物检查通过，遗留全库失败与跳过未变。详见[第十九轮报告](RESEARCH_REPORT_ITERATION19.md)、[存在性界推导](research/iteration19_final_state_mps_bound.md)、[完整诊断表](results/iteration19_analysis/summary_tables.md)。

## 第十八轮进展（历史）

将Meixner阈值投影重写为真空子集级数，消去光子多重指标枚举；证明物理均值参考使均匀参考族的展开谱半径最小。60输入、两种参考、D0…D64共7,800条记录完成。24个新种子确认输入中，均值参考32阶全部优于retained-k4，64阶最大TV约9.11e−5；原参考64阶也全部优于k4。

当前仍需全部2^m子集，完整分布构造比精确枚举更慢，未建立采样成本优势。12个四模式输入的全部阶数通过80/120位独立参考；新增8项测试、28,165项产物检查通过，遗留全库失败与跳过未变。详见[第十八轮报告](RESEARCH_REPORT_ITERATION18.md)、[真空级数与谱半径推导](research/iteration18_vacuum_series.md)、[完整对照表](results/iteration18_analysis/summary_tables.md)。

## 第十七轮进展（历史）

实现平移Wick高阶系数与阈值支持集变换。36个输入共324条唯一逐阶分布，八阶全部优于六阶，12项超过retained-k2，0项超过retained-k4；8/12模式八阶为看到六阶结果后的全面板追加探索。2个输入出现相邻升阶误差增大，原始负权重与正修复结果均保留。当前仍是显式系数和完整输出枚举，没有建立可扩展采样或强基线成本优势。

全部12个四模式输入通过80/120位独立路径核对；新增10项测试及5,363项产物复核通过，原有全库失败与跳过未变。详见[第十七轮报告](RESEARCH_REPORT_ITERATION17.md)、[方法推导](research/iteration17_shifted_wick.md)、[完整对照表](results/iteration17_analysis/summary_tables.md)。

## 第十六轮进展（历史）

加入逐层未截断检查点重试与滚动尾重分配。18次对照中8个小系统运行接受，10次较大系统搜索均未接受，原四个深层bright输入仍未解决；失败、候选扫描成本与峰值内存全部保留。四对小系统的实际态距离均改善，但未建立大系统成功成本优势。

独立24个切口谱、168条秩诊断显示：局域案例χ128的三个已检查切口距离下界估计约0.004–0.007，没有排除0.1目标；128模式Haar对照约0.974，χ512仍约0.955。它们仅约束当前纯核心自然排序下的MPS近似，不是阈值输出TV下界。

新增9项测试及2,259项产物复核通过，原有全库失败与跳过未变。详见[第十六轮报告](RESEARCH_REPORT_ITERATION16.md)、[方法推导](research/iteration16_checkpoint_budget.md)、[完整结果](results/iteration16_analysis_release/summary_tables.md)、[谱诊断](results/iteration16_schmidt/summary_tables.md)。

## 第十五轮进展（历史）

自适应构造在34个目标配置中接受30个，固定d8 χ16接受21个；68次方法运行和167份尝试账本全部保留。128模式浅层bright选择d12/cap32后预算降至0.065–0.079，但需要更多计算；512模式浅层lossy选择d6/cap16达到0.085–0.089，单次含重建成本与存储更低。4个深层bright配置仍未解决，其中2个预算超标、2个时间上限中止。

32项小系统原Gaussian态距离检查均符合预算；两方法均接受的12个小系统比较中，自适应有8项实际误差更大，不能把相同预算当作相同实际质量。新增8项测试、2,925项产物复核通过；原有全库失败与跳过未变。这是理想构造预算的浮点估计，尚非完整输出TV证书。

详见[第十五轮报告](RESEARCH_REPORT_ITERATION15.md)、[自适应预算推导](research/iteration15_adaptive_budget.md)、[全部结果及失败耗时](results/iteration15_analysis_release/summary_tables.md)。下一步研究深层高亮度的键截断与预算分配，并恢复同质量强基线对照。

## 第十四轮进展（历史）

实现相关Gaussian位移沿局域门直接传播，避免全局稠密U，并提供分批生成。20项传播质量检查及8项小系统完整条件分布比较完成，最大坐标差8.16×10^−16、最大条件分布TV约9.94×10^−17。

180次隔离进程成本运行全部完成。1024模式、深度4、2048条位移的两个输入上，局域传播比稠密全批快1.35–1.42倍，峰值工作集约195→97 MiB；其中单独分批已降至约113 MiB。深度12或128模式时局域版本更慢，完整GBS采样没有稳定速度优势。512模式完整采样完成，但构造预算0.123–0.128仍高于0.1；1024模式只验证了位移生成。

新增7项测试、1,165项产物复核全部通过；原有全库失败与依赖跳过未变。详见[第十四轮报告](RESEARCH_REPORT_ITERATION14.md)、[物理与复杂度说明](research/iteration14_local_displacement.md)、[全部成本表](results/iteration14_analysis/summary_tables.md)。下一步优先推进自适应构造质量与同质量强基线比较。

## 第十三轮进展（历史）

完成49项条件采样律的隐藏分支完整枚举，保留失败符号，并以有理数/整数累计连续CDF与53位独立均匀网格模型下的概率。48项常规小系统输入中，网格分布与120位参考的最大TV约4.13×10^−16；80/120位参考收敛复核完成，无枚举到的失败质量。此结果固定MPS与位移，尚不覆盖128模式、Gaussian位移随机数生成或固定种子PRNG的分布。

解析极小概率例子显示：exp(−64)≈1.60×10^−28的首区间事件在53位网格下获得2^−53概率，绝对误差约1.11×10^−16而相对误差约6.92×10^11。推导了网格离散化TV≤Σd_i/2^53的耦合界；该界不控制浮点收缩误差。新增10项测试及561项产物检查全部通过，原有4失败、1导入错误及依赖跳过仍在。

详见[第十三轮报告](RESEARCH_REPORT_ITERATION13.md)、[算术与耦合推导](research/iteration13_sampling_arithmetic.md)、[完整结果](results/iteration13_analysis_release/summary_tables.md)、[验证记录](results/iteration13_validation/validation.json)。下一步推进局域门直接传播相关位移、自适应构造与同质量强基线对照。

## 第十二轮进展（历史）

直接从局域光学电路构造相干MPS，避免全局Fock稠密初始化，已完成128模式采样。逐门与整层两种实现合计300个方法运行、38,400个样本；96项小系统态距离与56个解析位移平均完整分布均完成。128模式、深度4、lossy配置的d8 χ16整层构造预算约0.0068–0.0076；这是理想构造态误差估计，尚非有限精度采样器TV证书。

在两个新128模式浅层低亮度输入上，整层d6 χ8与逐门d8 χ16均满足0.1名义构造预算，串行3次重复的端到端成本比分别为1.88和1.73，最终MPS系数存储相差5.31倍。未测进程峰值内存，也没有与retained/Frugal建立同质量速度优势。高亮度或深电路仍有明显预算瓶颈。

新增17项测试全部通过；全库仍有第十一轮相同的4失败、1导入错误及原生依赖跳过。详见[第十二轮报告](RESEARCH_REPORT_ITERATION12.md)、[误差预算推导](research/iteration12_truncation_budget.md)、[完整结果表](results/iteration12_analysis/summary_tables.md)、[源代码绑定验证](results/iteration12_validation/validation.json)。

## 第十一轮进展（历史）

完成672项Gaussian切口谱、48项小系统相干MPS压缩/采样实验，以及54项低阶投影和36项retained强基线对照。64模式自然排序中心切口中，Haar bright配置即使保留2048维仍丢失约30%–32%权重；深度4局域配置达到单切口尾权重0.01所需维数仅2–12（本批三档亮度、两个种子）。这是纯态压缩诊断，不是完整输出TV结论。

二阶Meixner投影在18/18输入上优于匹配均值热态，但全部落后于retained-k2和retained-k4。MPS原型保留相干性及相关位移，12个小系统解析位移平均分布完成；初始化仍用稠密张量，且普通TT-SVD会扩展原总光子截断支持。没有建立可扩展构造或联合质量/速度优势。

新增20项测试通过，900项产物数值复核通过。本轮Windows全库验证有4失败、1导入错误、29单项及7整类跳过；扩展精度和原生依赖尚待恢复，不沿用第十轮零跳过门禁。下一步优先研究局域网络直接压缩构造、全局测量误差预算，以及高阶展开的隐式求值。

详见[第十一轮研究报告](RESEARCH_REPORT_ITERATION11.md)、[完整结果表](results/iteration11_analysis_release/summary_tables.md)、[冻结汇总](results/iteration11_analysis_release/summary.json)、[当前平台验证](results/iteration11_validation/validation.json)。

## 第十轮结论与方向（历史）

**第十轮完成了新的相干核心采样实现，但没有获得质量与速度的联合优势。** 新 C++ 原型保留核心不同总光子数之间的相干性，施加原相关 Gaussian 位移，通过秩一隐藏通道实现阈值测量。完整项目门禁 **298 项通过，0 失败、0 错误、0 跳过**。5 个不同 Haar 干涉仪对应 9 个协方差输入，完成 51/51 个质量分布及 76/76 项交叉检查。

在 16 个有双方完整质量参考和计时的配对单元中，core 的 TV 均更大、完整原生成本也均更高，**0/16 满足 TV 不高且更快**。例如 m8 lossy 的 1024 样本批次，core K2 用时 0.034565 秒、K4 用时 0.285667 秒，retained k4 为 0.004727 秒；对应 TV 分别为 0.00562895、0.000285452 和 0.0000289980。这里 core TV 明确是导出归一化系数在**理想相关位移与阈值通道**下的 `construction_reference_tv`，retained 是完整枚举的扩展原生条件律；不能把两个证据范围写成相同的有限精度采样器全局保证。

66 次预定计时全部执行，60 次完整，6 次失败均在 m32 retained-k6；它们是同三个种子在两种批长下重现的失败前缀，不是六次独立观察，失败单元不补跑、不报告成功子集的中位数。core 的 30 次计时批次全部完成，但其全局 native 失败质量仍未知。m32 K4 的 58,905 维稠密状态生成 128 样本需 13.9428 秒，没有完整 Q；m64 K6 的 131,115,985 维状态在分配前被资源上限拒绝。

**下一步需要研究保留相干性的压缩表示，并分别验证状态近似与实际测量采样误差。** 秩一隐藏通道避免保存条件混合态的全部分量，却没有消除稠密 Fock 向量的规模瓶颈。当前未实现 MPS 压缩，没有有限精度原生输出的全局 TV 证书，也没有建立通用低复杂度或完整九章模拟优势。

| 当前交付 | 入口 |
|---|---|
| 第十轮方法、全部质量与成本结果 | [研究报告](RESEARCH_REPORT_ITERATION10.md)、[完整结果表](results/iteration10_analysis/summary_tables.md)、[冻结汇总](results/iteration10_analysis/summary.json) |
| 第十轮独立验证与理论 | [相干 POVM 推导](research/iteration10_coherent_povm_theory.md)、[独立参考与正式证据审查](research/iteration10_independent_audit.md)、[298 项完整门禁](results/iteration10_validation/validation.json) |
| 第八轮实现、质量、失败与完整成本 | [研究报告](RESEARCH_REPORT_ITERATION8.md)、[完整结果表](results/iteration8_analysis/summary_tables.md) |
| 第八轮独立复核 | [数值审查](research/iteration8_numerical_audit.md)、[理论/协议审查](research/iteration8_theory_and_protocol_audit.md)、[最终产物审计](research/iteration8_final_audit.md) |
| 第九轮理论与固定资源面板 | [研究报告](RESEARCH_REPORT_ITERATION9.md)、[完整推导与先前工作](research/iteration9_core_budget_theory.md)、[资源计算说明](results/iteration9_core_budget/README.md) |
| 排除的快捷方案 | [边界路线审查](research/iteration9_boundary_route_review.md) |

![第十轮：构造参考质量与完整原生成本](results/iteration10_analysis/quality_cost.png)

第十轮原生构建、完整门禁、正式质量与计时、独立分析的复现入口见 [报告第 13 节](RESEARCH_REPORT_ITERATION10.md#13-文件与复现入口)。新编译必须建立新的源码/二进制绑定门禁，全部复跑使用新输出目录；旧失败和资源拒绝不覆盖。core 的 `TV <= sqrt(tail)` 仅属于精确相干截断态在相同理想通道下的构造，数值状态与实际采样误差需另行预算，不能套到现有 retained/forest 分布。

## 第八、九轮（历史）

**第八轮实现了共享 Schur 事件内核，提高了小系统质量与数值可靠性，但速度目标失败。** 在 12 个新输入（4 个独立 Haar 干涉仪、各 3 种亮度）上完成 72 个完整分布，全部零失败质量；72 项 Python 编排核对和 24 项独立表事件核对均通过。四种 chain/tree 候选相对各自较强 retained 基线的 TV 比值中位数为 0.726–0.915，但完整原生成本中位比为 34.73–40.43，24 个配对计时单元全部更慢。历史 4 行数值失败已消除；正式面板未触发 quad，改善发生于新的 long-double 共享/缩放路径。该轮完整回归 **268 项通过，0 失败、0 错误、0 跳过**。

**第九轮随即转向纯态核心分解、全局截断预算与压缩表示。** 已核对有损 Gaussian 态的已知纯态核心加经典位移分解，推导保留相干性的总光子数截断界 `TV <= sqrt(tail)`，并完成 42 行固定资源比较及 20 项检查。1024 模式最轻配置的核心截断为 K=36（TV 预算 0.1），但稠密偶数 Fock 表仍约有 10^67 个复振幅；因此排除大规模稠密展开。第十轮只实现其中小窗口的稠密验证原型，压缩表示仍待研究；第九轮理论不为现有 forest 分布提供 TV 证书。

![第八轮：质量改善与实际成本失败](results/iteration8_analysis/quality_cost.png)

第八轮构建与复跑命令见 [报告第 7 节](RESEARCH_REPORT_ITERATION8.md#7-文件与复现)，第九轮解析预算入口见 [报告第 8 节](RESEARCH_REPORT_ITERATION9.md#8-复现来源与完整数据)。新编译必须建立新的源码/二进制绑定门禁；全部复跑使用新输出目录，保留原始成功与失败结果。所有结论均限于所声明的模型、参数和实验，不构成通用低复杂度 GBS 算法、方法原创性或完整九章任务的模拟突破。

## 第七轮（历史）

**第七轮以条件依赖树补入遗漏点击之间的关联，9 个新输入上的 72 个完整分布、27 项交叉检查均完成。** chain-k2 在 8/9 项比 retained-k4 的 TV 更低，tree-k2 为 7/9 项，两个 k2 方法均无数值失败；TV 比值中位数分别为 0.918156、0.880217。不过 Python 准备加完整枚举的成本中位比分别为 1.183、1.119，尚未建立速度优势。

两个 k4 树均在 6/9 项低于 retained-k6，但低亮度输入有显式失败质量，且相对 pair-k4 并非普遍更准。独立保存的 18 项事后表参考诊断表明事件求值值得优先改进；完整表不能冒充可扩展后端。另给出同一、二、三模式阈值边缘但四模式联合概率不同的合法 Gaussian 相位例，明确低阶近似的信息限制；它不构成计算困难性证明。

本轮已推导共享 Schur 事件内核的具体复用公式与成本，尚未实现或实测。全套回归 **240 项通过，0 失败、0 错误、0 跳过**。没有固定全局 TV 的普适多项式保证，也没有战胜完整九章任务或建立方法原创性。

先读 [第七轮研究报告](RESEARCH_REPORT_ITERATION7.md)、[完整结果表](results/iteration7_analysis/summary_tables.md)、[理论审查](research/iteration7_theory_audit.md)、[误差预算说明](research/iteration7_error_budget_note.md) 和 [共享内核推导](research/iteration7_batched_event_kernel.md)。

![第七轮：条件树与强基线的完整TV比较](results/iteration7_analysis/forest_tv_ratios.png)

## 第六轮（历史）

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
| `positive_completion.py` | 对所有遗漏点击做单点或不相交块的条件似然补全 | 无通用 TV 保证；直接包含排除有数值失败，失败质量不删除 |
| `positive_forest.py` | 用固定链或静态无条件 MI 树补入跨块依赖 | 无条件 MI 不优化条件 TV；Python 直接事件参考与原生共享核分别保留 |
| `batched_forest.py`、`cpp_forest.py` | 第八轮共享 Schur 事件核与完整原生 forest 采样 | 数值保护仍非严格区间；固定面板完整采样明显更慢 |
| `coherent_core.py` 与原生核心采样器 | 相干总光子截断、相关位移及秩一隐藏通道阈值采样 | 稠密全奇偶 Fock 空间；不输出冒充边缘似然的隐藏路径 logQ |
| `experiments/iteration10_reference.py` | 小系统 Gaussian tilt/Wick 平均完整构造参考 | 指数规模的独立数值参考；不积分全部有限 native 失败路径 |
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
- `results/iteration6_completion/`、`results/iteration7_forest/`：本轮连续探索的冻结完整 P/Q、元数据、源码快照与全部失败记录；各自事后诊断另存。
- `results/iteration8_quality/`、`iteration8_timing/`、`iteration9_core_budget/`：第八轮原生质量/成本及第九轮固定理论资源预算。
- `results/iteration10_quality/`、`iteration10_timing/`、`iteration10_analysis/`：第十轮完整构造参考、实际采样成本、失败轨迹和独立汇总；`iteration10_validation/` 保存绑定源码与二进制的完整门禁。
- `.cache/targets/`：可重建的精确参考缓存；不跟踪到 Git。历史独立归档可能保留缓存；最新恢复包只包含Git跟踪的研究成果。

未下载或复制论文全文。`third_party/FrugalGBS`（MIT）和 `third_party/xsimd`（BSD-3-Clause）保存用于可复现对赛的上游源码子集，文件均核对固定提交的 Git blob；参见 [上游说明](third_party/README.md)。本项目自有新增代码尚未选择对外开放源码许可证。

## GitHub 与完整研究历史

公开仓库为 **[qiyuanhuakai/deziroGBS](https://github.com/qiyuanhuakai/deziroGBS)**。项目包名仍为 `gbslab`。本次按项目所有者明确授权，将本地 GBS 研究代码、正式与探索结果、失败记录、研究报告、历史草稿、复现信息和原始 Git 提交一起公开。

完整研究目录直接保存在主分支；原始研究提交通过合并保留，不重写既有提交。第 1—9 轮完整历史另有一份经过 SHA-256 核验的分卷 Git bundle，永久保存在 [公开归档提交 d5fa167](https://github.com/qiyuanhuakai/deziroGBS/tree/d5fa1679fbfbe2523af62750dd75bbdffddf6efc)。它包含原研究头 `8675497ae2ff16f7f6fbaba4ad6eb0545c1612ff` 的全部 4,169 个跟踪文件及此前提交历史；归档中的 `restore_complete_history.py` 可独立恢复并逐卷核验。

[PUBLICATION.md](PUBLICATION.md) 记录公开范围、补漏材料、构建产物和归档入口。未提交历史草稿独立归档，并明确标记为历史材料，不覆盖正式实验版本。实际用于实验的原生可执行文件和构建记录位于 `native_builds/`；它们依赖构建平台，跨平台请按源码重新构建。

此前发布受阻的过程保留在历史提交中；该旧状态不代表当前发布状态。上游 FrugalGBS 和 xsimd 的许可证继续保留；公开研究资料不替代各自许可证，本项目自有新增代码尚未指定开放源码许可证。
