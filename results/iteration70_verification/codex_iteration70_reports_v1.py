from pathlib import Path
import json,statistics,math
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def put(name,text):
    p=ROOT/name;assert not p.exists();p.write_bytes(text.encode('utf-8'))
bench=read('results/iteration70_benchmark/summary.json');times={m:statistics.median(r['compute_seconds'] for r in bench['rows'] if r['method']==m) for m in ('direct','pnr','frugal_K2','frugal_K3')}
speed_table='\n'.join(f"| {label} | {times[m]:.6f} | {512/times[m]:.2f} | {times[m]/times['direct']:.3f} 倍 |" for m,label in [('direct','直接点击 C++'),('pnr','PNR 后转点击 C++'),('frugal_K2','官方 Frugal K2 CPU'),('frugal_K3','官方 Frugal K3 CPU')])
quality=read('results/iteration70_quality_extension_v2/summary.json');lo,hi=quality['nominal_full_pnr_tv_interval'];assert quality['below_one_percent'] and hi<.01
direct=read('results/iteration70_archive/iteration70_direct_click_pilot_v1_20261110/summary.json');small=read('results/iteration70_archive/iteration70_frugal_speed_v2_20261110/summary.json');rows=[]
for c,(r,e) in enumerate([(.7,.1),(.7,.3),(.7,.65),(1.1,.65)]):
    d=next(x for x in direct['rows'] if x['case']==c and x['label']=='displaced');f2=next(x for x in small['rows'] if x['case']==c and x['method']=='frugal_K2');f3=next(x for x in small['rows'] if x['case']==c and x['method']=='frugal_K3');rows.append(f"| {r} | {e} | {100*d['observed_threshold_tv']:.4f}% | {100*f2['induced_threshold_tv']:.4f}% | {100*f3['induced_threshold_tv']:.4f}% |")
quality_table='\n'.join(rows)
gap_direct=times['direct']/512/1.27e-6;gap_pnr=times['pnr']/512/1.27e-6
report=f'''# 第70轮 C++ 双路线采样与完整分布检验

本轮把整数光子计数与直接二值点击实现为两条独立 C++ 路线。144输出、50压缩输入的新合成矩阵上，16线程、每批512样本、三轮交错计时，直接点击比 PNR 后转点击快 **{times['pnr']/times['direct']:.3f}倍**，比当前通用 CPU 构建的 Frugal K2/K3 快 **{times['frugal_K2']/times['direct']:.3f}/{times['frugal_K3']/times['direct']:.3f}倍**。这是不同近似律的运行速度对照，未建立大系统同误差优势。

准确性方面，一个固定四模、r=1.4、η=0.43 输入新增8,000万条独立种子区域的 PNR 记录，完整联合计数 TV 的名义99%区间为 **{100*lo:.4f}%～{100*hi:.4f}%**，在所声明的独立随机流及数值目标假设下低于1%。另一较难四模输入的 PNR 误差仍有4.09%的名义下界。没有取得任意维度的普遍准确性或对九章的综合胜出。

## 两条可调用的计算核心

- [PNR C++ 核心](native/gaussian_reset_parallel.cpp)和[接口](src/gbslab/native_gaussian_reset_parallel.py)：保留整数光子数，Newton 等待时间反演、谱求解、端口选择、条件更新及多核轨迹全部原生执行。
- [直接点击 C++ 核心](native/gaussian_direct_click.cpp)和[接口](src/gbslab/native_direct_click.py)：逐端口判断点击，不生成光子数；每步只求解2×2条件矩问题并更新低秩状态，工作量 O(MN²)。
- [binary128 PNR 参考器](native/gaussian_pgf_quad.cpp)：小系统完整计数概率，用于检验误差；成本随模式数和光子界限组合增长。

两条采样路线都存在 Gaussian 矩近似。直接点击的无点击分支条件更新是 Gaussian 的，点击分支只保留一、二阶矩；PNR 路线也在光子减法后重置为 Gaussian 态。[方法文档](research/iteration70_methods.md)给出公式、已有文献来源、数值保护和统计区间的准确含义。

## 144输出的同机速度对照

输入为新种子705001的144×50复等距矩阵，r=1.4、η=0.43。四种方法使用同一协方差、16个 CPU 工作线程和512条完整输出，重复三次并轮换次序。计时包含模型或目标准备；进程启动和 Frugal JSON 输出另列。主要编译设置相同，PNR 的全端口扫描另调用明确标识的单线程 OpenBLAS。

| 方法 | 每批中位秒数 | 样本每秒 | 相对直接点击的耗时 |
|---|---:|---:|---:|
{speed_table}

这是[公开复现入口](experiments/iteration70_benchmark.py)的[最终运行记录](results/iteration70_benchmark/summary.json)。较早的完整三轮单线程及16线程面板、第一次双路线计时也全部保留，没有选择最快一次作为成绩。新公开直接点击输出与原型记录逐条复核。

Frugal 的本机指令集构建出现访问异常，因此正式比较使用其可运行的通用 x86-64 双通道 SIMD 构建；失败二进制及诊断原样归档。此结果没有覆盖 Frugal 最优 CPU 指令集和 GPU 配置，也没有证明144模完整分布误差相等。

## 小系统仍有不利结果

同机单线程四模面板中，Frugal K4 每20万样本约0.017～0.019秒，优化后的 PNR 路线约0.84～2.69秒，Frugal 快约44～154倍；K2/K3 的速度优势最高约177倍。小系统与大系统的排序不同，不能由其中一组外推另一组。

直接点击的四模位移版本，每个输入100万样本，其完整16项点击分布观测 TV 如下；Frugal 列为官方程序的数值诱导律误差。

| r | η | 直接点击 TV | Frugal K2 TV | Frugal K3 TV |
|---:|---:|---:|---:|---:|
{quality_table}

八个直接点击面板在理想 IID 假设下的名义联合99% TV 半径为0.2981个百分点。最后一行位移版本区间为0.4072%～1.0035%，低于 K2/K3；第三行仍落后于 K3。**四模 Frugal K4 的数值误差约10⁻¹⁵，仍优于本轮所有这些矩闭合近似。** 不能把胜过部分截断配置表述成胜过最佳 Frugal 配置。

中心直接点击版本可枚举自身完整诱导律；四例 TV 为0.0749%、0.6084%、3.5229%、5.9139%。因此保留输入位移噪声仍很重要，但并未消除所有输入的近似偏差。

## 完整 PNR 的进展与重尾

五组输入用 C++ binary128 生成625,675项完整计数模式概率，并保留超出参考表的全部尾质量。先进行35组各10万样本的噪声分解初筛，再用15组各100万样本独立复验。把噪声分到两个正交分量没有带来普遍改善；较难输入增加纯核压缩后误差和过重尾部明显增大。

r=1.1、η=0.65 的最小纯核分解，完整计数经验 TV 为4.9108%，名义区间4.0927%～5.1109%。这比只看总光子数的第69轮诊断更直接，与1%目标至少相差约3.09个百分点。

r=1.4、η=0.43 的固定四模输入，2,000万记录的上界仍为1.0242%，故继续执行预先固定8,000万记录的新实验。最终经验 TV 为{100*quality['empirical_joint_binned_tv']:.4f}%，名义区间{100*lo:.4f}%～{100*hi:.4f}%。该区间包含未列举尾箱的修正，但仍依赖理想独立流和数值目标，不是有限 PRNG 与全局舍入的严格证书，也不适用于其他矩阵或维度。

8,000万记录中总光子数≥128出现11次，频率1.375×10⁻⁷；解析目标概率约2.190×10⁻¹⁵，相差约6,279万倍。最大记录为7,511，全部保留。较小的整体 TV 仍可伴随极端稀有事件和高阶矩的严重失真，不能省略这一点。[20M原始结果](results/iteration70_quality_extension/summary.json)和[80M独立结果](results/iteration70_quality_extension_v2/summary.json)都可重算。

## 并行和正确性证据

PNR 预试验的144/1152/8176输出案例中，16线程相对同二进制一线程分别约快5.23/6.74/6.59倍，所有测试记录在1、4、8、16线程下逐项一致。Newton 在所测轨迹中通常4～5次迭代，替代固定80次二分；LAPACKE 谱求解没有稳定胜过 Eigen，默认保留 Eigen。

直接点击的240项完整物理空间条件矩检查通过；真空、相干态和单模压缩态的解析检查通过。公开接口重放800,576条原型记录，全部一致。完整测试集828项运行、799通过、0失败、0错误；29个历史跳过测试及7个跳过测试类保留，新PNR并行9项、直接点击7项均通过。

原始私有实验773个文件、96,785,170字节按字节归档，涵盖编译异常、语法错误、预试验和负面质量结果。公开参考器、公开重放、最终计时及两次大样本扩展单独保存；科学清单还包含后续核验结果。

## 距离九章与后续方向

本轮144输出直接点击的原始时间约为九章3.0报告1.27微秒的 **{gap_direct:,.0f}倍**，PNR路线约为 **{gap_pnr:,.0f}倍**。我们的矩阵为合成输入，测量响应和质量条件也不同，这些是量级差而非同任务排名。第70轮没有实现对九章的量子优势反转。[前沿对照](research/iteration70_frontier_comparison.md)列出具体范围。

下一轮按[两条路线的计划](research/iteration71_plan.md)推进：直接点击保留有限带符号 Gaussian 分支，PNR 保留光子减法产生的非 Gaussian 信息；继续补强 Frugal 指令集/GPU 对照和实际实验输入，而非只增加参数扫描。
'''
put('RESEARCH_REPORT_ITERATION70.md',report)
frontier=f'''# 第70轮 Frugal 与九章的对照范围

本轮同时测量 PNR 后转点击和直接点击。速度表对应同一144×50合成输入、r=1.4、η=0.43、16个 CPU 工作线程、每批512样本的三次交错中位值。

| 方法 | 每批秒数 | 样本每秒 | 相对直接点击耗时 |
|---|---:|---:|---:|
{speed_table}

数据来自[公开基准](../results/iteration70_benchmark/summary.json)。Frugal 使用作者固定提交的未修改采样入口；其通用 CPU 构建可正常运行，本机指令集构建的访问异常已保留。没有拿这份 CPU 结果代表全部 Frugal 实现的最高吞吐率。[Frugal论文](https://arxiv.org/abs/2511.14923)的统计基准与完整 TV 也不是同一准确性定义。

四模同机面板中，Frugal K4 同时更快、数值诱导律更准确。大系统中直接点击速度领先这份 CPU 构建，但未取得完整144模 TV 的上界，所以不能给出同误差综合胜出结论。直接点击的四模误差和 PNR 的四模误差分别有完整分布检验；二者不能混为一个质量成绩。

[九章2.0](https://arxiv.org/pdf/2106.15534)为144输出、50个等效单模压缩输入，最高113个探测事件。本轮覆盖矩阵维度，但使用合成矩阵和统一参数，未复现它的实验样本。250kHz泵浦频率没有被当作有效样本吞吐率。

[九章3.0](https://arxiv.org/html/2304.12240v2)使用144物理模式及1152个分流探测箱，报告1.27微秒每样本。当前直接点击的144端口时间为{times['direct']/512*1000:.4f}毫秒，原始比例约慢{gap_direct:,.0f}倍；PNR为{times['pnr']/512*1000:.4f}毫秒，约慢{gap_pnr:,.0f}倍。直接144位点击、完整PNR和1152箱伪PNR的测量输出不同，这些比例只说明时间量级。

8176输出的PNR多核预试验最多使用64个压缩输入，仍比[九章4.0](https://arxiv.org/html/2508.09092v3)的1024输入少16倍；输出数相同并未覆盖其计算难度。

[PCO与Gaussian SVD](https://arxiv.org/html/2512.10643v2)及已有 MPS 路线继续作为另一类基线。第68轮的 C++/CUDA 读出结果针对给定有限态，必须再纳入态构建和截断误差后才能与本轮完整采样成本相比。两条新路线的下一步分别是条件差态分支保留和PNR非Gaussian修正，见[第71轮计划](iteration71_plan.md)。
'''
put('research/iteration70_frontier_comparison.md',frontier)
p=ROOT/'README.md';text=p.read_text(encoding='utf-8');header=f'''\n\n## 第70轮 C++ 光子计数与直接点击双路线

新增独立的PNR多核采样和直接点击C++核心。144输出、50输入、16线程同机交错计时，直接点击每512样本中位{times['direct']:.3f}秒，相对PNR转点击快{times['pnr']/times['direct']:.2f}倍，相对当前通用CPU Frugal K2/K3快{times['frugal_K2']/times['direct']:.2f}/{times['frugal_K3']/times['direct']:.2f}倍；不同近似律的速度优势不等于同质量胜出。

固定四模PNR输入新增8,000万记录，完整TV名义区间{100*lo:.3f}%～{100*hi:.3f}%，但其他输入仍有超过4%的误差及过重尾部。四模Frugal K4仍更快、更准确。公开接口800,576条记录重放一致；全库828运行、799通过、0失败、0错误，历史跳过保留。

详见[第70轮报告](RESEARCH_REPORT_ITERATION70.md)、[两个C++接口和复现](native/README.md)、[方法与误差](research/iteration70_methods.md)、[Frugal及九章对照](research/iteration70_frontier_comparison.md)、[双路线后续](research/iteration71_plan.md)。
'''
first,rest=text.split('\n',1);text=first+header+'\n'+rest;text=text.replace('当前进展：**第69轮C++ Gaussian重置采样与质量对照，2026-11-10**','当前进展：**第70轮C++光子计数与直接点击双路线，2026-11-10**');p.write_bytes(text.encode())
print(json.dumps(dict(times=times,direct_over_pnr=times['pnr']/times['direct'],direct_over_frugal2=times['frugal_K2']/times['direct'],direct_over_frugal3=times['frugal_K3']/times['direct'],jz3_raw_direct_ratio=gap_direct),indent=2))
