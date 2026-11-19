"""Produce tables and a report only from the completed independent94 audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,hashlib
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def put(p,text):
    with (ROOT/p).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
v=read('results/iteration94_verification_v1/validation.json');assert v['passed'] and v['quality_records']==1704064 and v['confidence_events']==82
rows={r['name']:r for r in v['rows']};timing={r['job']['timing_name']:r for r in v['timings']};a=v['actual'];ab=v['actual_branch']
def us(name):return timing[name+'_t1']['median_seconds']/2048*1e6
def ci(x):return f"{100*x['lower']:.4f}%–{100*x['upper']:.4f}%"
def speed(name,other):return us(name)/us(other)
case_names=['四模式零位移','四模式位移','八模式零位移','损耗双模']
labels={'mps':'PNR MPS → 点击','branch4':'直接分支 K4','branch64':'直接分支 K64','pe':'PE PNR → 点击','frugal2':'Frugal K2','frugal3':'Frugal K3','frugal4':'Frugal K4','frugal5':'Frugal K5','direct16':'旧直接91 B16'}
table=['| 输入 | 方法 | 完整点击TV区间 | 每条微秒，CPU1 | 失败/65536 |','|---|---|---:|---:|---:|']
for r in v['rows']:
    suffix=r['name'].split('_',1)[1] if r['case']!=3 else r['name'][len('lossy_tmsv_'):]
    table.append(f"| {case_names[r['case']]} | {labels[suffix]} | {ci(r['click'])} | {us(r['name']):.6f} | {r['failures']} |")
pnr=['| 输入 | 方法 | 指标 | 名义联合99%区间 |','|---|---|---|---:|']
for r in v['rows']:
    if r['family'] in ('mps','pe'):
        metric='pnr' if 'pnr' in r else 'total_cdf';pnr.append(f"| {case_names[r['case']]} | {r['family'].upper()} | {'完整PNR TV' if metric=='pnr' else '总光子数CDF'} | {ci(r[metric])} |")
improvement=rows['pure4_frugal2']['click']['lower']/rows['pure4_branch64']['click']['upper']
frontier=dict(scope='Scoped completed94 comparisons; no arbitrary-input or Jiuzhang advantage. Nominal joint99 conditional on iid and numerical references.',quality_records=v['quality_records'],events=82,branch64_vs_frugal2_pure4=dict(guaranteed_click_error_ratio=improvement,time_ratio=speed('pure4_branch64','pure4_frugal2')),branch64_vs_frugal4_pure4=dict(time_ratio=speed('pure4_branch64','pure4_frugal4'),accuracy_intervals_overlap=True),branch64_vs_direct91_pure4=dict(speedup=speed('pure4_direct16','pure4_branch64'),accuracy_intervals_overlap=True),mps_vs_pe_pure4=dict(time_ratio=speed('pure4_mps','pure4_pe'),pnr_intervals_overlap=True),actual_pnr=dict(attempts=64,failures=a['failures'],seconds=a['seconds'],seconds_per_attempt=a['seconds']/64,raw_time_ratio_to_jz3=a['seconds']/64/1.27e-6,cdf=a['cdf'],failure_upper=a['failure_upper'],ideal_arithmetic_tv_upper=a['ideal_arithmetic_compression_tv_upper']),actual_branch=ab,sources=dict(verification_sha256=hashlib.sha256((ROOT/'results/iteration94_verification_v1/validation.json').read_bytes()).hexdigest(),jiuzhang='https://arxiv.org/html/2304.12240v3'))
hardware_path = ROOT / 'results/iteration81_jiuzhang_accuracy_v3/histograms.npz'
hardware_histogram = np.load(hardware_path)['total_click_histogram']
hardware_n = int(hardware_histogram.sum())
hardware_cdf = np.cumsum(hardware_histogram) / hardware_n
hardware_comparison = {}
for family, folder, field in (
    ('pnr', 'iteration94_actual_quality_v1', 'counts'),
    ('direct', 'iteration94_branch_actual_quality_v1', 'clicks'),
):
    archive = ROOT / 'results' / folder / 'combined.npz'
    with np.load(archive) as records:
        total_clicks = (records[field] > 0).sum(axis=1)
        total_clicks[records['failed']] = 1153
        assert len(total_clicks) == 64 and len(hardware_histogram) == 1153
    thresholds = np.arange(len(hardware_cdf))
    sample_cdf = np.searchsorted(np.sort(total_clicks), thresholds, side='right') / len(total_clicks)
    difference = np.abs(sample_cdf - hardware_cdf)
    threshold = int(np.argmax(difference))
    # Cross-check using histogram accumulation, including the failure bin.
    histogram = np.bincount(total_clicks, minlength=1154)
    assert np.array_equal(sample_cdf, np.cumsum(histogram[:-1]) / 64)
    hardware_comparison[family] = dict(empirical_distance=float(difference[threshold]),
        threshold=threshold, simulated_cdf=float(sample_cdf[threshold]),
        experimental_cdf=float(hardware_cdf[threshold]), raw_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
frontier['experimental_click_count'] = dict(hardware_records=hardware_n,
    hardware_histogram_sha256=hashlib.sha256(hardware_path.read_bytes()).hexdigest(),
    results=hardware_comparison, scope='Descriptive empirical CDF distances with failures above all physical counts. Outside the frozen confidence family.')
put('research/iteration94_frontier_v1.json',json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
report=f'''# 第94轮：PNR等待根优化与Gaussian分支直接点击

第83轮随机流勘误继续有效：旧PNR名义99%保证已撤回，第84轮已域分离并重采。第94轮新增的全部结论来自新的实验域和样本；地址不复用不代表数学证明iid，也不表示偏差消失。[勘误](RESEARCH_ERRATUM_ITERATION83_RNG.md)、[第84轮重采](RESEARCH_REPORT_ITERATION84.md)。

本轮把直接点击的Gaussian分支表示接成了完整C++采样器，不再只是矩阵接口；PNR核心则减少等待根反复构造规范态的工作。**任意输入优势和九章规模优势仍未达到。** 新直接核在实际规模64次尝试全部失败；PNR有{a['failures']}次失败，必须连同失败一起评价。

## 1. 已完成的计算与验证

26个小系统配置各65536条，另有实际PNR和直接点击各64次，共**1704064次正式尝试**。PE、Frugal和旧直接91基线均重新生成样本。预定82事件共享0.01失败概率；完整点击TV、可得PNR TV、CDF及失败概率分别标注，不把CDF上界称作全分布上界。

独立审计通过**{v['checks']}项检查**，重建全部82个具名事件、原始直方图、失败界、计时中位数和文件哈希。正式样本实际消耗的**{v['native_used_domains']}个原生用途域**无重复；PE的{v['pe_distinct_pcg_states']}个PCG初态、Frugal的224个实际工作种子分别核对，后者与第93轮也无重复。控制重放不计入独立质量N。

全库1076项运行，1047成功，零失败/错误，29项案例和7个整类跳过；10项MPS及6项新分支测试通过。科学源码/二进制的冻结哈希未变。[独立核验](results/iteration94_verification_v1/validation.json)、[全库测试](results/iteration94_tests_v1/validation.json)、[随机流审计](results/iteration94_rng_audit_v1/validation.json)。

## 2. PNR路线：降低重复计算，精度仍须单独测量

核心为`native/gaussian_mps_core94_v2.cpp`。中间等待根只做正范数QR，在接受点才构造完整规范态；使用对数割线及括区间回退，保留残差、迭代上限、误差预算和随机数消耗。101项独立根检查和8项物理门控通过；最大状态差5.8692e−9，属于数值检验，不是浮点实现的严格认证。

实际主谱同一92光子控制轨迹，v1到v2三次串行中位由26.0683秒降到21.3443秒，**快1.2213倍**，光子数、错误码和随机数消耗相同。此前93到94v1的另一组控制为1.0918倍；不跨不同实验相乘，也不推广成完整双谱吞吐加速。[控制与分项测量](research/iteration94_checkpoint_v2.md)。

新质量结果如下；每个方法/输入65536条，均无失败。八模式只给CDF，不能替代其完整PNR TV。

{chr(10).join(pnr)}

四模式MPS与PE的完整PNR精度区间重叠；同CPU1，MPS每条{us('pure4_mps'):.6f}微秒，PE {us('pure4_pe'):.6f}微秒，MPS仍慢**{speed('pure4_mps','pure4_pe'):.4f}倍**。第93轮相对旧R3的已证精度改善保留为历史结果，本轮未给R3重新采样，不把历史不同置信族冒充本轮联合比较。

## 3. 直接点击：可计量投影，但实际规模表示仍不够

`native/gaussian_click_branch94_v1.cpp`以初始Gaussian态的真空投影纯化向量为基底；利用联合真空概率计算Gram内积。每端口直接抽取0/1，再压缩Gaussian分支子空间，没有隐含光子数采样。显式计算残差并收取纯态迹距离sqrt(丢弃范数平方)，超预算即保存失败，不裁剪负范数。

48棵完整点击树及27项80位投影距离检查通过；最大距离恒等式差5.5512e−17。下表是完整浮点数值律的距离，含失败输出，不是抽样置信区间或区间算术认证。

| 输入/配置 | 完整点击TV数值 | 失败概率数值 |
|---|---:|---:|
| 四模式 K4，预算1 | 0.28644194% | 0 |
| 位移四模式 K4，预算1 | 0.69140497% | 0 |
| 八模式 K4，预算1 | 1.46162599% | 0.79099404% |
| 四模式 K64，预算0.01 | 5.0307e−17 | 0 |
| 位移四模式 K64，预算0.01 | 3.0358e−17 | 0 |
| 八模式 K64，预算0.01 | 0.00053530485% | 0.00053530485% |

八模式K64在65536条正式样本中未遇失败，不等于其失败概率为零；完整树已显示一个非零小概率失败事件。八模式K4正式样本有535条预算失败，全部计入TV和失败界。[完整树/投影审计](results/iteration94_branch_audit_v1/validation.json)。

## 4. Frugal、PE与旧直接核：速度和精度一起比较

下表均为同输入完整点击TV的名义联合99%区间。CPU1时间为每批2048条、三轮严格串行中位除以2048；原生CPU16数据和所有重复值另存。计时使用已准备模型，含输出、随机数和采样；外部准备、编译及文件I/O排除，Frugal ABI内部目标构建包含在内。PE/MPS额外生成光子数后阈值化，读者需计入其输出能力差别。

{chr(10).join(table)}

四模式新K64的TV上界{rows['pure4_branch64']['click']['upper']*100:.4f}%小于Frugal K2下界{rows['pure4_frugal2']['click']['lower']*100:.4f}%，在声明假设下，该例误差至少小**{improvement:.4f}倍**，但耗时是K2的**{speed('pure4_branch64','pure4_frugal2'):.4f}倍**。与Frugal K4精度区间重叠，新核慢**{speed('pure4_branch64','pure4_frugal4'):.4f}倍**，不能宣布更准或综合获胜。

相对旧直接91 B16，四模式新K64同线程快**{speed('pure4_direct16','pure4_branch64'):.4f}倍**，两者抽样精度区间重叠。位移输入本轮没有Frugal支持对照，表中留空，不用零位移结果替代。所有速度比仅覆盖列出的固定输入和配置。[正式计时](results/iteration94_timing_v1/summary.json)、[比较数据](research/iteration94_frontier_v1.json)。

![四类输入的完整点击精度与时间](results/iteration94_figures_v1/click_frontier.png)

## 5. 1152输出双谱实际任务与九章差距

PNR固定d6、键维32、两谱预算0.1/0.01、16工作线程。64次尝试中**{64-a['failures']}成功、{a['failures']}失败**；所有失败保留为额外输出。两谱采样调用总耗时**{a['seconds']:.6f}秒**，每次尝试**{a['seconds']/64:.6f}秒**，不是只统计成功子集。该时间包含核心模型准备与采样，排除前置损耗分解、数据I/O及独立参考计算；单次新质量运行与第93轮的种子不同，不作受控算法加速比。

总光子数CDF误差区间为**{ci(a['cdf'])}**；失败概率上界**{100*a['failure_upper']:.4f}%**。预算和失败给出的理想算术完整TV上界为**{100*a['ideal_arithmetic_compression_tv_upper']:.4f}%**，未认证浮点/求根总误差，且不把它混同于CDF上界。[实际PNR记录](results/iteration94_actual_quality_v1/summary.json)。

直接分支K8在相同实际模型下**64/64主谱预算失败，成功样本为0**。失败概率及完整点击TV下界均为**{100*ab['failure_probability_lower']:.4f}%**。5.597656秒是失败尝试累计时长，不能当成功样本吞吐，也不能与九章计算速度优势。[实际直接点击失败证据](results/iteration94_branch_actual_quality_v1/summary.json)。

先行4条控制已在第3–4次点击附近遇到约0.93–0.98的投影收费。次谱K4试验在126.0513秒后中断，未产出完整次谱记录；原始主谱结果和中断文件保留。这暴露两项独立问题：保留已测端口历史使纯化表示冗余，稠密子集分解又随无点击集合增长。独立热态控制中，未来律可保持精确而当前纯化压缩仍收费，支持下一步研究约化条件态。[容量试验及中断记录](results/iteration94_branch_actual_pilot_v1/interruption.json)。

九章3.0的144个物理输出各8个阈值格合成1152位；每原端口0–8个点击是有限伪PNR，不是理想无界光子计数。论文报告1.27微秒/记录；本轮PNR原始每次尝试仍慢约**{a['seconds']/64/1.27e-6:,.0f}倍**。两者硬件、质量和模型口径不同，这个比值只量化原始时间差距，绝非同精度比较。[九章3.0原论文](https://arxiv.org/html/2304.12240v3)。

将模型输出逐格阈值化后，用全部64次尝试与九章实测的{hardware_n:,}条记录比较总点击数。PNR的经验CDF最大距离为**{hardware_comparison['pnr']['empirical_distance']*100:.4f}%**，在阈值{hardware_comparison['pnr']['threshold']}处达到。直接核的经验距离为**{hardware_comparison['direct']['empirical_distance']*100:.4f}%**，因为全部尝试失败。失败计数统一放在所有物理点击数之上。该描述性对照位于预定82个事件之外。[实测直方图](results/iteration81_jiuzhang_accuracy_v3/histograms.npz)、[阈值与两侧CDF](research/iteration94_frontier_v1.json)。

## 6. 结论范围与后续

本轮两个检查点、所有执行版本、失败和最终质量样本均保留。密集QR、SVD、条件轨迹、Gram及压缩在C++中；Python承担编排、独立参考和统计，仓库脚本数量不是计算密集程度的度量。

PE比较固定于0.2.2官方CPU v2，Frugal固定于已核对的1.1 AVX2 FP64构建。[既有源码来源核验](results/iteration93_baseline_provenance_v2/validation.json)及[上游核查](results/iteration93_upstream_check_v1/validation.json)保留。2026-11-14再次读取的[Gaussian MPS论文v2](https://arxiv.org/abs/2512.10643v2)和[Gaussian分支编码论文v2](https://arxiv.org/abs/2603.15258v2)仍是相关方向；本轮未执行这些论文作者的完整采样实现，不宣称击败所有最新算法。张量构建时间、每键丢弃质量与完整采样时间/TV必须区分。

下一轮优先研究迹掉已测端口后的正条件态压缩、光学低秩真空概率更新，以及PNR的Schmidt预算和失败率；不靠放宽预算或删除失败换取速度。[方法](research/iteration94_methods.md)、[第95轮计划](research/iteration95_plan.md)。总体研究目标保持进行中。
'''
put('RESEARCH_REPORT_ITERATION94.md',report)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');marker='## 第94轮：PNR等待根与直接Gaussian分支\n\n'
assert marker not in text
entry=marker+f'1704064次正式尝试、82事件名义联合99%族及{v["native_used_domains"]}个原生用途域完成独立审计。新C++直接点击四模式K64相对Frugal K2的误差至少小{improvement:.4f}倍，但慢{speed("pure4_branch64","pure4_frugal2"):.4f}倍；与K4精度区间重叠且慢{speed("pure4_branch64","pure4_frugal4"):.4f}倍。PNR等待根实际配对控制加速1.2213倍。\n\n'+f'实际1152输出双谱PNR为{64-a["failures"]}/64成功，总数CDF区间{ci(a["cdf"])}；原始每次尝试比九章3.0慢约{a["seconds"]/64/1.27e-6:,.0f}倍。新直接分支64/64失败，完整点击TV至少{100*ab["failure_probability_lower"]:.4f}%，不报告成功吞吐。全库1076运行、1047成功，零失败/错误。详见[第94轮报告](RESEARCH_REPORT_ITERATION94.md)、[方法](research/iteration94_methods.md)、[独立核验](results/iteration94_verification_v1/validation.json)及[第95轮计划](research/iteration95_plan.md)。\n\n'
index=text.find('\n## ');assert index>=0;text=text[:index+1]+entry+text[index+1:];readme.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps(frontier,ensure_ascii=False,indent=2))
