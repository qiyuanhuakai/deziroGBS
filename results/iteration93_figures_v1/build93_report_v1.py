from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('.');v=json.loads((P/'results/iteration93_verification_v1/validation.json').read_text());q={r['name']:r for r in v['rows']};tm={r['job']['timing_name']:r for r in v['timings']};a=v['actual'];pct=lambda x:f'{100*x:.4f}%';interval=lambda x:f"{pct(x['lower'])}–{pct(x['upper'])}";seconds=lambda name,th=1:tm[f'{name}_t{th}']['median_seconds'];figs=P/'results/iteration93_figures_v1';figs.mkdir(exist_ok=False)
names={'mps0':'MPS小核心','mps1':'MPS大核心','r3':'旧PNR R3','r5':'旧PNR R5','pe':'Photonic Engines'}
pnr_table='\n'.join('| '+names[method]+' | '+' | '.join(interval(q[f'{case}_{method}']['pnr']) for case in ('pure4','displaced4','lossy_tmsv'))+' |' for method in names)
all_table=[]
for r in v['rows']:
    name=r['name'];extra=interval(r['pnr']) if 'pnr' in r else ('总数CDF '+interval(r['total_cdf']) if 'total_cdf' in r else '—');all_table.append(f"| {name} | {r['failures']} | {extra} | {interval(r['click'])} | {seconds(name):.7f} |")
speed_table='\n'.join(f"| {case} | {seconds(case+'_mps1'):.7f} | {seconds(case+'_mps1',16):.7f} | {seconds(case+'_pe'):.7f} | {seconds(case+'_mps1')/seconds(case+'_pe'):.4f} |" for case in ('pure4','displaced4','pure8','lossy_tmsv'))
ratio=q['pure4_r3']['pnr']['lower']/q['pure4_mps1']['pnr']['upper'];frugal_ratio=seconds('pure4_mps1')/seconds('pure4_frugal4');attempt=a['seconds']/64;jzgap=attempt/1.27e-6
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13,4.7),sharey=True)
labels=['MPS small','MPS large','PNR R3','PNR R5','PE CPU v2']
for ax,case,title in zip(axes,('pure4','displaced4','lossy_tmsv'),('4 modes: zero mean','4 modes: displaced','2 modes: lossy TMSV')):
    for k,method in enumerate(names):
        r=q[f'{case}_{method}']['pnr'];mid=r['empirical_tv']*100;ax.errorbar(mid,k,xerr=[[mid-r['lower']*100],[r['upper']*100-mid]],fmt='o',color=('#2563eb' if method.startswith('mps') else '#52525b'),capsize=4)
    ax.set_title(title);ax.set_xlabel('Full PNR TV (%)');ax.set_yticks(range(5),labels);ax.grid(axis='x',alpha=.2);ax.set_xlim(left=-.3)
axes[0].invert_yaxis();fig.suptitle('Same-input PNR accuracy: 65,536 attempts per method');fig.text(.5,.015,'Nominal joint 99% family (111 events); failures included. Numerical reference caveats apply.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(figs/'pnr_accuracy.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,case,title in zip(axes,('pure4','pure8'),('4 modes: complete click law','8 modes: complete click law')):
    selected=[r for r in v['rows'] if r['case']==(0 if case=='pure4' else 2)]
    for k,r in enumerate(selected):
        short=r['name'].replace(case+'_','');x=seconds(r['name'])/2048*1e6;y=r['click']['upper']*100;color='#dc2626' if r['family']=='frugal' else ('#2563eb' if r['family']=='mps' else '#52525b');ax.scatter(x,y,color=color,s=30);ax.annotate(short,(x,y),xytext=(5,5 if k%2==0 else -13),textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set_title(title);ax.set_xlabel('Microseconds per attempt, CPU 1 worker');ax.set_ylabel('Click TV upper bound (%)');ax.grid(alpha=.2);ax.margins(x=.16,y=.25)
fig.suptitle('Click accuracy and measured cost');fig.text(.5,.012,'Timing: median of 3 serial batches of 2,048. Quality: 65,536 fresh attempts; failures included.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(figs/'click_cost_accuracy.png',dpi=160);plt.close(fig)
shutil.copy2(Path(__file__),figs/Path(__file__).name);(figs/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figs.iterdir() if p.is_file()},indent=2)+'\n')
report=f'''# 第93轮：一般Gaussian相干MPS核心及完整精度对照

本轮完成了从固定秩二核心到多输入、复干涉、位移及相关Gaussian混态的C++轨迹实现，并对截断质量显式记账。33组小系统正式批次各65536条，另有64条实际1152输出双谱记录，共{v['quality_records']}条尝试。新大核心在四个小系统案例中均无失败；四模式零位移完整PNR TV上界{pct(q['pure4_mps1']['pnr']['upper'])}，旧R3下界{pct(q['pure4_r3']['pnr']['lower'])}，限定同输入可确认至少{ratio:.4f}倍改善。

总体目标仍未达到。新大核心同线程在四模式任务上比Photonic Engines慢；其真实规模64条中只有49条完成，15条预算耗尽。一般输入的相干更新和可计算投影误差已实现，但尚没有任意维度、同质量超越传统工具或九章的结果。

## 1. 实现变化与误差含义

Gaussian背景保留解析表示，非Gaussian核心存入潜在输入模式的MPS；每次探测施加完整相干算子之和。无探测映射支持任意位移和正负压缩，混态通过相关经典位移抽样。局部Fock越界与同次Schmidt截断合并为一次精确算术迹距离，跨物理跳跃继续累加。接近零奇异值的自适应缩减也收费。预算耗尽、光子上限或数值异常都保留为失败结果，不补抽。

新的端口选择把概率拆成正的相干率与中心化余项率，用有上界的拒绝采样减少高位移态的计算；选定端口后的相干更新完整保留。QR/SVD采用固定1线程的LAPACKE。当前代码上限为128输入、16384输出；键维仍可能迅速增长，不能以接口维度代替复杂度保证。[公式、约定和证明范围](research/iteration93_methods.md)。

直接点击沿用第91轮条件核心。本轮增加了同输入的完整点击精度面板，以及逐字一致的C++批量随机数接口；直接阈值后的受控混态核心仍需后续实现。

## 2. 精度：完整PNR与完整点击分别报告

以下为预先固定111事件族的名义联合99%区间，条件于iid及固定数值参考。域分离不证明真iid；数值交叉核对不等同于区间算术认证。全部失败映射到目标概率为零的专用结果。四模式参考枚举814385种光子数模式，并以100位精度配对和、总数卷积及真空子集独立核对；解析尾界低于3.66e−9。

| 方法 | 四模式零位移PNR TV | 四模式位移PNR TV | 损耗双模PNR TV |
|---|---:|---:|---:|
{pnr_table}

与PE的区间重叠，不能按上界大小排序真实精度。旧R5在四模式的下界也不足以建立其与新核心的严格排序。只需要点击时，四模式Frugal K4上界{pct(q['pure4_frugal4']['click']['upper'])}，新大核心上界{pct(q['pure4_mps1']['click']['upper'])}，同样未区分精度高低。

![完整PNR精度](results/iteration93_figures_v1/pnr_accuracy.png)

## 3. 速度：同输入、同线程，不混用质量结论

每次2048条，三轮严格串行，报告秒数中位；随机数、输出分配及采样计入，模型准备单独记录，编译和I/O排除。各方法均使用表中对应配置，失败仍计入工作量。CPU16列用于显示并行效果，不拿它冒充同线程优势。

| 输入 | 新大核心CPU1 | 新大核心CPU16 | PE CPU1 | 新核心CPU1 / PE CPU1 |
|---|---:|---:|---:|---:|
{speed_table}

四模式只采点击时，Frugal K4比新大核心CPU1快{frugal_ratio:.2f}倍，两者点击精度区间重叠。旧第92轮秩二特例的3.8248倍PE速度优势不能推广到本轮一般四输入。完整逐配置数据见文末；[原始计时](results/iteration93_timing_v1/summary.json)。

![点击精度与成本](results/iteration93_figures_v1/click_cost_accuracy.png)

## 4. 实际1152输出、50输入、双谱任务

固定64条、16个工作线程，主谱预算0.1、次谱0.01，局部维数6、键维32。原始协方差与子空间重构核对通过。49条完成，15条因预算耗尽停止；本批没有数值异常。总采样时间{a['seconds']:.6f}秒，每次尝试{attempt:.6f}秒。部分光子记录不计为成功样本，也不对49条成功结果作目标分布条件化。

总光子数CDF经验距离{pct(a['cdf']['empirical_distance'])}，名义联合99%区间{interval(a['cdf'])}。这个CDF上界不是完整PNR TV上界。失败概率上界{pct(a['failure_upper'])}；精确算术截断模型加失败概率的全分布上界{pct(a['ideal_arithmetic_compression_tv_upper'])}，尚未包含程序全部浮点和求根误差，不能称为完整实现的认证。

九章3.0论文报告1.27微秒/记录；本配置每次尝试原始耗时比其慢约{jzgap:,.0f}倍，且存在失败。九章3.0是144物理输出各8个阈值格、共1152位的伪光子数分辨读出，不是理想无上限整数PNR。[原论文](https://arxiv.org/html/2304.12240v3)。既有R3/直接点击完整任务的25358.69/58041.29倍差距仍单列保留；本轮MPS没有改善这一吞吐缺口。[本批原始结果](results/iteration93_actual_quality_v1/summary.json)。

## 5. 验证、随机流和文献范围

全库1070项运行，1041成功，零失败/错误，29项案例及7个类跳过；新增10项MPS测试全部通过。独立分析重建111个事件，完成{v['checks']}项检查，核对{v['native_used_domains']}个实际原生用途域无复用。PE的128个批次初始PCG状态不同；Frugal的224个实际工作线程种子不同，并以独立Python MT19937-64与C++派生核对。C++批量随机数的19908个值与原接口逐值一致。第83轮失效保证继续撤回，[勘误](RESEARCH_ERRATUM_ITERATION83_RNG.md)保持有效。

独立矩阵指数参考曾因截止过小失败；参考空子集适配与基线旧清单字段也各有一版执行失败，均保留源码和证据，使用新版本修正。最慢的旧Eigen容量探测在超过589秒后停止，保留未完成记录，不计成功或速度优势。未用失败后改写旧脚本的方法覆盖证据。

本轮在线核对了PE0.2.2与Frugal1.1当前主要采样源码，并核对历史二进制和源清单。已有Gaussian MPS文献的约一分钟指标对应中央张量构建，且其约0.03指标是切口丢弃质量；不能与本报告整条采样耗时或完整分布TV直接比较。[Gaussian MPS原论文](https://arxiv.org/html/2512.10643v2)。本轮没有执行该论文完整采样器，不能宣称胜过它。

[完整测试](results/iteration93_tests_v1/validation.json) · [独立审计](results/iteration93_verification_v1/validation.json) · [固定质量计划](research/iteration93_quality_plan_v1.json) · [上游核对](results/iteration93_upstream_check_v1/validation.json) · [下一轮计划](research/iteration94_plan.md)。

## 附：全部同输入配置

每行65536条质量尝试；耗时为CPU1的2048条中位。pure8只有总数CDF和完整点击TV，不提供完整PNR TV。较小核心失败总数1589；较大核心四组均为零。

| 配置 | 失败数 | 完整PNR TV／注明的总数CDF | 完整点击TV | CPU1秒/2048条 |
|---|---:|---:|---:|---:|
{chr(10).join(all_table)}
'''
(P/'RESEARCH_REPORT_ITERATION93.md').write_text(report,encoding='utf-8')
(P/'research/iteration93_frontier_comparison.md').write_text(f'''# 第93轮前沿比较

本轮实际执行PE0.2.2 CPU v2、Frugal1.1 AVX2 FP64、旧R3/R5与直接点击B4/B16，共33组同输入质量数据。具体参数、精度和三轮串行耗时见[完整报告](../RESEARCH_REPORT_ITERATION93.md)。

四模式零位移，新大核心PNR上界{pct(q['pure4_mps1']['pnr']['upper'])}、PE上界{pct(q['pure4_pe']['pnr']['upper'])}，均含零，无法排出精度优劣。新大核心同线程比PE慢{seconds('pure4_mps1')/seconds('pure4_pe'):.4f}倍。Frugal K4点击上界{pct(q['pure4_frugal4']['click']['upper'])}与新核心上界{pct(q['pure4_mps1']['click']['upper'])}重叠，但Frugal快{frugal_ratio:.2f}倍。第92轮低秩特例优势不能外推。

真实1152输出双谱64条中15条失败；平均每次尝试{attempt:.6f}秒，比九章3.0论文原始1.27微秒慢{jzgap:.2f}倍。这不是同质量比较，新实现未获得真实规模优势。旧R3/直接点击25358.69/58041.29倍原始差距继续保留。

[Gaussian MPS文献](https://arxiv.org/html/2512.10643v2)约一分钟指中央张量构建，约0.03指切口丢弃质量；这两个量不能替代整条采样时间与完整TV。本轮没有执行该论文完整采样器。下一步优先减少等待根的重复QR，并针对实际50输入的键维和局部投影分别测量误差成本；直接点击继续研究受控条件混态核心。
''',encoding='utf-8')
(P/'research/iteration93_worklog.md').write_text(f'''# 第93轮执行记录

起点main为bfa67e34303db5731a9b99ffd4d037c47d1016c1，工作分支research/iteration93-gaussian-mps-core。总体目标保持active。

1. 推广一般Gaussian无探测映射，独立矩阵指数检查；保留初版参考截止不足的失败证据。
2. 实现C++ MPS核心、显式局部越界和Schmidt误差。v2合并同次嵌套投影，v3接入固定线程LAPACKE，v4增加中心化正率提议，v5增加收费的自适应键维。
3. 24个显式核心案例核对实际迹距离；40个案例核对中心化率与包络。测试保留跨线程/分批重放、预算失败及极小非零奇异值收费。
4. 冻结111事件、33组65536条及64条真实双谱，执行全部{v['quality_records']}条尝试。参考空子集适配失败保留v1，正式使用v2参考；实际批次15条预算失败全部保留。
5. 执行PE、Frugal、R3/R5、直接点击同输入质量与三轮串行CPU1/CPU16计时。实际工作线程种子和原生域单独核验；不把控制重放计入质量N。
6. 全库1070运行/1041成功，零失败错误；独立核验{v['checks']}项，核对{v['native_used_domains']}个实际用途域。基线清单审计字段错误保留v1，v2通过117个Frugal文件、63个PE源码和3个旧二进制核对。
7. 报告和图由独立核验结果生成。发布时还需确认图像、字节清单、远端commit/tree和两条分支引用；发布回执记录最终状态。
''',encoding='utf-8')
(P/'research/iteration94_plan.md').write_text(f'''# 第94轮：降低一般输入的重复计算与压缩成本

总体目标继续active：任意输入维度的速度与精度优势、两条路线并行推进、C++密集计算、Python调度、持续公开证据。第93轮的四/八模式结果不替代实际1152输出50输入双谱目标；当前新核心64条有15条失败，原始耗时仍比九章3.0慢约{jzgap:.0f}倍。

1. 用冻结控制轨迹细分局部Fock越界、Schmidt丢弃、等待根QR和跳跃SVD成本。保留失败和完整消耗日志，不用成功子集给新精度结论。
2. 先研究以当前平均光子率初始化等待根，保留严格括区间、残差门控与原映射。比较逐值控制重放、根调用次数及完整串行耗时，再生成独立质量批次。
3. 测量潜在模式排序、Gaussian框架调整对键维的影响。局部幺正不能降低模式间Schmidt秩；若使用非局部变换，变换本身的近似与成本必须计入。
4. 直接点击推进阈值后条件混态核心。保留非Gaussian信息与正性，先在完整点击律可枚举输入上核验，再与Frugal K2/K3/K5及91核心比较。当前批量随机数优化不能替代这项算法工作。
5. 继续PE、Frugal和最新MPS文献的准确口径；区分张量构建、采样、完整PNR/点击TV及局部丢弃质量。未执行的论文方法不宣称被超越。
6. 每项新方案预先固定样本、域与置信事件；旧脚本/失败证据不覆盖。一次一个重任务，正式速度严格串行，完成独立审计与GitHub发布。
''',encoding='utf-8')
intro=f'''## 第93轮 一般Gaussian相干MPS核心

新增C++多输入、位移与相关混态轨迹，显式记录截断误差。2162752条正式尝试完成；四模式零位移PNR TV上界{pct(q['pure4_mps1']['pnr']['upper'])}，旧R3下界{pct(q['pure4_r3']['pnr']['lower'])}，限定同输入至少{ratio:.4f}倍改善。新核心在该例比PE同线程慢{seconds('pure4_mps1')/seconds('pure4_pe'):.4f}倍；Frugal K4点击精度区间与新核心重叠，速度快{frugal_ratio:.2f}倍。

真实1152输出50输入双谱64条中49条完成、15条预算耗尽；每次尝试{attempt:.4f}秒，未获得真实规模优势。1070项测试运行/1041成功，独立核验38148项、3097149个实际原生用途域。详见[第93轮报告](RESEARCH_REPORT_ITERATION93.md)、[方法](research/iteration93_methods.md)及[第94轮计划](research/iteration94_plan.md)。

'''
readme=P/'README.md';data=readme.read_bytes();marker='## 第92轮'.encode();assert data.count(marker)==1;assert '## 第93轮'.encode() not in data;newline='\r\n' if b'\r\n' in data else '\n';readme.write_bytes(data.replace(marker,intro.replace('\n',newline).encode()+marker,1));print(dict(r3_precision_ratio=ratio,core_to_pe=seconds('pure4_mps1')/seconds('pure4_pe'),core_to_frugal=frugal_ratio,actual_to_jiuzhang=jzgap))
