"""Read frozen artifacts, independently recompute metrics, save tables/figure."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    mp=ROOT/'results/iteration11_mps_probe'
    me=ROOT/'results/iteration11_meixner'
    co=ROOT/'results/iteration11_retained_control'
    spectra=read(mp/'spectral_rows.json')
    compression=read(mp/'compression_rows.json')
    meixner=read(me/'rows.json')
    control=read(co/'rows.json')
    checks=[]
    support=[]
    for row in compression:
        label,bond=row['input'],row['max_bond']
        data=np.load(mp/(label+f'_D{bond}.npz'))
        x=data['A0']
        for j in range(1,row['modes']):
            x=np.tensordot(x,data[f'A{j}'],axes=(-1,0))
        x=x[0,...,0]
        original=np.load(mp/(label+'_input.npz'))['tensor']
        overlap=np.vdot(original.ravel(),x.ravel())
        err2=np.linalg.norm(original-x/(overlap/abs(overlap)))**2
        distance=math.sqrt(max(0,err2*(1-err2/4)))
        checks.append(abs(distance-row['state_trace_distance'])<1e-12)
        checks.append(abs(np.linalg.norm(x)-1)<1e-12)
        totals=np.indices(x.shape).sum(axis=0)
        support.append({'input':label,'bond':bond,
            'above_total_cutoff_mass':float(np.sum(np.abs(x[totals>row['cutoff']])**2)),
            'odd_mass':float(np.sum(np.abs(x[totals%2==1])**2)),
            'mps_elements':row['stored_complex_elements'],
            'total_budget_elements':math.comb(row['modes']+row['cutoff'],row['cutoff'])})
        if row.get('averaged_status')=='complete':
            a=np.load(mp/(label+f'_D{bond}_averaged.npz'))
            checks.append(abs(.5*np.abs(a['target']-a['compressed']).sum()-row['averaged_tv_vs_target'])<1e-12)
            checks.append(row['averaged_contraction_check'])
    # Independent non-Hermitian spectrum checks the Hermitian sqrt-V method.
    for row in spectra:
        v=np.load(mp/(row['input']+'.npz'))['pure_covariance']
        m=row['modes']; left=row['order'][:row['cut']]
        if len(left)>m//2:
            left=[j for j in range(m) if j not in left]
        idx=left+[m+j for j in left]; n=len(left)
        omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
        values=np.linalg.eigvals(1j*omega@v[np.ix_(idx,idx)])
        nu=np.sort(values.real)[n:]
        q=(nu-1)/(nu+1)
        checks.append(np.max(np.abs(values.imag))<1e-10 and np.max(np.abs(q-row['geometric_parameters']))<1e-10)
    for row in meixner:
        label,d=row['input'],row['degree']
        p=np.load(me/(label+'_input.npz'))['target']
        q=np.load(me/(label+f'_D{d}.npz'))
        checks.append(abs(.5*np.abs(p-q['repaired']).sum()-row['tv_repaired'])<1e-12)
        checks.append(abs(q['raw'].sum()-1)<1e-12)
    by_control={(r['input'],r['k']):r for r in control}
    degree2=[r for r in meixner if r['degree']==2]
    validation=read(ROOT/'results/iteration11_validation/validation.json')
    class_skips=sum('setUpClass' in r['test'] for r in validation['skip_details'])
    summary={'spectral_rows':len(spectra),'compression_rows':len(compression),
        'averaged_laws':sum(r.get('averaged_status')=='complete' for r in compression),
        'quality_rows_meixner':len(meixner),'control_rows':len(control),
        'independent_numeric_checks':len(checks),'independent_numeric_passed':sum(bool(v) for v in checks),
        'degree2_wins_matched_thermal':sum(r['tv_repaired']<r['matched_mean_thermal_tv'] for r in degree2),
        'degree2_wins_retained_k2':sum(r['tv_repaired']<by_control[r['input'],2]['tv_extended'] for r in degree2),
        'degree2_wins_retained_k4':sum(r['tv_repaired']<by_control[r['input'],4]['tv_extended'] for r in degree2),
        'degree2_negative_laws':sum(r['negative_mass']>1e-14 for r in degree2),
        'control_nonzero_failure_rows':sum(r.get('failure_mass',0)>0 for r in control),
        'max_compression_mass_above_original_cutoff':max(r['above_total_cutoff_mass'] for r in support),
        'mps_smaller_than_total_budget_basis_rows':sum(r['mps_elements']<r['total_budget_elements'] for r in support),
        'validation':{'tests_run':validation['run'],'passed':validation['passed'],
            'failures':validation['failures'],'errors':validation['errors'],
            'skipped_test_cases':validation['skipped']-class_skips,'skipped_classes':class_skips,
            'new_tests_passed':validation['new_passed'],'full_gate_passed':False}}
    if not all(checks):
        raise AssertionError('Independent artifact check failed')
    summary['bound_inputs']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [mp/'spectral_rows.json',mp/'compression_rows.json',me/'rows.json',me/'budgets.json',co/'rows.json',ROOT/'results/iteration11_validation/validation.json']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.output/'support_diagnostics.json').write_text(json.dumps(support,indent=2),encoding='utf-8')
    lines=['# 第十一轮完整结果表','', '## 64 模式自然排序中心切口','',
        '| 网络 | 配置 | 种子 | 尾权重≤0.01所需维数 | 2048维剩余尾权重 |',
        '|---|---|---:|---:|---:|']
    for r in spectra:
        if r['modes']==64 and r['ordering']=='natural' and r['cut']==32:
            rank=r['required_rank']['0.01']
            lines.append(f"| {r['ensemble']} | {r['profile']} | {r['seed']} | {rank if rank else '>2048'} | {r['tail_at_cap']:.9g} |")
    lines+=['','## 二阶投影与强基线','', '| 输入 | 二阶修正后TV | 匹配均值热态TV | retained k2 TV | retained k4 TV | 二阶负质量 |', '|---|---:|---:|---:|---:|---:|']
    for r in degree2:
        c2,c4=by_control[r['input'],2],by_control[r['input'],4]
        lines.append(f"| {r['input']} | {r['tv_repaired']:.8g} | {r['matched_mean_thermal_tv']:.8g} | {c2['tv_extended']:.8g} | {c4['tv_extended']:.8g} | {r['negative_mass']:.8g} |")
    lines+=['','## MPS完整位移平均的构造参考','', '| 输入 | 键维数上限 | 相对目标TV | 未压缩核心TV | 压缩态迹距离 |', '|---|---:|---:|---:|---:|']
    for r in compression:
        if r.get('averaged_status')=='complete':
            lines.append(f"| {r['input']} | {r['max_bond']} | {r['averaged_tv_vs_target']:.9g} | {r['uncompressed_averaged_tv_vs_target']:.9g} | {r['state_trace_distance']:.9g} |")
    lines+=['','## 新论文的有符号L1截断预算','', '误差是有符号分布的L1目标，Haar失败概率0.05；朴素系数数不是论文随机算法的运行成本。','',
        '| 模式 | 配置 | L1预算 | 充分阶数D | 朴素系数数log10 |', '|---:|---|---:|---:|---:|']
    for r in read(me/'budgets.json'):
        lines.append(f"| {r['modes']} | {r['profile']} | {r['l1_error']} | {r['degree']} | {math.log10(int(r['naive_coefficient_count'])):.3f} |")
    (args.output/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    colors={'lossy':'#2274a5','bright':'#e28f22','stress':'#c94855'}
    for profile in colors:
        for ensemble,style in [('haar','-'),('local_depth4','--')]:
            xx,yy=[],[]
            for m in [8,16,32,64]:
                rs=[r for r in spectra if r['modes']==m and r['cut']==m//2 and r['ordering']=='natural' and r['profile']==profile and r['ensemble']==ensemble]
                vals=[r['required_rank']['0.01'] or 2048 for r in rs]
                xx.append(m); yy.append(np.median(vals))
                if any(r['required_rank']['0.01'] is None for r in rs):
                    axes[0].scatter([m],[np.median(vals)],marker='^',color=colors[profile],s=65)
            axes[0].plot(xx,yy,style,marker='o',color=colors[profile],label=f'{profile}, {ensemble}')
    axes[0].set(xscale='log',yscale='log',xlabel='Modes',ylabel='Central-cut rank for discarded weight <= 0.01',title='Gaussian core compression requirement')
    axes[0].axhline(2048,color='gray',linewidth=.8)
    axes[0].set_title('Gaussian core compression requirement\nTriangles: at least one seed exceeds rank 2048',fontsize=11)
    axes[0].set_ylim(.8,5500)
    axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
    for profile in colors:
        rs=[r for r in degree2 if r['profile']==profile]
        axes[1].scatter([by_control[r['input'],2]['tv_extended'] for r in rs],
            [r['tv_repaired'] for r in rs],label=profile,color=colors[profile])
    axes[1].plot([0,.3],[0,.3],color='gray',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Retained k2 TV (including failure mass)',ylabel='Degree-2 repaired projection TV',
        title='Stronger quality control on 18 inputs',xlim=(-.005,.3),ylim=(0,.3))
    axes[1].legend()
    fig.savefig(args.output/'research_summary.png',dpi=180)
    fig.savefig(args.output/'research_summary.svg')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
