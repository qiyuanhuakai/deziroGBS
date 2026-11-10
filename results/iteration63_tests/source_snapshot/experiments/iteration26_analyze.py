"""Summarize threshold-output lower bounds and separately audited regressions."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    out=ROOT/'results/iteration26_analysis'; out.mkdir(exist_ok=False); sha=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    rows=read('results/iteration26_panel/rows.json'); audit=read('results/iteration26_validation/validation.json'); tests=read('results/iteration26_tests/validation.json'); native=read('results/iteration26_native/probe.json'); regression=read('results/iteration26_forest_regression/summary.json'); rp=read('results/iteration26_forest_regression/plan.json'); rr=read('results/iteration26_forest_regression/rows.json'); boundaries=read('results/iteration26_forest_regression/boundaries.json')
    assert audit['count']==audit['passed'] and len(rows)==31 and all(r['status']=='passed' for r in rows)
    checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    for f,s in rp['source_hashes'].items(): check('regression source '+f,digest(ROOT/f)==s and digest(ROOT/'results/iteration26_forest_regression/source_snapshot'/f)==s)
    for f,s in regression['dependencies'].items(): check('regression historical dependency '+f,digest(ROOT/f)==s)
    check('regression frozen plan',digest(ROOT/'research/iteration26_regression_plan.md')==digest(ROOT/'results/iteration26_forest_regression/frozen_plan.md'))
    check('regression complete counts',len(rr)==len(rp['jobs'])==regression['finished']==regression['passed']==16 and regression['unchanged_sources'])
    for r,j in zip(rr,rp['jobs']):
        i=r['sequence']; check(f'paired {i} config',r['status']=='passed' and all(r[k]==v for k,v in j.items()))
        path=ROOT/f'results/iteration26_forest_regression/case_{i:02d}.npz'; check(f'paired {i} bytes',digest(path)==r['output_sha256'])
        with np.load(path,allow_pickle=False) as z:
            p=z['old_distribution']; q=z['new_distribution']; a=z['old_samples']; b=z['new_samples']
            check(f'paired {i} complete laws',p.shape==q.shape==(1<<r['modes'],) and min(p.min(),q.min())>=0 and max(abs(p.sum()-1),abs(q.sum()-1))<1e-11 and abs(.5*np.sum(abs(p-q))-r['law_tv'])<1e-15 and r['law_tv']<1e-11)
            check(f'paired {i} seeded samples',a.shape==b.shape==(32,r['modes']) and np.array_equal(a,b) and np.all((a==0)|(a==1)) and r['samples_equal'])
    check('regression maximum and samples',abs(max(r['law_tv'] for r in rr)-regression['maximum_law_tv'])<1e-20 and regression['paired_samples']==512)
    check('twelve retained boundary controls',len(boundaries)==regression['boundary_checks']==regression['boundary_passed']==12 and all(r['passed'] and r.get('all_queried_events_positive',True) for r in boundaries))
    save(out/'regression_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks)); assert all(c['passed'] for c in checks)
    candidates=[r for r in rows if r['panel']=='candidate']; small=[r for r in rows if r['panel']=='quality']; masses=[]
    with np.load(ROOT/'results/iteration26_panel/gaussian.npz') as z: p=z['probabilities']; masses.append([p[0],p[1:145].sum(),p[145:].sum(),1-p.sum()])
    for r in candidates:
        with np.load(ROOT/f"results/iteration26_panel/case_{r['sequence']:02d}.npz") as z: p=z['mps_probabilities']; masses.append([p[0],p[1:145].sum(),p[145:].sum(),1-p.sum()])
    summary=dict(candidate_rows=candidates,small_cases=24,small_max_full_threshold_tv=max(r['full_threshold_tv'] for r in small),candidate_masses=masses,artifact_checks=audit['count'],artifact_checks_passed=audit['passed'],independent_gaussian_events=audit['independent_gaussian_events'],independent_gaussian_max_error=audit['independent_gaussian_max_error'],direct_mps_events=audit['direct_mps_events'],direct_mps_amplitudes=audit['direct_mps_amplitudes'],regression=regression,regression_checks=len(checks),regression_checks_passed=sum(c['passed'] for c in checks),tests={k:v for k,v in tests.items() if not isinstance(v,list)},native=native,analyzer_sha256=sha)
    save(out/'summary.json',summary)
    lines=['# 第26轮完整结果','','## 指定144模零位移输出的直接阈值下界','','| 有限态 | 真空概率 | 恰好1点击 | 恰好2点击 | 选定质量 | 阈值TV粗粒化下界 |','|---|---:|---:|---:|---:|---:|']
    for r,mass in zip(candidates,masses[1:]): lines.append(f"| {r['family']} | {mass[0]:.12g} | {mass[1]:.12g} | {mass[2]:.12g} | {r['left_selected_mass']:.12g} | {r['threshold_coarse_tv_lower_estimate']:.12g} |")
    lines+=['',f"Gaussian候选态对应三项质量为{masses[0][0]:.12g}、{masses[0][1]:.12g}、{masses[0][2]:.12g}；余项{masses[0][3]:.12g}。",'', '下界直接针对指定阈值分布，是浮点诊断，不是原实验输出误差或区间证书。选定事件不重新归一化。','','## 小系统全分布对照','','| 模式 | 种子 | d | 秩 | 粗粒化TV | 全阈值TV |','|---:|---:|---:|---:|---:|---:|']
    for r in small: lines.append(f"| {r['modes']} | {r['seed']} | {r['dimension']} | {r['rank']} | {r['threshold_coarse_tv_lower_estimate']:.12g} | {r['full_threshold_tv']:.12g} |")
    lines+=['','## 回归与门禁','',f"16个旧新森林配对全部通过，最大完整代理分布TV={regression['maximum_law_tv']:.12g}，512条配对样本相同，12项边界控制通过。额外{len(checks)}项回归产物检查通过。",'', '全库408运行、379通过、0失败、0错误、29案例和7类依赖跳过。原4失败实际通过，原导入错误模块13项测试实际通过。新增11项测试全部通过。', '', 'Windows本机编译成功，但C++ long double为16字节/64位有效精度，Python为8字节/53位；适配器正确拒绝，未创建内核。旧构建入口失败日志保留。','']
    (out/'summary_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8)); masses=np.asarray(masses); bottom=np.zeros(8)
    for k,(label,color) in enumerate(zip(['0 clicks','1 click','2 clicks','More than 2'],['#286ba6','#7ab3d4','#459f85','#d2d6dc'])):
        axes[0].bar(np.arange(8),masses[:,k],bottom=bottom,label=label,color=color); bottom+=masses[:,k]
    labels=['Gaussian','New32','New64','New128','New200','Own200','PCO200','Haf200']; axes[0].set_xticks(range(8),labels,rotation=45,ha='right',fontsize=8); axes[0].set(ylabel='Probability mass',ylim=(0,1),title='Complete click-count categories'); axes[0].legend(fontsize=7,ncol=2)
    axes[1].bar(np.arange(7),[r['threshold_coarse_tv_lower_estimate'] for r in candidates],color='#286ba6'); axes[1].axhline(.1,color='#aa403d',ls='--',label='0.1 reference'); axes[1].set_xticks(range(7),labels[1:],rotation=45,ha='right',fontsize=8); axes[1].set(ylabel='Threshold TV lower estimate',title='Direct output discrepancy'); axes[1].legend(fontsize=8)
    x=np.array([r['full_threshold_tv'] for r in small]); y=np.array([r['threshold_coarse_tv_lower_estimate'] for r in small]); axes[2].scatter(x,y,color='#459f85'); lim=max(x.max(),y.max())*1.1; axes[2].plot([0,lim],[0,lim],'k--',lw=1); axes[2].set(xlabel='Complete threshold TV',ylabel='Coarse lower estimate',title='24 small-system controls',xlim=(0,lim),ylim=(0,lim))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Specified zero-displacement states; floating diagnostics, not experimental-state identification',fontsize=11); fig.tight_layout(); fig.savefig(out/'threshold_bounds_and_controls.png',dpi=180); plt.close(fig)
    assert digest(__file__)==sha; print(json.dumps(dict(artifact_checks=audit['count'],regression_checks=len(checks),candidate_bounds=[r['threshold_coarse_tv_lower_estimate'] for r in candidates],tests_run=tests['tests_run']),indent=2))


if __name__=='__main__': main()
