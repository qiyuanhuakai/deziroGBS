"""Summarize physical-rank limits without confusing target and reference gaps."""
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
    out=ROOT/'results/iteration25_analysis'; out.mkdir(exist_ok=False); sha=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    bounds=read('results/iteration25_spectra/bounds.json'); rows=read('results/iteration25_panel_retry/rows.json'); pairs=read('results/iteration25_compare/rows.json'); val=read('results/iteration25_validation/validation.json'); tests=read('results/iteration25_tests/validation.json')
    assert val['count']==val['passed'] and len(pairs)==18 and all(r['status']=='passed' for r in pairs)
    candidate=[{k:v for k,v in r.items() if k!='info'} for r in rows if r['panel']=='candidate']; quality=[r for r in rows if r['panel']=='quality']; rankbounds=[r for r in bounds if r['dimension']==4]
    for r in candidate:
        r['pure_rank_distance_lower_estimate']=next(x['pure_rank_distance_lower_estimate'] for x in rankbounds if x['rank']==r['rank'])
    summary=dict(physical_cuts=143,budget_rows=21,small_quality_runs=16,small_quality_accepted=sum(r['status']=='accepted' for r in quality),small_quality_all_actual_distances_bounded=all(r['actual_bound_check'] for r in quality),small_quality_max_actual_distance=max(r['actual_original_distance'] for r in quality),candidates=candidate,rank_bounds=rankbounds,full_comparisons=pairs,validation_checks=val['count'],validation_passed=val['passed'],complement_spectrum_max_error=val['max_complement_spectrum_error'],direct_low_amplitudes=val['direct_low_amplitudes'],independent_hafnian_coefficients=val['independent_hafnian_coefficients'],tests={k:v for k,v in tests.items() if not isinstance(v,list)},preserved_initial_path_failures=17,analyzer_sha256=sha)
    save(out/'summary.json',summary)
    lines=['# 第二十五轮结果','','## 144模式候选态：物理谱必要下界','','| 秩 | 纯态距离必要下界 | Schmidt尾和 | d4投影上界 |','|---:|---:|---:|---:|']
    for r in rankbounds: lines.append(f"| {r['rank']} | {r['pure_rank_distance_lower_estimate']:.12g} | {r['schmidt_tail_sum']:.12g} | {r['original_projector_trace_upper_estimate']:.12g} |")
    lines+=['','必要下界适用于相同站点顺序的任意该秩纯MPS；不是输出TV下界。所有d4/d6/d8充分上界均为1。','','## 实际构造与低光子诊断','','| 秩 | 单次构造秒数 | 真空概率差 | PNR粗粒化TV下界 | 低光子纯态距离下界 | 状态 |','|---:|---:|---:|---:|---:|---|']
    for r in candidate:
        l=r['low_sector']; lines.append(f"| {r['rank']} | {r['construction_seconds']:.6g} | {l['vacuum_probability_gap']:.12g} | {l['pnr_coarse_tv_lower_estimate']:.12g} | {l['pure_trace_lower_estimate']:.12g} | {r['status']} |")
    lines+=['','单次诊断计时不构成隔离重复性能证据。较小下界不证明实际目标误差更小。','','## 完整有限MPS比较','','| 两个有限态 | 纯态距离 |','|---|---:|']
    for r in pairs: lines.append(f"| {r['pair']} | {r['pure_trace_distance_from_difference']:.12g} |")
    lines+=['','有限态之间接近不能证明其接近Gaussian目标，更不能认定该目标就是原实验态。','','## 小系统实际质量','','| 模式 | 种子 | d | 秩 | 实际Gaussian距离 | 充分上界 | 状态 |','|---:|---:|---:|---:|---:|---:|---|']
    for r in quality: lines.append(f"| {r['modes']} | {r['seed']} | {r['dimension']} | {r['rank']} | {r['actual_original_distance']:.12g} | {r['construction_trace_budget']:.12g} | {r['status']} |")
    lines+=['',f"产物检查{val['passed']}/{val['count']}；独立hafnian系数{val['independent_hafnian_coefficients']}项；直接低光子收缩{val['direct_low_amplitudes']}项。新增5项测试通过；全库385运行、351通过、4失败、1错误、29案例及7类跳过。首次路径错误的17项失败及中止记录完整保留。",'']
    (out/'summary_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].semilogx([r['rank'] for r in rankbounds],[r['pure_rank_distance_lower_estimate'] for r in rankbounds],'o-',color='#286ba6',label='Necessary pure-state lower bound')
    axes[0].axhline(.1,color='#aa403d',ls='--',label='0.1 target'); axes[0].set(xlabel='MPS bond rank (fixed order)',ylabel='Pure-state trace distance',ylim=(0,.7),title='Physical Schmidt spectrum'); axes[0].legend(fontsize=7)
    x=np.arange(4); width=.36
    axes[1].bar(x-width/2,[r['low_sector']['left_vacuum_probability'] for r in candidate],width,label='Rebuilt finite MPS',color='#459f85')
    axes[1].bar(x+width/2,[r['low_sector']['right_vacuum_probability'] for r in candidate],width,label='Gaussian candidate',color='#97b9d8')
    axes[1].set_xticks(x,[str(r['rank']) for r in candidate]); axes[1].set(xlabel='Bond rank',ylabel='Vacuum probability',title='Zero-displacement diagnostic'); axes[1].legend(fontsize=7)
    for family,marker in [('author_pco','o'),('author_hafnian','s')]:
        rs=[next(p for p in pairs if p['pair']==f'new_{r}:{family}') for r in (32,64,128,200)]
        axes[2].semilogy([32,64,128,200],[p['pure_trace_distance_from_difference'] for p in rs],marker+'-',label=family)
    axes[2].set(xlabel='Rebuilt bond rank',ylabel='Full finite-state trace distance',title='Distance to finite references'); axes[2].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('144-mode kernel-chain candidate: rank limits and finite-state comparisons',fontsize=12); fig.tight_layout(); fig.savefig(out/'physical_rank_and_reference.png',dpi=180); plt.close(fig)
    assert digest(__file__)==sha
    print(json.dumps({k:v for k,v in summary.items() if k not in ('candidates','rank_bounds','full_comparisons','tests')},indent=2))


if __name__=='__main__': main()
