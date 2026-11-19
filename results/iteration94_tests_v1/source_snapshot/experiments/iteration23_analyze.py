"""Complete paired pipeline timings and controlled precision summary."""
import json
from pathlib import Path
import shutil
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration23_analysis'; out.mkdir(exist_ok=False)
    sha=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    quality=json.loads((ROOT/'results/iteration23_quality/rows.json').read_text())
    rows=json.loads((ROOT/'results/iteration23_performance/rows.json').read_text())
    precision=json.loads((ROOT/'results/iteration23_precision/rows.json').read_text())
    validation=json.loads((ROOT/'results/iteration23_validation/validation.json').read_text())
    pairs=json.loads((ROOT/'results/iteration23_validation/comparisons.json').read_text())
    tests=json.loads((ROOT/'results/iteration23_tests/validation.json').read_text())
    grouped={}
    for r in rows: grouped.setdefault((r['modes'],r['seed']),{}).setdefault(r['backend'],[]).append(r)
    timings=[]
    for key,group in sorted(grouped.items()):
        entry=dict(modes=key[0],seed=key[1],backends={})
        for backend in ('pco','recurrence'):
            selected=group[backend]; assert len(selected)==3 and all(r['status']=='accepted' for r in selected)
            values=[r['pipeline_seconds'] for r in selected]
            representative=sorted(selected,key=lambda r:r['pipeline_seconds'])[1]
            entry['backends'][backend]=dict(pipeline_median_seconds=statistics.median(values),pipeline_min_seconds=min(values),pipeline_max_seconds=max(values),
                construction_median_seconds=statistics.median(r['phase_seconds']['construction'] for r in selected),
                local_tensor_median_seconds=statistics.median(r['info']['phase_seconds']['local_tensors'] for r in selected),
                sampling_median_seconds=statistics.median(r['phase_seconds']['sampling'] for r in selected),
                process_wall_median_seconds=statistics.median(r['process_wall_seconds'] for r in selected),
                maximum_pipeline_peak_bytes=max(r['pipeline_peak_working_set_bytes'] for r in selected),
                median_run_sequence=representative['sequence'],median_run_phase_seconds=representative['phase_seconds'],median_run_construction_phases=representative['info']['phase_seconds'])
        entry['pipeline_ratio']=entry['backends']['pco']['pipeline_median_seconds']/entry['backends']['recurrence']['pipeline_median_seconds']
        entry['local_tensor_ratio']=entry['backends']['pco']['local_tensor_median_seconds']/entry['backends']['recurrence']['local_tensor_median_seconds']
        entry['budget']=group['pco'][0]['construction_trace_budget']
        entry['paired_repetition_ratios']=[next(r['pipeline_seconds'] for r in group['pco'] if r['repetition']==i)/next(r['pipeline_seconds'] for r in group['recurrence'] if r['repetition']==i) for i in range(3)]
        timings.append(entry)
    summary=dict(performance_runs=len(rows),performance_accepted=sum(r['status']=='accepted' for r in rows),quality_runs=len(quality),quality_accepted=sum(r['status']=='accepted' for r in quality),quality_budget_missed=sum(r['status']=='budget_missed' for r in quality),
        quality_actual_max=max(r['actual_original_distance'] for r in quality),quality_bound_checks=sum(r['actual_bound_check'] for r in quality),timings=timings,
        pipeline_ratio_min=min(r['pipeline_ratio'] for r in timings),pipeline_ratio_max=max(r['pipeline_ratio'] for r in timings),
        max_quality_pair_distance=max(r['comparison']['pure_trace_distance_from_difference'] for r in pairs if r['phase']=='quality'),
        max_performance_pair_distance=max(r['comparison']['pure_trace_distance_from_difference'] for r in pairs if r['phase']=='performance'),
        all_paired_samples_equal=all(r['samples_equal'] for r in pairs),generated_outputs=validation['samples'],
        new_state_files=len({r['state']['path'] for r in quality+rows if r['state']['path'].startswith('results/iteration23_')}),
        old_state_files_reused=sorted({r['state']['path'] for r in rows if r['state']['path'].startswith('results/iteration21_')}),
        precision=precision,checks=validation['count'],checks_passed=validation['passed'],tests={k:v for k,v in tests.items() if not isinstance(v,list)},analyzer_sha256=sha)
    save(out/'summary.json',summary)
    lines=['# 第二十三轮完整配对结果','', '| 模式 | 输入种子 | PCO秒，中位[最小,最大] | 递推秒，中位[最小,最大] | 流程比值 | 局部系数比值 | 原态预算 |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in timings:
        a,b=r['backends']['pco'],r['backends']['recurrence']
        lines.append(f"| {r['modes']} | {r['seed']} | {a['pipeline_median_seconds']:.5f} [{a['pipeline_min_seconds']:.5f}, {a['pipeline_max_seconds']:.5f}] | {b['pipeline_median_seconds']:.5f} [{b['pipeline_min_seconds']:.5f}, {b['pipeline_max_seconds']:.5f}] | {r['pipeline_ratio']:.5f} | {r['local_tensor_ratio']:.5f} | {r['budget']:.8f} |")
    lines+=['','流程包括输入、构造、预算、位移及16条阈值样本；排除启动、归档和独立审计。每输入每后端3次独立进程，重复使用固定随机种子，不是独立样本统计实验。','', '| 精度变换比较 | 稳定纯态距离 |', '|---|---:|']
    for r in precision: lines.append(f"| {r['pair']} | {r['pure_trace_distance_from_difference']:.12g} |")
    lines+=['','受控末端量化与乘法变化不能隔离未知的早期计算、规范条件数或截断误差。','']
    (out/'summary_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.9)); x=np.arange(len(timings)); width=.36
    for offset,backend,label in [(-width/2,'pco','PCO'),(width/2,'recurrence','Recurrence')]:
        med=np.array([r['backends'][backend]['pipeline_median_seconds'] for r in timings]); low=np.array([r['backends'][backend]['pipeline_min_seconds'] for r in timings]); high=np.array([r['backends'][backend]['pipeline_max_seconds'] for r in timings])
        axes[0].bar(x+offset,med,width,label=label,yerr=np.vstack([med-low,high-med]),capsize=2)
    axes[0].set_xticks(x,[f"{r['modes']} / {r['seed']}" for r in timings],rotation=45,ha='right',fontsize=8)
    axes[0].set(ylabel='Seconds: input through 16 samples',title='Three fresh processes per case/backend'); axes[0].legend(); axes[0].grid(axis='y',alpha=.2)
    labels=['PCO storage cast','Hafnian product cast','PCO vs single-product HF','Cast PCO vs single-product HF']
    axes[1].barh(labels,[r['pure_trace_distance_from_difference'] for r in precision],color=['#2878b5','#459f85','#d98535','#c25353'])
    axes[1].set_xscale('log'); axes[1].invert_yaxis(); axes[1].set(xlabel='Pure-state distance (complex128 diagnostic)',title='Controlled terminal precision changes'); axes[1].tick_params(axis='y',labelsize=8); axes[1].grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(out/'pipeline_and_precision.png',dpi=180); plt.close(fig)
    assert digest(__file__)==sha
    print(json.dumps({k:v for k,v in summary.items() if k not in ('timings','precision','tests','old_state_files_reused')},indent=2))


if __name__=='__main__': main()
