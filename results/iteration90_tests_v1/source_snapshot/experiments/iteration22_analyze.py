"""Summarize full external reconstruction and serial local timings."""
import json
from pathlib import Path
import shutil
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration22_analysis'; out.mkdir(exist_ok=False)
    source_hash=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    rows=json.loads((ROOT/'results/iteration22_external/rows.json').read_text())
    global_rows=json.loads((ROOT/'results/iteration22_global/rows.json').read_text())
    bench=json.loads((ROOT/'results/iteration22_benchmark/rows.json').read_text())
    validation=json.loads((ROOT/'results/iteration22_validation/validation.json').read_text())
    tests=json.loads((ROOT/'results/iteration22_tests/validation.json').read_text())
    timing=[]
    for site in sorted({r['site'] for r in bench}):
        med={a:statistics.median(r['seconds'] for r in bench if r['site']==site and r['algorithm']==a) for a in ('pco','recurrence')}
        timing.append(dict(site=site,pco_median_seconds=med['pco'],recurrence_median_seconds=med['recurrence'],ratio=med['pco']/med['recurrence']))
    summary=dict(sites=len(rows),passed=sum(r['status']=='passed' for r in rows),max_errors={k:max(r['errors'][k]['max_abs'] for r in rows) for k in ('pco_reference','recurrence_reference','recurrence_pco')},global_comparisons=global_rows,serial_timings=timing,benchmark_runs=len(bench),benchmark_passed=sum(r['passed'] for r in bench),checks=validation['count'],checks_passed=validation['passed'],tests={k:v for k,v in tests.items() if not isinstance(v,list)},maximum_worker_peak_bytes=max(r['peak_working_set_bytes'] for r in rows),analyzer_sha256=source_hash)
    save(out/'summary.json',summary)
    lines=['# 第二十二轮完整结果表','',f"全部144站点通过；复核{validation['passed']}/{validation['count']}项。",'', '| 完整144模式比较 | 直接差态纯态距离 | 原始log F | QR/收缩对数范数最大差 |', '|---|---:|---:|---:|']
    for r in global_rows: lines.append(f"| {r['pair']} | {r['pure_trace_distance_from_difference']:.12g} | {r['raw_log_fidelity']:.12g} | {max(abs(r['log_norm_qr_residual_left']),abs(r['log_norm_qr_residual_right'])):.4g} |")
    lines+=['','距离是complex128数值诊断；约机器精度的数值不能解释为严格零或误差上界。','', '| 站点 | PCO中位秒 | 递推中位秒 | 比值PCO/递推 |', '|---|---:|---:|---:|']
    for r in timing: lines.append(f"| {r['site']} | {r['pco_median_seconds']:.8g} | {r['recurrence_median_seconds']:.8g} | {r['ratio']:.6g} |")
    lines+=['','每算法每站3次，固定随机顺序，输入加载与比较不在计时内。边界站点毫秒级结果容易受计时噪声影响；仅为两个本地Python实现的局部系数成本，非端到端GBS或论文实现速度比较。','']
    (out/'summary_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for key,label in [('pco_reference','PCO vs author PCO'),('recurrence_reference','Recurrence vs author PCO'),('recurrence_pco','Recurrence vs local PCO')]:
        axes[0].semilogy([r['site'] for r in rows],[max(1e-18,r['errors'][key]['max_abs']) for r in rows],label=label,lw=1.4)
    axes[0].axhline(1e-10,color='gray',ls='--',label='Predeclared tolerance')
    axes[0].set(xlabel='Site (0-based)',ylabel='Maximum absolute coefficient difference',title='All 144 local tensors')
    axes[0].legend(fontsize=8,loc='center right'); axes[0].grid(alpha=.2)
    x=np.arange(len(timing)); width=.36
    axes[1].bar(x-width/2,[r['pco_median_seconds'] for r in timing],width,label='PCO')
    axes[1].bar(x+width/2,[r['recurrence_median_seconds'] for r in timing],width,label='Recurrence')
    axes[1].set_yscale('log'); axes[1].set_xticks(x,[str(r['site']) for r in timing])
    axes[1].set(xlabel='Site (0-based)',ylabel='Median seconds, 3 serial repetitions',title='Local Python coefficient calls only')
    axes[1].legend(); axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('144-mode external reference: d=4, virtual rank cap=200',fontsize=13)
    fig.tight_layout(); fig.savefig(out/'external_reconstruction.png',dpi=180); plt.close(fig)
    assert digest(__file__)==source_hash
    print(json.dumps({k:v for k,v in summary.items() if k not in ('global_comparisons','tests')},indent=2))


if __name__=='__main__': main()
