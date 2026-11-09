"""Summarize Gaussian candidate reconstruction and finite-reference gaps."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration24_analysis'; out.mkdir(exist_ok=False); sha=digest(__file__); shutil.copy2(__file__,out/Path(__file__).name)
    rows=json.loads((ROOT/'results/iteration24_chain/rows.json').read_text()); external=rows[-1]
    low=json.loads((ROOT/'results/iteration24_posthoc/rows.json').read_text()); geometry=json.loads((ROOT/'results/iteration24_geometry/summary.json').read_text()); grams=json.loads((ROOT/'results/iteration24_geometry/grams.json').read_text()); cuts=json.loads((ROOT/'results/iteration24_geometry/cuts.json').read_text())
    val=json.loads((ROOT/'results/iteration24_validation/validation.json').read_text()); tests=json.loads((ROOT/'results/iteration24_tests/validation.json').read_text()); glob=json.loads((ROOT/'results/iteration24_posthoc/global.json').read_text())
    assert val['count']==val['passed'] and all(r['status']=='passed' for r in rows+low)
    summary=dict(chain_cases=len(rows),chain_passed=len(rows),known_and_historical=len(rows)-1,max_original_pairing_error=max(d['original_pairing_max_error'] for r in rows[:-1] for d in r['directions'].values()),max_original_log_normalization_error=max(d['original_log_normalization_error'] for r in rows[:-1] for d in r['directions'].values()),external={k:v for k,v in external.items() if k not in ('directions','bond_modes')},external_gaussian={k:v for k,v in external['directions']['forward'].items() if k!='steps'},global_check=glob,low_sector=low,geometry_checks=geometry['count'],geometry_checks_passed=geometry['passed'],geometry_checks_failed=geometry['count']-geometry['passed'],original_projector_budget_available=False,max_local_gram_error=max(r['gram_pairing_error'] for r in grams),max_virtual_thermal_error=max(r['thermal_covariance_residual'] for r in cuts),max_physical_prefix_spectrum_error=max(r['physical_prefix_spectrum_error'] for r in cuts),validation_checks=val['count'],validation_passed=val['passed'],direct_low_amplitudes=val['direct_low_amplitudes'],tests={k:v for k,v in tests.items() if not isinstance(v,list)},analyzer_sha256=sha)
    save(out/'summary.json',summary)
    lines=['# 第二十四轮结果','','| 有限链 | 真空概率 | 0/2光子质量 | 纯态距离下界 | PNR粗粒化TV下界 | 真空概率差 |','|---|---:|---:|---:|---:|---:|']
    for r in low: lines.append(f"| {r['family']} | {r['left_vacuum_probability']:.12g} | {r['left_selected_mass']:.12g} | {r['pure_trace_lower_estimate']:.12g} | {r['pnr_coarse_tv_lower_estimate']:.12g} | {r['vacuum_probability_gap']:.12g} |")
    lines+=['',f"核链Gaussian候选态真空概率{low[0]['right_vacuum_probability']:.12g}，0/2光子质量{low[0]['right_selected_mass']:.12g}。",'', '以上是固定零位移纯态之间的浮点下界诊断，不是实验混合输出误差。301项几何前提检查中145项失败，上界不适用；完整失败记录保留。','']
    (out/'summary_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.9)); reference=low[0]
    masses=[[reference['right_vacuum_probability'],reference['right_selected_mass']-reference['right_vacuum_probability'],1-reference['right_selected_mass']]]+[[r['left_vacuum_probability'],r['left_selected_mass']-r['left_vacuum_probability'],1-r['left_selected_mass']] for r in low]
    masses=np.array(masses); bottom=np.zeros(4)
    for k,(label,color) in enumerate(zip(['Vacuum','Exactly two photons','Other counts'],['#2878b5','#459f85','#d0d6dc'])):
        axes[0].bar(np.arange(4),masses[:,k],bottom=bottom,label=label,color=color); bottom+=masses[:,k]
    axes[0].set_xticks(range(4),['Gaussian candidate','Own recurrence','Author PCO','Author hafnian'],rotation=20,ha='right',fontsize=8); axes[0].set(ylabel='Probability mass',ylim=(0,1),title='Normalized full states, zero displacement'); axes[0].legend(fontsize=8)
    axes[1].semilogy([r['site'] for r in grams],[max(r['gram_pairing_error'],1e-16) for r in grams],label='Local isometry residual')
    axes[1].semilogy([r['cut'] for r in cuts],[max(r['physical_prefix_spectrum_error'],1e-16) for r in cuts],label='Physical/virtual spectrum residual')
    axes[1].axhline(1e-8,color='black',ls='--',lw=1,label='Diagnostic threshold'); axes[1].set(xlabel='Site / cut index',ylabel='Absolute residual',title='Original-projector prerequisites fail'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'candidate_and_projection_checks.png',dpi=180); plt.close(fig); assert digest(__file__)==sha
    print(json.dumps({k:v for k,v in summary.items() if k not in ('external','external_gaussian','global_check','low_sector','tests')},indent=2))


if __name__=='__main__': main()
