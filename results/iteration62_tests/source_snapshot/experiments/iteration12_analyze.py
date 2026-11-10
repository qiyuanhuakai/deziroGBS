"""Frozen-artifact audit, complete tables, and local MPS figures."""
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


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    folders={'gate_quality':'iteration12_quality','gate_scaling':'iteration12_scaling',
        'layer_quality':'iteration12_layerwise_quality','layer_scaling':'iteration12_layerwise_scaling',
        'timing':'iteration12_serial_timing'}
    groups={}; checks=[]; bindings={}
    for key,name in folders.items():
        folder=ROOT/'results'/name
        rows=read(folder/'rows.json'); groups[key]=rows
        bindings[str((folder/'rows.json').relative_to(ROOT))]=hashlib.sha256((folder/'rows.json').read_bytes()).hexdigest()
        for r in rows:
            if r['status']!='complete':
                raise AssertionError(f'Incomplete row {key} {r["input"]}')
            directory=folder/f"run_{r['sequence']:02d}" if key=='timing' else folder
            z=np.load(directory/(r['input']+'.npz'))
            tensors=[z[f'A{i}'] for i in range(r['modes'])]
            checks.append(abs(np.linalg.norm(tensors[0])-1)<1e-9)
            for a in tensors[1:]:
                flat=a.reshape(a.shape[0],-1)
                checks.append(np.max(np.abs(flat@flat.conj().T-np.eye(len(flat))))<1e-9)
            checks.append(z['samples'].shape==(r['completed_samples'],r['modes']))
            checks.append(np.all((z['samples']==0)|(z['samples']==1)))
            checks.append(sum(a.size for a in tensors)==r['stored_complex_elements'])
            if 'layers' in r:
                loss=0.
                offset=0
                for layer in r['layers']:
                    gates=r['gates'][offset:offset+layer['gate_count']]; offset+=layer['gate_count']
                    projection=-math.expm1(sum(math.log1p(-g['local_projection_tail']) for g in gates))
                    rounding=-math.expm1(sum(math.log1p(-v) for v in layer['rounding_conditional_losses']))
                    checks.append(abs(projection-layer['local_projection_tail'])<1e-13)
                    checks.append(abs(rounding-layer['rounding_tail'])<1e-13)
                    loss+=math.sqrt(projection)+math.sqrt(rounding)
                checks.append(offset==len(r['gates']))
            else:
                loss=sum(math.sqrt(g['local_projection_tail']+(1-g['local_projection_tail'])*g['bond_tail']) for g in r['gates'])
            budget=min(1.,math.sqrt(r['joint_input_tail'])+loss)
            checks.append(abs(budget-r['construction_trace_budget'])<1e-12)
            timed=sum(r[k] for k in ['circuit_and_parameters_seconds','construction_seconds','displacement_seconds','sampling_seconds'])
            checks.append(r['end_to_end_compute_seconds']>=timed-1e-6)
            if 'quality' in key:
                data=np.load(directory/(r['input']+'_reference.npz'))
                overlap=np.vdot(data['gaussian_box'].ravel(),data['mps_tensor'].ravel())
                actual=math.sqrt(max(0.,1-abs(overlap)**2))
                checks.append(abs(actual-r['state_trace_distance'])<1e-12)
                checks.append(actual<=budget+1e-10)
                checks.append(np.max(np.abs(data['fixed_law']-data['reference_fixed_law']))<1e-10)
                if r.get('averaged_status')=='complete':
                    avg=np.load(directory/(r['input']+'_averaged.npz'))
                    tv=.5*np.abs(avg['distribution']-avg['target']).sum()
                    checks.append(abs(tv-r['averaged_tv'])<1e-12)
                    checks.append(tv<=actual+1e-10)
        meta=read(folder/'source_hashes.json') if key=='timing' else read(folder/'metadata.json')['source_hashes']
        for f,h in meta.items():
            checks.append(hashlib.sha256((folder/'source_snapshot'/f).read_bytes()).hexdigest()==h)
    if not all(checks): raise AssertionError('Artifact audit failed')
    gs,ls=groups['gate_scaling'],groups['layer_scaling']
    gq,lq=groups['gate_quality'],groups['layer_quality']
    timing=[]
    for seed in sorted({r['seed'] for r in groups['timing']}):
        item={'seed':seed}
        for method in ['gatewise','layerwise']:
            rows=[r for r in groups['timing'] if r['seed']==seed and r['builder']==method]
            item[method]={'median_seconds':float(np.median([r['end_to_end_compute_seconds'] for r in rows])),
                'min_seconds':min(r['end_to_end_compute_seconds'] for r in rows),
                'max_seconds':max(r['end_to_end_compute_seconds'] for r in rows),
                'construction_budget':rows[0]['construction_trace_budget'],
                'stored_complex_elements':rows[0]['stored_complex_elements'],'repeats':len(rows)}
        item['gate_over_layer_time_ratio']=item['gatewise']['median_seconds']/item['layerwise']['median_seconds']
        timing.append(item)
    summary={'date':'2026-11-08','rows_by_group':{k:len(v) for k,v in groups.items()},
        'all_completed':True,'sample_count':sum(r['completed_samples'] for rows in groups.values() for r in rows),
        'quality_full_averaged_laws':sum(r.get('averaged_status')=='complete' for r in gq+lq),
        'quality_averaged_failures':sum(r.get('averaged_status')=='failed' for r in gq+lq),
        'quality_state_checks':len(gq+lq),
        'gate_scaling_budget_at_most_01':sum(r['construction_trace_budget']<=.1 for r in gs),
        'layer_scaling_budget_at_most_01':sum(r['construction_trace_budget']<=.1 for r in ls),
        'layer_quality_actual_state_improved':sum(b['state_trace_distance']<a['state_trace_distance']-1e-10 for a,b in zip(gq,lq)),
        'layer_quality_actual_state_worsened':sum(b['state_trace_distance']>a['state_trace_distance']+1e-10 for a,b in zip(gq,lq)),
        'artifact_checks':len(checks),'artifact_checks_passed':sum(bool(x) for x in checks),
        'serial_timing':timing,'source_bindings':bindings,
        'scope':'Construction bounds are ideal-arithmetic state bounds, not certified finite-precision sampler TV.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# 第十二轮完整结果表','','## 128模式d8 χ16的构造预算','',
        '| 方案 | 深度 | 配置 | 种子 | 构造预算 | 单次端到端秒 | 最终复数系数数 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for method,rows in [('gatewise',gs),('layerwise',ls)]:
        for r in rows:
            if r['modes']==128 and r['d']==8:
                lines.append(f"| {method} | {r['depth']} | {r['profile']} | {r['seed']} | {r['construction_trace_budget']:.9g} | {r['end_to_end_compute_seconds']:.6f} | {r['stored_complex_elements']} |")
    lines+=['','## 新种子串行计时','', 'gatewise为d8 χ16，layerwise为d6 χ8；两者均以理想构造预算≤0.1为筛选条件，非完整采样TV对赛。','',
        '| 种子 | gate秒中位数 | layer秒中位数 | gate/layer | gate预算 | layer预算 |','|---:|---:|---:|---:|---:|---:|']
    for r in timing:
        a,b=r['gatewise'],r['layerwise']
        lines.append(f"| {r['seed']} | {a['median_seconds']:.6f} | {b['median_seconds']:.6f} | {r['gate_over_layer_time_ratio']:.4f} | {a['construction_budget']:.9g} | {b['construction_budget']:.9g} |")
    lines+=['','## 小系统完整位移平均分布','', '| 方案 | 输入 | 实际纯态迹距离 | 构造预算 | 输出TV |', '|---|---|---:|---:|---:|']
    for method,rows in [('gatewise',gq),('layerwise',lq)]:
        for r in rows:
            if r.get('averaged_status')=='complete':
                lines.append(f"| {method} | {r['input']} | {r['state_trace_distance']:.9g} | {r['construction_trace_budget']:.9g} | {r['averaged_tv']:.9g} |")
    (args.output/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    colors={4:'#da8e29',6:'#248b70',8:'#3268ae'}
    for ax,depth in zip(axes[0],[4,12]):
        for name,rows,style in [('gatewise',gs,'--'),('layerwise',ls,'-')]:
            for d in [4,6,8]:
                y=[np.median([r['construction_trace_budget'] for r in rows if r['depth']==depth and r['modes']==m and r['profile']=='lossy' and r['d']==d]) for m in [8,32,64,128]]
                ax.plot([8,32,64,128],y,style,marker='o',color=colors[d],label=f'{name}, d={d}')
        ax.axhline(.1,color='gray',linestyle=':',label='Budget 0.1')
        ax.set(xscale='log',yscale='log',xlabel='Modes',ylabel='Ideal construction trace-distance budget',
            title=f'Local depth {depth}, lossy profile',ylim=(.0007,1.5))
    axes[0,0].legend(fontsize=7,ncol=2)
    for name,rows,marker in [('gatewise',gq,'o'),('layerwise',lq,'x')]:
        axes[1,0].scatter([r['construction_trace_budget'] for r in rows],[r['state_trace_distance'] for r in rows],s=25,marker=marker,label=name,alpha=.7)
    axes[1,0].plot([.001,1],[.001,1],'k:',linewidth=1)
    axes[1,0].set(xscale='log',yscale='log',xlabel='Construction bound',ylabel='Exact-reference state trace distance',
        title='96 small-system checks against full Gaussian state')
    axes[1,0].legend()
    x=np.arange(len(timing)); width=.32
    for j,method in enumerate(['gatewise','layerwise']):
        med=[r[method]['median_seconds'] for r in timing]
        low=[r[method]['median_seconds']-r[method]['min_seconds'] for r in timing]
        high=[r[method]['max_seconds']-r[method]['median_seconds'] for r in timing]
        axes[1,1].bar(x+(j-.5)*width,med,width,yerr=[low,high],capsize=3,label=method)
    axes[1,1].set(xticks=x,xticklabels=[str(r['seed']) for r in timing],ylabel='Circuit to 128 samples (seconds)',
        xlabel='New circuit seed; 3 serial repeats each',title='128 modes, depth 4, nominal budget <= 0.1')
    axes[1,1].legend()
    fig.savefig(args.output/'local_mps_progress.png',dpi=170)
    fig.savefig(args.output/'local_mps_progress.svg')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_bindings'},indent=2))


if __name__=='__main__': main()
