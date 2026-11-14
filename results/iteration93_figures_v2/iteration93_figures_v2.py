from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('.');v=json.loads((P/'results/iteration93_verification_v1/validation.json').read_text());q={r['name']:r for r in v['rows']};tm={r['job']['timing_name']:r for r in v['timings']};a=v['actual'];pct=lambda x:f'{100*x:.4f}%';interval=lambda x:f"{pct(x['lower'])}–{pct(x['upper'])}";seconds=lambda name,th=1:tm[f'{name}_t{th}']['median_seconds'];figs=P/'results/iteration93_figures_v2';figs.mkdir(exist_ok=False)
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
offsets={'pure4':{'r3':(5,12),'r5':(5,4),'pe':(5,-14),'direct4':(-18,-19),'direct16':(-15,-17),'mps0':(5,5),'mps1':(5,-14),'frugal2':(5,5),'frugal3':(5,5),'frugal4':(5,-14)},'pure8':{'r3':(5,6),'r5':(-17,-16),'direct16':(5,-24),'mps1':(5,-14),'frugal5':(5,-14)}}
for ax,case,title in zip(axes,('pure4','pure8'),('4 modes: complete click law','8 modes: complete click law')):
    selected=[r for r in v['rows'] if r['case']==(0 if case=='pure4' else 2)]
    for k,r in enumerate(selected):
        short=r['name'].replace(case+'_','');x=seconds(r['name'])/2048*1e6;y=r['click']['upper']*100;color='#dc2626' if r['family']=='frugal' else ('#2563eb' if r['family']=='mps' else '#52525b');ax.scatter(x,y,color=color,s=30);ax.annotate(short,(x,y),xytext=offsets[case].get(short,(5,5)),textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set_title(title);ax.set_xlabel('Microseconds per attempt, CPU 1 worker');ax.set_ylabel('Click TV upper bound (%)');ax.grid(alpha=.2);ax.margins(x=.16,y=.25)
fig.suptitle('Click accuracy and measured cost');fig.text(.5,.012,'Timing: median of 3 serial batches of 2,048. Quality: 65,536 fresh attempts; failures included.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(figs/'click_cost_accuracy.png',dpi=160);plt.close(fig)
shutil.copy2(Path(__file__),figs/Path(__file__).name);(figs/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figs.iterdir() if p.is_file()},indent=2)+'\n')
