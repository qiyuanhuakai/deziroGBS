"""Standalone export of scoped confidence intervals, with separate families."""
from pathlib import Path
import json,hashlib,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
paths=[ROOT/'results/iteration88_small_pnr_quality_v1/summary.json',ROOT/'results/iteration88_joint_quality_v1/summary.json'];small,full=[json.loads(p.read_text()) for p in paths];fig,axs=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for case,color,y in [('paired_weak','#2374ab',0),('paired_strong','#b54531',4),('paired_lossy','#32835a',8)]:
 for j,r in enumerate(x for x in small['rows'] if x['case']==case):
  lo,hi=[100*r[f'nominal_joint99_full_pnr_tv_{k}'] for k in ('lower','upper')];axs[0].plot([lo,hi],[y+j]*2,color=color,lw=4);axs[0].plot([lo,hi],[y+j]*2,'|',color=color,ms=10);axs[0].text(hi+.4,y+j,f"{r['failed']} failed",fontsize=8,va='center')
axs[0].set(yticks=[0,1,2,4,5,6,8,9,10],yticklabels=['Weak R3','Weak R4','Weak R5','Strong R3','Strong R4','Strong R5','Lossy R3','Lossy R4','Lossy R5'],xlabel='Complete two-port photon-pattern TV (%)',title='PNR: 65,536 records per method / case',xlim=(-1,38));axs[0].invert_yaxis()
names=['pnr','block16','project16','block64','project64'];labels=['PNR R3 (reused)','Block B16','Project B16','Block B64','Project B64']
for i,n in enumerate(names):
 r=full['rows'][n]['experimental_click_count'];lo,hi=[100*r[f'nominal_joint99_cdf_{k}'] for k in ('lower','upper')];color='#2374ab' if n=='pnr' else ('#32835a' if n.startswith('project') else '#777777');axs[1].plot([lo,hi],[i]*2,color=color,lw=4);axs[1].plot([lo,hi],[i]*2,'|',color=color,ms=10);axs[1].text(hi+.3,i,f"N={full['rows'][n]['records']}",fontsize=8,va='center')
axs[1].set(yticks=range(5),yticklabels=labels,xlabel='Distance from experimental total-click CDF (%)',title='Full 1,152-output trajectories',xlim=(0,48));axs[1].invert_yaxis()
for ax in axs:ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Iteration 88: measured accuracy and retained failure outcomes',fontsize=14);fig.supxlabel('Nominal joint 99% within each panel family; at least 98% across both.\nLocal/count metrics do not bound full 1,152-bit pattern TV.',fontsize=10)
fig.savefig(out/'accuracy.png',dpi=180);fig.savefig(out/'accuracy.svg');plt.close(fig)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'plan.json').write_text(json.dumps(dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*paths]}),indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
