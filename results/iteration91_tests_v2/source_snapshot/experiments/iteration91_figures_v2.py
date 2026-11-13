"""Separate full-trajectory quality, conditional readout errors and serial costs."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[Path(__file__),ROOT/'results/iteration91_joint_quality_v2/summary.json',ROOT/'results/iteration91_fock_quality_v1/summary.json',ROOT/'results/iteration91_timing_v2/summary.json'];q=read(files[1]);f=read(files[2]);t=read(files[3]);fig,axes=plt.subplots(1,3,figsize=(17,5.1),layout='constrained')
names=['pnr3','pnr5','block16','physical16','optical16','compact16','compact16_fixed'];labels=['PNR R3 (reused)','PNR R5 (reused)','Block B16 (reused)','Canonical89 (reused)','Optical90 (reused)','Compact91 v2 (superseded)','Compact91 v3 (new)']
for i,name in enumerate(names):
 r=q['rows'][name]['experimental_click_count'];x=100*r['empirical_cdf_distance'];lo=100*r['nominal_joint99_cdf_lower'];hi=100*r['nominal_joint99_cdf_upper'];axes[0].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color='#145a8d' if name.startswith('pnr') else '#bb6729')
axes[0].set_yticks(range(7),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Experimental total-click CDF distance (%)');axes[0].set_title('Full trajectories: nominal joint 99%\n69 events, iid and reference assumptions');axes[0].grid(axis='x',alpha=.25);axes[0].set_xlim(left=0)
rows=[r for r in f['comparisons'] if r['case']==0 and r['target_photons']==4];x=np.arange(len(rows))
for shift,kind,label,color in ((-.18,'pnr','PNR','#145a8d'),(.18,'click','Click','#bb6729')):
 lo=np.array([100*r[kind+'_tv_lower'] for r in rows]);hi=np.array([100*r[kind+'_tv_upper'] for r in rows]);mid=(lo+hi)/2;axes[1].bar(x+shift,mid,.34,yerr=(hi-lo)/2,capsize=3,label=label,color=color)
axes[1].set_xticks(x,['Gaussian\nL=1','Parity\nL=2','L=3','L=4','Fock\nL=8']);axes[1].set_ylabel('Complete conditional model TV (%)');axes[1].set_title('Same 3-mode target, k=4\nPNR improvement can worsen clicks');axes[1].legend();axes[1].grid(axis='y',alpha=.2);axes[1].set_ylim(0,85)
med={(r['method'],r['cache'],r['records']):r['median_seconds'] for r in t['medians']}
for i,method in enumerate(('optical90','compact91')):axes[2].bar(np.arange(2)+(i-.5)*.34,[med[method,cache,64] for cache in ('cold','warm')],.34,label=method,color=('#777777','#145a8d')[i])
axes[2].set_xticks(range(2),['Cold cache','Warm cache']);axes[2].set_ylabel('Seconds / 64 full dual-spectrum records');axes[2].set_title('1152-grid direct sampling\n3 serial repeats, median, 16 threads');axes[2].legend();axes[2].grid(axis='y',alpha=.2);fig.suptitle('Iteration 91: compact optical cache and explicit occupation cores',fontsize=15);fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
(out/'plan.json').write_text(json.dumps(dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},scope=__doc__,limits='Middle panel concerns conditional models; Fock analytic closure error is zero, executable sampler still has explicit truncation tail and floating-point error.'),indent=2)+'\n',encoding='utf-8');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
