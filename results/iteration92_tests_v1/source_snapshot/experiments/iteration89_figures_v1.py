"""Static scientific figures from completed frozen quality and serial timing tables."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
files=[Path(__file__),ROOT/'results/iteration89_joint_quality_v1/summary.json',ROOT/'results/iteration89_pnr_timing_v1/summary.json',ROOT/'results/iteration89_direct_timing_v1/summary.json'];q=read(files[1]);pnr=read(files[2]);direct=read(files[3]);fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
names=['pnr3','pnr5','block16','physical16','unchecked16'];labels=['PNR R3 (reused)','PNR R5 (new)','Block B16','Conservative B16','Unchecked B16 (reused)']
for i,name in enumerate(names):
 r=q['rows'][name]['experimental_click_count'];x=100*r['empirical_cdf_distance'];low=100*r['nominal_joint99_cdf_lower'];high=100*r['nominal_joint99_cdf_upper'];axes[0].errorbar(x,i,xerr=[[x-low],[high-x]],fmt='o',capsize=5,color='#145a8d' if name.startswith('pnr') else '#bb6729')
axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Distance to experimental total-click CDF (%)');axes[0].set_title('One nominal joint 99% family (53 events)');axes[0].grid(axis='x',alpha=.25);axes[0].set_xlim(left=0)
pt={(r['method'],r['retained']):r['median_seconds'] for r in pnr['medians']};dt={(r['method'],r['records']):r['median_seconds'] for r in direct['medians']};times=[pt['old',3],pt['new',3],pt['old',5],pt['new',5],dt['block16',64],dt['physical16',64]];lab=['Old PNR R3','New PNR R3','Old PNR R5','New PNR R5','Block B16','Conservative B16'];axes[1].barh(range(6),times,color=['#a8bac9','#145a8d','#a8bac9','#145a8d','#e6b48e','#bb6729']);axes[1].set_yticks(range(6),lab);axes[1].invert_yaxis();axes[1].set_xlabel('Seconds per 64 dual-spectrum records');axes[1].set_title('16 threads; 3 serial repeats, median');axes[1].grid(axis='x',alpha=.25)
for i,t in enumerate(times):axes[1].text(t+.03,i,f'{t:.3f}',va='center',fontsize=9)
axes[1].set_xlim(0,max(times)*1.2);fig.suptitle('Iteration 89: accuracy and measured implementation cost',fontsize=14);fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
(out/'plan.json').write_text(json.dumps(dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},scope=__doc__,limits='CDF intervals do not bound the full pattern TV. Replays do not add quality samples. Timing and quality use different fixed records.'),indent=2)+'\n',encoding='utf-8');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
