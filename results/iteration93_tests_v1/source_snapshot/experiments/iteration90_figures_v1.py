"""Full-trajectory count-CDF intervals and exact conditional-model TV, separately."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[Path(__file__),ROOT/'results/iteration90_joint_quality_v1/summary.json',ROOT/'results/iteration90_parity_gates_v1/summary.json'];q=read(files[1]);pg=read(files[2]);fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained');names=['pnr3','pnr5','block16','physical16','optical16'];labels=['PNR R3 (reused)','PNR R5 (reused)','Block B16 (reused)','Canonical B16 (reused)','Optical B16 (new)']
for i,name in enumerate(names):
 r=q['rows'][name]['experimental_click_count'];x=100*r['empirical_cdf_distance'];lo=100*r['nominal_joint99_cdf_lower'];hi=100*r['nominal_joint99_cdf_upper'];axes[0].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=5,color='#145a8d' if name.startswith('pnr') else '#bb6729')
axes[0].set_yticks(range(5),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Distance to experimental total-click CDF (%)');axes[0].set_title('Full trajectories: nominal joint 99%, 53 events');axes[0].grid(axis='x',alpha=.25);axes[0].set_xlim(left=0)
r=pg['fock_comparisons'];axes[1].plot([x['photons'] for x in r],[100*x['gaussian_tv'] for x in r],'s-',label='Moment-matched Gaussian',color='#777777');axes[1].plot([x['photons'] for x in r],[100*x['parity_tv'] for x in r],'o-',label='Moment-matched parity projection',color='#145a8d');axes[1].set_xlabel('Photons removed from one TMSV mode');axes[1].set_ylabel('Complete conditional two-mode PNR TV (%)');axes[1].set_title('Conditional models only: exact rational TV');axes[1].grid(alpha=.25);axes[1].legend(fontsize=9);axes[1].set_ylim(65,100);fig.suptitle('Iteration 90: optical-space constraints and closure limitations',fontsize=14);fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
(out/'plan.json').write_text(json.dumps(dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},scope=__doc__,limits='Right panel is conditional model error, not the error of the complete sampling algorithm; parity inverse is not yet an end-to-end sampler.'),indent=2)+'\n',encoding='utf-8');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
