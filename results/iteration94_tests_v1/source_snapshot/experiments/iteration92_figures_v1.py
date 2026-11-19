"""Scientific figure from independently recomputed60-event quality and timings."""
import argparse
from iteration92_common_v2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);v=read(ROOT/'results/iteration92_verification_v1/validation.json');plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,3,figsize=(16,5.8),gridspec_kw={'width_ratios':[1.2,1.1,1.]})
q=[next(x['pnr'] for x in v['historical88'] if x['name']=='paired_strong' and x['retained']==k) for k in (3,5)]+[v['new_quality'][1]['pnr'],next(x['pnr'] for x in v['baselines'] if x['name']=='strong' and x['method']=='photonic-engines')];names=['Moment reset R3','Moment reset R5','Coherent core','photonic-engines'];colors=['#b65342','#d48b57','#147d92','#4e639e']
for j,(row,c) in enumerate(zip(q,colors)):
 lo,hi=100*row['lower'],100*row['upper'];axs[0].plot([lo,hi],[j,j],lw=6,color=c,solid_capstyle='round');axs[0].text(hi+.6,j,f'{lo:.2f}–{hi:.2f}%',va='center',fontsize=10)
axs[0].set_yticks(range(4),names);axs[0].invert_yaxis();axs[0].set_xlim(-1,38);axs[0].set_xlabel('Complete PNR TV (%)');axs[0].set_title('Strong paired state, r = 0.85\nOne nominal joint 99% family',fontsize=12)
times={(r['case'],r['method']):r['seconds']/4096*1e6 for r in v['timings']['v2']};x=np.arange(3);labels=['Coherent core\nPNR + click','photonic-engines\nPNR + click','Frugal K4\nclick only']
for case,color in ((0,'#57a2b1'),(1,'#243c63')):
 vals=[times[case,m] for m in ('core1','pe1','frugal1')];axs[1].bar(x+(case-.5)*.34,vals,width=.34,label=f'r = {(.6,.85)[case]}',color=color)
axs[1].set_yscale('log');axs[1].set_xticks(x,labels,fontsize=9);axs[1].set_ylabel('Microseconds per record (log scale)');axs[1].set_title('Same physical input, one CPU thread\nKnown vacuum ports removed',fontsize=12);axs[1].legend(frameon=False)
s=[r for r in v['scaling'] if r['threads']==16];axs[2].plot([r['modes'] for r in s],[r['seconds']/4096*1e6 for r in s],'o-',color='#147d92');axs[2].set_xscale('log');axs[2].set_xlabel('Output modes');axs[2].set_ylabel('Microseconds per record');axs[2].set_title('No output occupation table\nFixed rank 2, k = 4, r = 0.3',fontsize=12);axs[2].grid(alpha=.2)
fig.suptitle('Coherent finite-core trajectories: scoped accuracy and measured cost',fontsize=16,y=.98);fig.text(.5,.025,'393,216 fresh core records; failures retained. Timings: median of 3 serial runs, 4,096 records.\nRank-two results do not establish general GBS or Jiuzhang-scale superiority.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.11,1,.91]);fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.svg');save(out,'plan.json',dict(source_sha256=H(Path(__file__)),verification_sha256=H(ROOT/'results/iteration92_verification_v1/validation.json')));manifest(out)
