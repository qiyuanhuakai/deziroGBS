"""Plot independently audited full-click confidence intervals and serial timings."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'results/iteration94_verification_v1/validation.json';v=json.loads(source.read_text());assert v['passed'];t={r['job']['timing_name']:r for r in v['timings']}
colors={'mps':'#225ea8','branch':'#d95f0e','pe':'#238b45','frugal':'#756bb1','direct':'#555555'}
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained');titles=['Pure four-mode','Displaced four-mode','Pure eight-mode','Lossy two-mode']
for j,ax in enumerate(axes.flat):
    selected=[r for r in v['rows'] if r['case']==j]
    for i,r in enumerate(selected):
        timing=t[r['name']+'_t1'];x=timing['median_seconds']/2048*1e6;c=r['click'];lo=c['lower']*100;hi=c['upper']*100;mid=(lo+hi)/2;short=r['name'].split('_',1)[1] if j!=3 else r['name'][len('lossy_tmsv_'):]
        label={'mps':'MPS -> clicks','pe':'PE -> clicks','branch4':'Branch K4','branch64':'Branch K64','frugal2':'Frugal K2','frugal3':'Frugal K3','frugal4':'Frugal K4','frugal5':'Frugal K5','direct16':'Direct91 B16'}[short]
        ax.errorbar(x,mid,yerr=[[mid-lo],[hi-mid]],fmt='o',capsize=4,color=colors[r['family']],label=label,markersize=5)
    ax.set_xscale('log');ax.set_title(titles[j],loc='left',fontweight='bold');ax.set_xlabel('Microseconds / attempt, CPU1 (log scale)');ax.set_ylabel('Full-click TV confidence interval (%)');ax.set_ylim(bottom=-.15);ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2,loc='upper right');ax.margins(x=.2)
fig.suptitle('Iteration 94: click accuracy and runtime on the same inputs',fontsize=17,fontweight='bold')
fig.get_layout_engine().set(rect=(0,.05,1,.96))
fig.text(.5,.012,'Nominal joint 99% family, conditional on iid and numerical references. Markers are interval midpoints, not error estimates.\n65,536 quality attempts / configuration; failures retained. Timing: median of 3 serial batches of 2,048; prepared-model scope.',ha='center',fontsize=9)
fig.savefig(out/'click_frontier.png',dpi=160);fig.savefig(out/'click_frontier.pdf');plt.close(fig)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'plan.json').write_text(json.dumps(dict(scope=__doc__,hashes={Path(__file__).relative_to(ROOT).as_posix():H(Path(__file__)),source.relative_to(ROOT).as_posix():H(source)}),indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({p.name:H(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
print(out)
