"""Full-output mean-click witness for the fixed two-common-sector model.
This diagnoses model versus experimental data, separately from sampler error.
The confidence statement assumes iid records and fixed supplied model inputs.
"""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
files=[ROOT/f'results/iteration83_full_quality_v1/covariance_{s}.npz' for s in range(2)];histpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*files,histpath]},delta=.01))
vac=[];m=1152;ids=np.arange(m)
for path in files:
 c=np.load(path)['cov'];det=(1+c[ids,ids])*(1+c[m+ids,m+ids])/4-c[ids,m+ids]**2/4;assert np.all(det>=1);vac.append(1/np.sqrt(det))
clicks=1-vac[0]*vac[1];reference_mean=float(clicks.sum());hist=np.load(histpath)['total_click_histogram'];N=int(hist.sum());empirical_mean=float(hist@np.arange(len(hist))/N);radius=m*math.sqrt(math.log(2/.01)/(2*N));difference=abs(reference_mean-empirical_mean);row=dict(records=N,reference_expected_total_clicks=reference_mean,empirical_mean_total_clicks=empirical_mean,absolute_mean_difference=difference,empirical_full_tv_lower_bound_from_bounded_mean=difference/m,nominal99_mean_radius=radius,nominal99_reference_vs_experiment_full_tv_lower_bound=max(0,difference-radius)/m,scope=__doc__);save('summary.json',row);np.savez_compressed(OUT/'single_port_law.npz',reference_click_probabilities=clicks,vacuum=np.array(vac));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(row),flush=True)
