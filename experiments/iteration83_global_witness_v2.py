"""Simultaneous clipped-count expectation witnesses from Gaussian first moments.
For any variable C with mean mu and variance v,
E[(C-U)+] <= (sqrt(v+(U-mu)^2)-(U-mu))/2 by Cauchy-Schwarz.
Thus E[clip(C-L,0,U-L)] >= mu-L-that_bound.
DKW bounds every normalized monotone clipped function simultaneously. Optimizing
L,U after observing the data does not consume another confidence union budget.
"""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration83_global_quality_v1';summary=json.loads((base/'summary.json').read_text());mu=summary['reference_mean'];var=summary['reference_variance'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),base/'summary.json',base/'reference_moments.json']},delta_per_method=.005));rows=[]
for row in summary['rows']:
 if not row['success']:rows.append(dict(budget=row['budget'],success=False,reason='Failed batch has no assigned valid sampler law'));continue
 d=np.load(base/f"combined_{row['budget']}.npz");hist=d['total_histogram'];count=int(hist.sum());cdf=np.cumsum(hist)/count;survival=1-cdf;integral=np.r_[0,np.cumsum(survival)];eps=math.sqrt(math.log(2/.005)/(2*count));best=dict(lower_bound=0.)
 for U in range(1,len(hist)):
  L=np.arange(U);width=U-L;emp=(integral[U]-integral[L])/width;stoploss=(np.sqrt(var+(U-mu)**2)-(U-mu))/2;lower=np.maximum(0,(mu-L-stoploss)/width);bound=lower-emp-eps;j=int(np.argmax(bound))
  if bound[j]>best['lower_bound']:best=dict(lower_bound=float(bound[j]),clip_lower=int(L[j]),clip_upper=U,reference_normalized_expectation_lower=float(lower[j]),empirical_normalized_expectation=float(emp[j]),dkw_radius=eps,reference_stoploss_upper=float(stoploss))
 result=dict(budget=row['budget'],success=True,records=count,nominal_joint99_full_tv_lower_bound=best.pop('lower_bound'),**best);rows.append(result);save(f"row_{row['budget']}.json",result);print(result,flush=True)
save('summary.json',dict(rows=rows,scope='Full 1152-bit sampler versus fixed two-sector reference; iid and numerical-moment assumptions explicit. Not an experimental-data accuracy bound.'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
