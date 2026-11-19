"""Full-output count-distribution gap to actual Jiuzhang records, without a model.
DKW plus contraction gives a lower bound on full-bit TV. Confidence is joint99
over the two generated iid datasets and the experimental iid dataset, conditional
on those assumptions; no claim of experimentally verified independence.
"""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
real=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';base=ROOT/'results/iteration83_global_quality_v1';files=[real,base/'combined_4.npz',base/'combined_8.npz'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*files]},delta_per_dataset=.01/3))
h=np.load(real)['total_click_histogram'];N=int(h.sum());cdf=np.cumsum(h)/N;real_radius=math.sqrt(math.log(600)/(2*N));rows=[]
for budget in (4,8):
 d=np.load(base/f'combined_{budget}.npz')['total_histogram'];n=int(d.sum());q=np.cumsum(d)/n;diff=q-cdf;j=int(np.argmax(abs(diff)));radius=math.sqrt(math.log(600)/(2*n));row=dict(budget=budget,generated_records=n,experimental_records=N,empirical_count_tv=float(abs(d/n-h/N).sum()/2),empirical_count_ks=float(abs(diff[j])),threshold=j,generated_cdf=float(q[j]),experimental_cdf=float(cdf[j]),generated_dkw_radius=radius,experimental_dkw_radius=real_radius,nominal_joint99_full_bit_tv_lower_bound=max(0,float(abs(diff[j]))-radius-real_radius));rows.append(row);save(f'row_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
