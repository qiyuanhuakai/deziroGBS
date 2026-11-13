"""Exact rational law of every exported small factor alias/rejection plan.
These bounds concern discrete implementation vs normalized binary64 factors,
not its quantum approximation or the independence of PRNG streams.
"""
from pathlib import Path
import argparse,json,hashlib,math
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def exact(x):return F.from_float(float(x))
def form(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),approximation=float(x))
save('plan.json',dict(scope=__doc__,source_sha256=H(Path(__file__)),fixed_total=34,random_model='independent uniform uint64 words for alias and acceptance'))
base=ROOT/'results/iteration82_factor_pilot_v1';inputs=ROOT/'results/iteration81_factor_pilot_v2';rows=[]
for path in sorted(base.glob('law_*.npz')):
 label=path.stem[4:];d=np.load(path);L,m,_=d['oriented_pairs'].shape;inp=np.load(inputs/f'input_{label}.npz');ff=np.load(inputs/f'factors_{label}_0.npz');K=[[exact(x) for x in row] for row in inp['K']];w=[exact(x) for x in inp['weights']];wt=sum(w);w=[v/wt for v in w];maximum=F(0);min_accept=F(1);actual_weights=None;maximum_export_difference=0.
 for l in range(L):
  f=np.load(base/f'integer_{label}_{l}.npz');offset=0;tables=[]
  for size in f['sizes']:
   size=int(size);levels=2**64//size;counts=[0]*size
   for i in range(size):
    cut=int(f['thresholds'][offset+i]);alias=int(f['aliases'][offset+i]);assert 0<=cut<=levels and 0<=alias<size;counts[i]+=cut;counts[alias]+=levels-cut
   assert sum(counts)==2**64;tables.append([F(n,2**64) for n in counts]);offset+=size
  assert offset==len(f['thresholds']);actual_weights=tables[0][:L];assert all(p==0 for p in tables[0][L:]);root=tables[1][:m];assert all(p==0 for p in tables[1][m:]);cut=[int(x) for x in f['accept']];assert all(0<=x<=2**53 for x in cut)
  actual=[]
  for i in range(m):
   q=tables[2+i][:m];assert all(p==0 for p in tables[2+i][m:]);den=sum(q[j]*cut[j] for j in range(m))
   if root[i]:assert den>0;min_accept=min(min_accept,den/2**53)
   actual.extend([root[i]*q[j]*cut[j]/den if root[i] else F(0) for j in range(m)])
  assert sum(actual)==1
  aa=[exact(x) for x in ff['left'][l]];bb=[exact(x) for x in ff['right'][l]];ideal=[aa[i]*K[i][j]*bb[j] for i in range(m) for j in range(m)];norm=sum(ideal);ideal=[x/norm for x in ideal];tv=sum(abs(x-y) for x,y in zip(actual,ideal))/2;maximum=max(maximum,tv);maximum_export_difference=max(maximum_export_difference,max(abs(float(p)-q) for p,q in zip(actual,d['oriented_pairs'][l].flat)))
 latent=sum(abs(x-y) for x,y in zip(w,actual_weights))/2;bound=latent+17*maximum;assert maximum_export_difference<1e-14 and float(bound)<1e-10
 row=dict(label=label,labels=L,modes=m,maximum_pair_tv=form(maximum),latent_tv=form(latent),fixed_total34_tv_upper=form(bound),minimum_acceptance=form(min_accept),log10_max_attempt_failure_bound_N34=None if min_accept==1 else math.log10(17)+1000000*math.log1p(-float(min_accept))/math.log(10),zero_attempt_failure_probability=min_accept==1,maximum_float_export_difference=maximum_export_difference);rows.append(row);save(label+'.json',row);print(label,'exact rational fixed N34 TV bound',float(bound),flush=True)
save('summary.json',dict(passed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
