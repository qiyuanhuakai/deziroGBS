from pathlib import Path
import json,hashlib,math
import numpy as np
import mpmath as mp
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.git/iteration69_count_quality_v2_20261110';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
mp.mp.dps=100
edges=[0,1,2,3,4,5,6,8,10,16,32,64,128,256,512,1024]
save('plan.json',dict(source_sha256=H(Path(__file__)),edges=edges,scope='Total PNR count marginal of the SAME two equal lossy squeezed inputs, independent of passive interferometer. PGF=(1-q)/(1-q*(1-eta+eta*z)^2). The recurrence evaluates this marginal without Fock truncation; last bin contains every count >=1024. These marginal TVs are LOWER-bound diagnostics on joint PNR TV, never upper bounds on full GBS error. Fixed seeded sample estimates, nominal joint 99% IID marginal bands. Tail confidence intervals are pointwise 99% Clopper-Pearson numerical evaluations, not simultaneous certificates.'))
rows=[];deps={};repetitions=16;radius=math.sqrt((len(edges)*math.log(2)+math.log(repetitions/.01))/(2*500000))
for case,(r,eta) in enumerate([(.7,.1),(.7,.3),(.7,.65),(1.1,.65)]):
    rr=mp.mpf(r);e=mp.mpf(eta);q=mp.tanh(rr)**2;a=1-e;den=1-q*a*a;aa=2*q*a*e/den;bb=q*e*e/den;p=[(1-q)/den];p.append(aa*p[0])
    for k in range(2,1024):p.append(aa*p[-1]+bb*p[-2])
    t=mp.tanh(rr);b=t*e/(1-t*a);c=-t*e/(1+t*a);cp=(1-q)/(2*(1-t*a));cm=(1-q)/(2*(1+t*a))
    def tail(k):return cp*b**k/(1-b)+cm*c**k/(1-c)
    bins=[sum(p[left:right]) for left,right in zip(edges[:-1],edges[1:])]+[tail(1024)]
    assert min(bins)>=0 and abs(sum(bins)-1)<mp.mpf('1e-90');target=np.array([float(x) for x in bins]);save(f'target_{case}.json',dict(probabilities=[str(x) for x in bins],tail_ge64=str(tail(64)),tail_ge128=str(tail(128))))
    for label,folder in [('centered','iteration69_quality_pilot_20261110'),('displaced','iteration69_displaced_v6_20261110')]:
        for method in (1,3):
            path=ROOT/'.git'/folder/f'case_{case}_method_{method}.npz';deps[path.relative_to(ROOT).as_posix()]=H(path);counts=np.load(path)['counts'];total=counts.sum(axis=1);hist=np.bincount(np.searchsorted(edges,total,side='right')-1,minlength=len(edges));tv=float(.5*sum(abs(hist/len(total)-target)));tails=[]
            for cutoff in (64,128):
                k=int(sum(total>=cutoff));n=len(total);lo=0. if k==0 else float(beta.ppf(.005,k,n-k+1));hi=1. if k==n else float(beta.ppf(.995,k+1,n-k));tails.append(dict(cutoff=cutoff,events=k,frequency=k/n,target=str(tail(cutoff)),pointwise_99_interval=[lo,hi]))
            rows.append(dict(case=case,law=label,method=method,empirical_binned_total_count_tv=tv,nominal_marginal_tv_interval=[max(0,tv-radius),min(1,tv+radius)],maximum_count=int(total.max()),tails=tails,histogram=hist.tolist()))
save('summary.json',dict(rows=rows,nominal_joint_radius=radius,dependencies=deps));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(dict(radius=radius,rows=[{k:v for k,v in row.items() if k!='histogram'} for row in rows]),indent=2))
