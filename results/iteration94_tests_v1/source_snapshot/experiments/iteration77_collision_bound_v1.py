"""Repair the target-tail bound without repeating or changing frozen observations.
The original 80-digit subtraction returned zero through cancellation. A positive
geometric majorant of the negative-binomial remainder replaces that value.
"""
from pathlib import Path
import json,hashlib,argparse,shutil,math
import numpy as np
import mpmath as mp
mp.mp.dps=100
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration77_terminal_collision_v1';d=np.load(base/'input.npz');v,t=d['v'],float(d['t']);a=mp.mpf(v.shape[1])/2;z=mp.mpf(t)**2
ell=[sum(mp.mpc(complex(x)).real**2+mp.mpc(complex(x)).imag**2 for x in row) for row in v];diag=[sum(mp.mpc(complex(x))**2 for x in row) for row in v];A=sum(abs(x)**2 for x in diag);L=sum(x*x for x in ell);term=(1-z)**a;expectation=mp.mpf(0)
for j in range(1,257):
    term*=z*(a+j-1)/j;f=(j*(a+j)*A+2*j*(j-1)*L)/(a*(a+1)*2*j*(2*j-1));expectation+=term*f
ratio=z*(a+256)/257;assert 0<ratio<1 and a>=1;tail=term*ratio/(1-ratio)
values=[]
for batch in range(10):
    d=np.load(base/f'counts_{batch}.npz');x=d['counts'].astype(float);totals=x.sum(axis=1);f=np.divide((x*(x-1)).sum(axis=1),totals*np.maximum(totals-1,0),out=np.zeros(len(x)),where=totals>=2);np.testing.assert_array_equal(f,d['collision']);values.extend(f)
N=len(values);mean=float(np.mean(values));variance=float(np.var(values,ddof=1));radius=math.sqrt(2*variance*math.log(400)/N)+7*math.log(400)/(3*(N-1));lower=max(0,abs(mean-float(expectation))-radius-float(tail));old=json.loads((base/'summary.json').read_text());assert abs(float(expectation)-old['target_mean'])<1e-15
result=dict(passed=True,scope=__doc__,records=N,target_partial_expectation=str(expectation),target_remainder_upper=str(tail),empirical_mean=mean,sample_variance=variance,nominal_two_sided99_radius=radius,nominal_full_tv_lower=lower,original_result_preserved=True,numerical_arithmetic_not_globally_certified=True)
(OUT/'validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(result),flush=True)
(OUT/'manifest.json').write_bytes((json.dumps({'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}},indent=2)+'\n').encode())
