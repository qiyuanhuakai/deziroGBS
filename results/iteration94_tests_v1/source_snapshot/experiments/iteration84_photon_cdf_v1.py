"""Numerical Gaussian reference for total PNR count and scoped CDF accuracy.
For zero-mean X=(Gamma-I)/2, G(z)=det(I+(1-z)X)^(-1/2).
Reuse the three photon-CDF events (delta=.001 each) already allocated in the
regenerated ten-event family; no additional stochastic event is introduced.
A total-count CDF upper bound is NOT a full-PNR TV upper bound.
"""
from pathlib import Path
import os,json,argparse,hashlib,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';base=ROOT/'results/iteration84_pnr_regenerated_v1';files=[Path(__file__),inp,base/'plan.json',base/'summary.json',*[base/f'{name}_combined.npz' for name in ('mixed','core0','core1')]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},cutoff=1024,decimal_digits=90,delta_per_dataset=.001));mp.mp.dps=90
def coefficients(lambdas,K):
 c=[x/(1+x) for x in lambdas];pref=mp.fprod(1/mp.sqrt(1+x) for x in lambdas);powers=[mp.mpf(1)]*len(c);traces=[];prob=[pref]
 for j in range(1,K+1):
  powers=[a*b for a,b in zip(powers,c)];traces.append(mp.fsum(powers)/2);prob.append(mp.fsum(traces[k-1]*prob[j-k] for k in range(1,j+1))/j)
 return prob,c
nu=mp.mpf('0.7');thermal,_=coefficients([nu,nu],64);t=mp.mpf('0.4');squeezed,_=coefficients([-t/(1+t),t/(1-t)],64);analytic=[]
for j in range(65):
 ref=nu**j/(1+nu)**(j+1);assert abs(thermal[j]-ref)<mp.mpf('1e-80');reference=mp.mpf(0) if j%2 else mp.sqrt(1-t*t)*mp.binomial(j,j//2)*(t*t/4)**(j//2);assert abs(squeezed[j]-reference)<mp.mpf('1e-80')
save('analytic_checks.json',dict(passed=True,coefficients_checked=130,models=['single-mode thermal','single-mode pure squeezed vacuum'],absolute_tolerance='1e-80'))
data=np.load(inp);eigenvalues=np.concatenate([np.linalg.eigvalsh(x) for x in data['x']]);lambdas=[mp.mpf(float(v)) for v in eigenvalues];prob,c=coefficients(lambdas,1024);assert min(prob)>=0;positive=max(c);left=mp.mpf(1);right=1/positive
for _ in range(200):
 radius=(left+right)/2;tilted=mp.fsum(q*radius/(1-q*radius) for q in c)/2
 if tilted<1025:left=radius
 else:right=radius
radius=(left+right)/2;tail=mp.exp(-mp.fsum(mp.log(1+x) for x in lambdas)/2-mp.fsum(mp.log(1-q*radius) for q in c)/2-1025*mp.log(radius));normalization=mp.fsum(prob);assert 0<=1-normalization<=tail+mp.mpf('1e-80');reference_mean=mp.fsum(j*prob[j] for j in range(1025));reference_second=mp.fsum(j*j*prob[j] for j in range(1025));trace_mean=mp.fsum(lambdas)/2;trace_var=mp.fsum(x*(1+x) for x in lambdas)/2;save('reference.json',dict(eigenvalues=eigenvalues.tolist(),probabilities=[str(x) for x in prob],tail_upper=str(tail),chernoff_radius=str(radius),normalization=str(normalization),truncated_mean=str(reference_mean),trace_mean=str(trace_mean),truncated_variance=str(reference_second-reference_mean**2),trace_variance=str(trace_var),scope='High precision arithmetic on float64 covariance eigenvalues; numerical reference, not directed-rounding certificate.'))
pmf=np.array([float(x) for x in prob]);cdf=np.cumsum(pmf);np.savez_compressed(OUT/'reference.npz',photon_pmf=pmf,photon_cdf=cdf,eigenvalues=eigenvalues);rows=[]
for panel in read(base/'summary.json')['rows']:
 if not panel['success']:rows.append(dict(method=panel['method'],success=False));continue
 d=np.load(base/f"{panel['method']}_combined.npz");values=d['counts'].sum(1).astype(int);assert values.max()<=1024;hist=np.bincount(values,minlength=1025);emp=np.cumsum(hist)/len(values);dif=abs(emp-cdf);j=int(np.argmax(dif));ks=float(dif[j]);eps=math.sqrt(math.log(2000)/(2*len(values)));row=dict(method=panel['method'],success=True,records=len(values),threshold=j,empirical_count_cdf_distance=ks,dkw_radius=eps,nominal_joint99_population_count_cdf_lower=max(0.,ks-eps),nominal_joint99_population_count_cdf_upper=min(1.,ks+eps+float(tail)),scope='Count CDF/Kolmogorov distance only. Lower also bounds full PNR TV; upper does not.');rows.append(row);np.savez_compressed(OUT/f"{panel['method']}.npz",histogram=hist,empirical_cdf=emp);print(row,flush=True)
save('summary.json',dict(rows=rows,reference_tail_upper=str(tail),scope=__doc__,confidence='Same photon-CDF events as existing regenerated joint99 family; conditional iid and fixed numerical reference assumptions.'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
