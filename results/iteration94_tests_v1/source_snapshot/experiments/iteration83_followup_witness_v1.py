"""Predeclared accuracy witnesses for new ordering, B32 weights and photon-first.
New-family joint99 coverage: four methods times two click regions, one photon
count observable, and the shared experimental count CDF, each delta=.001.
These bounds are separate from the earlier 17-region baseline confidence family.
Failed batches retain their status and receive no assigned complete sampling law.
"""
from pathlib import Path
import os,argparse,json,hashlib,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration83_full_followup_quality_v1';order=ROOT/'results/iteration83_order_pilot_v1';profilepath=ROOT/'results/iteration83_bias_profile_v1/summary.json';realpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),base/'summary.json',order/'summary.json',profilepath,realpath,inp]+sorted(base.glob('*_combined.npz'))+sorted(order.glob('combined_*.npz'));save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},regions=[[0,1152],[1024,1152]],delta_per_observable=.001));profile=read(profilepath);real=np.load(realpath)['total_click_histogram'];N=int(real.sum());realcdf=np.cumsum(real)/N;realradius=math.sqrt(math.log(2000)/(2*N));xs=np.load(inp)['x'];photon_mean=0.;photon_variance=0.
for x in xs:
 n=len(x)//2;normal=(x[:n,:n]+x[n:,n:]+1j*(x[n:,:n]-x[:n,n:]))/2;anomalous=(x[:n,:n]-x[n:,n:]+1j*(x[n:,:n]+x[:n,n:]))/2;photon_mean+=float(np.trace(normal).real);photon_variance+=float(np.trace(normal@(normal+np.eye(n))).real+np.sum(abs(anomalous)**2))
save('reference_photons.json',dict(mean=photon_mean,variance=photon_variance,scope='exact zero-mean Gaussian moment identity evaluated in float64; summed independent sectors'))
def bound(counts,mu,var,size):
 n=len(counts);eps=math.sqrt(math.log(2000)/(2*n));end=max(512,int(counts.max())) if size is None else size;hist=np.bincount(counts,minlength=end+1);best=dict(nominal_joint99_full_tv_lower=0.,direction=None);choices=[('increasing',hist,mu)]
 if size is not None:choices.append(('decreasing',hist[::-1],size-mu))
 for direction,h,mean in choices:
  integral=np.r_[0,np.cumsum(1-np.cumsum(h)/n)]
  for U in range(1,end+1):
   L=np.arange(U);width=U-L;emp=(integral[U]-integral[L])/width;stop=(math.hypot(math.sqrt(var),U-mean)-(U-mean))/2;ref=(mean-L-stop)/width;values=ref-emp-eps;j=int(np.argmax(values))
   if values[j]>best['nominal_joint99_full_tv_lower']:best=dict(nominal_joint99_full_tv_lower=float(values[j]),direction=direction,clip_lower=int(L[j]),clip_upper=U,reference_expectation_lower=float(ref[j]),empirical_expectation=float(emp[j]))
 return dict(records=n,reference_mean=mu,reference_variance=var,empirical_mean=float(counts.mean()),empirical_variance=float(counts.var(ddof=1)),dkw_radius=eps,**best)
panels=[(f'order{r["budget"]}',r,order/f'combined_{r["budget"]}.npz') for r in read(order/'summary.json')['rows']]+[(r['method'],r,base/f'{r["method"]}_combined.npz') for r in read(base/'summary.json')['rows']];rows=[]
for method,panel,path in panels:
 if not panel['success']:rows.append(dict(method=method,success=False,reason='Failed complete batch; no valid law assigned'));continue
 d=np.load(path);bits=d['clicks'];record=dict(method=method,success=True,click_regions=[])
 for first,end in ((0,1152),(1024,1152)):
  ref=next(r for r in profile['reference'] if (r['first'],r['end'])==(first,end));record['click_regions'].append(dict(first=first,end=end,**bound(bits[:,first:end].sum(1).astype(int),ref['mean'],ref['variance'],end-first)))
 totals=bits.sum(1).astype(int);h=np.bincount(totals,minlength=1153);cdf=np.cumsum(h)/len(bits);diff=abs(cdf-realcdf);j=int(np.argmax(diff));record['experimental_comparison']=dict(threshold=j,empirical_ks=float(diff[j]),nominal_joint99_full_bit_tv_lower=max(0.,float(diff[j])-record['click_regions'][0]['dkw_radius']-realradius),experimental_records=N,experimental_dkw_radius=realradius)
 if method=='pnr3':record['photons']=bound(d['counts'].sum(1).astype(int),photon_mean,photon_variance,None)
 rows.append(record);save(f'row_{method}.json',record);print(record,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__,confidence='conditional on iid generated/experimental datasets and fixed numerical Gaussian moments; not an interval arithmetic certificate'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
