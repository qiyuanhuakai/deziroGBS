"""Convex reference solve in a fixed passive pure-core basis, not a sampler.
Minimize tr diag(exp(-2s),exp(2s)) subject to that diagonal <= R^T V R.
Log-det barrier Newton steps preserve positive noise without eigenvalue clipping.
For PSD dual Y, 2 sum sqrt((1+Yqq)(1+Ypp))-tr(Y W) is a dual lower bound.
Only 100-dimensional independent reference algebra; native implementation pending.
"""
from pathlib import Path
import os,argparse,json,hashlib,math,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';split=ROOT/'results/iteration83_core_budget_v2';xs=np.load(inp)['x'];n=xs.shape[1]//2;save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),inp,split/'split_0.npz',split/'split_1.npz']},barriers=[1,.1,.01,.001,.0001,.00001,.000001,.0000001,.00000001]));mp.mp.dps=80;rows=[]
for sector,x in enumerate(xs):
 start=time.perf_counter();old=np.load(split/f'split_{sector}.npz');R=old['rotation'];s=np.arctanh(old['t']);V=np.eye(2*n)+2*x;W=R.T@V@R;W=(W+W.T)/2;trace=[]
 def evaluate(s,mu,derivatives=True):
  d=np.r_[np.exp(-2*s),np.exp(2*s)];B=W-np.diag(d);chol=cho_factor(B,lower=True,check_finite=False);f=float(d.sum()-mu*2*np.log(np.diag(chol[0])).sum())
  if not derivatives:return f
  A=cho_solve(chol,np.eye(2*n),check_finite=False);dq,dp=-2*d[:n],2*d[n:];g=dq+dp+mu*(dq*np.diag(A)[:n]+dp*np.diag(A)[n:]);Hess=mu*(np.outer(dq,dq)*A[:n,:n]**2+np.outer(dq,dp)*A[:n,n:]**2+np.outer(dp,dq)*A[n:,:n]**2+np.outer(dp,dp)*A[n:,n:]**2);Hess[np.diag_indices(n)]+=4*d[:n]*(1+mu*np.diag(A)[:n])+4*d[n:]*(1+mu*np.diag(A)[n:]);return f,g,Hess,A,d
 for mu in (1.,.1,.01,.001,.0001,.00001,.000001,.0000001,.00000001):
  for iteration in range(80):
   f,g,h,A,d=evaluate(s,mu);delta=np.linalg.solve(h,-g);decrement=float(-g@delta)
   if decrement<1e-12:break
   step=1.
   for line in range(60):
    try:
     candidate=evaluate(s+step*delta,mu,False)
     if candidate<=f+1e-4*step*float(g@delta):break
    except np.linalg.LinAlgError:pass
    step*=.5
   else:raise ArithmeticError('Barrier line search failed')
   s+=step*delta
  else:raise ArithmeticError('Barrier Newton iteration limit')
  trace.append(dict(mu=mu,iterations=iteration,decrement=decrement,core_mean=float((d.sum()-2*n)/4)))
 f,g,h,A,d=evaluate(s,mu);Y=mu*A;dual=2*np.sqrt((1+np.diag(Y)[:n])*(1+np.diag(Y)[n:])).sum()-np.sum(Y*W.T);pure=(R*d)@R.T;noise=(V-pure)/2;mineig=float(np.linalg.eigvalsh(noise).min());assert mineig>0;gap=float(d.sum()-dual);assert gap>-1e-9
 t=np.tanh(s);z=[mp.mpf(float(q))**2 for q in t];coef=[mp.fprod(mp.sqrt(1-q) for q in z)];powers=[mp.mpf(1) for _ in z];traces=[]
 for j in range(1,257):
  powers=[a*b for a,b in zip(powers,z)];traces.append(mp.fsum(powers)/2);coef.append(mp.fsum(traces[q-1]*coef[j-q] for q in range(1,j+1))/j)
 radius=1/mp.sqrt(max(z));remainder=mp.fprod(mp.sqrt((1-q)/(1-q*radius)) for q in z)/radius**257;budgets=[]
 for K in (0,2,4,8,12,16,24,32,48,64,96,128):
  tail=min(mp.mpf(1),mp.fsum(coef[K//2+1:])+remainder);budgets.append(dict(budget=K,ideal_tv_upper=str(mp.sqrt(tail)),all_fock_dimension=math.comb(n+K,K)))
 row=dict(sector=sector,original_core_mean=float(np.sum(old['t']**2/(1-old['t']**2))),optimized_core_mean=float((d.sum()-2*n)/4),fixed_basis_core_mean_dual_lower=float((dual-2*n)/4),numerical_duality_gap_core_mean=gap/4,minimum_noise_eigenvalue=mineig,gradient_norm=float(np.linalg.norm(g)),seconds=time.perf_counter()-start,budgets=budgets);rows.append(row);save(f'row_{sector}.json',row);save(f'newton_{sector}.json',trace);np.savez_compressed(OUT/f'split_{sector}.npz',rotation=R,t=t,s=s,pure=pure,noise=noise,dual=Y,W=W);print({k:v for k,v in row.items() if k!='budgets'},flush=True);print('bounds',[(b['budget'],float(b['ideal_tv_upper'])) for b in budgets],flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
