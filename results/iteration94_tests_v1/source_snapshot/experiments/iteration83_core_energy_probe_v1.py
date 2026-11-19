"""Small active-space independent reference for lower-energy pure/noise splits.
Given V=S diag(nu,nu) S^T, rotate each Williamson pair to diagonalize its column
Gram matrix and minimize a exp(-2u)+b exp(2u) for |u|<=log(nu)/2. This is optimal
only within this fixed Williamson pair-product family, not globally optimal.
A tiny interior margin gives strictly positive noise without eigenvalue clipping.
Reference linear algebra is 100-dimensional only; production kernel is pending.
"""
from pathlib import Path
import os,argparse,json,hashlib,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import schur
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);xs=data['x'];n=xs.shape[1]//2;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),inp]},interior_fraction=1e-8,decimal_digits=80));mp.mp.dps=80;rows=[]
for sector,x in enumerate(xs):
 V=np.eye(2*n)+2*x;e,U=np.linalg.eigh(V);root=(U*np.sqrt(e))@U.T;A=root@omega@root;tri,O=schur((A-A.T)/2,output='real');signed=np.array([tri[2*j,2*j+1] for j in range(n)]);nu=abs(signed);assert np.all(nu>1);S=root@np.c_[O[:,::2],O[:,1::2]*np.sign(signed)]/np.sqrt(np.r_[nu,nu]);assert np.max(abs(S@omega@S.T-omega))<1e-11
 rotated=S.copy();parameters=[]
 for j in range(n):
  pair=S[:,[j,n+j]];gram=pair.T@pair;ev,R=np.linalg.eigh(gram)
  if np.linalg.det(R)<0:R[:,0]*=-1
  pair=pair@R;a,b=ev;limit=.5*np.log(nu[j])*(1-1e-8);u=max(-limit,min(limit,.25*np.log(a/b)));rotated[:,j]=pair[:,0]*np.exp(-u);rotated[:,n+j]=pair[:,1]*np.exp(u);parameters.append(u)
 pure=rotated@rotated.T;noise=(V-pure)/2;noise=(noise+noise.T)/2;ev=np.linalg.eigvalsh(pure);noise_eigen=np.linalg.eigvalsh(noise);purity=np.max(abs(pure@omega@pure-omega));assert noise_eigen.min()>0 and purity<1e-10;coremean=float((np.trace(pure)-2*n)/4);original=float((np.sum(S*S)-2*n)/4);t=np.tanh(.5*np.log(ev[n:]));z=[mp.mpf(float(q))**2 for q in t];coef=[mp.fprod(mp.sqrt(1-q) for q in z)];powers=[mp.mpf(1) for _ in z];traces=[]
 for j in range(1,257):
  powers=[a*b for a,b in zip(powers,z)];traces.append(mp.fsum(powers)/2);coef.append(mp.fsum(traces[q-1]*coef[j-q] for q in range(1,j+1))/j)
 R=1/mp.sqrt(max(z));remainder=mp.fprod(mp.sqrt((1-q)/(1-q*R)) for q in z)/R**257;budgets=[]
 for K in (0,2,4,8,12,16,24,32,48,64,96,128):
  tail=min(mp.mpf(1),mp.fsum(coef[K//2+1:])+remainder);budgets.append(dict(budget=K,ideal_tv_upper=str(mp.sqrt(tail)),all_fock_dimension=math.comb(n+K,K)))
 row=dict(sector=sector,original_core_mean=original,optimized_core_mean=coremean,displacement_mean=float(np.trace(noise)/2),minimum_noise_eigenvalue=float(noise_eigen.min()),purity_residual=float(purity),williamson_reconstruction_error=float(abs((S*np.r_[nu,nu])@S.T-V).max()),budgets=budgets);rows.append(row);save(f'row_{sector}.json',row);np.savez_compressed(OUT/f'split_{sector}.npz',pure=pure,noise=noise,williamson=S,nu=nu,optimized_symplectic=rotated,parameters=parameters,t=t);print({k:v for k,v in row.items() if k!='budgets'},flush=True);print('bounds',[(b['budget'],float(b['ideal_tv_upper'])) for b in budgets],flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
