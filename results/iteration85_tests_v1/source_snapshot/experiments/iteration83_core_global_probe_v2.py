"""Guarded all-basis reference decomposition on the same 1152-output input.
Use the existing SDP oracle with the predeclared 1e-6 active covariance margin;
check the final canonical pairing state against the original input. Preserve v1
infeasible candidates. Theory tails and Schmidt resources are not sampler TV.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,time,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import guarded_resource,pure_pairing,resource_diagnostics
from gbslab.mps_probe import geometric_top_weights
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),inp,ROOT/'src/gbslab/gaussian_resource.py',ROOT/'src/gbslab/exact.py',ROOT/'src/gbslab/mps_probe.py',ROOT/'src/gbslab/banded_gaussian_mps.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},maximum_iterations=100,margin=1e-6,source='https://arxiv.org/html/2606.29739v1'));d=np.load(inp);xs,v=d['x'],d['v'];n=v.shape[1];omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);g=v[:576].conj().T@v[:576];gv,gu=np.linalg.eigh(g);root=(gu*np.sqrt(gv))@gu.conj().T;root=np.block([[root.real,-root.imag],[root.imag,root.real]]);mp.mp.dps=80;rows=[]
for sector,x in enumerate(xs):
 V=np.eye(len(x))+2*x;start=time.perf_counter();result=guarded_resource(V,margin=1e-6,max_iterations=100);save(f'candidate_{sector}.json',dict(accepted=result['numerically_accepted'],diagnostics=result['diagnostics'],optimizer_success=result['optimizer_success'],iterations=result['iterations'],evaluations=result['evaluations']));np.savez_compressed(OUT/f'candidate_{sector}.npz',pure=result['pure_covariance'],S=result['dual_multiplier'],L=result['dual_factor']);assert result['numerically_accepted'];B,P,roundtrip=pure_pairing(result['pure_covariance']);noise=(V-P)/2;noise=(noise+noise.T)/2;factor=np.linalg.cholesky(noise);diag=resource_diagnostics(V,P,result['dual_multiplier']);assert diag['minimum_remainder_eigenvalue']>0 and diag['purity_residual']<1e-9;t=np.linalg.svd(B,compute_uv=False);assert np.all(t<1);z=[mp.mpf(float(q))**2 for q in t];coef=[mp.fprod(mp.sqrt(1-q) for q in z)];powers=[mp.mpf(1)]*n;traces=[]
 for j in range(1,257):
  powers=[a*b for a,b in zip(powers,z)];traces.append(mp.fsum(powers)/2);coef.append(mp.fsum(traces[q-1]*coef[j-q] for q in range(1,j+1))/j)
 radius=1/mp.sqrt(max(z));remainder=mp.fprod(mp.sqrt((1-q)/(1-q*radius)) for q in z)/radius**257;budgets=[]
 for K in (8,12,16,24,32,48,64):budgets.append(dict(budget=K,conditional_ideal_tv_upper=str(mp.sqrt(min(mp.mpf(1),mp.fsum(coef[K//2+1:])+remainder)))))
 reduced=np.eye(2*n)+root@(P-np.eye(2*n))@root;L=np.linalg.cholesky((reduced+reduced.T)/2);nu=np.linalg.eigvalsh(1j*(L.T@omega@L))[n:];assert nu.min()>1;q=(nu-1)/(nu+1);weights=geometric_top_weights(q,4096);rankdistance=math.sqrt(1-math.fsum(weights));row=dict(sector=sector,seconds=time.perf_counter()-start,diagnostics=diag,pairing_roundtrip_error=roundtrip,minimum_noise_eigenvalue=float(np.linalg.eigvalsh(noise).min()),covariance_reconstruction_error=float(abs(P+2*factor@factor.T-V).max()),budgets=budgets,middle_rank4096_pure_state_distance_lower=rankdistance);rows.append(row);save(f'row_{sector}.json',row);save(f'history_{sector}.json',result['history']);np.savez_compressed(OUT/f'split_{sector}.npz',pure=P,S=result['dual_multiplier'],L=result['dual_factor'],pairing=B,t=t,noise=noise,noise_factor=factor,middle_nu=nu,middle_weights=weights);print({k:v for k,v in row.items() if k!='budgets'},flush=True);print('budgets',[(b['budget'],float(b['conditional_ideal_tv_upper'])) for b in budgets],flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
