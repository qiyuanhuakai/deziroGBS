"""Quantify pure-core truncation budgets on the actual lossy 50-input sectors.
Reuse the frozen C++ Williamson/noise decomposition. The trace-distance bound
sqrt(discarded core mass) applies only if that coherently projected core is
sampled exactly after the displacement channel. It does not certify current
Gaussian-reset or factor samplers, nor independent addition of core/noise PNRs.
"""
from pathlib import Path
import os,argparse,json,hashlib,shutil,ctypes as C,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';dll=ROOT/'results/iteration75_resplit_pilot_v1/resplit.dll';data=np.load(inp);xs=data['x'];n=xs.shape[1]//2;save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),inp,dll,ROOT/'native/gaussian_resplit75_v1.hpp']},decimal_digits=80,maximum_core_photons=128,positive_tail_series_pairs=256))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.split75_probe;fn.argtypes=[C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;mp.mp.dps=80;rows=[]
for s,x in enumerate(xs):
 rotation=np.zeros_like(x);t=np.zeros(n);noise=np.zeros_like(x);res=np.zeros(1);err=C.create_string_buffer(4096);rc=fn(n,ptr(x),ptr(rotation),ptr(t),ptr(noise),ptr(res),err,len(err));row=dict(sector=s,returncode=rc,error=err.value.decode())
 if rc:
  save(f'row_{s}.json',row);rows.append(row);continue
 pure=rotation@np.diag(np.r_[(1-t)/(1+t),(1+t)/(1-t)])@rotation.T;reconstructed=pure+2*noise@noise.T;row.update(covariance_reconstruction_error=float(abs(reconstructed-(np.eye(2*n)+2*x)).max()),native_residual=float(res[0]),core_mean_photons=float(np.sum(t*t/(1-t*t))),total_mean_photons=float(np.trace(x)/2),displacement_mean_photons=float(np.sum(noise*noise)/2));assert row['covariance_reconstruction_error']<1e-11
 z=[mp.mpf(float(q))**2 for q in t];coeff=[mp.fprod(mp.sqrt(1-q) for q in z)];power=[mp.mpf(1) for _ in z];traces=[]
 for j in range(1,257):
  power=[a*b for a,b in zip(power,z)];traces.append(mp.fsum(power)/2);coeff.append(mp.fsum(traces[q-1]*coeff[j-q] for q in range(1,j+1))/j)
 R=1/mp.sqrt(max(z));chernoff=mp.fprod(mp.sqrt((1-q)/(1-q*R)) for q in z)/R**257
 budgets=[]
 for K in (0,2,4,8,12,16,24,32,48,64,96,128):
  tail=mp.fsum(coeff[K//2+1:]);upper=min(mp.mpf(1),tail+chernoff);budgets.append(dict(core_photon_budget=K,positive_tail_partial_sum=str(tail),chernoff_remainder_upper=str(chernoff),tail_mass_upper=str(upper),ideal_click_and_pnr_tv_upper=str(mp.sqrt(upper)),all_fock_states_up_to_budget=math.comb(n+K,K),interpretation='conditional on exact coherent projected-core sampling; not a current sampler guarantee'))
 row['budgets']=budgets;rows.append(row);save(f'row_{s}.json',row);np.savez_compressed(OUT/f'split_{s}.npz',rotation=rotation,t=t,noise_factor=noise,pure=pure,reconstructed=reconstructed);print({k:v for k,v in row.items() if k!='budgets'},flush=True);print('bounds',[(b['core_photon_budget'],float(b['ideal_click_and_pnr_tv_upper'])) for b in budgets],flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
