"""Feed the optimized pure core and full correlated displacement into coherent
photon-jump conditioning. Finite photon-factor closure remains approximate.
Small coherent/thermal analytic checks precede two full-input pilot batches.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,subprocess,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.stats import poisson,geom
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';src=ROOT/'native/gaussian_pnr_correlated84_v1.cpp';gcc=Path(shutil.which('g++'));core=ROOT/'results/iteration83_core_global_probe_v2';files=[Path(__file__),inp,ROOT/'src/gbslab/native_direct_click.py',*[core/f'split_{s}.npz' for s in range(2)],*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.cu','.hpp','.h')]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},full_records=16,retained=3,resplit=[0,1],seeds=[840200,840300],threads=16,analytic_records=4096))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'correlated.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP)
def sample(label,v,t,L,count,seed,resplit,mean=None,workers=16):
 n=len(t);t=np.ascontiguousarray(t,float);L=np.ascontiguousarray(L,float);mean=np.zeros(2*n) if mean is None else np.ascontiguousarray(mean,float);x=np.diag(np.r_[-t/(1+t),t/(1-t)]);start=time.perf_counter()
 with NativeDirectClickSampler(v,x,OUT/'correlated.dll',runtime_dirs=[gcc.parent]) as model:
  fn=model.lib.pnr84_correlated_sample;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;counts=np.zeros((count,len(v)),np.uint32);stats=np.zeros((count,12));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096);rc=fn(model.handle,ptr(t),ptr(L),count,seed,workers,3,resplit,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
 elapsed=time.perf_counter()-start;row=dict(label=label,success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,1]==1)),returncode=rc,error=err.value.decode(),seconds=elapsed,records=count,failed_rows=int(np.count_nonzero(errors[:,0])),completed_rows=int(np.count_nonzero(stats[:,1]==1)),mean_photons=float(counts.sum(1).mean()),mean_clicks=float((counts>0).sum(1).mean()),statistics_sum=stats.sum(0).tolist());np.savez_compressed(OUT/f'{label}.npz',counts=counts,statistics=stats,errors=errors);save(f'{label}.json',row);print(row,flush=True);return row,counts
checks=[]
coherent=np.array([np.sqrt(2)*.7,np.sqrt(2)*.4]);row,c=sample('coherent',np.eye(1),np.zeros(1),np.zeros((2,2)),4096,840050,0,coherent);assert row['success'];counts=c.sum(1).astype(int);h=np.bincount(counts);ks=float(abs(np.cumsum(h)/len(c)-poisson.cdf(np.arange(len(h)),.65)).max());checks.append(dict(case='coherent',mean=.65,empirical_mean=float(counts.mean()),count_cdf_distance=ks,dkw99_radius=float(np.sqrt(np.log(200)/(2*len(c))))));assert ks<.04
L=np.zeros((4,4));L[0,0]=L[1,0]=np.sqrt(.7);L[2,1]=L[3,1]=np.sqrt(.7);row,c=sample('correlated_thermal',np.eye(2),np.zeros(2),L,4096,840051,0);assert row['success'];counts=c.sum(1).astype(int);h=np.bincount(counts);ks=float(abs(np.cumsum(h)/len(c)-geom.cdf(np.arange(len(h))+1,1/2.4)).max());checks.append(dict(case='correlated_thermal',mean=1.4,empirical_mean=float(counts.mean()),count_cdf_distance=ks,dkw99_radius=float(np.sqrt(np.log(200)/(2*len(c)))),empirical_cross_covariance=float(np.cov(c.T,ddof=1)[0,1]),reference_cross_covariance=.49));assert ks<.04;save('analytic_checks.json',dict(passed=True,checks=checks,scope='analytic count CDF tests; not a deterministic floating-point certificate'))
data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);models=[]
for s in range(2):
 P=np.load(core/f'split_{s}.npz')['pure'];e,Q=np.linalg.eigh(P);q=Q[:,:n];R=np.column_stack([q,-omega@q]);t=(1-e[:n])/(1+e[:n]);diag=np.r_[(1-t)/(1+t),(1+t)/(1-t)];canonical=(R*diag)@R.T;state=np.eye(2*n)+2*x[s];noise=(state-canonical)/2;noise=(noise+noise.T)/2;L=np.linalg.cholesky(noise);rotated=np.ascontiguousarray(R.T@L);U=R[:n,:n]+1j*R[n:,:n];outv=v@U;checks=dict(sector=s,passive_error=float(abs(R.T@R-np.eye(2*n)).max()),symplectic_error=float(abs(R@omega@R.T-omega).max()),pure_reconstruction_error=float(abs(canonical-P).max()),minimum_noise_eigenvalue=float(np.linalg.eigvalsh(noise).min()),final_covariance_error=float(abs(canonical+2*L@L.T-state).max()),mean_total_photons=float((np.trace(state)-2*n)/4));assert max(checks[k] for k in ('passive_error','symplectic_error','pure_reconstruction_error','final_covariance_error'))<1e-10;assert checks['minimum_noise_eigenvalue']>0;save(f'input_{s}.json',checks);np.savez_compressed(OUT/f'input_{s}.npz',v=outv,t=t,noise_factor=rotated,rotation=R,pure=canonical);models.append((outv,t,rotated))
rows=[]
for resplit,seed in ((0,840200),(1,840300)):
 parts=[];sectors=[]
 for s,(outv,t,L) in enumerate(models):
  row,c=sample(f'full_{resplit}_{s}',outv,t,L,16,seed+s,resplit);parts.append(c);sectors.append(row)
 row=dict(resplit=resplit,records=16,success=all(r['success'] for r in sectors),seconds=sum(r['seconds'] for r in sectors),sectors=sectors)
 if row['success']:
  counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;row.update(mean_photons=float(counts.sum(1).mean()),mean_clicks=float(bits.sum(1).mean()));np.savez_compressed(OUT/f'combined_{resplit}.npz',counts=counts,clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1),photon_histogram=np.bincount(counts.sum(1).astype(int)))
 rows.append(row);save(f'row_{resplit}.json',row)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
