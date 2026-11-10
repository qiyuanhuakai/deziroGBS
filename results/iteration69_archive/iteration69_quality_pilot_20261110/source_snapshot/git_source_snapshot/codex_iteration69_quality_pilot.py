from pathlib import Path
import os,sys,ctypes as C,hashlib,json,shutil,time,traceback,math,itertools
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mpmath as mp
from gbslab.exact import ThresholdModel
from gbslab.frugal import frugal_sample
OUT=ROOT/'.git/iteration69_quality_pilot_20261110';OUT.mkdir(exist_ok=False)
def H(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(n,v):
 p=OUT/n;assert not p.exists();p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
source=ROOT/'.git/iteration69_reset_diagonal_v4.cpp';dll=ROOT/'.git/iteration69_reset_diagonal_v4_20261110/native.dll';official=ROOT/'results/iteration38_frugal_runtime/frugal_adapter.exe'
sources=[Path(__file__),source,ROOT/'src/gbslab/exact.py',ROOT/'src/gbslab/frugal.py'];hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}
for p in sources:
 dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
count=500000;radius=math.sqrt((16*math.log(2)+math.log(8/.01))/(2*count))
save('plan.json',dict(sources=hashes,dependencies={str(p.relative_to(ROOT)):H(p) for p in [dll,official,ROOT/'results/iteration38_frugal_runtime/build.json']},seed=690401,samples_per_method=count,cases=[dict(r=r,eta=e) for r,e in [(.7,.1),(.7,.3),(.7,.65),(1.1,.65)]],methods=[1,3],nominal_joint_failure_probability=.01,nominal_tv_radius=radius,scope='Threshold coarsening of the approximate PNR reset law against full 4-mode target and official Frugal K2/K3/K4. Statistical bands assume ideal IID samples; fixed seeded PRNG runs and 80-digit determinant values are not formal TV certificates. Existing Frugal build flags differ, so no matched-compiler speed claim.'))
gcc=Path(shutil.which('g++')).parent;runtime=os.add_dll_directory(str(gcc));lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.gr_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,UP,DP,C.c_char_p,C.c_int]
def full_law(cov):
 mp.mp.dps=80;m=len(cov)//2;vac={0:mp.mpf(1)};matrix=mp.matrix([[mp.mpf(float(x)) for x in row] for row in cov])
 for mask in range(1,1<<m):
  ids=[i for i in range(m) if mask>>i&1];ids+= [i+m for i in ids.copy()];q=mp.matrix([[(matrix[i,j]+int(i==j))/2 for j in ids] for i in ids]);vac[mask]=1/mp.sqrt(mp.det(q))
 law=[]
 for clicks in range(1<<m):
  zeros=((1<<m)-1)^clicks;value=mp.mpf(0);sub=clicks
  while True:
   value+=(-1)**sub.bit_count()*vac[zeros|sub]
   if sub==0:break
   sub=(sub-1)&clicks
  assert value>=0;law.append(value)
 assert abs(sum(law)-1)<mp.mpf('1e-70');return np.array([float(x) for x in law]),[str(x) for x in law]
rng=np.random.default_rng(690401);rows=[]
for case,(r,eta) in enumerate([(.7,.1),(.7,.3),(.7,.65),(1.1,.65)]):
 q,rr=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)));v=np.ascontiguousarray(q*(np.diag(rr)/abs(np.diag(rr)))[None,:]);number=np.sinh(r)**2;pair=np.sinh(r)*np.cosh(r);x0=np.ascontiguousarray(np.diag([eta*(number+pair)]*2+[eta*(number-pair)]*2));o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.eye(8)+2*o@x0@o.T;cov=(cov+cov.T)/2;target,decimal=full_law(cov);reference=ThresholdModel(cov).full_distribution();assert np.max(abs(reference-target))<1e-12
 np.savez_compressed(OUT/f'input_{case}.npz',v=v,x0=x0,covariance=cov,target=target);save(f'target_{case}.json',dict(probabilities=decimal,maximum_float_reference_error=float(np.max(abs(reference-target)))))
 error=C.create_string_buffer(4096);handle=lib.gr_create(4,2,v.ctypes.data_as(DP),x0.ctypes.data_as(DP),error,4096);assert handle,error.value
 for method in (1,3):
  counts=np.zeros((count,4),np.uint32);stats=np.zeros(7);start=time.perf_counter();rc=lib.gr_sample(handle,count,690500+case,method,counts.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,4096);seconds=time.perf_counter()-start;row=dict(case=case,r=r,eta=eta,method=method,returncode=rc,error=error.value.decode(),seconds=seconds)
  np.savez_compressed(OUT/f'case_{case}_method_{method}.npz',counts=counts,stats=stats)
  if rc==0:
   mask=np.sum((counts>0)*(1<<np.arange(4)),axis=1);hist=np.bincount(mask,minlength=16);law=hist/count;tv=float(.5*np.sum(abs(law-target)));row.update(empirical_threshold_tv=tv,nominal_tv_interval=[max(0,tv-radius),min(1,tv+radius)],mean_photons=float(counts.sum(axis=1).mean()),target_mean_photons=float(2*eta*number),maximum_photons=int(counts.sum(axis=1).max()),proposals_per_photon=float(stats[1]/stats[0]),vacuum_frequency=float(hist[0]/count));np.savez_compressed(OUT/f'histogram_{case}_{method}.npz',histogram=hist,target=target)
  rows.append(row);save(f'case_{case}_method_{method}.json',row);print(row,flush=True)
 lib.gr_destroy(handle)
 for cutoff in (2,3,4):
  try:
   result=frugal_sample(cov,cutoff,sample_count=count,seed=690600+case,executable=official,threads=1);save(f'frugal_{case}_{cutoff}.json',result);tv=float(.5*np.sum(abs(np.array(result['distribution'])-target)));row=dict(case=case,r=r,eta=eta,method='frugal',cutoff=cutoff,induced_threshold_tv=tv,timing=result.get('timings_seconds'),status='completed')
  except Exception:row=dict(case=case,method='frugal',cutoff=cutoff,status='failed',traceback=traceback.format_exc())
  rows.append(row);save(f'frugal_{case}_{cutoff}_summary.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,nominal_radius=radius,samples_per_native_method=count,source_hashes=hashes))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
