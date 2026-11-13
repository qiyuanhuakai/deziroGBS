"""First frozen survival failure per affected small-PNR configuration.
Replay retains the exact original output. Independent100-digit real-quadrature
Wick expansion evaluates the actual exported eigensystem without probability
clipping. Examples diagnose their own failures, not every failed record.
"""
import argparse,subprocess,functools,itertools,mpmath as mp
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'native/gaussian_pnr_diagnostic89_v1.cpp';dll=out/'diagnostic.dll';old=ROOT/'results/iteration88_small_pnr_quality_v1';files=[Path(__file__),source,ROOT/'experiments/iteration89_common_v1.py',old/'manifest.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},digits=100,selection='First failed record in each configuration with at least one frozen failure'));gcc=runtime_dirs()[0]/'g++.exe';cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;mp.mp.dps=100;rows=[]
def independent(trace,s):
 vals=list(map(mp.mpf,trace['values']));center=list(map(mp.mpf,trace['mean']));F=[[mp.mpf(x) for x in row] for row in trace['forms']];d=len(vals);c=len(F[0]);mu=[mp.fsum(F[i][j]*center[i]/(1+s*vals[i]) for i in range(d)) for j in range(c)];cov=[[mp.fsum(F[i][j]*F[i][k]*vals[i]/(1+s*vals[i]) for i in range(d)) for k in range(c)] for j in range(c)]
 @functools.lru_cache(None)
 def moment(ids):
  if not ids:return mp.mpf(1)
  i,*rest=ids;result=mu[i]*moment(tuple(rest))
  for j,k in enumerate(rest):result+=cov[i][k]*moment(tuple(rest[:j]+rest[j+1:]))
  return result
 value=mp.fsum(moment(tuple(sorted(i for j,b in enumerate(bits) for i in (2*j+b,2*j+b)))) for bits in itertools.product((0,1),repeat=c//2))/2**(c//2);logp=-mp.fsum(mp.log1p(s*vals[i])+s*center[i]**2/(1+s*vals[i]) for i in range(d))/2;return mp.exp(logp)*value
for row in read(old/'summary.json')['rows']:
 if not row['failed']:continue
 j,R=row['case_index'],row['retained'];selected=None
 for offset in range(0,65536,1024):
  raw=np.load(old/f'raw_{j}_{R}_{offset}.npz');bad=np.flatnonzero(~raw['valid'])
  if len(bad):selected=(offset+int(bad[0]),{k:raw[k][bad[0]].copy() for k in raw.files});break
 record,frozen=selected;z=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');n=len(z['t']);d=2*n;m=len(z['v']);counts=np.zeros(m,np.uint32);cov=np.zeros((d,d));center=np.zeros(d);factors=np.zeros(8,np.int32);stats=np.zeros(7);buffer=C.create_string_buffer(262144);err=C.create_string_buffer(4096)
 with NativeDirectClickSampler(z['v'],z['x'],dll,runtime_dirs=runtime_dirs()) as model:
  fn=model.lib.pnr89_flow_diagnostic;fn.argtypes=[C.c_void_p,DP,DP,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,UP,DP,DP,IP,DP,C.c_char_p,C.c_int,C.c_char_p,C.c_int];rc=fn(model.handle,ptr(z['t']),ptr(z['L']),row['experiment'],0,record,R,counts.ctypes.data_as(UP),ptr(cov),ptr(center),factors.ctypes.data_as(IP),ptr(stats),buffer,len(buffer),err,len(err));assert rc==0,err.value
 trace=json.loads(buffer.value.decode());save(out,f'trace_{j}_{R}.json',trace);np.savez_compressed(out/f'state_{j}_{R}.npz',counts=counts,covariance=cov,mean=center,factors=factors,statistics=stats);same=np.array_equal(counts,frozen['counts']) and np.array_equal(stats[:2],frozen['errors']);assert same,(j,R,record,stats,frozen['errors']);normalization=independent(trace,mp.mpf(0));evaluations=[]
 for point in trace['points']:
  s,native=mp.mpf(point[0]),mp.mpf(point[1]);p=independent(trace,s)/(normalization if int(mp.mpf(point[5])) else 1);evaluations.append(dict(s=point[0],native=point[1],independent=mp.nstr(p,90),difference=mp.nstr(native-p,60),independent_in_range=bool(0<=p<=1) if int(mp.mpf(point[5])) else None))
 omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);minimum=float(np.linalg.eigvalsh(cov+np.eye(d)/2+1j*omega/2).min());result=dict(case=j,retained=R,record=record,identical_frozen_output=same,error=err.value.decode(),completed_photons=int(stats[1]),factors=int(stats[2]),base_uncertainty_minimum=minimum,evaluations=evaluations);rows.append(result);save(out,f'row_{j}_{R}.json',result);print('PNR failure diagnostic',j,R,record,evaluations[-1],flush=True)
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
