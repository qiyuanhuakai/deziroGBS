"""Targeted two-binary128 signed-weight pilot on the same full lossy input.
Preserve old failed seeds and all new failures. Scalar arithmetic is checked
against 100-decimal mpmath. Gaussian covariance arithmetic remains long double;
this is neither an exact sampler nor a complete numerical error certificate.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,shutil,subprocess,ctypes as C,time,random
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));src=ROOT/'native/gaussian_click_weights83_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';refpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/input_0.npz';files=[Path(__file__),src,ROOT/'native/gaussian_click_stable82_v1.cpp',ROOT/'native/gaussian_click_branches_v1.cpp',inp,refpath];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},count=4,panels=[[0,8],[0,32],[1,32],[2,32],[0,64]],workers=16,prefix_budgets=[8,32,64],sampling_seed='830200+2*rep+sector; matches frozen timing83, not independent quality replicas'))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-lquadmath','-o',str(OUT/'weights.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
handle=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'weights.dll'));probe=lib.weight83_probe;probe.argtypes=[C.c_int,*([C.c_char_p]*6),C.c_int,C.c_char_p,C.c_int];probe.restype=C.c_int;mp.mp.dps=100;rng=random.Random(831900);cases=[(mp.mpf(2)**180,mp.mpf(1),-mp.mpf(2)**180,mp.mpf(0)),(mp.mpf(1),mp.mpf(0),mp.mpf(3),mp.mpf(0)),(mp.mpf(1),mp.mpf(2)**-113,mp.mpf(1),-mp.mpf(2)**-113)]
for _ in range(40):
 ah=mp.mpf(rng.choice([-7,-3,1,3,7]))*mp.mpf(2)**rng.randrange(-100,101);bh=mp.mpf(rng.choice([-7,-3,1,3,7]))*mp.mpf(2)**rng.randrange(-100,101);cases.append((ah,ah*mp.mpf(2)**-115,bh,-bh*mp.mpf(2)**-117))
checks=[]
for case,values in enumerate(cases):
 for op in range(4):
  inputs=[mp.nstr(v,100).encode() for v in values];hi=C.create_string_buffer(512);lo=C.create_string_buffer(512);err=C.create_string_buffer(2048);rc=probe(op,*inputs,hi,lo,512,err,len(err));assert rc==0,err.value;a,b=values[0]+values[1],values[2]+values[3];expected=[lambda:a+b,lambda:a-b,lambda:a*b,lambda:a/b][op]();actual=mp.mpf(hi.value.decode())+mp.mpf(lo.value.decode());error=abs(actual-expected);scale=max(abs(expected),mp.mpf('1e-100'));relative=error/scale;row=dict(case=case,operation=op,inputs=[s.decode() for s in inputs],high=hi.value.decode(),low=lo.value.decode(),absolute_error=str(error),relative_error=str(relative));checks.append(row);assert relative<mp.mpf('1e-62'),row
save('arithmetic_checks.json',dict(passed=True,checks=checks));print('scalar high-precision checks',len(checks),flush=True)
data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);prefix_rows=[]
for budget in (8,32,64):
 laws=[];prefix_failures=[]
 for s in range(2):
  with NativeStableClickSampler(v,x[s],OUT/'weights.dll',branch_budget=budget,runtime_dirs=[gcc.parent]) as model:
   fn=model.lib.tc83w_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;law=np.zeros(256);stats=np.zeros((256,6));errors=[]
   for mask in range(256):
    bits=np.array([(mask>>i)&1 for i in range(8)],np.uint8);out=np.zeros(1);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),budget,ptr(out),ptr(stats[mask]),err,len(err));law[mask]=out[0]
    if rc:errors.append(dict(mask=mask,error=err.value.decode()))
  np.savez_compressed(OUT/f'prefix_{budget}_{s}.npz',law=law,statistics=stats);save(f'prefix_{budget}_{s}.json',dict(errors=errors));prefix_failures.extend(dict(sector=s,**e) for e in errors)
  if not errors:assert abs(law.sum()-1)<1e-12
  laws.append(law)
 if prefix_failures:
  row=dict(budget=budget,success=False,errors=prefix_failures,first_eight_tv=None);prefix_rows.append(row);print('prefix failure',row,flush=True);continue
 law=np.zeros(256,np.longdouble)
 for i in range(256):
  for j in range(256):law[i|j]+=np.longdouble(laws[0][i])*np.longdouble(laws[1][j])
 reference=np.load(refpath)['two_common_sectors'];tv=float(abs(law-reference).sum()/2);row=dict(budget=budget,success=True,first_eight_tv=tv);prefix_rows.append(row);np.savez_compressed(OUT/f'prefix_combined_{budget}.npz',law=law);print('prefix',row,flush=True)
save('prefix_summary.json',dict(rows=prefix_rows));rows=[]
for rep,budget in ((0,8),(0,32),(1,32),(2,32),(0,64)):
 start=time.perf_counter();outputs=[];failures=0;sector_rows=[]
 for s in range(2):
  with NativeStableClickSampler(v,x[s],OUT/'weights.dll',branch_budget=budget,runtime_dirs=[gcc.parent]) as model:
   fn=model.lib.tc83w_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((4,m),np.uint8);prob=np.zeros((4,m));errors=np.zeros((4,2),np.int32);stats=np.zeros((4,6));err=C.create_string_buffer(4096);rc=fn(model.handle,4,830200+2*rep+s,16,budget,None,bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
  failed=int(np.count_nonzero(errors[:,0]));failures+=failed;outputs.append(bits);sr=dict(sector=s,returncode=rc,error=err.value.decode(),failed_rows=failed);sector_rows.append(sr);np.savez_compressed(OUT/f'sector_{rep}_{budget}_{s}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats);save(f'sector_{rep}_{budget}_{s}.json',sr)
 row=dict(rep=rep,budget=budget,records=4,success=failures==0 and all(q['returncode']==0 for q in sector_rows),failed_sector_rows=failures,seconds=time.perf_counter()-start,sectors=sector_rows)
 if row['success']:
  combined=np.bitwise_or(*outputs);row['mean_clicks']=float(combined.sum(1).mean());np.savez_compressed(OUT/f'combined_{rep}_{budget}.npz',clicks=combined)
 rows.append(row);save(f'row_{rep}_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,prefix=prefix_rows,arithmetic_checks=len(checks),scope='feasibility and scalar/prefix checks; timings exploratory, 4 records not a full-law accuracy estimate'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
