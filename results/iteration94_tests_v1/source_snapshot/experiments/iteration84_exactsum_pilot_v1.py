"""Exact integer-dot candidate: dyadic scalar audit, first-eight law and
same-seed full 1152-output replay against frozen Weight83. Exploratory times.
No changed branch budget, normalization guard or sampling law in ideal arithmetic.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,subprocess,ctypes as C,time,random
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';src=ROOT/'native/gaussian_click_exactsum84_v1.cpp';gcc=Path(shutil.which('g++'));old=ROOT/'results/iteration83_weights_pilot_v1';files=[Path(__file__),inp,ROOT/'src/gbslab/native_stable_click.py',*[old/f'sector_{r}_{b}_{s}.npz' for r,b in ((0,8),(0,32),(1,32),(2,32),(0,64)) for s in range(2)],*[old/f'prefix_{b}_{s}.npz' for b in (8,32,64) for s in range(2)],*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.cu','.hpp','.h')]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=4,panels=[[0,8],[0,32],[1,32],[2,32],[0,64]],threads=4,scalar_random_cases=240,seed=840100))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-lquadmath','-o',str(OUT/'exactsum.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'exactsum.dll'));probe=lib.sum84_probe;probe.argtypes=[C.c_int,C.c_char_p,C.c_char_p,C.c_int,C.c_char_p,C.c_int];probe.restype=C.c_int;mp.mp.dps=500;rng=random.Random(840100);cases=[[],[(mp.mpf(2)**200,mp.mpf(1),mp.mpf(1)),(-mp.mpf(2)**200,mp.mpf(0),mp.mpf(1))],[(mp.mpf(2)**500,mp.mpf(0),mp.mpf(1)),(mp.mpf(1),mp.mpf(0),mp.mpf(1))]]
for j in range(240):
 rows=[]
 for k in range((1,3,8,32,64,128,256)[j%7]):
  e=rng.randrange(-170,71);hi=mp.mpf(rng.getrandbits(112)|1)*mp.mpf(2)**e*rng.choice([-1,1]);lo=mp.mpf(rng.getrandbits(112)|1)*mp.mpf(2)**(e-114)*rng.choice([-1,1]);x=mp.mpf(rng.getrandbits(63)|1)*mp.mpf(2)**rng.randrange(-100,40)*rng.choice([-1,1]);rows.append((hi,lo,x))
 if j%3==0:rows=[(a,b,c) for a,b,c in rows]+[(-a,-b,c) for a,b,c in rows] if len(rows)<=128 else rows
 cases.append(rows)
checks=[]
for index,rows in enumerate(cases):
 text=' '.join(mp.nstr(v,500) for row in rows for v in row);out=C.create_string_buffer(2048);err=C.create_string_buffer(4096);rc=probe(len(rows),text.encode(),out,len(out),err,len(err));assert rc==0,err.value;hi,lo,fallback=out.value.decode().split();expected=mp.fsum((a+b)*c for a,b,c in rows);actual=mp.mpf(hi)+mp.mpf(lo);scale=mp.fsum((abs(a)+abs(b))*abs(c) for a,b,c in rows);bound=scale*mp.mpf(2)**-210;error=abs(expected-actual);check=dict(case=index,count=len(rows),inputs=[[str(v) for v in row] for row in rows],expected=str(expected),high=hi,low=lo,absolute_error=str(error),scale=str(scale),absolute_tolerance=str(bound),fallback=int(fallback),passed=bool(error<=bound));checks.append(check)
 if not check['passed']:save('arithmetic_failure.json',check);raise AssertionError(check)
save('arithmetic_checks.json',dict(passed=True,checks=checks));print('scalar audits passed',len(checks),'fallbacks',sum(r['fallback'] for r in checks),flush=True)
data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);prefix=[]
for b in (8,32,64):
 for s in range(2):
  with NativeStableClickSampler(v,x[s],OUT/'exactsum.dll',branch_budget=b,runtime_dirs=[gcc.parent]) as model:
   fn=model.lib.tc84e_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;law=np.zeros(256);stats=np.zeros((256,8));failures=[]
   for mask in range(256):
    bits=np.array([(mask>>i)&1 for i in range(8)],np.uint8);out=np.zeros(1);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),b,ptr(out),ptr(stats[mask]),err,len(err));law[mask]=out[0]
    if rc:failures.append(dict(mask=mask,error=err.value.decode()))
  reference=np.load(old/f'prefix_{b}_{s}.npz')['law'];np.savez_compressed(OUT/f'prefix_{b}_{s}.npz',law=law,statistics=stats);row=dict(budget=b,sector=s,success=not failures,errors=failures,law_exactly_equal=bool(np.array_equal(law,reference)),law_difference_tv=float(abs(law-reference).sum()/2),fallback_dots=float(stats[:,7].sum()));prefix.append(row);save(f'prefix_{b}_{s}.json',row);print('prefix',row,flush=True)
save('prefix_summary.json',dict(rows=prefix));rows=[]
for rep,b in ((0,8),(0,32),(1,32),(2,32),(0,64)):
 for s in range(2):
  start=time.perf_counter()
  with NativeStableClickSampler(v,x[s],OUT/'exactsum.dll',branch_budget=b,runtime_dirs=[gcc.parent]) as model:
   fn=model.lib.tc84e_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((4,m),np.uint8);prob=np.zeros((4,m));errors=np.zeros((4,2),np.int32);stats=np.zeros((4,8));err=C.create_string_buffer(4096);rc=fn(model.handle,4,830200+2*rep+s,4,b,None,bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
  wall=time.perf_counter()-start;ref=np.load(old/f'sector_{rep}_{b}_{s}.npz');np.savez_compressed(OUT/f'sector_{rep}_{b}_{s}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats);row=dict(rep=rep,budget=b,sector=s,success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,5]==m)),returncode=rc,error=err.value.decode(),seconds=wall,bits_equal=bool(np.array_equal(bits,ref['clicks'])),probabilities_equal=bool(np.array_equal(prob,ref['conditional'])),maximum_probability_difference=float(abs(prob-ref['conditional']).max()),original_statistics_equal=bool(np.array_equal(stats[:,:6],ref['statistics'])),exact_dots=float(stats[:,6].sum()),fallback_dots=float(stats[:,7].sum()));rows.append(row);save(f'row_{rep}_{b}_{s}.json',row);print('replay',row,flush=True)
save('summary.json',dict(rows=rows,prefix=prefix,scalar_cases=len(checks),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
