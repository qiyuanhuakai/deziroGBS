"""Build shared-base direct click; complete eight-bit laws versus independent
80-digit vacuum determinants and the frozen budgeted implementation. Full-system
replays are separate, and neither agreement nor numerical repair proves exact GBS.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,argparse,shutil,ctypes as C,subprocess,time
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int)
def ptr(a):return a.ctypes.data_as(DP)

def reference(v,x):
 O=np.block([[v.real,-v.imag],[v.imag,v.real]]);X=O@x@O.T;k=len(v);vac=[]
 for mask in range(1<<k):
  ids=[j for j in range(k) if mask>>j&1];ids+= [j+k for j in ids];A=np.eye(len(ids))+X[np.ix_(ids,ids)];vac.append(1/mp.sqrt(mp.det(mp.matrix(A.tolist()))) if ids else mp.mpf(1))
 law=[]
 for mask in range(1<<k):
  zeros=((1<<k)-1)^mask;sub=mask;p=mp.mpf(0)
  while True:
   p+=(-1)**sub.bit_count()*vac[zeros|sub]
   if not sub:break
   sub=(sub-1)&mask
  law.append(p)
 assert min(law)>=-mp.mpf('1e-65') and abs(sum(law)-1)<mp.mpf('1e-65')
 return np.array([float(p) for p in law])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 source=ROOT/'native/gaussian_click_shared86_v1.cpp';gcc=Path(shutil.which('g++'));old=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),old,inp,*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.hpp','.h')]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},seed=860020,budgets=[1,8,32,64,128]))
 command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/'shared.dll'),'-lquadmath'];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
 rng=np.random.default_rng(860020);v=np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0];t=np.array([.2,.4,.6]);pure=np.diag(np.r_[-t/(1+t),t/(1-t)]);d=np.load(inp);w=d['v'][:8];projection=np.block([[w.real,-w.imag],[w.imag,w.real]]);local=projection@d['x'][0]@projection.T;local=(local+local.T)/2;cases=[('thermal',v,np.diag([.2,.4,.6,.2,.4,.6])),('lossy_squeezed',v,pure*.65),('physical_first8',np.eye(8,dtype=complex),local)];mp.mp.dps=80;rows=[]
 for name,v,x in cases:
  exact=reference(v,x);np.savez_compressed(out/f'input_{name}.npz',v=v,x=x,target=exact);outputs={}
  for implementation,dll in [('old',old),('shared',out/'shared.dll')]:
   with NativeStableClickSampler(v,x,dll,branch_budget=128,runtime_dirs=[gcc.parent]) as model:
    fn=getattr(model.lib,'tc84e_prefix' if implementation=='old' else 'tc86_shared_prefix');fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
    for budget in [1,8,32,64,128]:
     law=[];statistics=[];errors=[];start=time.perf_counter()
     for label in range(256):
      bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);p=np.zeros(1);stats=np.zeros(8 if implementation=='old' else 6);err=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),budget,ptr(p),ptr(stats),err,len(err));law.append(p[0]);statistics.append(stats);errors.append(dict(label=label,returncode=rc,error=err.value.decode())) if rc else None
     law=np.array(law);outputs[implementation,budget]=law;row=dict(case=name,implementation=implementation,budget=budget,errors=errors,mass=float(law.sum()),minimum=float(law.min()),numerical_tv_to_exact=float(abs(law-exact).sum()/2),seconds=time.perf_counter()-start);np.savez_compressed(out/f'law_{name}_{implementation}_{budget}.npz',law=law,statistics=statistics);save(f'law_{name}_{implementation}_{budget}.json',row);rows.append(row);assert not errors and law.min()>=0 and abs(law.sum()-1)<1e-11,row
     if budget==128:assert row['numerical_tv_to_exact']<2e-11,row
   print(name,implementation,'complete',flush=True)
  for budget in [1,8,32,64,128]:
   difference=float(abs(outputs['old',budget]-outputs['shared',budget]).sum()/2);save(f'comparison_{name}_{budget}.json',dict(numerical_tv_between_implementations=difference));assert difference<2e-10,(name,budget,difference)
 # Vacuum and impossible outcomes; thread/chunk invariance has its own replay.
 with NativeStableClickSampler(np.eye(2,dtype=complex),np.zeros((4,4)),out/'shared.dll',runtime_dirs=[gcc.parent]) as model:
  fn=model.lib.tc86_shared_prefix;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
  for bits0,expected in [([0,0],1.),([1,0],0.),([0,1],0.)]:
   bits=np.array(bits0,np.uint8);p=np.zeros(1);stats=np.zeros(6);err=C.create_string_buffer(4096);assert fn(model.handle,2,bits.ctypes.data_as(BP),64,ptr(p),ptr(stats),err,len(err))==0,err.value;assert p[0]==expected
 save('summary.json',dict(passed=True,rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
