"""Three serial rotations of Weight83 versus exact integer-dot accumulation.
Four complete 1152-bit records, B32/B64, common domain-separated uniforms.
Include loss QR, model setup, random generation, both spectra, OR and destruction.
Preload DLLs; exclude build, data I/O and result serialization. No heavy overlap.
These are timing/replay panels, not new independent accuracy datasets.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
paths=dict(old=ROOT/'results/iteration83_weights_pilot_v1/weights.dll',exact=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll',loss=ROOT/'results/iteration83_full_quality_v1/loss.dll',rng=ROOT/'results/iteration84_rng_gates_v1/rng.dll');inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),inp,*paths.values(),ROOT/'native/gaussian_click_exactsum84_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'src/gbslab/native_stable_click.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},methods=['old','exact'],budgets=[32,64],records=4,threads=4,repetitions=3,experiments=[0x84003001+i for i in range(3)],purpose=5));dirs=[Path(shutil.which('g++')).parent];runtime=[os.add_dll_directory(str(p)) for p in dirs];libs={k:C.CDLL(str(p)) for k,p in paths.items()};DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);UP=C.POINTER(C.c_uint64);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);prepare=libs['loss'].loss83_prepare;prepare.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];prepare.restype=C.c_int;rng=libs['rng'].rng84_stream;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,UP,C.c_char_p,C.c_int];rng.restype=C.c_int;data=np.load(inp);T,x0=data['T'],data['input_x'];m,n=T.shape;rows=[]
for rep in range(3):
 panels=[('old',32),('exact',32),('old',64),('exact',64)];panels=panels[rep:]+panels[:rep]
 for method,budget in panels:
  parts=[];raw=[];sectors=[];start=time.perf_counter();v=np.zeros_like(T);x=np.zeros_like(x0);meta=np.zeros(5);err=C.create_string_buffer(4096);rc=prepare(m,n,2,ptr(T),ptr(x0),ptr(v),ptr(x),ptr(meta),err,len(err));assert rc==0,err.value
  for s in range(2):
   words=np.zeros((4,m),np.uint64)
   for row in range(4):assert rng(0x84003001+rep,s,5,row,0,m,words[row].ctypes.data_as(UP),err,len(err))==0,err.value
   draws=np.ascontiguousarray((words>>11).astype(float)*2**-53)
   with NativeStableClickSampler(v,x[s],paths[method],branch_budget=budget,runtime_dirs=dirs) as model:
    fn=getattr(model.lib,'tc83w_sample' if method=='old' else 'tc84e_sample');fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((4,m),np.uint8);prob=np.zeros((4,m));stats=np.zeros((4,6 if method=='old' else 8));errors=np.zeros((4,2),np.int32);rc=fn(model.handle,4,0,4,budget,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
   sectors.append(dict(sector=s,returncode=rc,error=err.value.decode(),success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,5]==m))));parts.append(bits);raw.append(dict(clicks=bits,conditional=prob,statistics=stats,errors=errors))
  bits=np.bitwise_or(*parts);elapsed=time.perf_counter()-start;row=dict(rep=rep,method=method,budget=budget,records=4,seconds=elapsed,success=all(s['success'] for s in sectors),sectors=sectors);rows.append(row);save(f'row_{rep}_{method}_{budget}.json',row)
  for s,r in enumerate(raw):np.savez_compressed(OUT/f'sector_{rep}_{method}_{budget}_{s}.npz',**r)
  if row['success']:np.savez_compressed(OUT/f'combined_{rep}_{method}_{budget}.npz',clicks=bits)
  print({k:v for k,v in row.items() if k!='sectors'},flush=True)
comparisons=[]
for budget in (32,64):
 for rep in range(3):
  pair=[r for r in rows if r['budget']==budget and r['rep']==rep]
  if not all(r['success'] for r in pair):comparisons.append(dict(budget=budget,rep=rep,success=False));continue
  a=np.load(OUT/f'combined_{rep}_old_{budget}.npz')['clicks'];b=np.load(OUT/f'combined_{rep}_exact_{budget}.npz')['clicks'];diff=0.
  for s in range(2):
   a0=np.load(OUT/f'sector_{rep}_old_{budget}_{s}.npz')['conditional'];b0=np.load(OUT/f'sector_{rep}_exact_{budget}_{s}.npz')['conditional'];diff=max(diff,float(abs(a0-b0).max()))
  comparisons.append(dict(budget=budget,rep=rep,success=True,bits_equal=bool(np.array_equal(a,b)),maximum_conditional_difference=diff))
save('summary.json',dict(completed=True,rows=rows,paired_replay=comparisons,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
