"""Replay all fifteen frozen85 weak-spectrum B64 failures at the SAME prefix.
Also run the original uniforms through complete trajectories, with sixteen fixed
control rows. These intentionally reused domains are comparisons, not fresh iid
quality samples. All failures and raw conditional probabilities are retained.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,argparse,json,hashlib,ctypes as C,shutil,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);U64P=C.POINTER(C.c_uint64)
def ptr(x):return x.ctypes.data_as(DP)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 dll=ROOT/'results/iteration86_shared_gates_v3/shared.dll';rngdll=ROOT/'results/iteration84_rng_gates_v1/rng.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';base=ROOT/'results/iteration85_full_quality_v1';files=[Path(__file__),dll,rngdll,inp,*[base/f'exact64_1_{o}.npz' for o in range(0,512,64)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},experiment=0x85001002,sector=1,purpose=5,budget=64,control_rows=list(range(16)),threads=16))
 d=np.load(inp);v,x=d['v'],d['x'][1];oldbits=[];olderrors=[];oldprob=[]
 for o in range(0,512,64):
  z=np.load(base/f'exact64_1_{o}.npz');oldbits.append(z['clicks']);olderrors.append(z['errors']);oldprob.append(z['conditional'])
 oldbits=np.concatenate(oldbits);olderrors=np.concatenate(olderrors);oldprob=np.concatenate(oldprob);failed=np.flatnonzero(olderrors[:,0]);assert len(failed)==15;dirs=[Path(shutil.which('g++')).parent];runtime=os.add_dll_directory(str(dirs[0]));lib=C.CDLL(str(rngdll));rng=lib.rng84_stream;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64P,C.c_char_p,C.c_int];rng.restype=C.c_int;rows=[]
 with NativeStableClickSampler(v,x,dll,branch_budget=64,runtime_dirs=dirs) as model:
  fn=model.lib.tc86_shared_conditionals;fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
  for r in failed:
   length=int(olderrors[r,1]);bits=np.ascontiguousarray(oldbits[r,:length]);prob=np.full(length+1,np.nan);stats=np.zeros(6);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(model.handle,length,bits.ctypes.data_as(BP),64,ptr(prob),ptr(stats),err,len(err));seconds=time.perf_counter()-start;finite=np.isfinite(prob[:length])&np.isfinite(oldprob[r,:length]);row=dict(record=int(r),old_failure_mode=length,returncode=rc,error=err.value.decode(),completed=int(stats[5]),next_probability=float(prob[-1]) if np.isfinite(prob[-1]) else None,maximum_previous_conditional_difference=float(abs(prob[:length][finite]-oldprob[r,:length][finite]).max()) if finite.any() else None,seconds=seconds,condition=float(stats[3]),wide_calls=float(stats[4]));rows.append(row);np.savez_compressed(out/f'prefix_{r}.npz',bits=bits,conditional=prob,statistics=stats,old_conditional=oldprob[r,:length]);save(f'prefix_{r}.json',row);print('fixed prefix',r,'rc',rc,'next',row['next_probability'],flush=True)
  ids=np.r_[np.arange(16),failed];draws=np.empty((len(ids),len(v)));err=C.create_string_buffer(4096)
  for i,r in enumerate(ids):
   words=np.zeros(len(v),np.uint64);assert rng(0x85001002,1,5,int(r),0,len(v),words.ctypes.data_as(U64P),err,len(err))==0,err.value;draws[i]=(words>>11).astype(float)*2**-53
  sample=model.lib.tc86_shared_sample;sample.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];sample.restype=C.c_int;bits=np.zeros(draws.shape,np.uint8);prob=np.zeros(draws.shape);errors=np.zeros((len(ids),2),np.int32);stats=np.zeros((len(ids),6));start=time.perf_counter();rc=sample(model.handle,len(ids),16,64,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err));seconds=time.perf_counter()-start;np.savez_compressed(out/'trajectories.npz',record_ids=ids,draws=draws,clicks=bits,conditional=prob,errors=errors,statistics=stats);natural=dict(returncode=rc,error=err.value.decode(),records=len(ids),failed=int(np.count_nonzero(errors[:,0])),seconds=seconds,controls_bits_equal=bool(np.array_equal(bits[:16],oldbits[:16])),failed_row_completion={str(r):bool(errors[i,0]==0 and stats[i,5]==len(v)) for i,r in enumerate(ids) if r in failed});save('natural.json',natural)
  # Exact replay invariance with a different scheduling/chunk boundary.
  parts=[];partprob=[];parterrors=[];partstats=[]
  for lo,hi in [(0,3),(3,7)]:
   chunk=np.ascontiguousarray(draws[lo:hi]);b=np.zeros(chunk.shape,np.uint8);p=np.zeros(chunk.shape);e=np.zeros((hi-lo,2),np.int32);s=np.zeros((hi-lo,6));err=C.create_string_buffer(4096);sample(model.handle,hi-lo,1,64,ptr(chunk),b.ctypes.data_as(BP),ptr(p),e.ctypes.data_as(IP),ptr(s),err,len(err));parts.append(b);partprob.append(p);parterrors.append(e);partstats.append(s)
  invariant=all(np.array_equal(np.concatenate(chunks),reference[:7],equal_nan=True) for chunks,reference in [(parts,bits),(partprob,prob),(parterrors,errors),(partstats,stats)]);assert invariant
 save('summary.json',dict(passed=all(r['returncode']==0 for r in rows) and natural['failed']==0 and invariant,fixed_prefixes=rows,natural=natural,thread_chunk_invariance=invariant,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
