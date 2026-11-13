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
 dll=ROOT/'results/iteration86_shared_gates_v3/shared.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';prior=ROOT/'results/iteration86_shared_replay_v1/trajectories.npz';d=np.load(inp);z=np.load(prior);dirs=[Path(shutil.which('g++')).parent];files=[Path(__file__),dll,inp,prior];save('plan.json',dict(hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scope='Diagnose failed exact thread/chunk gate without changing native code.'))
 rows=[]
 with NativeStableClickSampler(d['v'],d['x'][1],dll,branch_budget=64,runtime_dirs=dirs) as model:
  fn=model.lib.tc86_shared_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int]
  for case,(first,count,threads) in enumerate([(0,1,1),(0,3,1),(3,4,1),(0,7,4)]):
   draws=np.ascontiguousarray(z['draws'][first:first+count]);bits=np.zeros(draws.shape,np.uint8);prob=np.zeros(draws.shape);errors=np.zeros((count,2),np.int32);stats=np.zeros((count,6));err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(model.handle,count,threads,64,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err));data=dict(clicks=bits,conditional=prob,errors=errors,statistics=stats);np.savez_compressed(out/f'case_{case}.npz',**data);comparison={}
   for key,a in data.items():
    b=z[key][first:first+count];different=np.argwhere(a!=b);comparison[key]=dict(equal=bool(np.array_equal(a,b,equal_nan=True)),maximum_difference=float(np.nanmax(abs(a.astype(float)-b.astype(float)))),first_difference=different[0].tolist() if len(different) else None,old=float(b[tuple(different[0])]) if len(different) else None,new=float(a[tuple(different[0])]) if len(different) else None)
   row=dict(case=case,first=first,count=count,threads=threads,returncode=rc,error=err.value.decode(),seconds=time.perf_counter()-start,comparison=comparison);rows.append(row);save(f'case_{case}.json',row);print(json.dumps(row),flush=True)
 save('summary.json',dict(rows=rows));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()
