"""Capture x87/MXCSR state around sequential shared-base records after a failed
exact replay gate. Diagnostic source preserves the original sampler unchanged.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,argparse,json,hashlib,subprocess,ctypes as C,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
src=ROOT/'native/gaussian_click_shared86_diagnostic_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';raw=ROOT/'results/iteration86_shared_replay_v1/trajectories.npz';save('plan.json',dict(hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),src,inp,raw]},records=[0,1,2]))
command=json.loads((ROOT/'results/iteration86_shared_gates_v3/build.json').read_text())['command'];command[command.index(str(ROOT/'native/gaussian_click_shared86_v1.cpp'))]=str(src);command[command.index('-o')+1]=str(out/'diagnostic.dll');r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
d=np.load(inp);z=np.load(raw);draws=np.ascontiguousarray(z['draws'][:3]);env=np.zeros((3,8));prob=np.zeros(draws.shape);DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);err=C.create_string_buffer(4096)
with NativeStableClickSampler(d['v'],d['x'][1],out/'diagnostic.dll',branch_budget=64,runtime_dirs=[Path(shutil.which('g++')).parent]) as model:
 fn=model.lib.tc86_environment_probe;fn.argtypes=[C.c_void_p,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];rc=fn(model.handle,3,ptr(draws),ptr(env),ptr(prob),err,len(err))
np.savez_compressed(out/'probe.npz',environment=env,conditional=prob);save('summary.json',dict(returncode=rc,error=err.value.decode(),environment=env.tolist(),maximum_probability_difference=float(abs(prob-z['conditional'][:3]).max())));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}));print((out/'summary.json').read_text(),flush=True)
