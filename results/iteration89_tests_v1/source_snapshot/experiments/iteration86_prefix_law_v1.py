"""Enumerate the actual full1152-mode sampler's first eight-bit B64 marginal,
both spectra and their OR. Compare the same two-spectrum numerical target used
by Frugal. This first-prefix law does not describe later/full-output accuracy.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,argparse,json,hashlib,ctypes as C,shutil,time
import numpy as np
crt=C.CDLL('ucrtbase');nearest=crt.fesetround;nearest.argtypes=[C.c_int];nearest.restype=C.c_int
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
paths=dict(old=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll',shared=ROOT/'results/iteration86_shared_gates_v4/shared.dll');inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';ref=ROOT/'results/iteration85_joint_quality_v1/reference_0.npz';frugal=ROOT/'results/iteration85_joint_quality_v1/summary.json';files=[Path(__file__),inp,ref,frugal,*paths.values()];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},budget=64,full_model=True,modes=1152,prefix=8,all_labels=256,sectors=[0,1]));d=np.load(inp);target=np.load(ref)['target'];BP=C.POINTER(C.c_uint8);DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);dirs=[Path(shutil.which('g++')).parent];rows=[];laws={}
for method,dll in paths.items():
 parts=[];start=time.perf_counter()
 for sector in range(2):
  values=np.zeros(256);statistics=np.zeros((256,8 if method=='old' else 6))
  with NativeStableClickSampler(d['v'],d['x'][sector],dll,branch_budget=64,runtime_dirs=dirs) as model:
   fn=getattr(model.lib,'tc84e_prefix' if method=='old' else 'tc86_shared_prefix');fn.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
   for label in range(256):
    bits=np.array([(label>>j)&1 for j in range(8)],np.uint8);prob=np.zeros(1);stats=np.zeros(statistics.shape[1]);error=C.create_string_buffer(4096);rc=fn(model.handle,8,bits.ctypes.data_as(BP),64,ptr(prob),ptr(stats),error,len(error));assert nearest(0)==0;assert rc==0,(method,sector,label,error.value);values[label]=prob[0];statistics[label]=stats
  assert values.min()>=0 and abs(values.sum()-1)<1e-11;parts.append(values);np.savez_compressed(out/f'{method}_{sector}.npz',law=values,statistics=statistics)
 labels=(np.arange(256)[:,None]|np.arange(256)[None,:]).ravel();law=np.bincount(labels,weights=(parts[0][:,None]*parts[1][None,:]).ravel(),minlength=256);np.savez_compressed(out/f'{method}_combined.npz',law=law,target=target);laws[method]=law;row=dict(method=method,numerical_tv=float(abs(law-target).sum()/2),mass=float(law.sum()),seconds=time.perf_counter()-start);rows.append(row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,numerical_tv_old_new=float(abs(laws['old']-laws['shared']).sum()/2),frugal=[r for r in json.loads(frugal.read_text())['frugal_standalone_marginals'] if r['group']==0],scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))
