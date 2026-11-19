"""Serial complete64-record paired PNR timings on identical explicit domains.
Includes model/tree creation, sampling, spectral sum and destruction. Excludes
input loading, allocations, DLL loading and artifact I/O. Not quality samples.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,hashlib,ctypes as C,shutil,time,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
old=ROOT/'results/iteration85_tree_pilot_v1/tree.dll';new=ROOT/'results/iteration87_blocktree_pilot_v1/blocktree.dll';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),old,new,*[core/f'input_{s}.npz' for s in range(2)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=64,experiment=0x87002002,blocks=[0,16,32,64],threads=16,repetitions=3,intentional_replay=True))
DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda a:a.ctypes.data_as(DP);dirs=[Path(shutil.which('g++')).parent];runtime=os.add_dll_directory(str(dirs[0]));libs=[C.CDLL(str(old)),C.CDLL(str(new))];inputs=[]
for sector in range(2):
 q=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(q['t']);inputs.append(dict(v=q['v'],t=t,L=np.ascontiguousarray(q['noise_factor']),mean=np.zeros(2*len(t)),x=np.diag(np.r_[-t/(1+t),t/(1-t)])))
rows=[];reference=None
for rep in range(3):
 order=[0,16,32,64];order=order[rep:]+order[:rep]
 for block in order:
  arrays=[dict(counts=np.zeros((64,len(q['v'])),np.uint32),statistics=np.zeros((64,20 if block else 18)),errors=np.zeros((64,2),np.int32)) for q in inputs];metadata=[];start=time.perf_counter()
  for sector,q in enumerate(inputs):
   with NativeDirectClickSampler(q['v'],q['x'],new if block else old,runtime_dirs=dirs) as model:
    suffix='87' if block else '85';create=getattr(model.lib,f'pnr{suffix}_tree_create');create.argtypes=[C.c_void_p,*([C.c_int] if block else []),C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=getattr(model.lib,f'pnr{suffix}_tree_destroy');destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);tree=create(model.handle,*([block] if block else []),err,len(err));assert tree,err.value;info=getattr(model.lib,f'pnr{suffix}_tree_info');info.argtypes=[C.c_void_p,DP];meta=np.zeros(7 if block else 4);assert info(tree,ptr(meta))==0;metadata.append(meta.tolist());r=arrays[sector];fn=getattr(model.lib,f'pnr{suffix}_correlated_sample');fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int]
    try:rc=fn(model.handle,tree,ptr(q['t']),ptr(q['L']),64,0x87002002,sector,0,16,3,0,ptr(q['mean']),None,r['counts'].ctypes.data_as(UP),ptr(r['statistics']),r['errors'].ctypes.data_as(IP),err,len(err))
    finally:destroy(tree)
    assert rc==0 and not r['errors'].any(),err.value
  combined=arrays[0]['counts'].astype(np.uint64)+arrays[1]['counts'];seconds=time.perf_counter()-start
  if reference is None:reference=arrays
  identical=all(np.array_equal(reference[s]['counts'],arrays[s]['counts']) and np.array_equal(reference[s]['statistics'][:,:18],arrays[s]['statistics'][:,:18]) and np.array_equal(reference[s]['errors'],arrays[s]['errors']) for s in range(2));row=dict(repetition=rep,block=block,records=64,seconds=seconds,identical=identical,tree_metadata=metadata);rows.append(row);save(f'row_{rep}_{block}.json',row)
  for sector,r in enumerate(arrays):np.savez_compressed(out/f'raw_{rep}_{block}_{sector}.npz',**r)
  assert identical,row;print('formal PNR',rep,block,seconds,flush=True)
medians={str(block):statistics.median(r['seconds'] for r in rows if r['block']==block) for block in (0,16,32,64)};save('summary.json',dict(passed=True,rows=rows,medians=medians,ratios_to_old={b:s/medians['0'] for b,s in medians.items()},jiuzhang3_time_ratio={b:s/64/1.27e-6 for b,s in medians.items()},scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}))
