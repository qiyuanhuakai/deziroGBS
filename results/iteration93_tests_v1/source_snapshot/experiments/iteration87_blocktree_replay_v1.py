"""Replay every frozen86 PNR quality record using block32 Gram storage.
Reused domains are intentional: this does not add32768 independent observations.
Every integer count, original18 statistics and error entry must match exactly.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,shutil,ctypes as C,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';base=ROOT/'results/iteration86_pnr_quality_v1';pilot=ROOT/'results/iteration87_blocktree_pilot_v1';assert json.loads((pilot/'summary.json').read_text())['passed'];dll=pilot/'blocktree.dll';files=[Path(__file__),dll,ROOT/'native/gaussian_pnr_blocktree87_v1.cpp',base/'manifest.json',*[core/f'input_{s}.npz' for s in range(2)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=32768,experiment=0x86001001,sectors=[0,1],global_offset=0,purposes=[1,2,3,4],block=32,retained=3,batch_size=256,threads=16))
DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda a:a.ctypes.data_as(DP);dirs=[Path(shutil.which('g++')).parent];rows=[]
for sector in range(2):
 q=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);mean=np.zeros(2*len(t));xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);m=len(q['v'])
 with NativeDirectClickSampler(q['v'],xp,dll,runtime_dirs=dirs) as model:
  create=model.lib.pnr87_tree_create;create.argtypes=[C.c_void_p,C.c_int,C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=model.lib.pnr87_tree_destroy;destroy.argtypes=[C.c_void_p];info=model.lib.pnr87_tree_info;info.argtypes=[C.c_void_p,DP];err=C.create_string_buffer(4096);tree=create(model.handle,32,err,len(err));assert tree,err.value;meta=np.zeros(7);assert info(tree,ptr(meta))==0;save(f'tree_{sector}.json',dict(metadata=meta.tolist()));fn=model.lib.pnr87_correlated_sample;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int]
  try:
   for offset in range(0,32768,256):
    counts=np.zeros((256,m),np.uint32);stats=np.zeros((256,20));errors=np.zeros((256,2),np.int32);start=time.perf_counter();rc=fn(model.handle,tree,ptr(t),ptr(L),256,0x86001001,sector,offset,16,3,0,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err));seconds=time.perf_counter()-start;reference=np.load(base/f'raw_{sector}_{offset}.npz');same=dict(counts=bool(np.array_equal(counts,reference['counts'])),statistics=bool(np.array_equal(stats[:,:18],reference['statistics'])),errors=bool(np.array_equal(errors,reference['errors'])));row=dict(sector=sector,offset=offset,returncode=rc,error=err.value.decode(),seconds=seconds,equal=same);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',counts=counts,statistics=stats,errors=errors);save(f'row_{sector}_{offset}.json',row);rows.append(row);assert rc==0 and all(same.values()),row
    if offset%1024==0:print('exact replay',sector,offset,'seconds',seconds,flush=True)
  finally:destroy(tree)
save('summary.json',dict(passed=True,records=32768,sectors=2,rows=rows,sampling_seconds=sum(r['seconds'] for r in rows),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}))
