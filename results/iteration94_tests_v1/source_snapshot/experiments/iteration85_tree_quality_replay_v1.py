"""Replay all2048 core quality records with the new detector tree.
Use the SAME explicit experiment/sector/global-record/purpose addresses as
the original core0 quality dataset. This is an implementation comparison,
NOT2048 additional independent samples; sample count remains2048.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration85_full_quality_v1';pilot=ROOT/'results/iteration85_tree_pilot_v1';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';dll=pilot/'tree.dll';files=[Path(__file__),dll,pilot/'summary.json',base/'plan.json',*[core/f'input_{s}.npz' for s in range(2)],*sorted(base.glob('core0_*.npz'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=0x85001003,records=2048,batch_size=64,threads=16,retained=3,domain_reuse='Intentional exact implementation replay; never add these records to the original dataset.'));dirs=[Path(shutil.which('g++')).parent];DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rows=[];parts=[]
for s in range(2):
 q=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);n=len(t);m=len(q['v']);mean=np.zeros(2*n);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);output=[]
 with NativeDirectClickSampler(q['v'],xp,dll,runtime_dirs=dirs) as model:
  create=model.lib.pnr85_tree_create;create.argtypes=[C.c_void_p,C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=model.lib.pnr85_tree_destroy;destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);tree=create(model.handle,err,len(err));assert tree,err.value;fn=model.lib.pnr85_correlated_sample;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int]
  try:
   for offset in range(0,2048,64):
    counts=np.zeros((64,m),np.uint32);stats=np.zeros((64,18));errors=np.zeros((64,2),np.int32);start=time.perf_counter();rc=fn(model.handle,tree,ptr(t),ptr(L),64,0x85001003,s,offset,16,3,0,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err));elapsed=time.perf_counter()-start;old=np.load(base/f'core0_{s}_{offset}.npz');equal=np.array_equal(counts,old['counts']) and np.array_equal(stats[:,:16],old['statistics']) and np.array_equal(errors,old['errors']);row=dict(sector=s,offset=offset,returncode=rc,error=err.value.decode(),success=rc==0 and not errors.any() and bool(np.all(stats[:,1]==1)),counts_statistics_errors_equal=bool(equal),seconds=elapsed,tree_calls=float(stats[:,16].sum()),tree_fallbacks=float(stats[:,17].sum()));np.savez_compressed(OUT/f'core0_{s}_{offset}.npz',counts=counts,statistics=stats,errors=errors);save(f'row_{s}_{offset}.json',row);rows.append(row);output.append(counts);print(s,offset,elapsed,row['success'],equal,flush=True)
  finally:destroy(tree)
 parts.append(np.concatenate(output))
counts=parts[0].astype(np.uint64)+parts[1];np.savez_compressed(OUT/'combined.npz',counts=counts,clicks=counts>0);save('summary.json',dict(passed=all(r['success'] and r['counts_statistics_errors_equal'] for r in rows),records=2048,sectors=2,rows=rows,sampling_seconds=sum(r['seconds'] for r in rows),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
