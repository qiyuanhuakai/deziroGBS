"""Preserve the concrete v2 sparse-cache failure and v3 corrected matrices."""
import argparse
from iteration91_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dlls={version:ROOT/f'results/iteration91_build_{version}/optical.dll' for version in ('v2','v3')};files=[Path(__file__),*dlls.values()];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));rows=[]
for version,dll in dlls.items():
 v=np.eye(2,dtype=complex)
 with NativeStableClickSampler(v,np.eye(4),dll,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
  h,d,meta=optical_cache(model);fn=model.lib.tc91_optical_export;fn.argtypes=[C.c_void_p,C.c_int,DP,C.c_char_p,C.c_int];maps=[]
  for j in range(2):
   raw=np.zeros((2-j,2,2));msg=C.create_string_buffer(4096);rc=fn(h,j,ptr(raw),msg,len(msg));assert rc==0,msg.value;a0=raw[:,:,0]+1j*raw[:,:,1];maps.append(a0);np.savez_compressed(out/f'{version}_suffix_{j}.npz',map=a0,target_gram=v[j:].conj().T@v[j:]);rows.append(dict(version=version,suffix=j,gram_error=float(np.linalg.norm(a0.conj().T@a0-v[j:].conj().T@v[j:]))))
  d(h)
assert all(r['gram_error']>0.9 for r in rows if r['version']=='v2');assert all(r['gram_error']==0 for r in rows if r['version']=='v3');save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
