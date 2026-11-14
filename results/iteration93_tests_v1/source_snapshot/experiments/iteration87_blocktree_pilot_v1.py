"""Build/gate binary-order-preserving block-leaf PNR Gram storage. Frozen85
conditional probes and full16/64-record paired trajectories; exploratory costs,
not formal timings or additional independent quality samples for each block size.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,argparse,json,hashlib,subprocess,shutil,ctypes as C,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
source=ROOT/'native/gaussian_pnr_blocktree87_v1.cpp';gcc=Path(shutil.which('g++'));old=ROOT/'results/iteration85_tree_pilot_v1/tree.dll';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';probes=ROOT/'results/iteration85_tree_pilot_v1';files=[Path(__file__),old,*[q for q in (ROOT/'native').iterdir() if q.suffix in ('.cpp','.hpp','.h')],*[core/f'input_{s}.npz' for s in range(2)],*sorted(probes.glob('probe_*.npz'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},blocks=[1,8,16,32,64],records=[16,64],experiments=[0x87002001,0x87002002],threads=16,retained=3))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/'blocktree.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);UP=C.POINTER(C.c_uint32);ptr=lambda a:a.ctypes.data_as(DP);dirs=[gcc.parent]
def tree(model,block):
 suffix='85' if block==0 else '87';create=getattr(model.lib,f'pnr{suffix}_tree_create');create.argtypes=[C.c_void_p,*([C.c_int] if block else []),C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=getattr(model.lib,f'pnr{suffix}_tree_destroy');destroy.argtypes=[C.c_void_p];info=getattr(model.lib,f'pnr{suffix}_tree_info');info.argtypes=[C.c_void_p,DP];err=C.create_string_buffer(4096);start=time.perf_counter();handle=create(model.handle,*([block] if block else []),err,len(err));assert handle,err.value;seconds=time.perf_counter()-start;meta=np.zeros(7 if block else 4);assert info(handle,ptr(meta))==0;return handle,destroy,meta,seconds
checks=[]
for path in sorted(probes.glob('probe_*.npz')):
 z=np.load(path);x=np.ascontiguousarray((z['x']+z['x'].T)/2);mean=np.ascontiguousarray(z['mean']);modes=np.ascontiguousarray(z['modes'],np.int32);draws=np.ascontiguousarray(z['draws']);v=z['v']
 with NativeDirectClickSampler(v,x,out/'blocktree.dll',runtime_dirs=dirs) as model:
  fn=model.lib.pnr87_detector_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,IP,DP,C.c_char_p,C.c_int]
  for block in (1,8,32):
   handle,destroy,meta,seconds=tree(model,block);selected=np.zeros((len(draws),2),np.int32);stats=np.zeros(4);err=C.create_string_buffer(4096)
   try:rc=fn(model.handle,handle,ptr(x),ptr(mean),len(modes),modes.ctypes.data_as(IP),len(draws),ptr(draws),selected.ctypes.data_as(IP),ptr(stats),err,len(err))
   finally:destroy(handle)
   row=dict(probe=path.name,block=block,returncode=rc,error=err.value.decode(),equal=bool(np.array_equal(selected[:,0],selected[:,1])),statistics=stats.tolist());checks.append(row);assert rc==0 and row['equal'],row
save('probes.json',dict(passed=True,checks=checks));print('conditional probes passed',len(checks),flush=True);rows=[];comparisons=[]
for rep,count in enumerate((16,64)):
 reference=None
 for block in (0,1,8,16,32,64):
  raw=[];sectors=[];start=time.perf_counter()
  for sector in range(2):
   q=np.load(core/f'input_{sector}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);n=len(t);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.zeros(2*n);m=len(q['v']);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,20 if block else 18));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096)
   with NativeDirectClickSampler(q['v'],xp,old if block==0 else out/'blocktree.dll',runtime_dirs=dirs) as model:
    handle,destroy,meta,build_seconds=tree(model,block)
    try:
     if block==32 and count==16:
      audit=model.lib.pnr87_gram_audit;audit.argtypes=[C.c_void_p,C.c_void_p,DP,C.c_char_p,C.c_int];auditstats=np.zeros(3);rc=audit(model.handle,handle,ptr(auditstats),err,len(err));save(f'gram_audit_{sector}.json',dict(returncode=rc,error=err.value.decode(),compared=float(auditstats[0]),different=float(auditstats[1]),maximum_difference=float(auditstats[2])));assert rc==0 and auditstats[1]==0 and auditstats[2]==0
     fn=getattr(model.lib,'pnr87_correlated_sample' if block else 'pnr85_correlated_sample');fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];rc=fn(model.handle,handle,ptr(t),ptr(L),count,0x87002001+rep,sector,0,16,3,0,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
    finally:destroy(handle)
   row=dict(sector=sector,returncode=rc,error=err.value.decode(),success=rc==0 and not errors[:,0].any() and bool((stats[:,1]==1).all()),tree_metadata=meta.tolist(),tree_build_seconds=build_seconds);sectors.append(row);raw.append(dict(counts=counts,statistics=stats,errors=errors));np.savez_compressed(out/f'raw_{count}_{block}_{sector}.npz',**raw[-1]);assert row['success'],row
  row=dict(records=count,block=block,seconds=time.perf_counter()-start,sectors=sectors);rows.append(row);save(f'row_{count}_{block}.json',row)
  if block==0:reference=raw
  else:
   equal=all(np.array_equal(reference[s]['counts'],raw[s]['counts']) and np.array_equal(reference[s]['statistics'],raw[s]['statistics'][:,:18]) and np.array_equal(reference[s]['errors'],raw[s]['errors']) for s in range(2));comparisons.append(dict(records=count,block=block,equal=equal));assert equal,(count,block)
  print('full replay',count,block,row['seconds'],flush=True)
save('summary.json',dict(passed=True,rows=rows,comparisons=comparisons,probes=len(checks),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}))
