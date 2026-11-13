"""32768 fresh full1152-output correlated-core PNR records, fixed in advance.
Explicit new experiment domain; preserve every row including errors. Fifteen
nominal joint99 events cover three local click TV pairs, two optical-pair PNR
TV pairs, full/last128 photon CDFs, full/last128 click CDFs and experimental CDF.
Each local lower event uses first2048 pilot rows and remaining30720 validation.
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
core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';dll=ROOT/'results/iteration85_tree_pilot_v1/tree.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';N=32768;experiment=0x86001001;files=[Path(__file__),dll,ROOT/'native/gaussian_pnr_tree85_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',inp,*[core/f'input_{s}.npz' for s in range(2)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=experiment,records=N,batch_size=256,sectors=[0,1],global_offset=0,purposes=[1,2,3,4],threads=16,retained=3,groups=[list(range(8)),list(range(1144,1152)),list(range(0,1152,144))],optical_pairs=[[0,143],[142,143]],regions=[[0,1152],[1024,1152]],confidence=dict(events=15,delta_per_event=.01/15,pilot_records=2048),failure_policy='Any incomplete sector makes the whole record an explicit failure symbol. Never delete or refill.'))
dirs=[Path(shutil.which('g++')).parent];DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rows=[];parts=[];validparts=[]
for s in range(2):
 q=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);n=len(t);m=len(q['v']);mean=np.zeros(2*n);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);output=[];valids=[]
 with NativeDirectClickSampler(q['v'],xp,dll,runtime_dirs=dirs) as model:
  create=model.lib.pnr85_tree_create;create.argtypes=[C.c_void_p,C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=model.lib.pnr85_tree_destroy;destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);start=time.perf_counter();tree=create(model.handle,err,len(err));assert tree,err.value;save(f'prepare_{s}.json',dict(model_seconds=model.model_seconds,tree_seconds=time.perf_counter()-start));fn=model.lib.pnr85_correlated_sample;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int]
  try:
   for offset in range(0,N,256):
    count=min(256,N-offset);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,18));errors=np.zeros((count,2),np.int32);start=time.perf_counter();rc=fn(model.handle,tree,ptr(t),ptr(L),count,experiment,s,offset,16,3,0,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err));elapsed=time.perf_counter()-start;valid=(errors[:,0]==0)&(stats[:,1]==1);row=dict(sector=s,offset=offset,records=count,returncode=rc,error=err.value.decode(),completed=int(valid.sum()),failed=int((~valid).sum()),seconds=elapsed,tree_calls=float(stats[:,16].sum()),tree_fallbacks=float(stats[:,17].sum()));np.savez_compressed(OUT/f'raw_{s}_{offset}.npz',counts=counts,statistics=stats,errors=errors);save(f'row_{s}_{offset}.json',row);rows.append(row);output.append(counts);valids.append(valid)
    if offset%1024==0 or not valid.all():print(s,offset,'complete',row['completed'],'seconds',elapsed,flush=True)
  finally:destroy(tree)
 parts.append(np.concatenate(output));validparts.append(np.concatenate(valids));print('SECTOR COMPLETE',s,flush=True)
counts=parts[0].astype(np.uint64)+parts[1];valid=validparts[0]&validparts[1];np.savez_compressed(OUT/'combined.npz',counts=counts,clicks=counts>0,valid=valid);save('summary.json',dict(completed=True,records=N,failed_records=int((~valid).sum()),mean_photons_successful=float(counts[valid].sum(1).mean()),mean_clicks_successful=float((counts[valid]>0).sum(1).mean()),unique_domains=N*2*4,rows=rows,sampling_seconds=sum(r['seconds'] for r in rows),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
