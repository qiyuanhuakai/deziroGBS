"""Fresh complete-output quality panels with explicit disjoint random domains.
B32/B64 direct click and correlated-core PNR, sequential heavy work only.
Freeze a 30-event nominal joint99 family before any sampling. Three methods
times: full/last128 click CDF, three local click-TV upper bounds, three fixed
pilot-event lower bounds, and two regional photon-count CDF events (PNR only;
unused direct-click allocations remain unused). Domain separation is not iid proof.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
paths=dict(exact=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll',core=ROOT/'results/iteration84_rng_gates_v1/correlated.dll',rng=ROOT/'results/iteration84_rng_gates_v1/rng.dll');inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';panels=[('exact32',512,0x85001001),('exact64',512,0x85001002),('core0',2048,0x85001003)];groups=[list(range(8)),list(range(1144,1152)),list(range(0,1152,144))];files=[Path(__file__),inp,*paths.values(),*[core/f'input_{s}.npz' for s in range(2)],ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'native/gaussian_click_exactsum84_v1.cpp',ROOT/'native/gaussian_pnr_correlated84_v2.cpp'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},panels=[dict(method=m,records=n,experiment=e,sectors=[0,1],global_offset=0) for m,n,e in panels],workers=16,batch_size=64,groups=groups,regions=[[0,1152],[1024,1152]],confidence=dict(total_events=30,delta_per_event=.01/30,pilot_records=128,pilot_rule='First128 records used only to choose each local excess-probability event; lower bounds use remaining records. Upper TV bounds use full empirical TV with one-sided bounded differences and convexity. No pilot-selected threshold is reused on its own data.')))
dirs=[Path(shutil.which('g++')).parent];runtime=[os.add_dll_directory(str(d)) for d in dirs];lib=C.CDLL(str(paths['rng']));DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);UP=C.POINTER(C.c_uint32);U64P=C.POINTER(C.c_uint64);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rng=lib.rng84_stream;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64P,C.c_char_p,C.c_int];rng.restype=C.c_int;d=np.load(inp);v,x=d['v'],d['x'];m,n=v.shape;rows=[];domains=set()
for method,count,experiment in panels:
 print('START',method,count,flush=True);parts=[];sectors=[];seconds=0.
 for s in range(2):
  raw=[];batchrows=[];ledger=[]
  if method.startswith('exact'):
   budget=int(method[5:]);model=NativeStableClickSampler(v,x[s],paths['exact'],branch_budget=budget,runtime_dirs=dirs);fn=model.lib.tc84e_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
  else:
   q=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.zeros(2*n);model=NativeDirectClickSampler(q['v'],xp,paths['core'],runtime_dirs=dirs);fn=model.lib.pnr84d_correlated_sample;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int
  with model:
   for offset in range(0,count,64):
    size=min(64,count-offset);err=C.create_string_buffer(4096);errors=np.zeros((size,2),np.int32);start=time.perf_counter()
    if method.startswith('exact'):
     words=np.zeros((size,m),np.uint64)
     for r in range(size):assert rng(experiment,s,5,offset+r,0,m,words[r].ctypes.data_as(U64P),err,len(err))==0,err.value
     draws=np.ascontiguousarray((words>>11).astype(float)*2**-53);output=np.zeros((size,m),np.uint8);prob=np.zeros((size,m));stats=np.zeros((size,8));rc=fn(model.handle,size,0,16,budget,ptr(draws),output.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err));data=dict(clicks=output,conditional=prob,statistics=stats,errors=errors);complete=stats[:,5]==m;purposes=[5]
    else:
     output=np.zeros((size,m),np.uint32);stats=np.zeros((size,16));rc=fn(model.handle,ptr(t),ptr(L),size,experiment,s,offset,16,3,0,ptr(mean),None,output.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err));data=dict(counts=output,statistics=stats,errors=errors);complete=stats[:,1]==1;purposes=range(1,5)
    elapsed=time.perf_counter()-start;seconds+=elapsed
    for r in range(size):
     for purpose in purposes:
      address=(experiment,s,offset+r,purpose);assert address not in domains;domains.add(address);ledger.append(dict(experiment=experiment,sector=s,record=offset+r,purpose=purpose,words_generated=m if purpose==5 else int(stats[r,11+purpose])))
    np.savez_compressed(OUT/f'{method}_{s}_{offset}.npz',**data);row=dict(offset=offset,records=size,returncode=rc,error=err.value.decode(),success=rc==0 and not errors[:,0].any() and bool(np.all(complete)),completed=int(np.count_nonzero(complete)),failed_rows=int(np.count_nonzero(errors[:,0])),seconds=elapsed);save(f'{method}_{s}_{offset}.json',row);batchrows.append(row);raw.append(output);print(method,s,offset,row,flush=True)
  sectors.append(dict(sector=s,success=all(r['success'] for r in batchrows),batches=batchrows));parts.append(np.concatenate(raw));save(f'ledger_{method}_{s}.json',ledger)
 row=dict(method=method,records=count,experiment=experiment,success=all(s['success'] for s in sectors),sampling_seconds=seconds,sectors=sectors)
 if row['success']:
  if method.startswith('exact'):bits=np.bitwise_or(*parts);data=dict(clicks=bits)
  else:counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;data=dict(clicks=bits,counts=counts);row['mean_photons']=float(counts.sum(1).mean())
  data['total_histogram']=np.bincount(bits.sum(1).astype(int),minlength=m+1);np.savez_compressed(OUT/f'{method}_combined.npz',**data);row['mean_clicks']=float(bits.sum(1).mean())
 rows.append(row);save(f'row_{method}.json',row);print('END',method,row['success'],row.get('mean_clicks'),seconds,flush=True)
save('summary.json',dict(rows=rows,unique_domains=len(domains),domain_duplicates=0,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
