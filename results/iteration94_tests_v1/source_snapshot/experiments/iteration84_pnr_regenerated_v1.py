"""Fresh PNR baseline and correlated-core records after RNG domain correction.
key=experiment; counter=(draw block, global record, sector, purpose). Every
experiment/sector/record/purpose domain is explicit. No failed-row deletion.
Numerical model inputs from the quarantined pilot are deterministic covariance
decompositions, not reused samples. Timings here are exploratory.
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
gate=ROOT/'results/iteration84_rng_gates_v1';validation=json.loads((gate/'validation.json').read_text());assert validation['passed'] and validation['known_answers']==55 and validation['domain_duplicates']==0 and validation['pnr_replay_methods']==2
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';panels=[('mixed',1024,0x84001001),('core0',2048,0x84001002),('core1',256,0x84001003)];files=[Path(__file__),inp,*[gate/f'{name}.dll' for name in ('mixed','correlated')],gate/'validation.json',gate/'domain_checks.json',*[core/f'input_{s}.npz' for s in range(2)],ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'native/gaussian_pnr_domain84_v1.cpp',ROOT/'native/gaussian_pnr_correlated84_v2.cpp',ROOT/'src/gbslab/native_direct_click.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},panels=[dict(method=m,records=n,experiment=e,sector_ids=[0,1],global_record_offset=0,purposes=dict(waiting=1,detector=2,displacement=3,resplitting=4)) for m,n,e in panels],threads=16,retained=3,confidence_plan=dict(events=10,delta_per_event=.001,observables='Three methods times full click count, last-128 click count and total photon count, plus shared experimental click CDF.')))
d=np.load(inp);v,x=d['v'],d['x'];m,n=v.shape;DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);dirs=[Path(shutil.which('g++')).parent];rows=[];domains=set()
for method,count,experiment in panels:
 print('START',method,count,experiment,flush=True);parts=[];sectors=[];start=time.perf_counter()
 for s in range(2):
  counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,16));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096);mean=np.zeros(2*n)
  if method=='mixed':
   with NativeDirectClickSampler(v,x[s],gate/'mixed.dll',runtime_dirs=dirs) as model:
    variance=np.zeros(2*n);fn=model.lib.pnr84_domain_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,count,experiment,s,0,16,3,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
  else:
   modelinput=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(modelinput['t']);L=np.ascontiguousarray(modelinput['noise_factor']);xp=np.diag(np.r_[-t/(1+t),t/(1-t)])
   with NativeDirectClickSampler(modelinput['v'],xp,gate/'correlated.dll',runtime_dirs=dirs) as model:
    fn=model.lib.pnr84d_correlated_sample;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,ptr(t),ptr(L),count,experiment,s,0,16,3,int(method[-1]),ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
  np.savez_compressed(OUT/f'{method}_{s}.npz',counts=counts,statistics=stats,errors=errors);ledger=[]
  for row in range(count):
   for purpose in range(1,5):
    used=int(stats[row,11+purpose]);identity=(experiment,s,row,purpose);assert identity not in domains;domains.add(identity);ledger.append(dict(experiment=experiment,sector=s,record=row,purpose=purpose,draws_used=used,blocks_used=(used+1)//2))
  save(f'ledger_{method}_{s}.json',ledger);success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,1]==1));row=dict(sector=s,success=success,returncode=rc,error=err.value.decode(),completed=int(np.count_nonzero(stats[:,1]==1)),failed_rows=int(np.count_nonzero(errors[:,0])),maximum_draws_by_purpose=stats[:,12:16].max(0).tolist());sectors.append(row);parts.append(counts);save(f'{method}_{s}.json',row);print(method,row,flush=True)
 row=dict(method=method,records=count,experiment=experiment,success=all(r['success'] for r in sectors),seconds=time.perf_counter()-start,sectors=sectors)
 if row['success']:
  counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;row.update(mean_photons=float(counts.sum(1).mean()),mean_clicks=float(bits.sum(1).mean()));np.savez_compressed(OUT/f'{method}_combined.npz',counts=counts,clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1),photon_histogram=np.bincount(counts.sum(1).astype(int)))
 rows.append(row);save(f'row_{method}.json',row);print('END',row,flush=True)
save('summary.json',dict(rows=rows,unique_domains=len(domains),domain_duplicates=0,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
