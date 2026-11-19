"""Three serial paired rotations of frozen84 PNR versus optical-tree85 PNR.
Use the actually published old DLL, not the legacy symbol recompiled in tree85.
Archive this v2 driver under the first formal timing result directory.
Prepared correlated-core representation is the common input. Include model
construction, optical tree construction, random draws, both spectra, integer
sum, thresholding and destruction; exclude precomputed loss/core decomposition,
DLL loading, builds and I/O. No heavy concurrent work. Replay is not quality data.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
pilot=ROOT/'results/iteration85_tree_pilot_v1';assert json.loads((pilot/'summary.json').read_text())['passed'];dll=pilot/'tree.dll';old_dll=ROOT/'results/iteration84_rng_gates_v1/correlated.dll';core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),dll,old_dll,pilot/'summary.json',ROOT/'native/gaussian_pnr_tree85_v1.cpp',*[core/f'input_{s}.npz' for s in range(2)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},methods=['old','tree'],records=[4,64],repetitions=3,threads=16,retained=3,experiments=[0x85003001+i for i in range(3)]));dirs=[Path(shutil.which('g++')).parent];runtime=[os.add_dll_directory(str(p)) for p in dirs];lib=C.CDLL(str(dll));oldlib=C.CDLL(str(old_dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);create=lib.pnr85_tree_create;create.argtypes=[C.c_void_p,C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=lib.pnr85_tree_destroy;destroy.argtypes=[C.c_void_p];common=[DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];old=oldlib.pnr84d_correlated_sample;old.argtypes=[C.c_void_p,*common];new=lib.pnr85_correlated_sample;new.argtypes=[C.c_void_p,C.c_void_p,*common];models=[]
for s in range(2):
 q=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(q['t']);models.append(dict(v=q['v'],t=t,L=np.ascontiguousarray(q['noise_factor']),x=np.diag(np.r_[-t/(1+t),t/(1-t)]),mean=np.zeros(2*len(t))))
rows=[]
for rep in range(3):
 panels=[('old',4),('tree',4),('old',64),('tree',64)];panels=panels[rep:]+panels[:rep]
 for method,count in panels:
  parts=[];raw=[];sectors=[];start=time.perf_counter()
  for s,q in enumerate(models):
   counts=np.zeros((count,len(q['v'])),np.uint32);stats=np.zeros((count,16 if method=='old' else 18));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096)
   with NativeDirectClickSampler(q['v'],q['x'],old_dll if method=='old' else dll,runtime_dirs=dirs) as model:
    args=[ptr(q['t']),ptr(q['L']),count,0x85003001+rep,s,0,16,3,0,ptr(q['mean']),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err)]
    if method=='old':rc=old(model.handle,*args)
    else:
     tree=create(model.handle,err,len(err));assert tree,err.value
     try:rc=new(model.handle,tree,*args)
     finally:destroy(tree)
   sectors.append(dict(sector=s,returncode=rc,error=err.value.decode(),success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,1]==1))));parts.append(counts);raw.append(dict(counts=counts,statistics=stats,errors=errors))
  counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;elapsed=time.perf_counter()-start;row=dict(rep=rep,method=method,records=count,seconds=elapsed,success=all(s['success'] for s in sectors),sectors=sectors);rows.append(row);save(f'row_{rep}_{method}_{count}.json',row)
  for s,data in enumerate(raw):np.savez_compressed(OUT/f'sector_{rep}_{method}_{count}_{s}.npz',**data)
  if row['success']:np.savez_compressed(OUT/f'combined_{rep}_{method}_{count}.npz',counts=counts,clicks=bits)
  print(rep,method,count,elapsed,row['success'],flush=True)
comparisons=[];speeds=[]
for count in (4,64):
 for rep in range(3):
  equal=True
  for s in range(2):
   a=np.load(OUT/f'sector_{rep}_old_{count}_{s}.npz');b=np.load(OUT/f'sector_{rep}_tree_{count}_{s}.npz');equal &= np.array_equal(a['counts'],b['counts']) and np.array_equal(a['statistics'],b['statistics'][:,:16]) and np.array_equal(a['errors'],b['errors'])
  comparisons.append(dict(rep=rep,records=count,counts_statistics_errors_equal=bool(equal)))
 for method in ('old','tree'):
  selected=[r for r in rows if r['method']==method and r['records']==count];speeds.append(dict(method=method,records=count,successful=sum(r['success'] for r in selected),median_seconds=statistics.median(r['seconds'] for r in selected) if all(r['success'] for r in selected) else None))
save('summary.json',dict(rows=rows,replay=comparisons,medians=speeds,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
