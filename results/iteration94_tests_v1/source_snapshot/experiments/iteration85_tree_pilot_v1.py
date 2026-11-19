"""Build and gate quadratic-moment optical-tree PNR detector routing.
Compare every detector probability against the frozen polynomial evaluation,
then identical domain streams through complete old/new trajectories. Exploratory
times include native setup and tree construction; formal repetitions are separate.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time,subprocess
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
src=ROOT/'native/gaussian_pnr_tree85_v1.cpp';gcc=Path(shutil.which('g++'));core=ROOT/'results/iteration84_correlated_pnr_pilot_v1';files=[Path(__file__),*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.hpp','.h')],*[core/f'input_{s}.npz' for s in range(2)]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},probe_seed=850020,replay_experiments=[0x85002001,0x85002002],replay_records=[16,64],threads=16,retained=3));command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'tree.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);UP=C.POINTER(C.c_uint32);ptr=lambda x:x.ctypes.data_as(DP);dirs=[gcc.parent];rng=np.random.default_rng(850020);checks=[]
def tree_api(model):
 create=model.lib.pnr85_tree_create;create.argtypes=[C.c_void_p,C.c_char_p,C.c_int];create.restype=C.c_void_p;destroy=model.lib.pnr85_tree_destroy;destroy.argtypes=[C.c_void_p];info=model.lib.pnr85_tree_info;info.argtypes=[C.c_void_p,DP];err=C.create_string_buffer(4096);start=time.perf_counter();tree=create(model.handle,err,len(err));assert tree,err.value;seconds=time.perf_counter()-start;meta=np.zeros(4);assert info(tree,ptr(meta))==0;return tree,destroy,seconds,meta
for case,(m,n) in enumerate(((3,3),(8,3),(32,8))):
 v=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];t=np.linspace(.1,.6,n);xp=np.diag(np.r_[-t/(1+t),t/(1-t)])
 with NativeDirectClickSampler(v,xp,OUT/'tree.dll',runtime_dirs=dirs) as model:
  tree,destroy,seconds,meta=tree_api(model);fn=model.lib.pnr85_detector_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,DP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
  for drawcase in range(4):
   A=rng.normal(size=(2*n,2*n))*.1;x=np.ascontiguousarray(xp+A@A.T);mean=np.ascontiguousarray(rng.normal(size=2*n));draws=np.r_[0.,np.nextafter(1.,0.),rng.random(15)]
   for k in range(4):
    modes=np.ascontiguousarray(rng.integers(m,size=k),np.int32);old=np.zeros(m);new=np.zeros(m);selected=np.zeros((len(draws),2),np.int32);stats=np.zeros(2);err=C.create_string_buffer(4096);rc=fn(model.handle,tree,ptr(x),ptr(mean),k,modes.ctypes.data_as(IP),len(draws),ptr(draws),ptr(old),ptr(new),selected.ctypes.data_as(IP),ptr(stats),err,len(err));row=dict(case=case,drawcase=drawcase,retained=k,returncode=rc,error=err.value.decode(),maximum_probability_difference=float(abs(old-new).max()),selection_equal=bool(np.array_equal(selected[:,0],selected[:,1])),fallbacks=float(stats[1]));np.savez_compressed(OUT/f'probe_{case}_{drawcase}_{k}.npz',v=v,x=x,mean=mean,modes=modes,draws=draws,old=old,new=new,selected=selected,statistics=stats);checks.append(row);save(f'probe_{case}_{drawcase}_{k}.json',row);assert rc==0 and row['selection_equal'] and row['maximum_probability_difference']<2e-12,row
  destroy(tree)
save('probe_checks.json',dict(passed=True,checks=checks));print('detector checks passed',len(checks),flush=True);rows=[]
for rep,count in enumerate((16,64)):
 outputs={}
 for method in ('old','tree'):
  parts=[];sectors=[];start=time.perf_counter()
  for s in range(2):
   q=np.load(core/f'input_{s}.npz');t=np.ascontiguousarray(q['t']);L=np.ascontiguousarray(q['noise_factor']);n=len(t);m=len(q['v']);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.zeros(2*n);err=C.create_string_buffer(4096);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,16 if method=='old' else 18));errors=np.zeros((count,2),np.int32)
   with NativeDirectClickSampler(q['v'],xp,OUT/'tree.dll',runtime_dirs=dirs) as model:
    common=[DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];args=[ptr(t),ptr(L),count,0x85002001+rep,s,0,16,3,0,ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err)]
    if method=='old':fn=model.lib.pnr84d_correlated_sample;fn.argtypes=[C.c_void_p,*common];fn.restype=C.c_int;rc=fn(model.handle,*args);meta=np.zeros(4);buildseconds=0.
    else:
     tree,destroy,buildseconds,meta=tree_api(model);fn=model.lib.pnr85_correlated_sample;fn.argtypes=[C.c_void_p,C.c_void_p,*common];fn.restype=C.c_int;rc=fn(model.handle,tree,*args);destroy(tree)
   success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,1]==1));row=dict(sector=s,success=success,returncode=rc,error=err.value.decode(),tree_build_seconds=buildseconds,tree_nodes=float(meta[0]),tree_bytes=float(meta[1]),tree_calls=float(stats[:,16].sum()) if method=='tree' else 0,tree_fallbacks=float(stats[:,17].sum()) if method=='tree' else 0);sectors.append(row);parts.append(counts);np.savez_compressed(OUT/f'replay_{rep}_{method}_{s}.npz',counts=counts,statistics=stats,errors=errors);print(rep,method,s,row,flush=True)
  row=dict(rep=rep,method=method,records=count,success=all(r['success'] for r in sectors),seconds=time.perf_counter()-start,sectors=sectors);rows.append(row);outputs[method]=np.stack(parts);save(f'row_{rep}_{method}.json',row)
 equal=np.array_equal(outputs['old'],outputs['tree']);save(f'comparison_{rep}.json',dict(equal=bool(equal),records=count));assert equal,'Complete PNR replay differs'
save('summary.json',dict(passed=all(r['success'] for r in rows),rows=rows,probe_cases=len(checks),scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
