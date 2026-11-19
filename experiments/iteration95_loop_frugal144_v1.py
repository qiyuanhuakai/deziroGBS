"""Same-input 144-bin comparison: retain the first time bin in every physical loop."""
import os
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[k]="1"
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy.special import ndtr
from iteration92_common_v2 import ROOT,H,save,read,manifest
from iteration95_loop_tables_v1 import integer_row,discrete
from iteration95_loop_covariance_v1 import calibrate,thresholds
sys.path.insert(0,str(ROOT/"src"))
from gbslab.frugal_law import full_law

p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
actual=ROOT/"results/iteration94_actual_quality_v1";layout=ROOT/"results/iteration83_loss_pilot_v1/input.npz"
old=ROOT/"results/iteration95_pair_copula_v1/optical_4.npz"
alloc=ROOT/"results/iteration95_loop_residual_v1/residual.dll"
frugal=ROOT/"results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll"
prefix=ROOT/"results/iteration81_prefix_build_v1/prefix.dll"
sources=[Path(__file__),layout,old,alloc,frugal,prefix,ROOT/"src/gbslab/frugal_law.py",
         *[actual/f"input_{s}.npz" for s in (0,1)]]
save(out,"plan.json",dict(hashes={x.relative_to(ROOT).as_posix():H(x) for x in sources},
  task="144 physical detector bins 0,8,...1144 from same original input, both independent spectra, output 144 clicks in both methods. Exact Gaussian reduction; no projection onto approximate optical modes.",
  accuracy="Full eight-bit prefix on 144-dimensional targets. Enumerate Frugal FP64 K2 upstream thresholds on each spectrum and OR convolve. Deterministic local optical law and our finite-table law.",
  timing="one worker, warmup then five alternating repetitions of 256. Upstream official generate_samples plus unpack and OR shown separately from setup and wall. Own record includes output/temporary allocation and expansion. No iid interval from Frugal timing samples.",
  rng="Frugal 0x95210000+2*(rep+1)+spectrum; proposal 0x95211000+(rep+1); independent frame setup 0x95212000"))
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
new=ct.CDLL(str(alloc));f=ct.CDLL(str(frugal));prefix_lib=ct.CDLL(str(prefix))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,B,W,T,J=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32,np.int32)]
def bind(lib,name,args,result=I):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    return fn
optical=bind(new,"window95_moments",[I,I,D,D,D,D,D,D]);window=bind(new,"window95_prepare",[I,I,I,I,D,D,D],P)
get_law=bind(new,"window95_law",[P,D]);drop=bind(new,"window95_destroy",[P],None)
center=bind(new,"cube95_centers",[I,I,I,D,U,D]);prepare=bind(new,"cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
sample=bind(new,"cube95_sample",[P,I,U,U,I,I,B]);release=bind(new,"cube95_destroy",[P],None)
expand=bind(new,"residual95_expand",[I,I,I,I,I,U,U,I,I,W,W,I,J,W,T,B,B])
external=bind(f,"fc73_sample",[I,I,I,ct.c_uint,I,D,B,D,ct.c_char_p,I])
tree=bind(prefix_lib,"fc81_tree",[I,I,I,I,D,D,D,D,D,ct.c_char_p,I])
start=perf_counter();ports=8*np.arange(144);perm=np.argsort(np.load(layout)["order"])[ports]
q=np.empty((2,288,288));mu=np.empty((2,288))
for s in (0,1):
    z=np.load(actual/f"input_{s}.npz")
    assert optical(144,len(z["t"]),np.ascontiguousarray(z["u"][perm]).view(float),z["t"],
                   z["alpha"].view(float),z["noise"],q[s],mu[s])==0
cov=np.ascontiguousarray(2*q-np.eye(288))
cov_seconds=perf_counter()-start
start=perf_counter();truth=[];masks=np.zeros((18,9,256),np.uint64);port=np.zeros((18,9))
bits=(np.arange(256)[:,None]>>np.arange(8))&1
pop=bits.sum(1)
for g in range(18):
    j=np.arange(8*g,8*g+8);ix=np.r_[j,j+144]
    plan=window(8,8,0,2,np.ascontiguousarray(q[:,ix][:,:,ix]),np.ascontiguousarray(mu[:,ix]),np.zeros(4))
    v=np.empty(256);assert plan and get_law(plan,v)==0;drop(plan);truth.append(v)
    for k in range(9):
        sel=(pop==k);port[g,k]=v[sel].sum();masks[g,k]=integer_row(v*sel if port[g,k]>0 else sel)
oracle=np.load(old);single=oracle["single"][ports];joint=oracle["joint"][np.ix_(ports,ports)]
group_cov=(joint-np.outer(single,single)).reshape(18,8,18,8).sum((1,3))
cuts=thresholds(port,True);corr,stats=calibrate(cuts,group_cov);val,vec=np.linalg.eigh(corr)
assert min(val)>0;variance=.98*val[0]
grid=np.empty((512,18))
assert center(18,64,4,np.ascontiguousarray(vec*np.sqrt(val-variance)),0x95212000,grid)==0
marg=np.empty_like(cuts);shift=np.empty_like(cuts)
ptr=prepare(18,512,8,grid,np.sqrt(variance),cuts,1,1,marg,shift)
assert ptr
prep=perf_counter()-start
actual_marg=np.r_[1,marg[0]]-np.r_[marg[0],0]
new_law=sum(actual_marg[k]*discrete(masks[0,k]) for k in range(9))
assert abs(new_law.sum()-1)<1e-11
def message():return ct.create_string_buffer(2048)
sector_laws=[];trees=[]
for s in (0,1):
    trace=np.empty(255);past=trace.copy();proposed=trace.copy();stat=np.zeros(3);err=message()
    assert tree(144,2,8,1,cov[s],trace,past,proposed,stat,err,len(err))==0,err.value
    nodes=[dict(prefix=''.join(str((j>>k)&1) for k in range(depth)),threshold_hex=float(trace[(1<<depth)-1+j]).hex())
           for depth in range(8) for j in range(1<<depth)]
    dist=full_law(8,nodes,model="uint64")
    sector_laws.append(np.array([float(x) for x in dist["distribution"]]));trees.append(trace)
pattern=np.arange(256)
f_law=np.bincount((pattern[:,None]|pattern).ravel(),weights=np.outer(*sector_laws).ravel(),minlength=256)
dummy=np.zeros(1,np.uint64);idx=np.zeros(1,np.int32);codes=np.zeros(1,np.uint32)
clocks=[];evidence={"truth":truth,"new":new_law,"frugal":f_law,"thresholds":trees}
for rep in range(-1,5):
    n=8 if rep<0 else 256
    for name in (["loop","frugal"] if rep%2==0 else ["frugal","loop"]):
        begin=perf_counter()
        if name=="loop":
            totals=np.empty((n,18),np.uint8);a=np.empty((n,144),np.uint8);key=0x95211000+rep+1
            assert sample(ptr,n,key,0,1,1,totals)==0
            assert expand(18,n,8,8,0,key,0,1,3,dummy,masks,0,idx,dummy,codes,totals,a)==0
            wall=perf_counter()-begin
            sample_seconds=wall;summary=[]
        else:
            sectors=[];summary=[]
            for s in (0,1):
                a=np.empty((n,144),np.uint8);stat=np.zeros(7);err=message()
                assert external(144,2,n,0x95210000+2*(rep+1)+s,1,cov[s],a,stat,err,len(err))==0,err.value
                sectors.append(a);summary.append(stat.tolist())
            or_begin=perf_counter();a=np.bitwise_or(*sectors);merge=perf_counter()-or_begin
            wall=perf_counter()-begin
            sample_seconds=sum(v[3]+v[4] for v in summary)+merge
        if rep>=0:
            clocks.append(dict(method=name,rep=rep,n=n,wall=wall,sampling=sample_seconds,phases=summary))
            evidence[f"{name}_{rep}"]=a
    save(out,"timing.json",clocks)
    print("completed",rep,flush=True)
release(ptr)
np.savez_compressed(out/"audit.npz",**evidence)
save(out,"summary.json",dict(new_prefix_tv=float(abs(new_law-truth[0]).sum()/2),
    frugal_prefix_tv=float(abs(f_law-truth[0]).sum()/2),covariance_seconds=cov_seconds,
    loop_setup_seconds=prep,timings=clocks,latent=stats))
manifest(out)
