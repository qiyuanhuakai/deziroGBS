"""Separate the optical-model and sampler discrepancies in Jiuzhang's 144 fiber loops."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse
import ctypes as ct
import math
from pathlib import Path
import shutil

import numpy as np
from iteration92_common_v2 import ROOT,H,read,save,manifest

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
actual=ROOT/"results/iteration94_actual_quality_v1"
prior=ROOT/"results/iteration95_pair_copula_v1"
filters=ROOT/"results/iteration95_count_filter_v1"
audit=ROOT/"results/iteration95_readout_audit_v1"
order_file=ROOT/"results/iteration83_loss_pilot_v1/input.npz"
paths=[Path(__file__),actual/"input_0.npz",actual/"input_1.npz",order_file,prior/"optical_4.npz",
       filters/"filter.dll",filters/"summary.json",audit/"diagnostic.npz",audit/"validation.json",
       *[filters/f"filtered_4_{i}.npz" for i in range(4)]]
save(out,"plan.json",dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths},
    groups="All 144 physical loops, eight consecutive MSB-first detector bins per loop; full 256-outcome optical laws by determinant inclusion-exclusion",
    comparison="Empirical hardware heldout loop-count CDF vs fixed optical reference; shared simultaneous band for all 144 loops using .01/144 per loop. Also compute exact filtered eight-bit laws for the previously observed direct-filter worst loop.",
    selection="Worst loop index may depend on heldout hardware; all-loop simultaneous band covers selection. Optical comparison deterministic."))
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
lib=ct.CDLL(str(filters/"filter.dll"))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,J,W=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.int32,np.uint64)]
def bind(name,args,result=I):
    fn=getattr(lib,name)
    fn.argtypes=args
    fn.restype=result
    return fn
optical=bind("window95_moments",[I,I,D,D,D,D,D,D])
window=bind("window95_prepare",[I,I,I,I,D,D,D],P)
window_law=bind("window95_law",[P,D])
window_free=bind("window95_destroy",[P],None)
q=np.empty((2,2304,2304))
mu=np.empty((2,2304))
for s in range(2):
    z=np.load(actual/f"input_{s}.npz")
    assert optical(1152,len(z["t"]),np.ascontiguousarray(z["u"]).view(float),z["t"],z["alpha"].view(float),
                   z["noise"],q[s],mu[s])==0
perm=np.argsort(np.load(order_file)["order"])
opt=np.load(prior/"optical_4.npz")
single=opt["single"]
joint=opt["joint"]
outcomes=np.arange(256)
bits=(outcomes[:,None]>>np.arange(8))&1
population=bits.sum(1)
law=np.empty((144,256))
model=np.empty((144,9))
for g in range(144):
    ports=8*g+np.arange(8)
    idx=np.r_[perm[ports],perm[ports]+1152]
    plan=window(8,8,0,2,np.ascontiguousarray(q[:,idx][:,:,idx]),np.ascontiguousarray(mu[:,idx]),np.zeros(4))
    assert plan and window_law(plan,law[g])==0
    window_free(plan)
    assert abs(law[g].sum()-1)<1e-12 and abs(law[g]@bits-single[ports]).max()<1e-12
    assert abs((bits.T*law[g])@bits-joint[np.ix_(ports,ports)]).max()<1e-12
    model[g]=np.bincount(population,weights=law[g],minlength=9)
hardware=np.load(audit/"diagnostic.npz")["hardware_loop"]
n=int(hardware[0].sum())
assert np.all(hardware.sum(1)==n)
emp=hardware.cumsum(1)/n
gaps=model.cumsum(1)-emp
worst=int(np.abs(gaps).max(1).argmax())
radius=math.sqrt(math.log(2*144/.01)/(2*n))
audit_row=next(x for x in read(audit/"validation.json")["rows"] if x["method"]=="filter_0")
g=audit_row["worst_loop"]
ports=np.arange(8*g,8*g+8,dtype=np.int32)
eig=bind("pair95_eigen",[I,D,D,D])
centers=bind("cube95_centers",[I,I,I,D,U,D])
prepare=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
project=bind("filter95_local",[P,I,I,W,I,J,I,D])
release=bind("cube95_destroy",[P],None)
val=np.empty(1152)
vec=np.empty((1152,1152))
assert eig(1152,opt["correlation"],val,vec)==0
d=.98*val[0]
grid=np.empty((16384,1152))
assert centers(1152,2048,4,np.ascontiguousarray(vec*np.sqrt(val-d)),0x950e1004,grid)==0
z=np.ascontiguousarray(opt["thresholds"][:,:17])
ptr=prepare(1152,len(grid),16,grid,math.sqrt(d),z,1,16,np.empty_like(z),np.empty_like(z))
assert ptr
rows=[]
corrected=[]
for row in (x for x in read(filters/"summary.json")["rows"] if x["case"]==4):
    route=row["route"]
    cut=np.load(filters/f"filtered_4_{route}.npz")["accept"]
    v=np.empty(256)
    assert project(ptr,int(route>0),len(cut),cut,8,ports,16,v)==0
    assert abs(v.sum()-row["acceptance"])<2e-10
    v/=row["acceptance"]
    local=np.bincount(population,weights=v,minlength=9)
    ks=float(max(abs(local.cumsum()-emp[g])))
    rows.append(dict(route=route,full_8bit_tv_optical=float(abs(v-law[g]).sum()/2),
        count_ks_optical=float(max(abs(local.cumsum()-model[g].cumsum()))),
        count_ks_hw=ks,hardware_at_most_zero=float(emp[g,0]),model_at_most_zero=float(local[0])))
    corrected.append(v)
release(ptr)
np.savez_compressed(out/"loops.npz",optical_click=law,optical_counts=model,hardware_counts=hardware,
                    selected_detector_ports=ports,filtered_click=np.array(corrected))
maximum=float(abs(gaps).max())
save(out,"summary.json",dict(worst_optical_loop=worst,worst_threshold=int(abs(gaps[worst]).argmax()),
    optical_hw_count_ks=maximum,hardware_radius=radius,optical_hw_lower99=maximum-radius,
    optical_hw_median=float(np.median(abs(gaps).max(1))),selected_loop=g,selected_optical_ks=float(abs(gaps[g]).max()),
    selected_optical_zero=float(model[g,0]),selected_rows=rows))
manifest(out)
