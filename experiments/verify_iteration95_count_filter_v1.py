"""Replay filtered records and test their spatial click distribution on held-out hardware shots."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import ctypes as ct
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from scipy.special import ndtr
from iteration92_common_v2 import ROOT,H,save,read,manifest

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
tables=ROOT/"results/iteration95_pair_table_v1"
filters=ROOT/"results/iteration95_count_filter_v1"
prior=ROOT/"results/iteration95_pair_copula_v1"
third=ROOT/"results/iteration95_third_click_v1"
hardware=ROOT/"results/iteration81_jiuzhang_accuracy_v3"
raw=ROOT.parent/"external_sources/jiuzhang3_iteration81_sample_v1/extracted_3/jiuzhang3/data/high power/high power (1).bin"
optical=np.load(prior/"optical_4.npz")
source=ROOT/"native/gaussian_readout95_v1.cpp"
files=[Path(__file__),source,tables/"manifest.json",filters/"manifest.json",prior/"optical_4.npz",third/"quality_3.npz",
       hardware/"histograms.npz",third/"moments.json"]
pilot=32768
delta=.01/1024
save(out,"plan.json",dict(hashes={x.relative_to(ROOT).as_posix():H(x) for x in files},hardware_sha256=H(raw),
    training="32768 first experimental shots and 32768 first direct-filter shots, no other sample used. Features 1152 bin hits and per-loop >=2,>=3 bin collisions. Solve ridge LDA with 0.2 standardized diagonal ridge.",
    score="Remove pooled pilot linear regression on total clicks from LDA score. Choose signed threshold using maximum absolute training CDF difference over 999 pooled quantiles. Freeze score then evaluate holdouts.",
    holdout="All remaining 1459874 hardware shots and 229376 per filter; entire 65536 base shots and 8192 third-moment window shots are separate from training.",
    family="One two-sided Hoeffding event rate and 144 DKW loop CDFs for each of seven sources. Union budget 0.01/1024 each; shared hardware band; distinct records assumed iid within source. Hardware-file blocks are diagnostic only.",
    replay="Check all 128 archived SHA256 hashes and total/count/retry arrays for every filtered record; verify standalone stored array prefix. Independent scipy probability table, grid covariance, subset enumeration and model totals.",
    timing="Dependent inputs are fixed archived binary64 optical targets and calibration moments. Time a fresh factor/table preparation plus the relevant proposal total distribution. Excludes optical input preparation and calculation of the target photon PMF."))
assert H(raw)=="c786e6e64f76a599479eda75c7fdedf6906a3b41040eec49e5ea89caeb714e74"
for directory in (tables,filters):
    listed=read(directory/"manifest.json")
    assert set(listed)=={x.name for x in directory.iterdir()}-{"manifest.json"}
    assert all(H(directory/path)==digest for path,digest in listed.items())

cmd=[shutil.which("g++"),"-std=c++20","-O3","-ffp-contract=off","-fopenmp","-shared",str(source),"-o",str(out/"readout.dll")]
build=subprocess.run(cmd,capture_output=True,text=True)
save(out,"build.json",dict(command=cmd,code=build.returncode,stderr=build.stderr))
build.check_returncode()
runtime=os.add_dll_directory(str(Path(cmd[0]).parent))
lib=ct.CDLL(str(filters/"filter.dll"))
reader=ct.CDLL(str(out/"readout.dll"))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,B,W,T=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32)]
def bind(lib,name,args,result=I):
    fn=getattr(lib,name)
    fn.argtypes=args
    fn.restype=result
    return fn
eig=bind(lib,"pair95_eigen",[I,D,D,D])
centers=bind(lib,"cube95_centers",[I,I,I,D,U,D])
prepare=bind(lib,"cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
release=bind(lib,"cube95_destroy",[P],None)
draw=bind(lib,"filter95_draw",[P,I,U,U,I,I,I,W,B,T])
click_sum=bind(lib,"cube95_counts",[P,I,I,D])
photon_sum=bind(lib,"cube95_photon_law",[P,I,I,D])
readout=bind(reader,"readout95",[B,I,I,D,D,W,T])
hw=np.memmap(raw,mode="r",dtype=np.uint8).reshape(-1,144)
m=1152
saved=np.load(tables/"model_6.npz")
z=np.ascontiguousarray(optical["thresholds"][:,:17])
start=perf_counter()
eigval=np.empty(m)
eigvec=np.empty((m,m))
assert eig(m,optical["correlation"],eigval,eigvec)==0
d=.98*eigval[0]
grid=np.empty((16384,m))
assert centers(m,2048,4,np.ascontiguousarray(eigvec*np.sqrt(eigval-d)),0x950e1004,grid)==0
marg=np.empty_like(z)
shifts=np.empty_like(z)
ptr=prepare(m,len(grid),16,grid,math.sqrt(d),z,1,16,marg,shifts)
setup=perf_counter()-start
assert ptr and np.array_equal(marg,saved["marginal"])
assert np.array_equal(grid[:,saved["audit_ports"]],saved["centers_subset"])
setup_times=dict(table_from_stored_correlation=setup)
for which,method in [("click_count",click_sum),("photon_count",photon_sum)]:
    law=saved[which]
    actual=np.empty_like(law)
    start=perf_counter()
    assert method(ptr,len(law)-1,16,actual)==0
    setup_times[which]=perf_counter()-start
    assert np.array_equal(actual,law)
# Independent SciPy implementation of every integer-rounded two-port law.
q=np.rint(ndtr((grid-(z[:,0]+shifts[:,0]))/math.sqrt(d))*2**32).clip(0,2**32-1)/2**32
ex=q.mean(0)
joint=q.T@q/len(q)
np.fill_diagonal(joint,ex)
independent_pairs=float(abs(joint-saved["click_second"]).max())
assert independent_pairs<2e-12 and abs(ex-saved["click_first"]).max()<2e-12
del grid,q
filter_rows=[x for x in read(filters/"summary.json")["rows"] if x["case"]==4]
samples={}
for route,r in enumerate(filter_rows):
    f=np.load(filters/f"filtered_4_{route}.npz")
    cut=f["accept"]
    observed=f["observed_total"]
    clicks=f["observed_click"]
    attempts=f["attempts"]
    packed=np.empty((len(observed),144),np.uint8)
    for block,digest in enumerate(r["records_sha256"]):
        off=8192*block
        full=np.empty((8192,m),np.uint8)
        trial=np.empty(8192,np.uint32)
        assert draw(ptr,8192,0x95150000+128+route,off,int(route>0),8,len(cut),cut,full,trial)==0
        assert hashlib.sha256(full.tobytes()).hexdigest()==digest
        assert np.array_equal(full.sum(1),observed[off:off+8192]) and np.array_equal(trial,attempts[off:off+8192])
        assert np.array_equal((full>0).sum(1),clicks[off:off+8192])
        if off==0:assert np.array_equal(full,f["first_records"])
        packed[off:off+8192]=np.packbits(full>0,axis=1)
    samples[f"filter_{route}"]=packed
release(ptr)
samples["table"]=np.concatenate([np.packbits(np.load(tables/f"records_6_{off}.npz")["counts"]>0,axis=1)
                                  for off in range(0,65536,8192)])
window_little=np.load(third/"quality_3.npz")["packed_bits"]
samples["window"]=np.packbits(np.unpackbits(window_little,axis=1,bitorder="little"),axis=1)
pop=np.array([j.bit_count() for j in range(256)])
def features(packed):
    n=pop[packed]
    return np.column_stack((np.unpackbits(packed,axis=1),n>=2,n>=3)).astype(float)
a=features(hw[:pilot])
b=features(samples["filter_0"][:pilot])
ma=a.mean(0)
mb=b.mean(0)
a-=ma
b-=mb
cov=(a.T@a+b.T@b)/(2*(pilot-1))
cov.flat[::len(cov)+1]+=.2*np.maximum(cov.diagonal(),1e-5)
weight=solve(cov,ma-mb,assume_a="pos")
del a,b
bits=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)
lookup=(weight[:m].reshape(144,8)@bits.T+weight[m:m+144,None]*(pop>=2)+weight[m+144:,None]*(pop>=3))
lookup=np.ascontiguousarray(lookup)
def measure(packed):
    h=np.empty((144,9),np.uint64)
    score=np.empty(len(packed))
    total=np.empty(len(packed),np.uint32)
    assert readout(packed,len(packed),16,lookup,score,h,total)==0 and np.all(h.sum(1)==len(packed))
    return score,total,h
ha,hc,_=measure(hw[:pilot])
mb,mc,_=measure(samples["filter_0"][:pilot])
s=np.r_[ha,mb]
k=np.r_[hc,mc].astype(float)
slope=np.mean((s-s.mean())*(k-k.mean()))/k.var()
lookup-=slope*pop
ha-=slope*hc
mb-=slope*mc
cuts=np.quantile(np.r_[ha,mb],np.arange(1,1000)/1000)
gaps=(np.searchsorted(np.sort(ha),cuts,side="right")-np.searchsorted(np.sort(mb),cuts,side="right"))/pilot
j=int(np.argmax(abs(gaps)))
cut=cuts[j]
sign=int(np.sign(gaps[j]))
hw_score,hw_total,hh=measure(hw[pilot:])
assert np.array_equal(measure(hw[:32])[0],lookup[np.arange(144),hw[:32]].cumsum(1)[:,-1])
hw_event=float((hw_score<=cut).mean())
hw_loop=hh.cumsum(1)/len(hw_score)
radius=lambda n: math.sqrt(math.log(2/delta)/(2*n))
target_hw=np.load(hardware/"histograms.npz")["total_click_histogram"]
assert np.array_equal(np.bincount(np.r_[hc,hw_total],minlength=1153),target_hw)
audit=dict(lookup=lookup,threshold=cut,sign=sign,residual_slope=slope,hardware_loop=hh,hardware_blocks=np.array(
    [[len(block),np.count_nonzero(block<=cut)] for block in np.array_split(hw_score,16)]))
rows=[]
for name,packed in samples.items():
    if name.startswith("filter_"):packed=packed[pilot:]
    value,count,h=measure(packed)
    rate=float((value<=cut).mean())
    diff=sign*(hw_event-rate)
    error=radius(len(value))+radius(len(hw_score))
    loop=np.max(abs(h.cumsum(1)/len(value)-hw_loop),axis=1)
    worst=int(loop.argmax())
    audit[name+"_loop"]=h
    rows.append(dict(method=name,n=len(value),model_event=rate,hardware_event=hw_event,witness_gap=diff,
                     witness_tv_lower99=max(0.,diff-error),loop_ks_median=float(np.median(loop)),loop_ks_max=float(loop[worst]),
                     worst_loop=worst,worst_loop_lower99=max(0.,loop[worst]-error),simultaneous_radius=error))
np.savez_compressed(out/"diagnostic.npz",**audit)
save(out,"validation.json",dict(passed=True,pair_reconstruction=independent_pairs,replayed_records=4*262144,
    training_gap=float(abs(gaps[j])),holdout_hardware=len(hw_score),setup_seconds=setup_times,rows=rows))
manifest(out)
