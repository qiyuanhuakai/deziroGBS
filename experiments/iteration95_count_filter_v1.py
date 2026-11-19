"""Compare direct-click optical-moment filtering and PNR optical-total filtering."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"

import argparse
import ctypes as ct
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from iteration92_common_v2 import ROOT,H,save,read,manifest

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
base=ROOT/"results/iteration95_pair_table_v1"
optics=ROOT/"results/iteration95_pair_copula_v1"
reference=ROOT/"results/iteration93_reference_v2"
totals=ROOT/"results/iteration84_photon_cdf_v1"
hw=ROOT/"results/iteration81_jiuzhang_accuracy_v3"
third=ROOT/"results/iteration95_third_click_v1"
src=ROOT/"native/gaussian_count_filter95_v1.cpp"
files=[Path(__file__),src,base/"summary.json",base/"manifest.json",third/"moments.json",
       totals/"reference.npz",totals/"reference.json",hw/"histograms.npz"]
files.extend(ROOT/k for k in read(base/"plan.json")["hashes"])
save(out,"plan.json",dict(hashes={a.relative_to(ROOT).as_posix():H(a) for a in files},
    fits="Click fit only first three optical click count moments (all independent powers at m<3), with logistic acceptance .75 or .5 when infeasible. PNR probability ratio truncated at 1.25,2,4. No measured hardware data in either fit.",
    model="finite table four calibrated frames, full physical output, independent trial words per record and purpose",
    samples="large 262144 per route/bound, small 32768, store all photon/click sums and retries plus first 8192 full records per stage and hashes of all blocks",
    rng="quality keys 0x95150000+32*case+route, timing 0x95160000+1024*case+64*route+index; purposes 123,124,126 distinct",
    family="shared hardware DKW plus three independent large PNR-to-click model bands, alpha .01/4 each; all other reported table-law quantities numerical deterministic",
    timings="five interleaved repetitions incl allocation of counts and attempt counts, 32768 large, 2048 small; 101 serial single-shot calls. Proposal and filter preparation timed separately.",
    local="exact centerwise conditional convolution for three fixed physical eight-port groups and complete small click laws; global normalization audited against independent total-count PMF"))
cmd=read(base/"build.json")["command"]
cmd[-3]=str(src)
cmd[-1]=str(out/"filter.dll")
build=subprocess.run(cmd,capture_output=True,text=True)
save(out,"build.json",dict(command=cmd,code=build.returncode,stderr=build.stderr))
build.check_returncode()
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
lib=ct.CDLL(str(out/"filter.dll"))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,B,IDX,WORD,RETRY=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in
                      (np.float64,np.uint8,np.int32,np.uint64,np.uint32)]
def bind(name,args,result=I):
    fn=getattr(lib,name)
    fn.argtypes=args
    fn.restype=result
    return fn
eig=bind("pair95_eigen",[I,D,D,D])
make_centers=bind("cube95_centers",[I,I,I,D,U,D])
prepare=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
release=bind("cube95_destroy",[P],None)
sample=bind("filter95_draw",[P,I,U,U,I,I,I,WORD,B,RETRY])
project=bind("filter95_local",[P,I,I,WORD,I,IDX,I,D])
hardware=np.load(hw/"histograms.npz")
hist=hardware["total_click_histogram"]
hw_cdf=hist.cumsum()/hist.sum()
hardware_error=math.sqrt(math.log(2/(.01/4))/(2*hist.sum()))
rows=[]

def photon_target(case,small_law):
    if case==4:return np.load(totals/"reference.npz")["photon_pmf"]
    if case!=3:return small_law["total"]
    fock=small_law["pnr"]
    return np.bincount(np.indices(fock.shape).sum(axis=0).ravel(),weights=fock.ravel())

def optical_moments(case,truth):
    if case==4:return read(third/"moments.json")["optical"]
    k=np.array([x.bit_count() for x in range(len(truth))])
    center=k@truth
    return center,(k-center)**2@truth,(k-center)**3@truth

def fit_click(prob,moments):
    mean,var,skew=moments
    standardized=(np.arange(len(prob))-mean)/math.sqrt(var)
    degree=min(3,len(prob)-1)
    features=np.column_stack([standardized**k for k in range(degree+1)])
    unit=np.array([1.,0.,1.,skew/var**1.5])[:degree+1]
    for rate in [.75,.5]:
        def mismatch(b):
            return (prob*expit(features@b))@features-rate*unit
        opt=least_squares(mismatch,np.zeros(degree+1),gtol=1e-13,xtol=1e-14,ftol=1e-14)
        if max(abs(opt.fun))<1e-9:
            return expit(features@opt.x),dict(rate=rate,parameters=opt.x.tolist(),standardized_residual=float(max(abs(opt.fun))))
    raise ValueError("infeasible click moments in tested acceptance range")

def box_target(case,m,cutoff):
    ref=np.load(reference/f"law_{case}.npz")
    if case==3:return ref["pnr"][:cutoff+1,:cutoff+1].T.ravel()
    index=np.arange((cutoff+1)**m,dtype=np.uint64)
    keys=np.zeros_like(index)
    for j in range(m):
        keys|=(index%(cutoff+1))<<(8*j)
        index//=cutoff+1
    lookup=ref["keys"]
    pos=np.searchsorted(lookup,keys)
    valid=pos<len(lookup)
    valid[valid]&=lookup[pos[valid]]==keys[valid]
    truth=np.zeros(len(keys))
    truth[valid]=ref["probabilities"][pos[valid]]
    return truth

for case,config in enumerate([0,1,2,3,6]):
    data=np.load(base/f"model_{config}.npz")
    target=np.load(optics/f"optical_{case}.npz")
    corr=target["correlation"]
    m=len(corr)
    small_law=np.load(reference/f"law_{case}.npz") if case<4 else None
    target_photon=photon_target(case,small_law)
    moments=optical_moments(case,small_law["click"] if case<4 else None)
    cap=16 if case==4 else 64
    width=max(64,1<<(m-1).bit_length())
    z=np.ascontiguousarray(target["thresholds"][:,:cap+1])
    begin=perf_counter()
    val=np.empty(m)
    vec=np.empty_like(corr)
    assert eig(m,corr,val,vec)==0
    variance=.98*val[0]
    grid=np.empty((8*width,m))
    assert make_centers(m,width,4,np.ascontiguousarray(vec*np.sqrt(val-variance)),0x950e1000+case,grid)==0
    marginal=np.empty_like(z)
    shifts=np.empty_like(z)
    ptr=prepare(m,len(grid),cap,grid,math.sqrt(variance),z,1,16,marginal,shifts)
    base_prep=perf_counter()-begin
    assert ptr and np.array_equal(data["marginal"],marginal) and np.array_equal(data["shift"],shifts)
    try:
        for route,bound in enumerate([0,1.25,2.,4.]):
            kind=int(route>0)
            source=data["photon_count" if kind else "click_count"]
            begin=perf_counter()
            if kind:
                n=min(len(source),len(target_photon))
                accept=np.divide(target_photon[:n],bound*source[:n],out=np.zeros(n),where=source[:n]>0).clip(0,1)
                info=dict(bound=bound)
            else:
                accept,info=fit_click(source,moments)
            cut=np.rint(accept*2**32).astype(np.uint64)
            weight=cut.astype(float)/2**32
            normalize=source[:len(cut)]@weight
            law=source[:len(cut)]*weight/normalize
            fit_time=perf_counter()-begin
            row=dict(case=case,route=route,pnr=kind,setup_seconds=base_prep,fit_seconds=fit_time,
                     fit=info,acceptance=float(normalize),retry_limit=2048,
                     input_law_missing=max(0.,1-source.sum()))
            total=np.arange(len(law))
            mean=law@total
            row["moments"]=[mean,law@(total-mean)**2,law@(total-mean)**3]
            arrays=dict(accept=cut,count=law)
            if kind:
                n=min(len(law),len(target_photon))
                row["photon_total_ks"]=float(max(abs(law[:n].cumsum()-target_photon[:n].cumsum())))
                row["photon_total_tv"]=float((abs(law[:n]-target_photon[:n]).sum()+max(0.,1-target_photon[:n].sum()))/2)
                row["odd_mass"]=float(law[1::2].sum())
                if m<=4:
                    cutoff=64 if m==2 else 16
                    q=data["box"]
                    index=np.arange(len(q))
                    sums=np.zeros(len(q),int)
                    for j in range(m):
                        sums+=index%(cutoff+1)
                        index//=cutoff+1
                    v=q*np.pad(weight,(0,max(0,int(sums.max()+1)-len(weight))))[sums]/normalize
                    truth=box_target(case,m,cutoff)
                    inside=abs(v-truth).sum()/2
                    a,b=max(0.,1-v.sum()),max(0.,1-truth.sum())
                    slack=0 if case==3 else float(read(reference/f"row_{case}.json")["tail_chernoff"])
                    row["pnr_full_tv"]=[max(0.,inside+abs(a-b)/2-slack),min(1.,inside+(a+b)/2+slack)]
                    arrays["box"]=v
            elif case==4:
                ks=float(max(abs(law.cumsum()-hw_cdf[:len(law)])))
                row["hardware_click_ks"]=ks
                row["hardware_interval"]=[ks-hardware_error,ks+hardware_error]
            groups=hardware["ports"] if case==4 else [np.arange(m)]
            row["local_click_tv"]=[]
            for j,ports in enumerate(groups):
                ports=np.ascontiguousarray(ports,dtype=np.int32)
                joint=np.empty(1<<len(ports))
                assert project(ptr,kind,len(cut),cut,len(ports),ports,16,joint)==0
                assert abs(joint.sum()-normalize)<2e-10
                joint/=normalize
                truth=(np.load(hw/f"input_{j}.npz")["two_common_sectors"] if case==4 else small_law["click"])
                row["local_click_tv"].append(float(abs(joint-truth).sum()/2))
                arrays[f"local_{j}"]=joint
            def draw(n,key,off,workers):
                d=np.empty((n,m),np.uint8)
                tries=np.empty(n,np.uint32)
                assert sample(ptr,n,key,off,kind,workers,len(cut),cut,d,tries)==0
                return d,tries
            shots=262144 if case==4 else 32768
            key=0x95150000+32*case+route
            counts=[]
            retries=[]
            clicks=[]
            digests=[]
            for off in range(0,shots,8192):
                d,tries=draw(8192,key,off,16)
                if off==0:
                    test,t=draw(32,key,7,1)
                    assert np.array_equal(d[7:39],test) and np.array_equal(t,tries[7:39])
                    arrays["first_records"]=d
                counts.append(d.sum(1).astype(np.uint16))
                clicks.append((d>0).sum(1).astype(np.uint16))
                retries.append(tries)
                digests.append(hashlib.sha256(d.tobytes()).hexdigest())
            arrays.update(observed_total=np.concatenate(counts),observed_click=np.concatenate(clicks),
                          attempts=np.concatenate(retries))
            row["records_sha256"]=digests
            row["observed_acceptance"]=float(shots/arrays["attempts"].sum())
            if case==4 and kind:
                emp=np.bincount(arrays["observed_click"],minlength=len(hw_cdf)).cumsum()/shots
                ks=float(max(abs(emp-hw_cdf)))
                radius=hardware_error+math.sqrt(math.log(2/(.01/4))/(2*shots))
                row["hardware_click_ks"]=ks
                row["hardware_interval"]=[max(0.,ks-radius),min(1.,ks+radius)]
            batch=32768 if case==4 else 2048
            timing=[]
            index=0
            time_key=0x95160000+1024*case+64*route
            for rep in range(-1,5):
                workers=([1,16] if rep%2==0 else [16,1]) if case==4 else [1]
                for w in workers:
                    n=256 if rep<0 else batch
                    begin=perf_counter()
                    draw(n,time_key+index,0,w)
                    duration=perf_counter()-begin
                    index+=1
                    if rep>=0:timing.append(dict(workers=w,n=n,rep=rep,seconds=duration))
            row["timing"]=timing
            times=[]
            for off in range(101):
                begin=perf_counter()
                draw(1,time_key+index,off,1)
                times.append(perf_counter()-begin)
            arrays["latency"]=np.asarray(times)
            np.savez_compressed(out/f"filtered_{case}_{route}.npz",**arrays)
            rows.append(row)
            save(out,"summary.json",dict(rows=rows,hardware_radius=hardware_error))
            print(case,route,normalize,row.get("photon_total_ks"),row["local_click_tv"],flush=True)
    finally:
        release(ptr)
manifest(out)
