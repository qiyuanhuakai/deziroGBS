"""Whole-record count check for the 144-bin benchmark and optical-source replay."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
import argparse
import ctypes as ct
import math
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from scipy.special import ndtr
from iteration92_common_v2 import ROOT,H,save,read,manifest
from iteration95_loop_covariance_v1 import thresholds,calibrate
from iteration95_loop_tables_v1 import integer_row,discrete

p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
bench=ROOT/"results/iteration95_loop_frugal144_v1"
actual=ROOT/"results/iteration94_actual_quality_v1";layout=ROOT/"results/iteration83_loss_pilot_v1/input.npz"
ref=ROOT/"results/iteration95_pair_copula_v1/optical_4.npz"
photons=ROOT/"results/iteration84_photon_cdf_v1/reference.npz";binary=ROOT/"results/iteration95_loop_residual_v1/residual.dll"
raw=ROOT.parent/"external_sources/jiuzhang3_iteration81_sample_v1/extracted_3/jiuzhang3/data/high power/high power (1).bin"
sources=[Path(__file__),bench/"audit.npz",bench/"summary.json",ref,photons,binary,layout,
         *[actual/f"input_{s}.npz" for s in (0,1)]]
save(out,"plan.json",dict(hashes={x.relative_to(ROOT).as_posix():H(x) for x in sources},hardware=H(raw),
    projections="144 measured bits, same choice as the full sampler benchmark. Exact optical 8-bit marginals at prefix, middle, end and eight evenly spread groups; all three nonprefix Frugal TVs empirical on 1280 records.",
    accuracy="Entire 144-bit total CDF vs measured records. Exact loop count law, empirical Frugal DKW only under iid assumption. .01/8 per CDF; both raw sample count gaps also saved.",
    replay="Replay full optical covariance and all 662976 detector-pair moments from source. Independently regenerate the 512-photon projection target from the low-rank transfer Gram matrix."))
assert H(raw)=="c786e6e64f76a599479eda75c7fdedf6906a3b41040eec49e5ea89caeb714e74"
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent));lib=ct.CDLL(str(binary))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D=np.ctypeslib.ndpointer(dtype=float,flags="C_CONTIGUOUS")
def bind(name,args,result=I):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
opt=bind("window95_moments",[I,I,D,D,D,D,D,D]);pairs=bind("pair95_optical",[I,I,I,D,D,D,D,D,D,D])
win=bind("window95_prepare",[I,I,I,I,D,D,D],P);law=bind("window95_law",[P,D]);free=bind("window95_destroy",[P],None)
centers=bind("cube95_centers",[I,I,I,D,U,D]);prep=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
counts=bind("cube95_photon_law",[P,I,I,D]);release=bind("cube95_destroy",[P],None)
q=np.empty((2,2304,2304));mu=np.empty((2,2304));m=1152;perm=np.argsort(np.load(layout)["order"])
sectors=[dict(np.load(actual/f"input_{i}.npz")) for i in range(2)]
start=perf_counter()
for s,z in enumerate(sectors):
    assert opt(m,len(z["t"]),np.ascontiguousarray(z["u"][perm]).view(float),z["t"],z["alpha"].view(float),z["noise"],q[s],mu[s])==0
qsec=perf_counter()-start
start=perf_counter();single=np.empty(m);joint=np.empty((m,m));pn_cov=np.empty_like(joint);local=np.empty((m,42));cuts=np.empty_like(local)
assert pairs(m,2,41,q,mu,single,joint,local,cuts,pn_cov)==0
pairsec=perf_counter()-start
old=np.load(ref)
replay=max(abs(single-old["single"]).max(),abs(joint-old["joint"]).max(),abs(pn_cov-old["photon_cov"]).max())
assert replay<1e-12
start=perf_counter();ev=[]
for z in sectors:
    u=z["u"];t=z["t"];c=z["noise"]
    assert not z["alpha"].any()
    transfer=np.block([[u.real,-u.imag],[u.imag,u.real]])
    core=2*c@c.T+np.diag(np.r_[t/(1-t),-t/(1+t)])
    l=np.linalg.cholesky(transfer.T@transfer);ev.extend(np.linalg.eigvalsh(l.T@core@l))
ev=np.asarray(ev);beta=ev/(1+ev)
derivative=np.array([.5*np.sum(beta**k) for k in range(1,513)])
ph=np.zeros(513);ph[0]=np.exp(-.5*np.log1p(ev).sum())
for k in range(1,513):ph[k]=derivative[:k]@ph[k-1::-1]/k
pgftime=perf_counter()-start
oracle=np.load(photons)["photon_pmf"][:513];e=float(abs(ph-oracle).max());assert e<1e-13
data=np.load(bench/"audit.npz");port_bits=np.arange(144)*8
cov=joint[np.ix_(port_bits,port_bits)]-np.outer(single[port_bits],single[port_bits])
coarse=cov.reshape(18,8,18,8).sum((1,3))
bits=(np.arange(256)[:,None]>>np.arange(8))&1;pop=bits.sum(1);truth=data["truth"]
port=np.stack([np.bincount(pop,weights=v,minlength=9) for v in truth])
cuts=thresholds(port,True);corr,_=calibrate(cuts,coarse);eig,vec=np.linalg.eigh(corr);d=.98*eig[0]
grid=np.empty((512,18));assert centers(18,64,4,np.ascontiguousarray(vec*np.sqrt(eig-d)),0x95212000,grid)==0
first=np.empty_like(cuts);shift=np.empty_like(cuts)
ptr=prep(18,len(grid),8,grid,math.sqrt(d),cuts,1,1,first,shift);assert ptr
global_law=np.empty(145);assert counts(ptr,144,4,global_law)==0;release(ptr)
assert abs(global_law.sum()-1)<1e-12
tail=np.rint(ndtr((grid[:,:,None]-cuts-shift)/math.sqrt(d))*2**32).clip(0,2**32-1)/2**32
pmf=np.concatenate([np.ones((512,18,1)),tail],2)-np.concatenate([tail,np.zeros((512,18,1))],2)
kernel=np.zeros((18,10,256))
for g in range(18):
    for k in range(9):
        v=truth[g]*(pop==k);kernel[g,k]=discrete(integer_row(v if v.sum()>0 else pop==k))
samples={name:np.concatenate([data[f"{name}_{j}"] for j in range(5)]) for name in ["frugal","loop"]}
selectors=[np.arange(8),np.arange(65,73),np.arange(136,144),18*np.arange(8)]
projection=[];save_laws={}
for which,sel in enumerate(selectors):
    physical=port_bits[sel];ix=np.r_[physical,physical+m]
    plan=win(8,8,0,2,np.ascontiguousarray(q[:,ix][:,:,ix]),np.ascontiguousarray(mu[:,ix]),np.zeros(4));assert plan
    exact=np.empty(256);assert law(plan,exact)==0;free(plan)
    prob=np.ones((512,256))
    for g in np.unique(sel//8):
        where=np.flatnonzero(sel//8==g);inside=bits[:,sel[where]%8]@(1<<np.arange(len(where)))
        outcode=bits[:,where]@(1<<np.arange(len(where)))
        lut=np.stack([kernel[g][:,inside==j].sum(1) for j in range(1<<len(where))])
        prob*=(pmf[:,g]@lut.T)[:,outcode]
    fitted=prob.mean(0);assert abs(fitted.sum()-1)<1e-11
    row=dict(ports=physical.tolist(),new_deterministic_tv=float(abs(fitted-exact).sum()/2))
    for name,x in samples.items():
        emp=np.bincount(x[:,sel]@(1<<np.arange(8)),minlength=256)/len(x)
        row[name+"_empirical_tv"]=float(abs(emp-exact).sum()/2)
    if which==0:row["frugal_deterministic_tv"]=float(abs(data["frugal"]-exact).sum()/2)
    projection.append(row);save_laws[f"true_{which}"]=exact;save_laws[f"new_{which}"]=fitted
hw=np.memmap(raw,dtype=np.uint8,mode="r").reshape(-1,144);hist=np.zeros(145,np.int64)
for offset in range(0,len(hw),65536):
    total=(hw[offset:offset+65536]>>7).sum(1).astype(int)
    hist+=np.bincount(total,minlength=145)
ecdf=hist.cumsum()/len(hw);n=1280
radius=lambda n:math.sqrt(math.log(2/(.01/8))/(2*n))
results={}
for name,x in samples.items():
    h=np.bincount(x.sum(1).astype(int),minlength=145)
    save_laws[name+"_hist"]=h
    empirical=float(max(abs(h.cumsum()/len(x)-ecdf)))
    gap=float(max(abs(global_law.cumsum()-ecdf))) if name=="loop" else empirical
    err=radius(len(hw))+(radius(len(x)) if name=="frugal" else 0.)
    results[name]=dict(ks=gap,observed_ks=empirical,interval=[max(0,gap-err),min(1,gap+err)],
                       mean=float(x.sum(1).mean()),variance=float(x.sum(1).var()))
np.savez_compressed(out/"laws.npz",**save_laws,global_law=global_law,hardware=hist,photon_target_replay=ph)
save(out,"validation.json",dict(passed=True,full_source_q_seconds=qsec,full_all_pair_seconds=pairsec,full_pair_replay=float(replay),
                                full_photon_ref_seconds=pgftime,photon_pmf_max_error=e,hardware_records=len(hw),
                                optical_subset_mean=float(single[port_bits].sum()),optical_subset_variance=float(cov.sum()),
                                projections=projection,full_subset_count=results))
manifest(out)
