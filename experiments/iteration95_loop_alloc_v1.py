"""Physical-loop count sampling, exact low-order allocation, and conserved totals."""
import os
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[k]="1"
import argparse
import ctypes as ct
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit,ndtri
from iteration92_common_v2 import ROOT,H,save,read,manifest
from iteration95_loop_tables_v1 import WORD,make_optics,discrete,mask_given_photons,fock_probabilities

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
prior=ROOT/"results/iteration95_pair_table_v1"
filters=ROOT/"results/iteration95_count_filter_v1"
spatial=ROOT/"results/iteration95_spatial_optical_v1"
actual=ROOT/"results/iteration94_actual_quality_v1"
small=ROOT/"results/iteration93_reference_v2"
photon_ref=ROOT/"results/iteration84_photon_cdf_v1/reference.npz"
hw=ROOT/"results/iteration81_jiuzhang_accuracy_v3/histograms.npz"
third=ROOT/"results/iteration95_third_click_v1/moments.json"
layout=ROOT/"results/iteration83_loss_pilot_v1/input.npz"
source=ROOT/"native/gaussian_loop_alloc95_v1.cpp"
files=[Path(__file__),Path(__file__).with_name("iteration95_loop_tables_v1.py"),source,layout,photon_ref,hw,third,
       spatial/"loops.npz",*[actual/f"input_{s}.npz" for s in (0,1)],
       *[small/f"{name}_{j}.npz" for name in ("law","input") for j in range(4)]]
files.extend(ROOT/k for k in read(filters/"plan.json")["hashes"] if k.startswith("native/"))
save(out,"plan.json",dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},cases=[0,1,2,3,4],
    groups="Each physical loop is eight consecutive detector bins, single entire group on small fixtures. Decomposition supports groups of one to eight bins with ragged last group.",
    dependence="Invert exact probabilities of two nonempty groups, then four antithetic frames of 256 Hadamard axes at 144 groups; no copula approximation for one group.",
    pnr="Group-total PGF, bivariate sibling PGFs at all tree nodes, exact multivariate count coefficients through total 12 (four or two modes), 6 (eight modes) or 4 (1152). Above that order split by exact sibling totals. Overflow root above 64 small or 40 large represented by cap+1.",
    click="Exact 256-pattern optical law per group; sample conditional on each group click count. Output direct route never draws a photon count.",
    methods="raw direct click, direct calibrated to three optical click moments, raw PNR allocation, PNR global optical-total likelihood ratio capped at 2; also tree-only PNR ablation. No hardware fit.",
    quality="65536 small or 131072 large; save all total/click/retry and hashes, first 8192 records. Deterministic local mask and count laws, entire PNR box on 2/4, empirical PNR-to-click total only. Compare hardware with shared DKW conditional on iid.",
    timing="five alternating repetitions per batch after warmup; 2048 small, 32768 full; expanded output, group-count and attempts array allocations included. Median of 101 separate one-shot calls.",
    domains="quality 0x951a0000+32*case+method, timing 0x951b0000+1024*case+64*method+call, center 0x951c0000+case; purposes 123/124/125/126/130/131"))
cmd=read(prior/"build.json")["command"]
cmd[-3]=str(source);cmd[-1]=str(out/"alloc.dll");cmd.insert(2,"-fopenmp")
run=subprocess.run(cmd,capture_output=True,text=True)
save(out,"build.json",dict(command=cmd,code=run.returncode,stderr=run.stderr))
run.check_returncode()
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
dll=ct.CDLL(str(out/"alloc.dll"))
I,U,P,F=ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_double
D,B,W,T,J=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32,np.int32)]
def bind(name,args,result=I):
    f=getattr(dll,name);f.argtypes=args;f.restype=result
    return f
lib=SimpleNamespace(radial=bind("alloc95_radial",[I,I,I,I,I,D,D,D]),low=bind("alloc95_low",[I,I,I,I,D,D,D,D]),
    window=bind("window95_prepare",[I,I,I,I,D,D,D],P),window_law=bind("window95_law",[P,D]),
    window_free=bind("window95_destroy",[P],None))
optical=bind("window95_moments",[I,I,D,D,D,D,D,D])
invert=bind("pair95_invert",[I,D,D,D,D])
center=bind("cube95_centers",[I,I,I,D,U,D])
prepare=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
release=bind("cube95_destroy",[P],None)
sum_law=bind("cube95_photon_law",[P,I,I,D])
marginal=bind("alloc95_marginals",[P,I,W,I,D])
sample=bind("alloc95_draw",[P,I,U,U,I,I,I,W,I,I,I,W,W,B,T,B,I,J,W,T])
observed=np.load(hw)["total_click_histogram"]
hardware=observed.cumsum()/observed.sum()
hw_loops=np.load(spatial/"loops.npz")["hardware_counts"]
hw_loop=hw_loops.cumsum(1)/hw_loops.sum(1)[:,None]
alpha=.01/8
rad=lambda n:math.sqrt(math.log(2/alpha)/(2*n))
def fit_count(law,mom):
    mu,var,skew=mom
    if len(law)<=4: degree=len(law)-1
    else:degree=3
    x=(np.arange(len(law))-mu)/math.sqrt(var)
    f=np.column_stack([x**k for k in range(degree+1)])
    target=np.array([1.,0.,1.,skew/var**1.5])[:degree+1]
    for accept in (.75,.5,.25):
        solve=least_squares(lambda b:(law*expit(f@b))@f-accept*target,np.zeros(degree+1),
                            gtol=1e-13,ftol=1e-14,xtol=1e-14)
        if max(abs(solve.fun))<1e-8:return expit(f@solve.x)
    raise ValueError("infeasible moment fit")
def true_small_pnr(case):
    v=np.load(small/f"law_{case}.npz")
    if case==3:
        n=np.indices(v["pnr"].shape).reshape(2,-1).T
        return n,v["pnr"].ravel()
    keys=v["keys"]
    return np.column_stack([((keys>>(8*j))&255).astype(int) for j in range(4)]),v["probabilities"]
rows=[];setups=[]
for case in range(5):
    start=perf_counter()
    paths=[actual/f"input_{s}.npz" for s in (0,1)] if case==4 else [small/f"input_{case}.npz"]
    m=len(np.load(paths[0])["u"]);width=min(8,m);ng=math.ceil(m/width)
    groups=[np.arange(j,min(j+width,m)) for j in range(0,m,width)]
    q=np.empty((len(paths),2*m,2*m));mu=np.empty((len(paths),2*m))
    projection=[]
    for s,path in enumerate(paths):
        data=np.load(path);u=np.ascontiguousarray(data["u"])
        assert optical(m,len(data["t"]),u.view(float),data["t"],data["alpha"].view(float),data["noise"],q[s],mu[s])==0
    if case==4:
        perm=np.argsort(np.load(layout)["order"]);ix=np.r_[perm,perm+m]
        q=np.ascontiguousarray(q[:,ix][:,:,ix]);mu=np.ascontiguousarray(mu[:,ix])
        sing=np.linalg.svd(u[perm].reshape(144,8,-1),compute_uv=False)
        projection=(1-sing[:,0]**2/(sing**2).sum(1)).tolist()
    cap=40 if case==4 else 64
    degree=4 if case==4 else 6 if m==8 else 12
    stride=cap+2
    items,single,joint=make_optics(lib,q,mu,groups,degree,cap)
    optical_secs=perf_counter()-start
    masks=np.stack([v["mask"] for v in items]);split=np.stack([v["split"] for v in items])
    offsets=[];low=[];patterns=[]
    off=0
    for v in items:
        for c,pat in zip(v["low"],v["pattern"]):
            offsets.append(off);low.extend(c);patterns.extend(pat);off+=len(c)
        offsets.append(off)
    offsets=np.asarray(offsets,np.int32);low=np.asarray(low,np.uint64);patterns=np.asarray(patterns,np.uint32)
    root=np.stack([v["root"] for v in items])
    click=np.stack([v["click_count"] for v in items])
    truth=np.stack([v["click"] for v in items])
    if case==4:
        independent=np.load(spatial/"loops.npz")["optical_click"]
    else:independent=np.load(small/f"law_{case}.npz")["click"][None,:]
    assert abs(truth-independent).max()<1e-11
    start=perf_counter()
    if ng==1:
        grid=np.zeros((1,1));variance=1.;corr=np.eye(1);min_eigen=1.
    else:
        corr=np.empty((ng,ng))
        assert invert(ng,np.ascontiguousarray(-ndtri(single)),single,joint,corr)==0
        eig,vec=np.linalg.eigh(corr);min_eigen=float(eig.min())
        assert min_eigen>0
        variance=.98*eig[0];axes=1<<(ng-1).bit_length()
        grid=np.empty((8*axes,ng))
        assert center(ng,axes,4,np.ascontiguousarray(vec*np.sqrt(eig-variance)),0x951c0000+case,grid)==0
    dependence_secs=perf_counter()-start
    kernels=[]
    for item,g in zip(items,groups):
        kernels.append(mask_given_photons(len(g),stride,item["split"],degree,item["low"],item["pattern"]))
    setups.append(dict(case=case,groups=ng,centers=len(grid),cap=cap,degree=degree,optical_seconds=optical_secs,
                       dependence_seconds=dependence_secs,min_eigen=min_eigen,projection_residual=projection,
                       coefficient_min=min(x["min_coeff"] for x in items),low_mass_error=max(x["mass_error"] for x in items),
                       imag_error=max(x["max_imag"] for x in items),optical_overflow_union=max(0.,1-root[:,:cap+1].sum(1).min())*ng,
                       photon_tail_above_exact_order_max=float(max(1-root[:,:degree+1].sum(1)))))
    arrays=dict(root=root,click=click,truth=truth,split=split,mask=masks,offset=offsets,low=low,pattern=patterns,
                single=single,joint=joint,corr=corr,grid=grid,variance=variance,pnr_mask_kernel=np.stack(kernels))
    if case in (0,1,3):
        fock,target=true_small_pnr(case)
        idx=fock.sum(1)<=cap;fock=fock[idx];target=target[idx]
    photon_target=(np.load(photon_ref)["photon_pmf"] if case==4 else root[0])
    moments=(read(third)["optical"] if case==4 else None)
    for kind in (0,1):
        start=perf_counter()
        port= root[:,:cap+1] if kind else click
        # Compute upper tails directly, preserving small high-count probabilities.
        tail=np.concatenate((port[:,1:],np.maximum(0.,1-port.sum(1))[:,None]),axis=1)[:,::-1].cumsum(1)[:,::-1]
        if not kind:tail[:,width:]=0.
        cuts=np.ascontiguousarray(-ndtri(tail.clip(0,1)))
        first=np.empty_like(cuts);shift=np.empty_like(cuts)
        ptr=prepare(ng,len(grid),port.shape[1]-1,grid,math.sqrt(variance),cuts,1,16,first,shift)
        assert ptr
        limit=(512 if kind else min(m,384)) if case==4 else (cap+1 if kind else m)
        proposal=np.empty(limit+1)
        assert sum_law(ptr,limit,16,proposal)==0 and abs(proposal.sum()-1)<1e-8
        table_secs=perf_counter()-start
        arrays[f"proposal_{kind}"]=proposal
        for mode in ([2,3,4] if kind else [0,1]):
            use_filter=mode in (1,3)
            low_degree=0 if mode==4 else degree
            start=perf_counter()
            if mode==1 and case==4:
                acceptance=fit_count(proposal,moments)
            elif mode==3:
                ref=photon_target[:len(proposal)]
                acceptance=np.divide(ref,2*proposal,out=np.zeros_like(proposal),where=proposal>0).clip(0,1)
            else:
                acceptance=np.ones(len(proposal))
            cut=np.rint(WORD*acceptance).astype(np.uint64)
            rate=proposal@(cut/WORD)
            count_law=proposal*(cut/WORD)/rate
            fit_secs=perf_counter()-start
            one=np.zeros((ng,port.shape[1]+1))
            assert marginal(ptr,len(cut),cut,12,one)==0 and max(abs(one.sum(1)-rate))<2e-9
            one/=rate
            local=[]
            for g,item in enumerate(items):
                if kind:
                    kernel=kernels[g] if low_degree else mask_given_photons(len(groups[g]),stride,item["split"],0,item["low"],item["pattern"])
                    local.append(one[g]@kernel)
                else:
                    kernel=np.zeros((one.shape[1],len(item["click"])))
                    for n in range(len(groups[g])+1):kernel[n]=discrete(item["mask"][n,:len(item["click"])])
                    local.append(one[g]@kernel)
            local=np.stack(local)
            tv=np.abs(local-truth).sum(1)/2
            row=dict(case=case,mode=mode,table_seconds=table_secs,fit_seconds=fit_secs,rate=float(rate),
                     max_loop_click_tv=float(tv.max()),median_loop_click_tv=float(np.median(tv)),
                     local_photon_count_tv_max=float((abs(one[:,:-1]-port).sum(1)+abs(one[:,-1]-np.maximum(0,1-port.sum(1)))).max()/2) if kind else None,
                     overflow_model_union=float(one[:,-1].sum()),marginal_sha=hashlib.sha256(one.tobytes()).hexdigest())
            mode_out=dict(count_law=count_law,accept=cut,local=local,group_marginal=one,loop_tv=tv)
            if kind:
                reference=photon_target[:len(count_law)]
                row["photon_ks"]=float(max(abs(count_law.cumsum()-reference.cumsum())))
                if case in (0,1,3):
                    v=fock_probabilities(fock,one[0],split[0],low_degree,items[0]["low"],items[0]["pattern"])
                    err=abs(v-target).sum()/2
                    a,b=max(0.,1-v.sum()),max(0.,1-target.sum())
                    row["full_pnr_tv"]=[max(0.,err+abs(a-b)/2),min(1.,err+(a+b)/2)]
            N=131072 if case==4 else 65536
            def draw(n,key,off,threads):
                data=np.empty((n,m),np.uint8);attempt=np.empty(n,np.uint32);tot=np.empty((n,ng),np.uint8)
                rc=sample(ptr,n,key,off,kind,threads,len(cut) if use_filter else 0,cut,width,len(groups[-1]),
                          stride,split,masks,data,attempt,tot,low_degree,offsets,low,patterns)
                assert rc==0,rc
                return data,attempt,tot
            sample_key=0x951a0000+32*case+mode
            observed_click=[];observed_count=[];hashes=[];total_attempts=0;overflows=0
            for offset in range(0,N,8192):
                records,attempt,coarse=draw(8192,sample_key,offset,16)
                assert np.array_equal(records.reshape(len(records),ng,width).sum(2),coarse)
                if offset==0:
                    assert np.array_equal(draw(17,sample_key,21,1)[0],records[21:38])
                    mode_out["first"]=records
                hashes.append(hashlib.sha256(records.tobytes()).hexdigest())
                observed_click.append((records>0).sum(1).astype(np.uint16))
                observed_count.append(records.sum(1).astype(np.uint16))
                total_attempts+=attempt.sum();overflows+=np.count_nonzero((coarse>cap).any(1)) if kind else 0
            mode_out["observed_count"]=np.concatenate(observed_count)
            mode_out["observed_click"]=np.concatenate(observed_click)
            row.update(records=N,records_sha=hashes,observed_rate=float(N/total_attempts),overflows=int(overflows))
            if case==4:
                cdf=(np.bincount(mode_out["observed_click"],minlength=m+1).cumsum()/N if kind else count_law.cumsum())
                gap=float(max(abs(cdf-hardware[:len(cdf)])))
                row["hardware_ks"]=gap
                radius=rad(observed.sum())+(rad(N) if kind else 0.)
                row["hardware_interval"]=[max(0.,gap-radius),min(1.,gap+radius)]
                pop=np.array([k.bit_count() for k in range(256)])
                loop_cdf=np.stack([local[:,pop<=k].sum(1) for k in range(9)],axis=1)
                gaps=abs(loop_cdf-hw_loop).max(1)
                row.update(physical_loop54_ks=float(gaps[54]),physical_loop54_tv=float(tv[54]),hardware_worst_loop=int(gaps.argmax()))
            clocks=[];index=0;batch=32768 if case==4 else 2048
            key=0x951b0000+1024*case+64*mode
            for rep in range(-1,5):
                for workers in (([1,16] if rep%2==0 else [16,1]) if case==4 else [1]):
                    n=128 if rep<0 else batch
                    before=perf_counter();draw(n,key+index,0,workers);elapsed=perf_counter()-before;index+=1
                    if rep>=0:clocks.append(dict(n=n,workers=workers,seconds=elapsed,rep=rep))
            lat=[]
            for offset in range(101):
                before=perf_counter();draw(1,key+index,offset,1);lat.append(perf_counter()-before)
            row["timing"]=clocks;mode_out["latency"]=np.array(lat)
            np.savez_compressed(out/f"output_{case}_{mode}.npz",**mode_out)
            rows.append(row)
            save(out,"summary.json",dict(rows=rows,setups=setups))
            print(case,mode,row.get("full_pnr_tv"),row["max_loop_click_tv"],flush=True)
        release(ptr)
    np.savez_compressed(out/f"setup_{case}.npz",**arrays)
manifest(out)
