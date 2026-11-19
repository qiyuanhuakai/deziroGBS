"""Compare physical-loop count calibration with bounded and conditional sampling."""
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

import numpy as np
from scipy.special import ndtr
from iteration92_common_v2 import ROOT,H,save,read,manifest
from iteration95_loop_covariance_v1 import thresholds,calibrate
from iteration95_loop_tables_v1 import discrete

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/"results/iteration95_loop_alloc_v1"
old=ROOT/"results/iteration95_pair_copula_v1/optical_4.npz"
hardware=ROOT/"results/iteration81_jiuzhang_accuracy_v3/histograms.npz"
photon=ROOT/"results/iteration84_photon_cdf_v1/reference.npz"
click=ROOT/"results/iteration95_third_click_v1/quality_3.npz"
spatial=ROOT/"results/iteration95_spatial_optical_v1/loops.npz"
src=ROOT/"native/gaussian_loop_residual95_v1.cpp"
files=[Path(__file__),Path(__file__).with_name("iteration95_loop_covariance_v1.py"),src,old,hardware,photon,click,spatial,
       base/"setup_4.npz",base/"manifest.json",*[ROOT/k for k in read(base/"plan.json")["hashes"] if k.startswith("native/")]]
save(out,"plan.json",dict(hashes={x.relative_to(ROOT).as_posix():H(x) for x in files},
    targets="144 loop count covariances from all 1152 optical ports. Each loop count marginal from the optical PGF or full 8-bit optical law. Binary64 Hermite expansion order 96; independent Plackett quadrature on highest 16 and 24 deterministic spread pairs.",
    calibration="Direct output count teacher is the previous optical-only three-moment window PMF (truncated at 384); photon output count target is the independent Gaussian total-photon PMF (truncated at 512). Preserve proposal allocation conditional on total.",
    algorithm="bulk cap 1.25; use conditional subtree sampling for excess target mass; measure raw plus corrected models. Discarded optical PMF beyond limits reported.",
    quality="262144 fresh expanded records per model, same counts conserved per loop, exact finite-proposal local mask laws and exact reweighted loop count marginals; DKW bands allocating .01/8 per hardware or PNR-to-click CDF.",
    rng="0x951d0000+16*kind+corrected quality, 0x951e0000+128*kind+32*corrected+call timing, 0x951c0004 for common frame signs. Independent residual route choice and conditional allocation use purposes 132 and 133.",
    timing="five repeated batches of 32768, alternating one/16 workers; allocation and physical expansion included; 101 one-shot calls"))
cmd=read(base/"build.json")["command"];cmd[-3]=str(src);cmd[-1]=str(out/"residual.dll")
comp=subprocess.run(cmd,capture_output=True,text=True)
save(out,"build.json",dict(command=cmd,code=comp.returncode,stderr=comp.stderr));comp.check_returncode()
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent));dll=ct.CDLL(str(out/"residual.dll"))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,B,W,T,J=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32,np.int32)]
def bind(name,args,result=I):
    fn=getattr(dll,name);fn.argtypes=args;fn.restype=result
    return fn
center=bind("cube95_centers",[I,I,I,D,U,D]);prepare=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
release=bind("cube95_destroy",[P],None)
raw_draw=bind("cube95_sample",[P,I,U,U,I,I,B])
prep_res=bind("residual95_prepare",[P,I,D,F,I,D,D],P);free_res=bind("residual95_free",[P],None)
reweighted=bind("residual95_marginals",[P,I,D])
res_draw=bind("residual95_draw",[P,I,U,U,I,B,T,B])
expand=bind("residual95_expand",[I,I,I,I,I,U,U,I,I,W,W,I,J,W,T,B,B])
model=np.load(base/"setup_4.npz");op=np.load(old)
truth=model["truth"];split=model["split"];mask=model["mask"]
offsets=model["offset"];low=model["low"];patterns=model["pattern"]
h=np.load(hardware);hw_hist=h["total_click_histogram"];hw_cdf=hw_hist.cumsum()/hw_hist.sum()
hw_spatial=np.load(spatial)["hardware_counts"];hw_loop=hw_spatial.cumsum(1)/hw_spatial.sum(1)[:,None]
bits=(np.arange(256)[:,None]>>np.arange(8))&1;pop=bits.sum(1)
fiber=np.arange(1152)//8
groups=h["ports"]
small_opt=[np.load(hardware.with_name(f"input_{j}.npz"))["two_common_sectors"] for j in range(3)]
family=.01/8
radius=lambda n:math.sqrt(math.log(2/family)/(2*n))
rows=[]
for kind in [0,1]:
    start=perf_counter()
    port=np.ascontiguousarray(model["root"][:,:41] if kind else model["click"])
    cuts=thresholds(port,not kind)
    matrix=op["photon_cov"] if kind else op["joint"]-np.outer(op["single"],op["single"])
    target=matrix.reshape(144,8,144,8).sum((1,3))
    corr,stats=calibrate(cuts,target)
    eig,vec=np.linalg.eigh(corr)
    assert eig.min()>0
    d=.98*eig[0];grid=np.empty((2048,144))
    assert center(144,256,4,np.ascontiguousarray(vec*np.sqrt(eig-d)),0x951c0004,grid)==0
    first=np.empty_like(cuts);shifts=np.empty_like(cuts)
    ptr=prepare(144,2048,port.shape[1]-1,grid,math.sqrt(d),cuts,1,16,first,shifts)
    assert ptr
    fit_secs=perf_counter()-start
    # Independent evaluation of the untruncated covariance integral.
    ii,jj=np.triu_indices(144,1);selected=np.unique(np.r_[np.argsort(corr[ii,jj])[-16:],np.linspace(0,len(ii)-1,24).astype(int)])
    nodes,weights=np.polynomial.legendre.leggauss(32);independent=[]
    for n in selected:
        a,b=ii[n],jj[n];u=cuts[a,np.isfinite(cuts[a])][:,None];v=cuts[b,np.isfinite(cuts[b])][None,:]
        angle=np.arcsin(corr[a,b]);total=0.
        for z,w in zip((nodes+1)*angle/2,weights):
            total+=w*np.exp(-(u*u+v*v-2*u*v*np.sin(z))/(2*np.cos(z)**2)).sum()
        independent.append(abs(total*angle/(4*np.pi)-target[a,b]))
    stats.update(quadrature_error=max(independent),min_eigen=float(eig.min()))
    assert max(independent)<1e-10
    tail=np.rint(ndtr((grid[:,:,None]-(cuts+shifts)[None,:,:])/math.sqrt(d))*2**32).clip(0,2**32-1)/2**32
    pmf=np.concatenate((np.ones((*tail.shape[:2],1)),tail),axis=2)-np.concatenate((tail,np.zeros((*tail.shape[:2],1))),axis=2)
    assert pmf.min()>=0
    mean=tail.sum(2)
    empirical=(mean-mean.mean(0)).T@(mean-mean.mean(0))/len(grid)
    stats["finite_cov_rms"]=float(np.sqrt(np.mean((empirical[ii,jj]-target[ii,jj])**2)))
    stats["finite_cov_relative"]=float(np.linalg.norm(empirical[ii,jj]-target[ii,jj])/np.linalg.norm(target[ii,jj]))
    kernel=np.zeros((144,pmf.shape[2],256))
    if kind:kernel[:]=model["pnr_mask_kernel"]
    else:
        for g in range(144):
            for k in range(9):kernel[g,k]=discrete(mask[g,k])
    truth_total=np.load(photon if kind else click)["photon_pmf" if kind else "count"]
    limit=512 if kind else 384;mass=truth_total[:limit+1].copy()
    tail_mass=float(truth_total[limit+1:].sum());mass=np.maximum(mass,0);mass/=mass.sum()
    proposal=np.empty_like(mass);diag=np.zeros(4)
    before=perf_counter();rp=prep_res(ptr,limit,mass,1.25,16,proposal,diag);prep_time=perf_counter()-before
    assert rp
    corr_group=np.empty((144,port.shape[1]+1))
    assert reweighted(rp,16,corr_group)==0 and abs(corr_group.sum(1)-1).max()<1e-9
    # Compute the emitted total law, including integer acceptance and the
    # finite 53-bit CDF for the residual branch. Subtotals conserve every draw.
    prob=np.divide(mass,1.25*proposal,out=np.zeros_like(mass),where=proposal>0).clip(0,1)
    bulk=proposal*(np.floor(prob*2**32)/2**32)
    rem=np.maximum(0,mass-1.25*bulk);res_mass=rem.sum()
    assert abs(diag[0]-res_mass)<1e-12
    rounded=np.ceil(2**53*(rem.cumsum()/res_mass).clip(0,1))/2**53
    rounded[-1]=1.
    bulk_mix=np.ceil(2**53*(1-diag[0]))/2**53
    final=bulk_mix*bulk/bulk.sum()+(1-bulk_mix)*np.diff(np.r_[0.,rounded])
    # Independently reconstruct cross-loop binary marginals of the raw model.
    conditional=np.einsum("cgt,gtb->cgb",pmf,kernel@bits,optimize=True).reshape(len(grid),1152)
    first_bit=conditional.mean(0);joint=conditional.T@conditional/len(grid)
    raw_local=np.einsum("gt,gtb->gb",pmf.mean(0),kernel)
    for g in range(144):
        sl=slice(8*g,8*g+8)
        joint[sl,sl]=(bits.T*raw_local[g])@bits
    a,b=np.triu_indices(1152,1)
    dm=first_bit-op["single"];dj=(joint-op["joint"])[a,b]
    tv=(abs(dj)+abs(dm[a]-dj)+abs(dm[b]-dj)+abs(dj-dm[a]-dm[b]))/2
    stats.update(max_pair_tv=float(tv.max()),worst_pair=[int(a[tv.argmax()]),int(b[tv.argmax()])])
    # Exact raw 8-port laws across physical fibers, using conditional independence.
    projected=[]
    for ports in groups:
        law=np.ones((len(grid),256))
        for g in np.unique(fiber[ports]):
            sel=np.where(fiber[ports]==g)[0]
            code=bits[:,sel]@(1<<np.arange(len(sel)))
            true_code=bits[:,ports[sel]%8]@(1<<np.arange(len(sel)))
            lut=np.array([kernel[g,:,true_code==v].sum(0) for v in range(1<<len(sel))])
            law*= (pmf[:,g]@lut.T)[:,code]
        projected.append(law.mean(0))
    common=dict(corr=corr,grid=grid,cuts=cuts,shifts=shifts,proposal=proposal,target=mass,corrected=final,
                loop_cov=target,raw_first=first_bit,raw_second=joint,raw_projected=projected,diag=diag)
    np.savez_compressed(out/f"model_{kind}.npz",**common)
    for corrected in [0,1]:
        count=final if corrected else proposal
        group_law=corr_group if corrected else pmf.mean(0)
        local=np.einsum("gt,gtb->gb",group_law,kernel)
        tv_loop=abs(local-truth).sum(1)/2
        lc=np.stack([local[:,pop<=k].sum(1) for k in range(9)],axis=1)
        row=dict(pnr=kind,corrected=corrected,fit_seconds=fit_secs,residual_prep_seconds=prep_time,
                 calibration=stats,excess_mass=float(diag[0]),bulk_acceptance=float(diag[1]),tree_bytes=int(diag[3]),
                 target_tail=tail_mass,count_tv=float((abs(count-truth_total[:len(count)]).sum()+tail_mass)/2),
                 count_ks=float(max(abs(count.cumsum()-truth_total[:len(count)].cumsum()))),
                 max_loop_tv=float(tv_loop.max()),median_loop_tv=float(np.median(tv_loop)),loop54_hw=float(max(abs(lc[54]-hw_loop[54]))),
                 overflow_union=float(group_law[:,-1].sum()) if kind else 0.)
        if not corrected:row["raw_projected_tv"]=[float(abs(x-y).sum()/2) for x,y in zip(projected,small_opt)]
        def draw(n,key,off,workers):
            total=np.empty((n,144),np.uint8);data=np.empty((n,1152),np.uint8)
            retry=np.ones(n,np.uint32);branch=np.zeros(n,np.uint8)
            if corrected: assert res_draw(rp,n,key,off,workers,total,retry,branch)==0
            else: assert raw_draw(ptr,n,key,off,1,workers,total)==0
            assert expand(144,n,8,8,kind,key,off,workers,42,split,mask,4,offsets,low,patterns,total,data)==0
            return data,total,retry,branch
        N=262144;key=0x951d0000+16*kind+corrected
        digits=[];sums=[];branches=[];attempts=[];hashes=[]
        histogram=np.zeros((3,256),np.int64)
        result=dict(local=local,group_law=group_law)
        for off in range(0,N,8192):
            data,total,retry,branch=draw(8192,key,off,16)
            assert np.array_equal(data.reshape(-1,144,8).sum(2),total)
            if off==0:
                assert np.array_equal(draw(23,key,13,1)[0],data[13:36])
                result["first"]=data
            hashes.append(hashlib.sha256(data.tobytes()).hexdigest())
            record=data>0
            for j,ports in enumerate(groups):
                histogram[j]+=np.bincount(record[:,ports]@(1<<np.arange(8)),minlength=256)
            digits.append(record.sum(1).astype(np.uint16));sums.append(data.sum(1).astype(np.uint16))
            branches.append(branch);attempts.append(retry)
        result.update(click=np.concatenate(digits),total=np.concatenate(sums),branch=np.concatenate(branches),attempts=np.concatenate(attempts),groups=histogram)
        emp=np.bincount(result["click"],minlength=1153).cumsum()/N
        actual=(emp if kind else count.cumsum())
        gap=float(max(abs(actual-hw_cdf[:len(actual)])))
        error=radius(hw_hist.sum())+(radius(N) if kind else 0.)
        row.update(hardware_click_ks=gap,hardware_99=[max(0.,gap-error),min(1.,gap+error)],observed_excess=float(result["branch"].mean()),
                   observed_projected_tv=[float(abs(x/N-y).sum()/2) for x,y in zip(histogram,small_opt)],records_sha=hashes)
        times=[];i=0;key=0x951e0000+128*kind+32*corrected
        for rep in range(-1,5):
            for threads in ([1,16] if rep%2==0 else [16,1]):
                n=256 if rep<0 else 32768
                start=perf_counter();draw(n,key+i,0,threads);t=perf_counter()-start;i+=1
                if rep>=0:times.append(dict(n=n,threads=threads,seconds=t,rep=rep))
        latency=[]
        for off in range(101):
            start=perf_counter();draw(1,key+i,off,1);latency.append(perf_counter()-start)
        row["timings"]=times;result["latency"]=np.asarray(latency)
        np.savez_compressed(out/f"output_{kind}_{corrected}.npz",**result)
        rows.append(row);save(out,"summary.json",dict(rows=rows))
        print(kind,corrected,stats["max_pair_tv"],row["count_ks"],row["max_loop_tv"],flush=True)
    free_res(rp);release(ptr)
manifest(out)
