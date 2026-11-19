"""Build and validate finite conditional probability tables from the fixed optical copula targets."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import ctypes as ct
import math
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
import scipy
from scipy.special import ndtr
from iteration92_common_v2 import ROOT, H, read, save, manifest

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
out = parser.parse_args().output.resolve()
out.mkdir(parents=True, exist_ok=False)
prior = ROOT / "results/iteration95_pair_copula_v1"
small = ROOT / "results/iteration93_reference_v2"
hardware = ROOT / "results/iteration81_jiuzhang_accuracy_v3"
photons = ROOT / "results/iteration84_photon_cdf_v1"
source = ROOT / "native/gaussian_pair_table95_v1.cpp"
blas_path = next((Path(scipy.__file__).parent.parent / "scipy.libs").glob("*openblas*.dll"))
files = [Path(__file__), source, prior/"plan.json", prior/"summary.json", hardware/"histograms.npz",
         photons/"reference.npz", photons/"reference.json", *[prior/f"optical_{i}.npz" for i in range(5)],
         *[hardware/f"input_{i}.npz" for i in range(3)],
         *[small/f"law_{i}.npz" for i in range(4)]]
files.extend(ROOT/p for p in read(prior/"plan.json")["hashes"] if p.startswith("native/"))
N = 65536
cases = [(i,4,True) for i in range(4)] + [(4,1,False),(4,1,True),(4,4,True)]
save(out, "plan.json", dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},
    blas=H(blas_path), cases=cases, observations=N, conditional_variance="0.98 * min eigenvalue",
    width="small 64, else next power of two", shared_frame_seed="0x950e1000+case, nested frame prefix",
    independent_domains="key 0x950e2000+configuration for quality; 0x950e3000+256*configuration+timing_index for timing; purpose 123 center, 124 ports, 125 frame generation",
    count="Exact integer-table conditional product sums in physical detector order. CDF truncation <=384 for large click, <=512 for large PNR. Report missing mass and any-overflow upper bound.",
    calibration="Use optical single-port photon marginal at each threshold. No hardware samples are used to fit the table.",
    family="All large configurations share the same 99% DKW band on the hardware empirical CDF, conditional on iid hardware records. Other reported model-law metrics are deterministic numerical calculations.",
    timing="allocation included; warmup then alternate formats/order over five repetitions, 32768 shots (large) or 2048 (small); one worker and 16 on large; 101 separate single-record calls"))
dll = out/"table.dll"
command = [shutil.which("g++"),"-shared","-O3","-std=c++20","-DEIGEN_DONT_PARALLELIZE",
           "-ffp-contract=off","-fno-fast-math","-I",str(ROOT.parent/"external_sources/eigen_3_4_0_iteration38/eigen-3.4.0"),
           str(source),"-o",str(dll)]
build = subprocess.run(command, capture_output=True, text=True)
save(out,"build.json",dict(command=command,code=build.returncode,stderr=build.stderr))
build.check_returncode()
runtime_directory = os.add_dll_directory(str(Path(command[0]).parent))
blas_directory = os.add_dll_directory(str(blas_path.parent))
blas = ct.CDLL(str(blas_path))
blas.scipy_openblas_set_num_threads(1)
assert blas.scipy_openblas_get_num_threads()==1
lib = ct.CDLL(str(dll))
I,U,P,F = ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_double
D,B,Indices = [np.ctypeslib.ndpointer(dtype=t, flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.int32)]
def bind(name, args, result=I):
    fn=getattr(lib,name)
    fn.argtypes=args
    fn.restype=result
    return fn
assert bind("pair95_gemm",[P])(ct.cast(blas.scipy_cblas_dgemm,P))==0
eigen=bind("pair95_eigen",[I,D,D,D])
centers_native=bind("cube95_centers",[I,I,I,D,U,D])
prepare=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P)
destroy=bind("cube95_destroy",[P],None)
sample=bind("cube95_sample",[P,I,U,U,I,I,B])
pair_stats=bind("cube95_pair_stats",[P,I,D,D])
local=bind("cube95_local",[P,I,Indices,D])
counts=bind("cube95_counts",[P,I,I,D])
photon_counts=bind("cube95_photon_law",[P,I,I,D])
box=bind("cube95_box",[P,I,D])
hist=np.load(hardware/"histograms.npz")
hw_hist=hist["total_click_histogram"]
hw_cdf=hw_hist.cumsum()/hw_hist.sum()
hw_radius=math.sqrt(math.log(200)/(2*hw_hist.sum()))
photon_ref=np.load(photons/"reference.npz")["photon_cdf"]
reference_tail=float(read(photons/"reference.json")["tail_upper"])

def pair_errors(single, joint, first, second):
    i,j=np.triu_indices(len(single),1)
    d=second[i,j]-joint[i,j]
    a,b=(first-single)[i],(first-single)[j]
    tv=(abs(d)+abs(a-d)+abs(b-d)+abs(d-a-b))/2
    order=int(tv.argmax())
    return dict(max_pair_tv=float(tv[order]),mean_pair_tv=float(tv.mean()),
                worst_pair=[int(i[order]),int(j[order])])

def cdf_distance(law, reference):
    residual=max(0.,1-law.sum())
    cdf=law.cumsum()
    gap=float(np.max(abs(cdf-reference[:len(law)])))
    # A subprobability due to tail trimming can miss mass below any retained count.
    # Beyond the last retained count both CDFs are within their remaining tails of 1.
    high=max(gap+residual,residual,1-float(reference[len(law)-1]))
    return [max(0.,gap-residual),min(1.,high)]

def small_box_target(case,cutoff,modes):
    data=np.load(small/f"law_{case}.npz")
    if case==3:
        return data["pnr"][:cutoff+1,:cutoff+1].T.ravel()
    keys=data["keys"]
    index=np.arange((cutoff+1)**modes,dtype=np.uint64)
    code=np.zeros_like(index)
    for j in range(modes):
        code |= (index%(cutoff+1))<<(8*j)
        index//=cutoff+1
    pos=np.searchsorted(keys,code)
    inside=pos<len(keys)
    inside[inside]&=keys[pos[inside]]==code[inside]
    result=np.zeros(len(code))
    result[inside]=data["probabilities"][pos[inside]]
    return result

rows=[]
for config,(case,frames,calibrate) in enumerate(cases):
    opt=np.load(prior/f"optical_{case}.npz")
    corr=opt["correlation"]
    m=len(corr)
    cap=16 if case==4 else 64
    limits=np.ascontiguousarray(opt["thresholds"][:,:cap+1])
    width=max(64,1<<(m-1).bit_length())
    size=2*width*frames
    start=perf_counter()
    vals=np.empty(m)
    vecs=np.empty_like(corr)
    assert eigen(m,corr,vals,vecs)==0 and vals.min()>0
    noise=.98*vals[0]
    loadings=np.ascontiguousarray(vecs*np.sqrt(vals-noise))
    centers=np.empty((size,m))
    assert centers_native(m,width,frames,loadings,0x950e1000+case,centers)==0
    elapsed_factor=perf_counter()-start
    cov=centers.T@centers/size
    cov.flat[::m+1]+=noise
    factor_error=float(np.max(abs(cov-corr)))
    assert factor_error<1e-10 and abs(centers.mean(0)).max()<1e-12
    marginal=np.empty_like(limits)
    shift=np.empty_like(limits)
    start=perf_counter()
    table=prepare(m,size,cap,centers,math.sqrt(noise),limits,int(calibrate),16,marginal,shift)
    elapsed_prepare=perf_counter()-start
    assert table
    row=dict(case=case,frames=frames,calibrate=calibrate,centers=size,cap=cap,noise=float(noise),
             factor_seconds=elapsed_factor,table_seconds=elapsed_prepare,bytes_tables=size*m*(cap+1)*4,
             center_cov_max_error=factor_error,overflow_union_bound=float(marginal[:,-1].sum()),
             marginal_max=float(np.max(abs(marginal-ndtr(-limits)))))
    print("prepared",config,elapsed_prepare,flush=True)
    try:
        arrays=dict(marginal=marginal,shift=shift)
        for kind in [0,1]:
            first=np.empty(m)
            second=np.empty_like(corr)
            assert pair_stats(table,kind,first,second)==0
            covariance=second-np.outer(first,first)
            if kind:
                row["photon_variance"]=float(covariance.sum())
                row["optical_photon_variance"]=float(opt["photon_cov"].sum())
                row["photon_mean"]=float(first.sum())
                arrays["photon_covariance"]=covariance
            else:
                row.update(pair_errors(opt["single"],opt["joint"],first,second))
                arrays.update(click_first=first,click_second=second)
        groups=hist["ports"] if case==4 else [np.arange(m)]
        tvs=[]
        for index,ports in enumerate(groups):
            ports=np.ascontiguousarray(ports,dtype=np.int32)
            law=np.empty(1<<len(ports))
            assert local(table,len(ports),ports,law)==0 and abs(law.sum()-1)<1e-11
            truth=(np.load(hardware/f"input_{index}.npz")["two_common_sectors"] if case==4
                   else np.load(small/f"law_{case}.npz")["click"])
            tvs.append(float(abs(law-truth).sum()/2))
            arrays[f"local_{index}"]=law
        row["local_click_tv"]=tvs
        check_ports=np.unique(np.concatenate(groups)).astype(int)
        arrays.update(audit_ports=check_ports,centers_subset=centers[:,check_ports])
        if m<=4:
            cutoff=64 if m==2 else 16
            law=np.empty((cutoff+1)**m)
            assert box(table,cutoff,law)==0
            truth=small_box_target(case,cutoff,m)
            inside=float(abs(law-truth).sum()/2)
            left,right=max(0.,1-law.sum()),max(0.,1-truth.sum())
            # For cases 0,1 the known reference is a submeasure (global total cutoff).
            slack=0 if case==3 else float(read(small/f"row_{case}.json")["tail_chernoff"])
            row["pnr_full_tv"]=[max(0.,inside+abs(left-right)/2-slack),min(1.,inside+(left+right)/2+slack)]
            arrays["box"]=law
        click_law=np.empty(min(m,384)+1)
        photon_law=np.empty((512 if case==4 else 256)+1)
        assert counts(table,len(click_law)-1,16,click_law)==0
        assert photon_counts(table,len(photon_law)-1,16,photon_law)==0
        assert min(click_law)>=0 and min(photon_law)>=0
        assert max(click_law.sum(),photon_law.sum())<1+2e-12
        arrays.update(click_count=click_law,photon_count=photon_law)
        row["count_missing"]=[max(0.,1-click_law.sum()),max(0.,1-photon_law.sum())]
        if case in (0,2):
            row["odd_photon_mass"]=float(photon_law[1::2].sum())
        if case==4:
            interval=cdf_distance(click_law,hw_cdf)
            row["click_empirical_ks"]=interval
            row["click_population_99"]=[max(0.,interval[0]-hw_radius),min(1.,interval[1]+hw_radius)]
            lower,upper=cdf_distance(photon_law,photon_ref)
            slack=reference_tail+row["overflow_union_bound"]
            row["photon_optical_ks"]=[max(0.,lower-slack),min(1.,upper+slack)]
        def draw(n,exp,offset,kind,workers):
            data=np.empty((n,m),np.uint8)
            assert sample(table,n,exp,offset,kind,workers,data)==0
            return data
        exp=0x950e2000+config
        photon_hist=np.zeros(len(photon_law)+1,np.int64)
        click_hist=np.zeros(len(click_law)+1,np.int64)
        for offset in range(0,N,8192):
            data=draw(8192,exp,offset,1,16)
            if offset==0:
                assert np.array_equal(draw(21,exp,11,1,1),data[11:32])
                assert np.array_equal(draw(8192,exp,offset,0,1),data>0)
            total=data.sum(1).astype(int).clip(max=len(photon_law))
            nclick=(data>0).sum(1).clip(max=len(click_law))
            photon_hist+=np.bincount(total,minlength=len(photon_hist))
            click_hist+=np.bincount(nclick,minlength=len(click_hist))
            np.savez_compressed(out/f"records_{config}_{offset}.npz",counts=data)
        arrays.update(photon_hist=photon_hist,click_hist=click_hist)
        row["empirical_vs_table_ks"]=[float(max(abs(v.cumsum()/N-p.cumsum()))) for v,p in
                                     [(click_hist[:-1],click_law),(photon_hist[:-1],photon_law)]]
        timing=[]
        methods=[(0,1),(1,1)]+([(0,16),(1,16)] if case==4 else [])
        batch=32768 if case==4 else 2048
        ti=0
        for rep in range(-1,5):
            for kind,workers in (methods if rep%2==0 else methods[::-1]):
                n=256 if rep<0 else batch
                beg=perf_counter()
                data=draw(n,0x950e3000+256*config+ti,0,kind,workers)
                elapsed=perf_counter()-beg
                if rep>=0:
                    timing.append(dict(pnr=kind,workers=workers,rep=rep,n=n,seconds=elapsed))
                ti+=1
        row["timings"]=timing
        for kind in (0,1):
            latency=[]
            for offset in range(101):
                beg=perf_counter()
                draw(1,0x950e3000+256*config+ti,offset,kind,1)
                latency.append(perf_counter()-beg)
            ti+=1
            arrays[f"single_latency_{kind}"]=np.array(latency)
        np.savez_compressed(out/f"model_{config}.npz",**arrays)
        rows.append(row)
        save(out,"summary.json",dict(rows=rows,hardware_size=int(hw_hist.sum()),hardware_dkw99=hw_radius))
        print("verified",config,row["max_pair_tv"],row.get("photon_optical_ks"),flush=True)
    finally:
        destroy(table)
manifest(out)
