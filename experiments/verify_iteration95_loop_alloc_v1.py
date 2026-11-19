"""Independent Fock, generating-function and rare-branch checks for the loop sampler."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import ctypes as ct
import itertools
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.special import ndtr,ndtri
from iteration92_common_v2 import ROOT,H,save,read,manifest
from iteration95_loop_tables_v1 import discrete,fock_probabilities

p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
al=ROOT/"results/iteration95_loop_alloc_v1";res=ROOT/"results/iteration95_loop_residual_v1"
fr=ROOT/"results/iteration95_loop_frugal144_v1";small=ROOT/"results/iteration93_reference_v2"
source=ROOT/"native/gaussian_loop_audit95_v1.cpp"
hardware=ROOT/"results/iteration81_jiuzhang_accuracy_v3/histograms.npz"
files=[Path(__file__),source,hardware,al/"manifest.json",res/"manifest.json",fr/"manifest.json",
       *[ROOT/k for k in read(res/"plan.json")["hashes"] if k.startswith("native/")],
       *[small/f"input_{j}.npz" for j in (0,2)]]
save(out,"plan.json",dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},
    scope="Independent pure-state amplitude recurrence on even Fock patterns, independent FFT of all group PGFs with Chernoff alias bound, numerical full eight-bit laws across eight distinct loops after reweighting, forced three-group residual branch, explicit 16-mode twin-source counterexample.",
    control="262144 samples in independent Philox domain 0x95223000. Complete joint law of three nonidentical Bernoulli variables in a four-center mixture retargeted to totals [0.01,.04,.55,.4]."))
for path in (al,res,fr):
    listed=read(path/"manifest.json")
    assert set(listed)=={f.name for f in path.iterdir()}-{"manifest.json"}
    assert all(H(path/f)==v for f,v in listed.items())
cmd=read(res/"build.json")["command"];cmd[-3]=str(source);cmd[-1]=str(out/"audit.dll")
r=subprocess.run(cmd,capture_output=True,text=True)
save(out,"build.json",dict(command=cmd,code=r.returncode,stderr=r.stderr));r.check_returncode()
runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent));dll=ct.CDLL(str(out/"audit.dll"))
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
D,B,W,T,J=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32,np.int32)]
def bind(name,args,result=I):
    f=getattr(dll,name);f.argtypes=args;f.restype=result;return f
fock_native=bind("audit95_fock",[I,I,D,F,B,D])
prep=bind("cube95_prepare",[I,I,I,D,F,D,I,I,D,D],P);release=bind("cube95_destroy",[P],None)
retarget=bind("residual95_prepare",[P,I,D,F,I,D,D],P);release_r=bind("residual95_free",[P],None)
draw=bind("residual95_draw",[P,I,U,U,I,B,T,B]);masks=bind("audit95_mask",[P,J,D,I,I,D])
rows=[];truths={}
for case,last in [(0,32),(2,16)]:
    z=np.load(small/f"input_{case}.npz");u=z["u"];m=len(u);t=z["t"]
    assert not z["alpha"].any() and not z["noise"].any() and np.linalg.norm(u.conj().T@u-np.eye(m))<1e-11
    barg=np.ascontiguousarray((u*t)@u.T);vacuum=float(np.prod(np.sqrt(1-t*t)))
    parts=[]
    for total in range(0,last+1,2):
        marks=np.array(list(itertools.combinations(range(total+m-1),m-1)),int)
        parts.append(np.diff(np.column_stack([np.full(len(marks),-1),marks,np.full(len(marks),total+m-1)]),axis=1)-1)
    counts=np.ascontiguousarray(np.concatenate(parts),dtype=np.uint8);law=np.empty(len(counts))
    assert fock_native(m,len(law),barg.view(float),vacuum,counts,law)==0
    sums=counts.sum(1).astype(int);known=np.load(small/f"law_{case}.npz")["total"]
    check=np.bincount(sums,weights=law,minlength=last+1)
    assert np.max(abs(check-known[:last+1]))<1e-13
    st=np.load(al/f"setup_{case}.npz");ind=st["offset"];degree=len(ind)-2
    cdf=[st["low"][ind[k]:ind[k+1]] for k in range(degree+1)]
    codes=[st["pattern"][ind[k]:ind[k+1]] for k in range(degree+1)]
    mode=np.load(al/f"output_{case}_2.npz")
    probability=fock_probabilities(counts,mode["group_marginal"][0],st["split"][0],degree,cdf,codes)
    l1=abs(law-probability).sum()/2
    a,b=max(0.,1-law.sum()),max(0.,1-probability.sum())
    rows.append(dict(case=case,m=m,last=last,patterns=len(law),optical_tail=a,
                     full_tv=[max(0.,l1+abs(a-b)/2),min(1.,l1+(a+b)/2)],
                     low_order_l1=float(abs(law[sums<=degree]-probability[sums<=degree]).sum())))
    truths[f"totals_{case}"]=check
st=np.load(al/"setup_4.npz");groups=np.load(hardware)["ports"]
bits=(np.arange(256)[:,None]>>np.arange(8))&1
refs=np.stack([np.load(hardware.with_name(f"input_{j}.npz"))["two_common_sectors"] for j in range(3)])
large=[]
for kind in [0,1]:
    model=np.load(res/f"model_{kind}.npz");grid=model["grid"];cuts=np.ascontiguousarray(model["cuts"])
    noise=.98*np.linalg.eigvalsh(model["corr"])[0]
    # Avoid eigenvalue recomputation changing a rounding decision by one word.
    noise_direct=1-np.einsum("cg,cg->g",grid,grid)/len(grid)
    assert max(abs(noise_direct-noise))<2e-12
    marg=np.empty_like(cuts);shift=np.empty_like(cuts)
    ptr=prep(144,len(grid),cuts.shape[1]-1,grid,math.sqrt(noise),cuts,1,16,marg,shift)
    assert ptr and max(abs((shift-model["shifts"]).ravel()))<1e-11
    tail=np.rint(ndtr((grid[:,:,None]-(cuts+model["shifts"])[None,:,:])/math.sqrt(noise))*2**32).clip(0,2**32-1)/2**32
    prob=np.concatenate([np.ones((*tail.shape[:2],1)),tail],2)-np.concatenate([tail,np.zeros((*tail.shape[:2],1))],2)
    freq=1024;trans=np.zeros(freq//2+1,dtype=complex)
    for i in range(0,len(grid),8):
        trans+=np.fft.rfft(prob[i:i+8],n=freq,axis=2).prod(1).sum(0)/len(grid)
    convolution=np.fft.irfft(trans,n=freq)
    chernoff=np.exp(np.log(prob@np.exp(.15*np.arange(prob.shape[2]))).sum(1)-.15*freq).mean()
    limit=len(model["proposal"])-1
    diff=float(abs(convolution[:limit+1]-model["proposal"]).max())
    assert diff<1e-11 and chernoff<1e-20
    mass=model["target"];law=np.empty_like(mass);diag=np.empty(4)
    rp=retarget(ptr,limit,mass,1.25,16,law,diag);assert rp
    assert max(abs(law-model["proposal"]))<1e-12
    if kind:kernel=st["pnr_mask_kernel"]
    else:
        kernel=np.zeros((144,10,256))
        for g in range(144):
            for k in range(9):kernel[g,k]=discrete(st["mask"][g,k])
    targets=[]
    for weighted in [0,1]:
        local=np.load(res/f"output_{kind}_{weighted}.npz")["local"]
        per=[]
        for j,ports in enumerate(groups):
            group=np.ascontiguousarray(ports//8,dtype=np.int32)
            chance=np.ascontiguousarray(np.stack([kernel[g]@bits[:,p%8] for g,p in zip(group,ports)]))
            v=np.empty(256)
            assert masks(rp,group,chance,weighted,16,v)==0
            assert abs(v.sum()-1)<2e-10
            pred=np.array([local[g]@bits[:,p%8] for g,p in zip(group,ports)])
            assert abs(v@bits-pred).max()<2e-10
            if not weighted: assert abs(v-model["raw_projected"][j]).max()<2e-10
            truths[f"law_{kind}_{weighted}_{j}"]=v
            per.append(float(abs(v-refs[j]).sum()/2))
        targets.append(per)
    large.append(dict(pnr=kind,fft_pmf_max_error=diff,fft_alias_bound=float(chernoff),projected_tv=targets))
    release_r(rp);release(ptr)
# Small exact control: several centers, three nonidentical coins, non-power-of-two leaf count.
grid=np.array([[-.75,.2,-.5],[.75,-.2,.5],[.15,.45,-.8],[-.15,-.45,.8]])
threshold=np.column_stack([-ndtri([.2,.35,.5]),np.full(3,np.inf)])
marg=np.empty_like(threshold);shift=np.empty_like(threshold)
ptr=prep(3,4,1,grid,.8,threshold,1,2,marg,shift);assert ptr
target=np.array([.01,.04,.55,.4]);q=np.empty(4);diag=np.empty(4)
rp=retarget(ptr,3,target,1.25,2,q,diag);assert rp
p1=np.rint(ndtr((grid-threshold[:,0]-shift[:,0])/.8)*2**32)/2**32
binary=bits[:8,:3];nclick=binary.sum(1)
exact=np.prod(np.where(binary[None,:,:]>0,p1[:,None,:],1-p1[:,None,:]),axis=2).mean(0)
assert abs(np.bincount(nclick,weights=exact)-q).max()<1e-13
correct=exact*target[nclick]/q[nclick]
n=262144;output=np.empty((n,3),np.uint8);retry=np.empty(n,np.uint32);branch=np.empty(n,np.uint8)
assert draw(rp,n,0x95223000,0,4,output,retry,branch)==0
again=np.empty((127,3),np.uint8);t=np.empty(127,np.uint32);b=np.empty(127,np.uint8)
assert draw(rp,127,0x95223000,111,1,again,t,b)==0 and np.array_equal(again,output[111:238]) and np.array_equal(b,branch[111:238])
hist=np.bincount(output@np.array([1,2,4]),minlength=8)/n
assert abs(hist-correct).max()<.006 and abs(branch.mean()-diag[0])<.006
control=dict(expected_excess=float(diag[0]),observed_excess=float(branch.mean()),complete_empirical_tv=float(abs(hist-correct).sum()/2))
release_r(rp);release(ptr)
# Eight ideal twin-beam pairs, partitioned with every signal in the first block.
# Even granting perfectly matching block counts, independent within-count
# allocation has limited probability to reproduce the entire matching mask.
q=.25;sizes=np.array([int(v).bit_count() for v in range(256)])
ideal=np.zeros((256,256));np.fill_diagonal(ideal,q**sizes*(1-q)**(8-sizes))
broken=np.zeros_like(ideal)
for k in range(9):
    pick=np.flatnonzero(sizes==k)
    broken[np.ix_(pick,pick)]=math.comb(8,k)*q**k*(1-q)**(8-k)/len(pick)**2
gap=float(abs(ideal-broken).sum()/2)
assert abs(gap-(1-np.trace(broken)))<1e-14
# Grouping the two members of each independent pair gives the product of
# their exact two-bit laws; enumerate to check the factorization.
fixed=np.ones_like(ideal)
for j in range(8):
    a=bits[:,j];fixed*=np.where(a[:,None]==a,np.where(a,q,1-q),0)
assert np.array_equal(fixed,ideal)
np.savez_compressed(out/"laws.npz",**truths,control=correct,observed=hist)
save(out,"validation.json",dict(passed=True,small=rows,large=large,residual_control=control,
    paired_16=dict(pair_firing_probability=q,split_across_blocks_tv=gap,paired_blocks_tv=float(abs(fixed-ideal).sum()/2))))
manifest(out)
