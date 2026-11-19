"""Same-input single-worker PNR timing and independent photonic-engines quality sample."""
import argparse
import hashlib
import subprocess
from iteration93_baseline_common_v1 import ROOT,Path,np,C,IP,DP,UP,ptr,H,save,manifest,read,time
from iteration94_common_v2 import load93
from iteration92_pe_common_v1 import UPSTREAM,pe_sample
from iteration93_baseline_common_v1 import Baseline93
from iteration92_quality_v1 import confidence

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration95_adaptive_pnr_v1';ref=ROOT/'results/iteration93_reference_v2'
z=dict(np.load(ref/'input_0.npz'));dll=base/'adaptive.dll'
revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
save(out,'plan.json',dict(upstream=revision,threads=1,batch=2048,repeats=5,warmup=64,
    methods='PE CPU v2 window4 vs adaptive d4-D4 to d8-D16 budget .02. Both cap64; include call boundary, RNG, return buffers and native diagnostic trace.',
    quality='PE 16384 fresh records in 16 batches of 1024, seed 0x95c00000+batch. Compare with previously collected adaptive small records. Four simultaneous CV bounds (PNR and click lower/upper) for PE.',
    timing_domains='0x95c10000+2*(rep+1)+method; controls at previous addresses excluded',
    hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),dll,ref/'input_0.npz',ref/'law_0.npz',base/'small_1.npz']}))
legacy,runtime=load93();lib=C.CDLL(str(dll))
set_lapack=lib.mps93_set_lapack;set_lapack.argtypes=[C.c_void_p]*3
assert set_lapack(*[C.cast(getattr(legacy._blas,x),C.c_void_p) for x in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
fn=lib.mps95_sample;I=C.c_int
fn.argtypes=[I,I,DP,DP,DP,DP,I,I,C.c_double,I,I,C.c_uint64,C.c_uint,C.c_uint64,I,UP,DP,IP,C.c_char_p,I,I,I,C.c_double,DP]
u,t,alpha,noise=[np.ascontiguousarray(z[k]) for k in ('u','t','alpha','noise')]
adapter=Baseline93(dict(family='pe',case=0,threads=1),z)
def sample(method,n,seed,offset=0):
    if method=='pe':
        counts,stats=pe_sample(adapter.state,adapter.U,n,loss=adapter.loss,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
        return np.asarray(counts),np.asarray(stats.censored,dtype=bool)
    counts=np.empty((n,4),np.uint32);stats=np.empty((n,18));errors=np.empty(n,np.int32);trace=np.empty((n,64,14))
    msg=C.create_string_buffer(4096)
    rc=fn(4,len(t),ptr(u.view(float)),ptr(t),ptr(alpha.view(float)),ptr(noise),4,4,.02,64,n,seed,0,offset,1,
          counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),msg,len(msg),8,16,.0006,ptr(trace))
    assert rc==int(any(errors)) and np.isin(errors,[0,2]).all(),msg.value
    return counts,errors!=0
c,e=sample('adaptive',64,0x95a01000,113)
with np.load(base/'small_1.npz') as old:
    assert np.array_equal(c,old['counts'][113:177]) and np.array_equal(e,old['errors'][113:177]!=0)
parts=[];failed=[]
for batch in range(16):
    c,e=sample('pe',1024,0x95c00000+batch);parts.append(c);failed.append(e)
c=np.concatenate(parts);e=np.concatenate(failed);assert np.all(c>=0) and c.max()<256
np.savez_compressed(out/'pe_quality.npz',counts=c,failed=e)
law=np.load(ref/'law_0.npz');meta=read(ref/'row_0.json')
target=np.r_[law['probabilities'],float(meta['total_tail']),0.]
keys=law['keys'];packed=sum(c[:,i].astype(np.uint64)<<(8*i) for i in range(4))
pos=np.searchsorted(keys,packed);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==packed[inside]
labels=np.where(inside,pos,len(keys));labels[e]=len(keys)+1
click=sum((c[:,i]>0).astype(int)<<i for i in range(4));click[e]=16
a,_=confidence(labels,target,len(c),4096,.0025,float(meta['tail_chernoff']))
b,_=confidence(click,np.r_[law['click'],0.],len(c),4096,.0025)
save(out,'quality.json',dict(records=len(c),failures=int(e.sum()),pnr=a,click=b))
times=[]
for rep in range(-1,5):
    for index in ((0,1) if rep%2==0 else (1,0)):
        method=('pe','adaptive')[index];n=64 if rep<0 else 2048
        start=time.perf_counter();c,e=sample(method,n,0x95c10000+2*(rep+1)+index)
        elapsed=time.perf_counter()-start
        if rep>=0:
            times.append(dict(rep=rep,n=n,method=method,seconds=elapsed,failures=int(e.sum()),
                sha256=hashlib.sha256(c.tobytes()+e.tobytes()).hexdigest()))
adapter.close()
save(out,'timing.json',times);manifest(out)
