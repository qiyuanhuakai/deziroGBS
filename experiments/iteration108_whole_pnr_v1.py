"""Same complete 16-port PNR output from matched-pair and photonic-engines CPU samplers."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from scipy.linalg import null_space
from iteration92_pe_common_v1 import GaussianInput,pe_sample
from iteration100_baselines_v1 import pair_reference
from iteration106_width_v1 import case
from iteration106_field_v1 import optical
from iteration106_sample_v1 import benchmark
from iteration107_cliques_v1 import optical_x
from iteration108_compact_v1 import Compact
from iteration108_low_bench_v1 import weak


def run(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(exist_ok=True,parents=True)
    t,r=case(16,6);r=weak(r,1.2)
    local=Compact(t,r,4,512);dilated=local.u
    basis=np.empty_like(dilated)
    basis[:,::2]=1j*(dilated[:,::2]+dilated[:,1::2])/np.sqrt(2)
    basis[:,1::2]=(dilated[:,::2]-dilated[:,1::2])/np.sqrt(2)
    u=np.column_stack([basis,null_space(basis.conj().T)])
    state=GaussianInput(np.repeat(r[0],2),tuple(range(t.shape[1])),len(u))
    block,_,_=state.bargmann(u)
    cross=(dilated[:,::2]*np.tanh(r[0]))@dilated[:,1::2].T
    error=float(abs(block-cross-cross.T).max());assert error<1e-12
    n=8192;batch=256
    truth=pair_reference(np.eye(4)+optical_x(t[:2],r),12)
    e=optical(t,r);z=np.exp(-2j*np.pi*np.arange(513)/1024)
    ideal=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1)),n=1024)
    tm=benchmark(lambda b,s,o:local.draw(b,True,s,o),batch,108091000,15)
    hist=np.zeros((2,169),np.int64);tot=np.zeros((2,1024),np.int64);times=[]
    for j in range(n//batch):
        start=perf_counter()
        y,stat=pe_sample(state,u,batch,seed=108092000+j,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
        y=y[:,:16].copy();dt=(perf_counter()-start)*1e6/batch
        if j:times.append(dt)
        assert not np.any(stat.censored)
        x=local.draw(batch,True,108093000,j*batch)
        for d,records in enumerate((x,y)):
            hist[d]+=np.bincount(records[:,:2].clip(0,12)@np.array([1,13]),minlength=169)
            tot[d]+=np.bincount(records.sum(1).astype(int),minlength=1024)
    tv=abs(hist/n-truth).sum(1)/2;ks=abs(tot.cumsum(1)/n-ideal.cumsum()).max(1)
    np.savez_compressed(out/'histograms.npz',pair=hist,totals=tot,pair_truth=truth,total_truth=ideal)
    save(out,'summary.json',dict(ports=16,sources=6,mean_emitted_pairs=1.2,shots=n,batch=batch,cap=4,
        model_us=tm['us'],pe_us=float(np.median(times)),pe_quartiles=np.quantile(times,[.25,.75]).tolist(),
        pair_empirical_tv=tv.tolist(),total_empirical_ks=ks.tolist(),purification_error=error,censored=0));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
