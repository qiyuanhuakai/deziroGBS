"""Match the archived full-width Frugal K=2 benchmark to the present calibration."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ORDER
from iteration100_projection_v1 import physical
from iteration114_rank_v1 import cov
from iteration119_hardware import OUT,ROOT,hardware

def run():
    t,ss=hardware()
    previous=physical(dict(np.load(ORDER)),np.arange(1152))-np.eye(2304)
    covariance_error=[float(abs(x-cov(t,s)).max()) for x,s in zip(previous,ss)]
    assert max(covariance_error)<1e-14
    # Explicit vacuum determinants for every subset of the 16 physical prefix ports.
    masks=np.arange(65536);bits=(masks[:,None]>>np.arange(16))&1
    empty=np.ones(len(masks))
    for s in ss:
        q=cov(t[:16],s)+np.eye(32)
        for size in range(1,17):
            selected=masks[bits.sum(1)==size]
            for chunk in np.array_split(selected,max(1,(len(selected)+255)//256)):
                axes=np.where(bits[chunk])[1].reshape(-1,size)
                axes=np.column_stack((axes,axes+16))
                sign,log=np.linalg.slogdet(q[axes[:,:,None],axes[:,None,:]])
                assert min(sign)>0
                empty[chunk]*=np.exp(-log/2)
    optical=empty[::-1].astype(np.longdouble)
    for k in range(16):
        u=optical.reshape(-1,2,1<<k);u[:,1]-=u[:,0]
    source=ROOT/'results/iteration105_full_prefix_v1/law.npz'
    with np.load(source) as f:
        reference_error=float(abs(f['optical']-optical).max())
        tv=float(abs(f['frugal']-optical).sum()/2)
    metadata=json.loads((ROOT/'results/iteration103_frugal_full_v1/summary.json').read_bytes())
    assert reference_error<1e-10 and abs(tv-.10771072769668194)<1e-9
    result=dict(cutoff=2,full_ports=1152,causal_prefix_ports=16,covariance_max_errors=covariance_error,
                prefix_optical_max_error=reference_error,full_click_tv_lower=tv,
                archived_cpu_workers=metadata['threads'],archived_samples=metadata['records'],
                archived_seconds_per_full_record=metadata['total_us']*1e-6,
                archive_scope='CPU FP64 adapter with spectra processed sequentially. Full-input causal tree for the first 16 physical ports.')
    (OUT/'frugal_full.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
