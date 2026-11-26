"""Sample the archived Jiuzhang input or a user-supplied zero-mean Gaussian covariance."""
import argparse
import hashlib
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,np,Native,Tree,prepare
from iteration100_fast_v1 import Fast,make
from iteration100_relabel_v1 import Calibrated

DATA=ROOT/'results'
def checked(stage,name):
    folder=DATA/f'iteration100_{stage}_v1';p=folder/name
    digest=json.loads((folder/'manifest.json').read_text())[name]
    with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=digest:raise ValueError(f'archive hash mismatch for {name}')
    return p

def cached(calibrated=True):
    with np.load(checked('spatial','prepared.npz')) as f:base=dict(f)
    with np.load(checked('fast','shortcuts.npz')) as f:extra=dict(f)
    model=Fast(checked('fast','fast.dll'),base,extra)
    if calibrated:
        with np.load(checked('relabel','prepared.npz')) as f:correction=dict(f)
        model=Calibrated(model,checked('relabel','relabel.dll'),correction)
    return model

def custom(path,width):
    with np.load(path) as z:
        v=np.asarray(z['v'],complex);x=np.asarray(z['x'])
        if np.iscomplexobj(x) or any(np.any(z[k]) for k in ('alpha','mu','mean') if k in z):
            raise ValueError('expected real quadrature covariance and zero displacement')
        x=np.asarray(x,float)
    if x.ndim==2:x=x[None]
    if v.ndim!=2 or min(v.shape)<1 or x.ndim!=3 or len(x)<1 or x.shape[1:]!=(2*v.shape[1],)*2:
        raise ValueError('v must have shape (ports, sources), x must have shape (spectra, 2*sources, 2*sources)')
    if not np.isfinite(v).all() or not np.isfinite(x).all() or np.max(abs(x-x.transpose(0,2,1)))>1e-12:
        raise ValueError('nonfinite input or asymmetric covariance')
    n=v.shape[1];eye=np.eye(n);omega=np.block([[0*eye,eye],[-eye,0*eye]])
    if np.linalg.eigvalsh(v.conj().T@v).max()>1+1e-11 or np.linalg.eigvalsh(np.eye(2*n)+2*x+1j*omega).min()<-1e-10:
        raise ValueError('passivity or covariance uncertainty check failed')
    from iteration100_compression_v1 import occupation
    mean=((v@sum(map(occupation,x)))*v.conj()).sum(1).real
    # Exact dark coordinates can be reinserted in physical order after sampling.
    active=np.flatnonzero(mean!=0)
    if not len(active):return None,active,len(v)
    if len(active)%width:width=1
    if len(active)//width>16384 or len(active)>=65535:raise ValueError('input exceeds native tree index capacity')
    lib=Native(checked('spatial','spatial.dll'))
    arr,_=prepare(lib,v[active],x,width)
    if arr['caps'].max()>65535:raise ValueError('source count scale exceeds record dtype')
    ex,_=make(Tree(lib,arr))
    return Fast(checked('fast','fast.dll'),arr,ex),active,len(v)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    which=p.add_mutually_exclusive_group();which.add_argument('--input',type=Path);which.add_argument('--uncalibrated',action='store_true')
    p.add_argument('output',type=Path);p.add_argument('--route',choices=('click','pnr'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--width',type=int,choices=(1,2,4,8),help='physical group size on custom input, defaults to 1')
    a=p.parse_args()
    if not(0<=a.seed<2**64 and 1<=a.shots<2**31 and 0<=a.offset<=2**32-a.shots and 1<=a.workers<=64):
        p.error('invalid sample count, worker count or RNG domain')
    if a.width and not a.input:p.error('--width applies to custom optical inputs')
    if a.input:
        model,active,m=custom(a.input,a.width or 1);cal=False
    else:
        cal=not a.uncalibrated;model=cached(cal);active=None;m=1152
    if model is None:
        records=np.zeros((a.shots,m),np.uint16);flags=np.zeros(a.shots,np.uint8);updates=flags.copy()
    else:
        records,flags,*extra=model.sample(a.shots,int(a.route=='pnr'),a.seed,a.offset,a.workers)
        updates=extra[0] if extra else np.zeros(a.shots,np.uint8)
        if active is not None and len(active)<m:
            full=np.zeros((a.shots,m),np.uint16);full[:,active]=records;records=full
    with a.output.open('xb') as stream:
        np.savez_compressed(stream,records=records,flags=flags,correction=updates,route=a.route,
                            seed=np.uint64(a.seed),offset=a.offset,calibrated=cal)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
