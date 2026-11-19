"""Sample one optical input in physical port order using the round 96 native cores."""
import argparse
import ctypes as ct
from pathlib import Path
from threadpoolctl import threadpool_limits
from iteration96_group_common_v1 import ROOT,np,Native,Optical,partition,pair_information,block_weights,construct,D,B,T,J,I,U,P,bind
from iteration96_component_v1 import components
from iteration96_identify_herald_v1 import identify

def run(data,route,n,seed,offset,workers):
    if n<1 or workers not in range(1,65) or seed not in range(2**64) or offset<0 or offset+n>2**32:
        raise ValueError('invalid sample count, workers, experiment or record offset')
    if route not in ('click','pnr'):
        raise ValueError('route must be click or pnr')
    if 'barg' in data and 'q' not in data:
        b=np.asarray(data['barg'],complex)
        if b.ndim!=2 or not len(b) or b.shape[0]!=b.shape[1] or not np.isfinite(b).all() or np.max(abs(b-b.T))>1e-12 or np.linalg.norm(b,2)>=1 or np.any(data.get('mu',0)):
            raise ValueError('Bargmann-only input requires a pure undisplaced state')
        eye=np.eye(len(b))
        iq=np.block([[eye-b.real,-b.imag],[-b.imag,eye+b.real]])
        data=dict(barg=b,q=np.linalg.inv(iq)[None],mu=np.zeros((1,2*len(b))))
    lib=Native(ROOT/'results/iteration96_tree_build_v1/tree.dll')
    if route=='pnr' and 'barg' in data and not np.any(data.get('mu',0)):
        # A Bargmann matrix denotes one pure undisplaced spectrum. The standalone
        # .npz may alternatively contain just q,mu for the general Gaussian route.
        barg=np.asarray(data['barg'],complex);fit=identify(barg)
        if fit is not None:
            s,h,v,t,_,err=fit
            if 'q' in data:
                # Check optional Gaussian moments before using the pure-state description.
                q=np.asarray(data['q']);m=len(barg);eye=np.eye(m)
                if q.ndim==2:q=q[None]
                iq=np.block([[eye-barg.real,-barg.imag],[-barg.imag,eye+barg.real]])
                if q.shape!=(1,2*m,2*m) or np.max(abs(q[0]@iq-np.eye(2*m)))>1e-10:
                    raise ValueError('inconsistent pure-state Bargmann and Gaussian inputs')
            dll=ct.CDLL(str(ROOT/'results/iteration96_herald_v1/herald.dll'))
            prep=bind(dll,'herald96_prepare',[I,D,D,T],P);draw=bind(dll,'herald96_draw',[P,I,U,U,I,T,B]);free=bind(dll,'herald96_free',[P],None)
            flag=np.ones(1,np.uint32);d=len(t);ptr=prep(d,np.ascontiguousarray(v).view(float),t*t,flag)
            if not ptr:raise ValueError('invalid recovered optical network')
            try:
                x=np.empty((n,2*d),np.uint32);status=np.empty(n,np.uint8)
                if draw(ptr,n,seed,offset,workers,x,status):raise RuntimeError('invalid heralded sampling arguments')
                out=np.zeros((n,len(barg)),np.uint32);out[:,s]=x[:,:d];out[:,h]=x[:,d:]
                return dict(records=out,status=status,tail=np.zeros(n,np.uint8),method='herald',bargmann_fit_error=err)
            finally:free(ptr)
    if 'q' not in data or 'mu' not in data:raise ValueError('general input requires q and mu')
    q=np.ascontiguousarray(data['q'],dtype=float);mu=np.ascontiguousarray(data['mu'],dtype=float)
    if q.ndim==2:q=q[None]
    if mu.ndim==1:mu=mu[None]
    if q.ndim!=3 or q.shape[1]%2 or q.shape[1]!=q.shape[2] or mu.shape!=q.shape[:2] or not (np.isfinite(q).all() and np.isfinite(mu).all()):
        raise ValueError('invalid Gaussian moment arrays')
    opt=Optical(lib,q,mu);m=opt.m;out=np.empty((n,m),np.uint8);flags=np.zeros(n,np.uint8)
    if route=='click':
        groups=components(q)
        if max(map(len,groups))<=18:
            dll=ct.CDLL(str(ROOT/'results/iteration96_component_v1/component.dll'))
            prep=bind(dll,'component96_prepare',[I,I,J,J,D],P);draw=bind(dll,'component96_draw',[P,I,U,U,I,B]);free=bind(dll,'component96_free',[P],None)
            laws=np.concatenate([opt.law(tuple(g),2) for g in groups]);ports=np.concatenate(groups).astype(np.int32)
            ptr=prep(m,len(groups),np.array(list(map(len,groups)),np.int32),ports,laws)
            if not ptr:raise ValueError('invalid connected components')
            try:
                if draw(ptr,n,seed,offset,workers,out):raise RuntimeError('component draw failed')
                return dict(records=out,status=flags,tail=np.zeros_like(flags),method='component_click')
            finally:free(ptr)
    base,width,cap=(2,4,4) if route=='click' else (3,2,32)
    info,_=pair_information(q);groups=partition(m,width,True,info);table,_=construct(opt,groups,block_weights(groups,info),base,cap,True)
    # Unconditioned draws use the full local tables; this limit bounds only the auxiliary count messages.
    limit=min(m,512) if base==2 else 384;proposal=np.zeros(limit+1)
    h=lib.prepare(m,len(groups),table['transition'].shape[-1],cap,int(base==3),limit,*table.values(),proposal)
    if not h:raise ValueError('invalid optical tree')
    try:
        tries=np.zeros(n,np.uint32);tail=np.empty(n,np.uint8);rare=np.zeros_like(tail)
        if lib.draw(h,n,seed,offset,0,workers,out,tries,tail,rare):raise RuntimeError('tree draw failed')
        return dict(records=out,status=flags,tail=tail,method='tree_click' if base==2 else 'tree_pnr')
    finally:lib.free(h)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--route',choices=['pnr','click'],required=True);p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    with np.load(a.input) as data,threadpool_limits(limits=1,user_api='blas'):
        res=run(data,a.route,a.shots,a.seed,a.offset,a.workers)
    with a.output.open('xb') as f:np.savez_compressed(f,**res)
    print(str(res['method']),len(res['records']),'records,',np.count_nonzero(res['status']),'failed,',np.count_nonzero(res['tail']),'with a capped pair count')
if __name__=='__main__':main()
