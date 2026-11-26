"""Draw physical-port records from Husimi covariance q and displacement mu."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration97_conditionals_v1 import Kernel,Optical,pair_information,layout,fit,sample,I,J,W,SCALE
from iteration96_component_v1 import components

def prepare(kernel,q,mu,route,history,prep_workers):
    q=np.ascontiguousarray(q,dtype=float);mu=np.ascontiguousarray(mu,dtype=float)
    if q.ndim==2:q=q[None]
    if mu.ndim==1:mu=mu[None]
    if (q.ndim!=3 or q.shape[1]!=q.shape[2] or not q.shape[1] or q.shape[1]%2 or
        mu.shape!=q.shape[:2] or not np.isfinite(q).all() or not np.isfinite(mu).all()):
        raise ValueError('incompatible q and mu')
    if route not in ('click','pnr') or not 0<=history<=14 or not 1<=prep_workers<=64:raise ValueError('invalid configuration')
    opt=Optical(kernel.prior,q,mu);m=opt.m
    if route=='click':
        groups=components(q)
        if max(map(len,groups))<=18:
            laws=[kernel.joint(opt,[],group,2,prep_workers)[0].ravel() for group in groups]
            # Accumulate optical CDF in extended precision in C++ through the round-96 component API.
            from iteration96_group_common_v1 import ROOT,D,B,U,P,bind,ct
            dll=ct.CDLL(str(ROOT/'results/iteration96_component_v1/component.dll'))
            build=bind(dll,'component96_prepare',[I,I,J,J,D],P)
            draw=bind(dll,'component96_draw',[P,I,U,U,I,B]);free=bind(dll,'component96_free',[P],None)
            h=build(m,len(groups),np.array(list(map(len,groups)),np.int32),np.concatenate(groups).astype(np.int32),np.concatenate(laws))
            if not h:raise ValueError('invalid component tables')
            def emit(n,seed,offset,workers):
                x=np.empty((n,m),np.uint8)
                if draw(h,n,seed,offset,workers,x):raise ValueError('invalid record interval')
                return x,np.zeros(n,np.uint8)
            return emit,lambda:free(h),'schur_component'
    base,width=(2,4) if route=='click' else (3,2)
    info,_=pair_information(q);groups=layout(info,width)
    model=fit(kernel,opt,info,groups,base,history,parallel=prep_workers)
    return lambda n,seed,offset,workers:sample(kernel,model,m,n,seed,offset,workers),lambda:kernel.free(model['handle']),'conditional_'+route

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--route',choices=['click','pnr'],required=True);p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--offset',type=int,default=0);p.add_argument('--history',type=int,default=8)
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.shots<1 or a.offset<0 or a.offset+a.shots>2**32 or not 0<=a.seed<2**64:p.error('invalid RNG domain or shot count')
    with np.load(a.input) as data,threadpool_limits(limits=1,user_api='blas'):
        emit,close,method=prepare(Kernel(),data['q'],data['mu'],a.route,a.history,a.workers)
        try:x,tail=emit(a.shots,a.seed,a.offset,a.workers)
        finally:close()
    with a.output.open('xb') as file:np.savez_compressed(file,records=x,tail=tail,method=method)
    print(method,len(x),'records;',int(tail.sum()),'containing a capped pair total')
if __name__=='__main__':main()
