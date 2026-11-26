"""Structural full-PNR TV for a chain of balanced optical cells, verified by independent sampling."""
import argparse,ctypes as ct,math
from pathlib import Path
from time import perf_counter
from scipy.linalg import block_diag
from scipy.stats import beta,nbinom
from threadpoolctl import threadpool_limits
from iteration97_conditionals_v1 import ROOT,Kernel,np,I,D,B,W,U,P,bind,H,save,manifest
from iteration96_group_common_v1 import T,J
from iteration96_component_v1 import components

def network(cells,angle):
    v0=np.kron(np.eye(cells),np.exp(2j*np.pi*np.outer(np.arange(4),np.arange(4))/4)/2)
    v=v0.copy()
    for g in range(cells-1):
        a=4*g+3;b=a+1
        v[[a,b]]=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])@v[[a,b]]
    return v0,v

def covariance(v,p):
    d=len(v);zero=np.zeros_like(v);b=np.block([[zero,v*np.sqrt(p)],[v.T*np.sqrt(p),zero]])
    m=2*d;eye=np.eye(m)
    return np.linalg.inv(np.block([[eye-b.real,-b.imag],[-b.imag,eye+b.real]]))[None],b

def full_pnr_tv(p,theta,links):
    return float(-np.expm1(-.5*links*np.log1p(4*p*np.sin(theta)**2/(1-p)**2)))

def diagonal_stay(p,theta,cap=18):
    c=math.cos(theta);s=math.sin(theta);value=0.
    for n in range(cap+1):
        for m in range(cap+1):
            diag=sum((-1)**k*math.comb(n,k)*math.comb(m,k)*c**(n+m-2*k)*s**(2*k) for k in range(min(n,m)+1))
            value+=(1-p)**2*p**(n+m)*diag**2
    return value,2*p**(cap+1)

def main():
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);out=a.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';herald=data/'iteration96_herald_v1/herald.dll';component=data/'iteration96_component_v1/component.dll';quad=data/'iteration96_quad_v1/oracle.dll'
    files=[Path(__file__),herald,component,quad,ROOT/'native/heralded_fock96_v1.cpp',ROOT/'native/component_click96_v1.cpp',data/'iteration97_build_v1/conditional.dll']
    cases=[(2,.02),(2,.18),(16,.02),(144,.02),(144,.08)];p=.0576;records=131072
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},cases=cases,p=p,N=records,
        scope='disjoint terminal beamsplitters on an open path, two distinct endpoints per interior cell, uniform squeezing; five PNR escape and five threshold lower-bound events; family alpha=.01',
        rng=97100000))
    kernel=Kernel();hl=ct.CDLL(str(herald));cl=ct.CDLL(str(component));ql=ct.CDLL(str(quad))
    hp=bind(hl,'herald96_prepare',[I,D,D,T],P);hd=bind(hl,'herald96_draw',[P,I,U,U,I,T,B]);hf=bind(hl,'herald96_free',[P],None)
    cp=bind(cl,'component96_prepare',[I,I,J,J,D],P);cd=bind(cl,'component96_draw',[P,I,U,U,I,B]);cf=bind(cl,'component96_free',[P],None)
    cuts_read=bind(cl,'component96_cuts',[P,I,W]);oracle=bind(ql,'oracle96',[I,I,D,D,D,D])
    single_v,_=network(1,0);q1,_=covariance(single_v,p);single=np.zeros(256);diag=np.zeros(2);mean=np.zeros((1,16))
    assert oracle(8,1,q1,mean,single,diag)==0 and diag[0]<1e-13
    numerical=np.maximum(single,0);numerical/=numerical.sum()
    rows=[]
    for j,(cells,theta) in enumerate(cases):
        d=4*cells;m=2*d;v0,v=network(cells,theta);pr=np.full(d,p);error=np.ones(1,np.uint32)
        # Check that the true input has one connected component in its physical Bargmann graph.
        from scipy.sparse.csgraph import connected_components
        edge=abs(v)>0
        adjacency=np.block([[np.zeros_like(edge),edge],[edge.T,np.zeros_like(edge)]])
        assert connected_components(adjacency,directed=False,return_labels=False)==1
        start=perf_counter();true=hp(d,v.view(float),pr,error);complete_prep=perf_counter()-start
        start=perf_counter();cut=hp(d,v0.view(float),pr,error);pnr_prep=perf_counter()-start
        start=perf_counter()
        ports=np.concatenate([np.r_[np.arange(4*i,4*i+4),d+np.arange(4*i,4*i+4)] for i in range(cells)]).astype(np.int32)
        law=np.tile(numerical,cells);direct=cp(m,cells,np.full(cells,8,np.int32),ports,law);click_prep=perf_counter()-start
        assert true and cut and direct
        try:
            cuts=np.empty(256,np.uint64);assert cuts_read(direct,0,cuts)==0
            realized=np.diff(cuts,prepend=np.uint64(0)).astype(float)/2**53
            quant=float(abs(realized-single).sum()/2)
            bound=full_pnr_tv(p,theta,cells-1);stay,missing=diagonal_stay(p,theta)
            assert abs(stay-(1-full_pnr_tv(p,theta,1)))<missing+2e-14
            row=dict(cells=cells,ports=m,links=cells-1,angle=theta,pnr_tv=bound,click_upper=min(1.,bound+cells*quant),
                local_numerical_tv=quant,pnr_workspace_failure_upper=float(cells*nbinom.sf(60,4,1-p)),
                single_edge_direct_sum=stay,omitted_mass_bound=missing,preparation_s=[complete_prep,pnr_prep,click_prep])
            if cells==2:
                q,b=covariance(v,p);law16=np.empty(65536);assert oracle(16,1,q,np.zeros((1,32)),law16,diag)==0
                words=np.arange(65536);prod=np.ones(65536)
                for ids in ports.reshape(cells,8):
                    z=sum(((words>>a)&1)<<k for k,a in enumerate(ids));prod*=realized[z]
                row['full_click_tv']=float(abs(prod-law16).sum()/2)
                assert row['full_click_tv']<=row['click_upper']+1e-12
                np.savez_compressed(out/f'input_{j}.npz',v=v,v0=v0,p=pr,q=q,mu=np.zeros((1,32)),barg=b,click=law16,cut_click=prod)
            exp=97100000+20*j;imbalance=forbidden=0
            for rep in range(records//2048):
                x=np.empty((2048,m),np.uint32);flags=np.empty(2048,np.uint8)
                assert hd(true,len(x),exp,rep*2048,16,x,flags)==0 and not flags.any()
                signal=x[:,:d].reshape(-1,cells,4).sum(2);idler=x[:,d:].reshape(-1,cells,4).sum(2)
                assert np.all(signal.sum(1)==idler.sum(1))
                imbalance+=int((signal!=idler).any(1).sum());forbidden+=int(((signal>0)!=(idler>0)).any(1).sum())
            def interval(k):
                return [float(beta.ppf(.0005,k,records-k+1)) if k else 0.,float(beta.ppf(.9995,k+1,records-k)) if k<records else 1.]
            ci=interval(imbalance);row.update(attempts=records,imbalance=imbalance,forbidden_click=forbidden,escape_ci=ci,click_event_ci=interval(forbidden))
            # The confidence check allows the predeclared statistical failure probability.
            assert ci[0]<=bound<=ci[1],(ci,bound)
            times={}
            for name in ['exact_pnr','cut_pnr','direct_click']:
                for workers in [1,16]:
                    n=2048 if cells==2 else 8192;values=[]
                    for rep in range(5):
                        start=perf_counter();off=2**20+(workers*5+rep)*n
                        if name=='direct_click':
                            x=np.empty((n,m),np.uint8);rc=cd(direct,n,exp+3,off,workers,x)
                        else:
                            x=np.empty((n,m),np.uint32);flags=np.empty(n,np.uint8)
                            rc=hd(true if name=='exact_pnr' else cut,n,exp+(1 if name=='exact_pnr' else 2),off,workers,x,flags)
                        elapsed=perf_counter()-start;assert rc==0
                        if name!='direct_click':assert not flags.any()
                        values.append(elapsed*1e6/n)
                    times[f'{name}_{workers}']=dict(batch=n,us=values,median=float(np.median(values)))
            row['timing']=times;print(m,theta,bound,forbidden/records,times['direct_click_16']['median'],flush=True)
            rows.append(row);save(out,'summary.json',rows)
        finally:hf(true);hf(cut);cf(direct)
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
