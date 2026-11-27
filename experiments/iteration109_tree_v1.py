"""Build two-mode displaced Gaussian kernels for a maximum-weight spanning tree of outputs."""
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree,breadth_first_order
from iteration106_packed_v1 import compress
from iteration108_laws_v1 import modes
from iteration108_residual_v1 import split
from iteration109_model_v1 import axes,pure_field


def minors(x):
    # det(I+ZX) and the symmetric coefficient matrices of
    # det(I+ZX) d^T(I+ZX)^(-1) Z d, where Z=diag(u,u,v,v).
    pol=np.zeros((3,3));pol[0,0]=1
    num=np.zeros((3,3,4,4))
    for k in range(1,5):
        for ids in itertools.combinations(range(4),k):
            idx=np.array(ids);a=sum(i<2 for i in ids);b=k-a
            sub=x[np.ix_(idx,idx)];pol[a,b]+=np.linalg.det(sub)
            for i,j in itertools.product(range(k),repeat=2):
                cof=(-1)**(i+j)*np.linalg.det(np.delete(np.delete(sub,i,axis=0),j,axis=1))
                num[a,b,idx[i],idx[j]]+=cof
    return pol,num


def edges(row,rem,method,required=()):
    groups=len(row)
    if required and method!='tree':raise ValueError('Required edges need tree topology')
    if method not in ('tree','chain','independent'):raise ValueError('Invalid topology')
    if groups==1 and required:raise ValueError('Single vertex has no edges')
    if method=='independent' or groups==1:return np.full(groups,-1,np.int32),np.arange(groups,dtype=np.int32)
    if method=='chain':return np.arange(-1,groups-1,dtype=np.int32),np.arange(groups,dtype=np.int32)
    if groups>4096:raise ValueError('Use chain or independent topology above 4096 groups')
    w=np.zeros((groups,groups));v=row.reshape(2*groups,-1)
    for x in rem:
        mat=(v@x@v.T).reshape(groups,2,groups,2)
        w+=np.sum(mat**2,axis=(1,3))
    np.fill_diagonal(w,0);root=int(w.sum(0).argmax())
    # Required edges must form a forest. Making them strictly heavier
    # than every other edge leaves their inclusion independent of rounding.
    peak=w.max()+1
    for a,b in required:
        if not 0<=a<groups or not 0<=b<groups or a==b:raise ValueError('Invalid required edge')
        w[a,b]=w[b,a]=peak
    graph=minimum_spanning_tree(csr_matrix(-w-1e-12));order,parent=breadth_first_order(graph,root,directed=False)
    assert len(order)==groups
    parent[root]=-1
    if any(parent[a]!=b and parent[b]!=a for a,b in required):raise ValueError('Required edges contain a cycle')
    return parent.astype(np.int32),order.astype(np.int32)


def packed_matrix(a):
    return np.stack([a[...,i,j]*(1 if i==j else 2) for i in range(4) for j in range(i,4)],axis=-1)


def prepare(t,r,fan=8,method='tree',share=1.,required=()):
    t=np.asarray(t,complex);r=np.atleast_2d(r)
    if t.ndim!=2 or r.ndim!=2 or not t.size or not r.size:raise ValueError('Expected nonempty transfer and spectra')
    m=len(t)
    if not m or fan not in (1,2,4,8) or m%fan or not 0<=share<=1 or r.shape[1]*2!=t.shape[1] or not np.isfinite(t).all() or not np.isfinite(r).all():raise ValueError('Invalid paired-source layout')
    if np.linalg.eigvalsh(t.conj().T@t).max()>1+1e-9:raise ValueError('Non-passive transfer')
    if fan>1:
        base,w,_=compress(t,(np.sinh(r)**2).sum(0),1,width=fan)
        power=(abs(w.ravel())**2).reshape(-1,fan);norm=power.sum(1)
        weights=np.divide(power,norm[:,None],out=np.full_like(power,1/fan),where=norm[:,None]>0)
        base*=np.sqrt(norm[:,None])
    else:base=t;weights=np.ones((m,1))
    # Per-spatial-port orthogonal phases are a local passive gauge.
    u,x,_=modes(base,r)
    row=np.stack((np.c_[u.real,-u.imag],np.c_[u.imag,u.real]),axis=1)
    rem=[];noise=[];sur=[]
    for xs in x:
        v=np.eye(len(xs))+2*xs;p,n,_=split(v)
        noise.append(n/2);rem.append((p-np.eye(len(p)))/2);sur.append(pure_field(p))
    groups=len(base);parent,order=edges(row,rem,method,required)
    both=np.concatenate((row[np.maximum(parent,0)],row),axis=1)
    both[parent<0,:2]=0
    generators=[];vp=[];pn=[];meta=[];single=[];maxshare=[]
    omega=np.kron(np.eye(2),axes(1))
    for shared,res,extra in zip(noise,rem,sur):
        pair=both@res@both.transpose(0,2,1);candidate=both@extra@both.transpose(0,2,1)
        lo=0.;hi=1.
        for _ in range(43):
            mid=(lo+hi)/2
            value=np.linalg.eigvalsh(np.eye(4)+2*(pair-mid*candidate)+1j*omega)[:,0].min()
            if value>=-2e-13:lo=mid
            else:hi=mid
        gain=share*lo;maxshare.append(gain)
        pair-=gain*candidate
        e,v=np.linalg.eigh(shared+gain*extra);g=(v*np.sqrt(e.clip(0)/2)).T
        generators.append((g@row.reshape(-1,row.shape[-1]).T).reshape(len(e),groups,2))
        vparams=[];pparams=[];pars=[];marginals=[]
        for C in pair:
            C=(C+C.T)/2;D,N=minors(C)
            vparams.append(np.concatenate([D[:,:,None],packed_matrix(N)],axis=2))
            # Parent marginal and child conditional on a vacuum parent.
            cross=C[2:,:2]@np.linalg.inv(np.eye(2)+C[:2,:2])
            conditional=C[2:,2:]-cross@C[:2,2:]
            singles=[]
            for variance,projection in ((C[:2,:2],np.eye(4)[:2]),(conditional,np.c_[-cross,np.eye(2)])):
                eig,vec=np.linalg.eigh(variance)
                singles.append(np.c_[1/(1+eig),eig/(1+eig),vec.T@projection])
            marginals.append(singles)
            Q=np.eye(4)+C;inv=np.linalg.inv(Q);K=np.eye(4)-inv
            E,H=minors(-K)
            # Positive exponent at z>0 uses b = Q^{-1} d.
            def conv(a,b):
                out=np.zeros((5,5,*b.shape[2:]))
                for i,j in itertools.product(range(3),repeat=2):
                    out[i:i+3,j:j+3]+=a[i,j]*b
                return out
            degree=np.add.outer(np.arange(3),np.arange(3))
            s=conv(E,E);fixed=-.5*conv(E,degree*E)
            R=conv(E,degree[:,:,None,None]*H)-conv(degree*E,H)
            pparams.append(np.concatenate((s[:,:,None],fixed[:,:,None],packed_matrix(R)),axis=2))
            pars.append(np.r_[inv.ravel(),np.linalg.slogdet(Q)[1]/2,C[0,0],C[0,1],C[1,1],C[2,2],C[2,3],C[3,3]])
        vp.append(vparams);pn.append(pparams);meta.append(pars);single.append(marginals)
    subset=np.array([weights@((mask>>np.arange(fan))&1) for mask in range(1<<fan)]).T.copy()
    return dict(parent=parent,order=order,weights=weights,subset=subset,noise=np.ascontiguousarray(generators),
                vacuum=np.ascontiguousarray(vp),photon=np.ascontiguousarray(pn),meta=np.ascontiguousarray(meta),
                single=np.ascontiguousarray(single),retained_classical=np.array(maxshare),fan=np.int32(fan),physical_modes=base)
