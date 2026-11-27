"""Projected Gaussian state, residual matching and whole-circuit checks."""
import json
import networkx as nx
import numpy as np
from scipy.linalg import eigh, solve_triangular
from threadpoolctl import threadpool_limits

from iteration111_memory_v1 import realify
from iteration115_displacement_v1 import conditional
from iteration115_fidelity_v1 import root_fidelity
from iteration115_projection_review_v3 import project
from iteration115_visible_v1 import DATA, hardware

OUT=DATA/"iteration117_blocks"


def axes(banks):
    return (2*np.asarray(banks)[:,None]+np.arange(2)).ravel()


def decomposition():
    t,ss=hardware()
    u,reference,leak=project(t,ss,1)
    w=u[:,:,0]
    split=abs(w)**2
    idx=np.column_stack((np.arange(144),144+np.arange(144))).ravel()
    components=[]
    for s,(basis,ev) in zip(ss,reference):
        q,v,_,response,_=conditional(t,s)
        b=(w.conj()[:,:,None]*q.reshape(144,8,-1)).sum(1)
        e,rotation=eigh(np.eye(len(s))-b.conj().T@b)
        dark=realify((rotation*np.sqrt(e.clip(0)))@rotation.conj().T)
        bright=realify(b)
        excess=v-np.eye(len(v))
        cross=bright@excess@dark.T
        den=2*np.eye(len(dark))+dark@excess@dark.T
        rd=dark@response
        core=bright@excess@bright.T-cross@np.linalg.solve(den,cross.T)
        load=bright@response-cross@np.linalg.solve(den,rd)
        chol=np.linalg.cholesky(np.eye(response.shape[1])+rd.T@np.linalg.solve(den,rd))
        load=solve_triangular(chol,load.T,lower=True).T
        np.testing.assert_allclose(core/2+load@load.T/2,(basis*ev)@basis.T,atol=7e-13,rtol=0)
        # Vacuum covariance I; classical coherent amplitudes are half the mean quadratures.
        components.append((core[np.ix_(idx,idx)]/2,load[idx]/2))
    return split,components,leak


def matching(components):
    # Weight the omitted cross-block X covariance by its squared Frobenius norm.
    score=sum((x.reshape(144,2,144,2)**2).sum((1,3)) for x,_ in components)
    graph=nx.from_numpy_array(np.triu(score,1))
    edges=nx.max_weight_matching(graph,maxcardinality=True)
    pairs=np.array(sorted(tuple(sorted(p)) for p in edges))
    assert len(pairs)==72 and np.array_equal(np.sort(pairs.ravel()),np.arange(144))
    return pairs,score


def retained(c, groups):
    x,load=c
    keep=np.zeros_like(x)
    for group in groups:
        ax=axes(group)
        keep[np.ix_(ax,ax)]=x[np.ix_(ax,ax)]
    return keep+2*load@load.T


def assess():
    split,components,leak=decomposition()
    pairs,score=matching(components)
    full=[x+2*l@l.T for x,l in components]
    plain=np.array_split(np.arange(144),144)
    def distance(groups):
        mats=[retained(c,groups) for c in components]
        # Put canonical q's before p's for the symplectic fidelity convention.
        o=np.r_[np.arange(0,288,2),np.arange(1,288,2)]
        ident=.5*np.eye(len(o))
        f=np.prod([root_fidelity(ident+x[np.ix_(o,o)],ident+y[np.ix_(o,o)])
                   for x,y in zip(full,mats)])
        return dict(bright_root_fidelity=float(f),full_TV_upper=float(np.sqrt(1-(1-leak)*f*f)))
    out={"singles":distance(plain),"pairs":distance(pairs)}
    # Endpoints and several successively larger groups along this fixed pair ordering.
    for width in (4,8,16,36,72):
        out[str(width)]=distance([pairs.ravel()[i:i+width] for i in range(0,144,width)])
    out["full_projection_upper"]=float(np.sqrt(leak))
    out["pairs_chosen"]=pairs.tolist()
    out["sum_offdiag_squared_fraction_recovered"]=float(score[tuple(pairs.T)].sum()/np.triu(score,1).sum())
    OUT.mkdir(exist_ok=True)
    (OUT/"partition.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print({k:v for k,v in out.items() if k!="pairs_chosen"},flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3):assess()
