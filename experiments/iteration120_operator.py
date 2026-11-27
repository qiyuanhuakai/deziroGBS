"""Small dual certificates for a piecewise-linear quantum likelihood correction.

All terms are supported on the span of the nested normalized pair-cutoff states.
A feasible dual matrix for each sign bounds all POVMs, including the common
displacement channel and either refined detector. The solver is optional when
verifying a deposited certificate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from iteration119_hardware import DATA

OUT=DATA/'iteration120_contractions'
UP=list((i,j) for j in range(4) for i in range(j+1))


def operators(ret):
    shells=np.diff(np.r_[0.,ret,1.])
    assert min(shells)>0
    result=[]
    for i in range(1,5):
        a=np.sqrt(shells);a[i:]=0;a/=np.linalg.norm(a)
        result.append(np.outer(a,a))
    return result


def branches(mats,clamps):
    b,m,p,o=mats
    lo,hi,l,u=clamps
    for w,tests in [(l*m,[l*m-p]),(p,[p-l*m,u*m-p]),(u*m,[p-u*m])]:
        for g,more in [(lo*b,[lo*b-w]),(w,[w-lo*b,hi*b-w]),(hi*b,[w-hi*b])]:
            yield o-g,tests+more


def svec(a):
    return np.array([a[i,j]*(1 if i==j else np.sqrt(2)) for i,j in UP])


def matrix(x):
    y=np.zeros((4,4))
    for v,(i,j) in zip(x,UP):y[i,j]=y[j,i]=v/(1 if i==j else np.sqrt(2))
    return y


def solve(ret,clamps,sign):
    import clarabel
    from threadpoolctl import threadpool_limits
    terms=list(branches(operators(ret),clamps))
    count=10+sum(len(cs) for d,cs in terms)
    rows=[];rhs=[];cones=[];offset=10
    for d,cs in [(np.zeros((4,4)),[]),*terms]:
        A=np.zeros((10,count));A[:,:10]=-np.eye(10)
        for a in cs:
            A[:,offset]=svec(a);offset+=1
        rows.append(A);rhs.extend(svec(-sign*d));cones.append(clarabel.PSDTriangleConeT(4))
    A=np.zeros((count-10,count));A[:,10:]=-np.eye(count-10)
    rows.append(A);rhs.extend(np.zeros(count-10));cones.append(clarabel.NonnegativeConeT(count-10))
    q=np.zeros(count);q[:10]=svec(np.eye(4))
    cfg=clarabel.DefaultSettings();cfg.verbose=False
    cfg.tol_feas=cfg.tol_gap_abs=cfg.tol_gap_rel=1e-10
    with threadpool_limits(limits=1):
        sol=clarabel.DefaultSolver(sparse.csc_matrix((count,count)),q,sparse.csc_matrix(np.vstack(rows)),
                                   np.array(rhs),cones,cfg).solve()
    x=np.array(sol.x);y=matrix(x[:10]);offset=10;lam=[];least=np.linalg.eigvalsh(y)[0]
    for d,cs in terms:
        # A negative numerical multiplier cannot be used on a cone of nonnegative effects.
        vals=np.maximum(x[offset:offset+len(cs)],0);offset+=len(cs);lam.append(vals.tolist())
        constraint=y-sign*d-sum(a*v for a,v in zip(cs,vals))
        least=min(least,np.linalg.eigvalsh(constraint)[0])
    y+=(max(0,-least)+1e-10)*np.eye(4)
    return dict(sign=sign,Y=y.tolist(),multipliers=lam,trace=float(np.trace(y)),status=str(sol.status))


def run(ret,clamps):
    signed=[solve(ret,clamps,sign) for sign in (1,-1)]
    return dict(retention=ret,clamps=clamps,dual=signed,upper=max(x['trace'] for x in signed))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('limits',nargs='*',type=float);ap.add_argument('--save',type=Path)
    a=ap.parse_args();ret=json.loads((DATA/'iteration119_weights/clamped_bands_64.json').read_bytes())['retention']
    answer=run(ret,a.limits or [.5,4,.8,1.25])
    if a.save:a.save.write_text(json.dumps(answer,indent=2)+'\n')
    print(answer['clamps'],[v['trace'] for v in answer['dual']],flush=True)
