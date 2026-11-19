"""Independent sparse matrix exponential check of coherent-core no-jump identity.

This is a rank-two S2 background, not a general multimode Gaussian sampler.
The matrix-exponential reference does not use the proposed disentangling map.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import expm_multiply

def transform(core,r,eta):
    k=len(core)-1;rp=eta*r;c=math.sqrt((1-r*r)/(1-rp*rp));alpha=math.sqrt(eta)*c;beta=-r*(1-eta*eta)/(1-rp*rp)
    out=np.zeros_like(core)
    for a in range(k+1):
        for b in range(k+1-a):
            f=1.
            for l in range((k-a-b)//2+1):
                out[a,b]+=f*core[a+l,b+l]
                f*=beta*math.sqrt((a+l+1)*(b+l+1))/(l+1)
            out[a,b]*=c*alpha**(a+b)
    return out,rp

def check():
    rng=np.random.default_rng(920001);rows=[]
    for cutoff in (24,36):
        d=cutoff+1;a=diags(np.sqrt(np.arange(1,d)),1,shape=(d,d));aa=kron(a,eye(d));bb=kron(eye(d),a);generator=aa.T@bb.T-aa@bb;number=np.add.outer(np.arange(d),np.arange(d)).ravel()
        for k in (0,2,5):
            core=np.zeros((k+1,k+1),complex)
            for i in range(k+1):
                for j in range(k+1-i):core[i,j]=rng.normal()+1j*rng.normal()
            core/=np.linalg.norm(core)
            for r in (0.,.2,.45):
                padded=np.zeros((d,d),complex);padded[:k+1,:k+1]=core;state=expm_multiply(np.arctanh(r)*generator,padded.ravel())
                for eta in (0.,.17,.61,1.):
                    transformed,rp=transform(core,r,eta);small=np.zeros_like(padded);small[:k+1,:k+1]=transformed;actual=expm_multiply(np.arctanh(rp)*generator,small.ravel());expected=state*np.power(eta,number/2);err=float(np.linalg.norm(actual-expected))
                    assert err<2e-9,(cutoff,k,r,eta,err)
                    rows.append(dict(cutoff=cutoff,degree=k,r=r,eta=eta,error=err,survival=float(np.vdot(expected,expected).real)))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    rows=check();report=dict(passed=True,checks=len(rows),maximum_error=max(row['error'] for row in rows),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__)
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k] for k in ('passed','checks','maximum_error')}))
