"""Displaced single-mode Gaussian core map, independently checked by exponentials.
Keep Gaussian background analytic; apply a finite upper-triangular map to the
non-Gaussian core. Product maps also act on entangled multimode cores.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,math,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import expm_multiply

def local_map(d,t,displacement,eta):
    tp=eta*t;den=1-tp*tp;ap=math.sqrt(eta)*((1-eta*t*t)*displacement-t*(1-eta)*displacement.conjugate())/den
    scale=math.sqrt(eta*(1-t*t)/den);beta=-t*(1-eta*eta)/den;gamma=math.sqrt(1-t*t)*(math.sqrt(eta)*ap.conjugate()-displacement.conjugate());e=np.zeros(d,complex);e[0]=1
    for j in range(1,d):e[j]=(gamma*e[j-1]+(beta*e[j-2] if j>=2 else 0))/j
    M=np.zeros((d,d),complex)
    for a in range(d):
        for b in range(a,d):M[a,b]=scale**a*math.sqrt(math.factorial(b)/math.factorial(a))*e[b-a]
    g=lambda a,t:-abs(a)**2/2+t*a.conjugate()**2/2
    prefactor=((1-t*t)/den)**.25*np.exp(g(displacement,t)-g(ap,tp));return M,prefactor,tp,ap

def run():
    rng=np.random.default_rng(930001);rows=[];snapshots=[]
    for cut in (36,60):
        d=cut+1;a=diags(np.sqrt(np.arange(1,d)),1,shape=(d,d));number=np.arange(d)
        for t in (0.,.3,-.5):
            for alpha in (0j,.2+.3j):
                c=rng.normal(size=6)+1j*rng.normal(size=6);c/=np.linalg.norm(c);padded=np.pad(c,(0,d-len(c)));state=expm_multiply(alpha*a.T-alpha.conjugate()*a,expm_multiply(np.arctanh(t)*(a.T@a.T-a@a)/2,padded))
                for eta in (0.,.17,.7,1.,1.05):
                    M,f,tp,ap=local_map(len(c),t,alpha,eta);transformed=M@c;out=expm_multiply(ap*a.T-ap.conjugate()*a,expm_multiply(np.arctanh(tp)*(a.T@a.T-a@a)/2,np.pad(transformed,(0,d-len(c)))))*f;expected=state*np.power(eta,number/2);err=float(np.linalg.norm(out-expected));assert err<(2e-7 if cut==36 else 2e-11),(cut,t,alpha,eta,err);rows.append(dict(cutoff=cut,t=t,alpha=[alpha.real,alpha.imag],eta=eta,error=err,weight=float(np.vdot(expected,expected).real)));snapshots.append((c,M,f,tp,ap))
    for cut in (16,24):
        d=cut+1;a=diags(np.sqrt(np.arange(1,d)),1,shape=(d,d));ops=[kron(a,eye(d)),kron(eye(d),a)];t=[.2,-.3];alpha=[.1+.2j,-.2+.1j];core=np.zeros((4,4),complex)
        for i in range(4):
            for j in range(4-i):core[i,j]=rng.normal()+1j*rng.normal()
        core/=np.linalg.norm(core);padded=np.zeros((d,d),complex);padded[:4,:4]=core
        S=sum(np.arctanh(ti)*(op.T@op.T-op@op)/2 for ti,op in zip(t,ops));D=sum(al*op.T-al.conjugate()*op for al,op in zip(alpha,ops));state=expm_multiply(D,expm_multiply(S,padded.ravel()));number=np.add.outer(np.arange(d),np.arange(d)).ravel()
        for eta in (0.,.31,.8,1.):
            maps=[local_map(4,ti,al,eta) for ti,al in zip(t,alpha)];transformed=maps[0][0]@core@maps[1][0].T;padded[:]=0;padded[:4,:4]=transformed;Sp=sum(np.arctanh(z[2])*(op.T@op.T-op@op)/2 for z,op in zip(maps,ops));Dp=sum(z[3]*op.T-z[3].conjugate()*op for z,op in zip(maps,ops));out=expm_multiply(Dp,expm_multiply(Sp,padded.ravel()))*maps[0][1]*maps[1][1];expected=state*np.power(eta,number/2);err=float(np.linalg.norm(out-expected));assert err<(2e-6 if cut==16 else 2e-10),(cut,eta,err);rows.append(dict(multimode=True,cutoff=cut,eta=eta,error=err,weight=float(np.vdot(expected,expected).real)))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=run();result=dict(passed=True,checks=len(rows),max_error=max(r['error'] for r in rows),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('passed','checks','max_error')}))
