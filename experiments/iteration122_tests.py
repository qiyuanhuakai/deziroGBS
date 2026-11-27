"""Independent checks of the optical family examples and the sequential memory maps."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from flint import acb,acb_mat,arb,ctx

from iteration120_interval import exact,diagonal
from iteration122_family_certificate import positive_hermitian
from iteration122_loss import OUT,complete,json_write
from iteration122_memory import Memory


def case_certificate(t,ss,eta,k):
    m,r=t.shape
    a=acb_mat(m,r,[acb(exact(z.real),exact(z.imag)) for z in t.ravel()])
    e=arb(eta.numerator)/eta.denominator
    g=a.conjugate().transpose()*a
    pivot=positive_hermitian(diagonal([e]*r)-g)
    mean=arb(0);keep=arb(1)
    for s in ss:
        mass=[arb(1)]+[arb(0) for _ in range(k)]
        for i,sj in enumerate(s):
            z=exact(sj);d=z.tanh()
            q=(e*d/(1+(1-e)*d))**2
            v=[(1-q).sqrt()]
            for j in range(1,k+1):v.append(v[-1]*q*(2*j-1)/(2*j))
            mass=[sum(v[h]*mass[j-h] for h in range(j+1)) for j in range(k+1)]
            mean+=g[i,i].real*z.sinh()**2
        keep*=sum(mass)
    return dict(minimum_gram_slack_pivot=pivot,model_mean_ball=str(mean),
                input_specific_full_tv_ball=str((1-keep).sqrt()))


def kraus(eng,site,row):
    off=eng.left.offset
    A=np.zeros((len(eng.initial),len(eng.initial)),complex)
    for k,block in enumerate(eng.maps[site]):
        pos=0
        for j in range(k+1):
            n=off[k-j+1]-off[k-j]
            A[off[k-j]:off[k-j+1],off[k]:off[k+1]]+=row[j]*block[pos:pos+n]
            pos+=n
    return A


def independent():
    eta=.4
    t=np.array([[.42,.11j],[.06j,.29]])
    s=np.array([1.7,.73]);L=4
    eng=Memory(t,s,eta,2);q=complete(t,eta)
    a=1-eta+eta*np.exp(-2*s)
    B=(q*((a-1)/(a+1)))@q.T
    vec=np.zeros((5,5,5,5),complex);vec[0,0,0,0]=1
    def create(x,i):
        y=np.zeros_like(x)
        u=np.moveaxis(x,i,0);v=np.moveaxis(y,i,0)
        v[1:]=u[:-1]*np.sqrt(np.arange(1,5))[:,None,None,None]
        return y
    def add(x):
        return sum(B[i,j]*create(create(x,j),i)/2 for i,j in product(range(4),repeat=2))
    two=add(vec);ket=vec+two+add(two)/2;ket/=np.linalg.norm(ket)
    alpha=np.array([.45-.31j,.29+.11j])
    create1=np.diag(np.sqrt(np.arange(1,36)),-1)
    D=[expm(z*create1-z.conjugate()*create1.T)[:,:5] for z in alpha]
    psi=np.einsum('xi,yj,ijab->xyab',*D,ket)
    ref=(abs(psi)**2).sum((2,3))
    def get(rows):
        A=[[kraus(eng,j,row) for row in values] for j,values in enumerate(rows)]
        return np.array([[np.linalg.norm(b@a@eng.initial)**2 for b in A[1]] for a in A[0]])
    estimated=get([x[:12] for x in D])
    err=float(abs(estimated-ref[:12,:12]).max())
    rows=[]
    for d in D:
        v=d[0]
        denom=1+np.sqrt(max(0.,1-np.vdot(v,v).real))
        rows.append(np.vstack([v,np.eye(5)-np.outer(v.conj(),v)/denom]))
    paths=get(rows)
    def grouped(p):
        return np.array([[p[0,0],p[0,1:].sum()],[p[1:,0].sum(),p[1:,1:].sum()]])
    err2=float(abs(grouped(paths)-grouped(ref)).max())
    assert max(err,err2)<2e-12
    return dict(pnr_cells=estimated.size,click_refinements=paths.size,
                maximum_pnr_error=err,maximum_click_error=err2)


def run():
    ctx.prec=160;ctx.threads=2
    data={}
    for name,eta,cap in [('unequal_loss',Fraction(7,100),2),('bright',Fraction(7,100),2),('memory',Fraction(51,100),15)]:
        f=OUT/name/'input.npz' if name!='memory' else OUT/'memory.npz'
        with np.load(f) as z:data[name]=case_certificate(z['T'],z['squeezing'],eta,cap)
    data['independent_four_mode_fock']=independent()
    json_write(OUT/'optical_checks.json',data)
    print(data,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
