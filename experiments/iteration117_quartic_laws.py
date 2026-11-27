"""Exact finite Bargmann contractions of a displaced quartic core against number PGFs."""
import json
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from iteration115_displacement_v1 import conditional
from iteration115_visible_v1 import DATA,hardware
from iteration116_sector_floor import BANKS
from iteration117_quartic import OUT

GRID=512


def physical_models(optical=None,transport=False):
    t,ss=hardware() if optical is None else optical
    models=[]
    for s in ss:
        q,v,_,resp,_=conditional(t,s)
        d=len(s);eye=np.eye(d)
        j=np.vstack((eye,1j*eye))/np.sqrt(2)
        S=j.T@(np.eye(2*d)-np.linalg.inv((v+np.eye(2*d))/2))@j
        k=S@S.conj().T
        pair=np.trace(k).real/2
        fourth=.5*pair**2+.25*np.trace(k@k).real
        if transport:
            from iteration117_transport import corrected
            retained=float(np.sqrt(np.linalg.det(np.eye(d)-k).real)*(1+pair+fourth))
            resp,_=corrected(S,resp,retained)
        noise=resp@resp.T/2
        models.append((q,S,noise,1+pair+fourth))
    return models


def exponential(log):
    value=np.zeros((3,3),complex);value[0,0]=1
    for i in range(3):
        for j in range(3):
            if i+j:
                value[i,j]=sum((a+b)*log[a,b]*value[i-a,j-b]
                   for a in range(i+1) for b in range(j+1) if a+b)/(i+j)
    return value


def contraction(S,normalization,t,metric,vacuum,coefficients=False):
    # Polarization of the diagonal coherent-state kernel
    # exp(|z|^2) * vacuum * exp(-[Re(Tz),Im(Tz)]^T metric [...] ).
    r=len(S)
    v=.5*np.vstack((t,-1j*t))
    a=v.conj();b=v
    av=a.T@metric;bv=b.T@metric
    A=-2*S.conj()@(av@a); B=S.conj()@(np.eye(r)-2*av@b)
    C=S@(np.eye(r)-2*bv@a); D=-2*S@(bv@b)
    BC=B@C; AB=A@B; DC=D@C
    trprod=lambda x,y:np.einsum("ij,ji",x,y)
    l=np.zeros((3,3),complex)
    l[1,0]=np.trace(A)/2;l[0,1]=np.trace(D)/2
    l[2,0]=trprod(A,A)/4;l[0,2]=trprod(D,D)/4;l[1,1]=np.trace(BC)/2
    l[2,1]=trprod(A,BC)/2;l[1,2]=trprod(B,DC)/2
    l[2,2]=trprod(AB,DC)/2+trprod(BC,BC)/4
    value=vacuum*exponential(l)/normalization
    return value if coefficients else value.sum()


def marginal(model,sites):
    q,S,noise,norm=model;r=len(S)
    if sites is None:
        t=np.eye(r);x=noise
    else:
        t=q[sites]
        real=np.block([[t.real,-t.imag],[t.imag,t.real]])
        x=real@noise@real.T
    return S,norm,t,x


def total_curve(args,grid):
    S,norm,t,x=args
    e,u=eigh(x)
    ans=[]
    for z in np.exp(-2j*np.pi*np.arange(grid//2+1)/grid):
        w=1-z
        metric=(u*(w/(1+w*e)))@u.T
        ans.append(contraction(S,norm,t,metric,np.exp(-.5*np.log1p(w*e).sum())))
    return np.array(ans)


def void(model,sites):
    if len(sites)==0:return 1.
    S,norm,t,x=marginal(model,sites)
    den=np.eye(len(x))+x
    v=contraction(S,norm,t,np.linalg.inv(den),np.exp(-.5*np.linalg.slogdet(den)[1]))
    assert abs(v.imag)<3e-12
    return v.real


def run():
    mods=physical_models()
    total=np.prod([total_curve(marginal(m,None),GRID) for m in mods],axis=0)
    table=np.fft.irfft(total,n=GRID)
    assert abs(total[0]-1)<1e-12 and table.min()>-1e-12
    # Positive real PGF gives a Chernoff bound on the probability aliased by FFT.
    radius=1.08;g=1.
    for model in mods:
        S,norm,t,x=marginal(model,None)
        e,u=eigh(x);w=1-radius
        assert np.all(1+w*e>0)
        g*=contraction(S,norm,t,(u*(w/(1+w*e)))@u.T,np.exp(-.5*np.log1p(w*e).sum()))
    tail=float(g.real*radius**(-GRID))
    archived=json.loads((DATA/"iteration115_visible_v1/high_power.json").read_text())
    truth=np.convolve(*[s["p"] for s in archived["spectra"]])
    n=np.arange(len(truth));fold=np.bincount(n%GRID,weights=truth,minlength=GRID)
    tv=float(np.abs(table-fold).sum()/2)
    # Bound optical truncation and modulo alias separately.
    delta=max(0,1-truth.sum())+truth[GRID:].sum()
    v=[]
    for mask in range(256):
        banks=np.array([b for j,b in enumerate(BANKS) if mask & (1<<j)],int)
        sites=(8*banks[:,None]+np.arange(8)).ravel()
        v.append(np.prod([void(m,sites) for m in mods]))
    word=np.array(v)
    for j in range(8):
        for mask in range(256):
            if not mask&(1<<j):word[mask]-=word[mask|(1<<j)]
    assert word.min()>0 and abs(word.sum()-1)<1e-12
    p=np.load(DATA/"iteration116_sectors/joint.npz")["optical"].sum(0)
    # Every single-detector void also tests total expected click count.
    mean_click=0.
    for site in range(1152):mean_click+=1-np.prod([void(m,np.array([site])) for m in mods])
    results=dict(total_number_tv_lower=tv-delta,total_number_tv_upper=tv+delta+tail,
                 grid=GRID,alias_tail=tail,mean_photons=float(table@np.arange(GRID)),
                 mean_clicks=float(mean_click),eight_bank_word_tv=float(abs(word-p).sum()/2))
    np.savez_compressed(OUT/"laws.npz",number=table,word=word)
    (OUT/"laws.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")
    print(results,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3):run()
