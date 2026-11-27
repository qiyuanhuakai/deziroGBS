"""Resolve low-count 16-detector PNR atoms and bound the remaining sectors separately."""
from argparse import ArgumentParser
from functools import lru_cache
from itertools import combinations_with_replacement
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits

from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration115_panel_accuracy_v2 import dark_projection
from iteration117_quartic import OUT
from iteration117_quartic_laws import physical_models,marginal,contraction
from iteration115_visible_v1 import DATA

KEYS=[(i,j) for i in range(3) for j in range(3)]
MUL=[(3*i+j,3*k+l,3*(i-k)+j-l) for i,j in KEYS for k in range(i+1) for l in range(j+1)]


def polarization(m):
    eye=np.eye(m)
    return np.block([[eye,eye],[1j*eye,-1j*eye]])/np.sqrt(2)


def gaussian_kernel(x):
    m=len(x)//2;L=polarization(m)
    den=np.eye(len(x))+x
    return (L.T@(np.eye(len(x))-np.linalg.inv(den))@L)[None,:,:],np.array([np.exp(-.5*np.linalg.slogdet(den)[1])])


def core_kernel(model,ports):
    S,norm,t,x=marginal(model,ports)
    m,r=t.shape;L=polarization(m)
    den=np.eye(2*m)+x
    metric=np.linalg.inv(den)
    C=.5*np.vstack((t,-1j*t));C=np.column_stack((C.conj(),C))
    E=metric@C
    J=np.kron(np.array([[0,1],[1,0]]),np.eye(r));H=J-2*C.T@E
    A=np.zeros_like(H);D=A.copy();A[:r,:r]=S.conj();D[r:,r:]=S
    ah=A@H;dh=D@H
    terms={};b=np.zeros((9,2*m,2*m),complex)
    b[0]=L.T@(np.eye(2*m)-metric)@L
    for i,j in KEYS[1:]:
        T=np.zeros_like(H)
        if (i,j)==(1,0):T+=A
        if (i,j)==(0,1):T+=D
        if i and (i-1,j) in terms:T+=ah@terms[i-1,j]
        if j and (i,j-1) in terms:T+=dh@terms[i,j-1]
        terms[i,j]=T;b[3*i+j]=2*L.T@E@T@E.T@L
    p0=contraction(S,norm,t,metric,np.exp(-.5*np.linalg.slogdet(den)[1]),True).ravel()
    return b,p0


def probabilities(kernel,orders,denominator):
    b,p0=kernel;size=len(p0);samples,n=orders.shape
    if n==0:return np.full(samples,float(p0.sum().real))
    vertices=np.column_stack((orders,orders+b.shape[-1]//2))
    def product(a,b):
        if size==1:return a*b
        c=np.zeros((9,samples),complex)
        for k,i,j in MUL:c[k]+=a[i]*b[j]
        return c

    @lru_cache(None)
    def haf(subset):
        if not subset:
            v=np.zeros((size,1),complex);v[0,0]=1.
            return v
        first=(subset&-subset).bit_length()-1
        rest=subset^(1<<first);choice=rest
        val=np.zeros((size,samples),complex)
        while choice:
            flag=choice & -choice;second=flag.bit_length()-1;choice-=flag
            val+=product(b[:,vertices[:,first],vertices[:,second]],haf(rest^flag))
        return val
    values=product(p0[:,None],haf((1<<(2*n))-1)).sum(0)/denominator
    haf.cache_clear()
    assert abs(values.imag).max()<2e-12 and values.real.min()>-1e-13
    return values.real


def both_spectra(laws,order):
    # Split this *occupation* between the two unresolved spectra in every distinct way.
    terms=[((),())]
    for i,count in enumerate(order):
        if count:
            terms=[(a+(i,)*k,b+(i,)*(count-k)) for a,b in terms for k in range(count+1)]
    return sum(laws[0][a]*laws[1][b] for a,b in terms)


def run(cap,transport=False):
    prefix="transport_" if transport else ""
    pair=[34,35];ports=(8*np.array(pair)[:,None]+np.arange(8)).ravel()
    raw,r=actual_input();raw=raw[ports]
    optical=optical_x(raw,r)
    kernels=[*(gaussian_kernel(x) for x in optical),*(core_kernel(m,ports) for m in physical_models(transport=transport))]
    tables=[{} for _ in kernels]
    qarchive=np.load(OUT/(prefix+"panel_34_35.npz"))
    parchive=np.load(DATA/"iteration115_panel_accuracy_v2/pnr_laws.npz")
    bank=[parchive["optical_bank"],qarchive["pnr"]]
    bright=[parchive["projected_bank"],qarchive["projected_pnr"]]
    meta=json.loads((OUT/(prefix+"panel_34_35.json")).read_text())
    keep=[1-meta["optical_dark"],1-meta["candidate_dark"]]
    tv_low=0.;bycount=[]
    for n in range(cap+1):
        patterns=list(combinations_with_replacement(range(16),n))
        dist=[[],[]]
        for start in range(0,len(patterns),256):
            pats=patterns[start:start+256]
            arr=np.array(pats,dtype=int).reshape(len(pats),n)
            occ=[tuple(np.bincount(v,minlength=16)) for v in arr]
            den=np.array([math.prod(math.factorial(k) for k in o if k>1) for o in occ])
            for db,kernel in zip(tables,kernels):
                db.update(zip(pats,probabilities(kernel,arr,den)))
            for j in (0,1):
                dist[j].extend(both_spectra(tables[2*j:2*j+2],o) for o in occ)
        which=np.array([sum(i<8 for i in p) for p in patterns],int)
        residual=0.
        for j in (0,1):
            residual=max(residual,float(abs(np.bincount(which,weights=dist[j],minlength=n+1)-
                               bank[j][np.arange(n+1),n-np.arange(n+1)]).max()))
        assert residual<2e-11,(n,residual)
        tv_low+=.5*abs(np.array(dist[0])-dist[1]).sum()
        total=np.indices(bank[0].shape).sum(0)
        high=total>n
        masses=[d[high] for d in bank]
        projected=[f*d[high] for f,d in zip(keep,bright)]
        # Joint (n_A,n_B) projectors commute with vacuum in dark modes. Purify each
        # subnormalized sector to get ||rho - P rho P||_1/2 <= sqrt(p*(p-k)).
        # Pool sectors for the disturbance bound to avoid amplification of roundoff in tiny cells.
        disturbance=sum(np.sqrt(max(0.,m.sum()*(m.sum()-v.sum()))) for m,v in zip(masses,projected))
        high_bright=.5*abs(projected[0]-projected[1]).sum()
        coarse=.5*abs(masses[0]-masses[1]).sum()
        # Aliasing below 128 perturbs each high mass, including its dark difference.
        # The high block totals come from the independently computed modulo laws.
        eps=max(meta["candidate_alias"],2e-16)
        # Four pooled masses can change by eps; also cover the two bright laws.
        margin=2*np.sqrt(2*eps)+4*eps
        row=dict(max_count=n,atoms=len(patterns),low_tv=float(tv_low),coarse_high=float(coarse),
                 high_bright=float(high_bright),disturbance=float(disturbance),alias_margin=float(margin),
                 pnr_lower=float(tv_low+coarse-4*eps),pnr_upper=float(tv_low+high_bright+disturbance+margin),
                 bank_mass_crosscheck=residual)
        bycount.append(row);print(n,row["pnr_lower"],row["pnr_upper"],flush=True)
    (OUT/(prefix+"resolved_pnr.json")).write_text(json.dumps(bycount,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=ArgumentParser();parser.add_argument("--cap",type=int,default=5)
    parser.add_argument("--transport",action="store_true");opts=parser.parse_args()
    with threadpool_limits(limits=1):run(opts.cap,opts.transport)
