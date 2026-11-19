"""Exact small-box references for one or two independent Gaussian drives."""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import numpy as np
from .exact_mps_law import integer_tensors
from .gaussian_fock_bridge import physical_pairing,positive_inverse,root_interval,cm
from .extended_displaced_fock import add,multiply,cmul,cadd,reciprocal_root_integer
from .rank_one_noise import direction_values,interval_law

def parameters(directions,sigmas,n):
    if not 1<=len(directions)<=2 or len(sigmas)!=len(directions):raise ValueError('One or two Gaussian drives required')
    v=[direction_values(v,n) for v in directions];s=[F(x) for x in sigmas]
    if not all(0<=x<=F(1,2) for x in s):raise ValueError('Noise scales in0..1/2 required')
    return v,s

def polynomial_product(a,b):
    out={}
    for ka,va in a.items():
        for kb,vb in b.items():
            key=tuple(x+y for x,y in zip(ka,kb));x,y=cm(va,vb);u,v=out.get(key,(F(0),F(0)));out[key]=(u+x,v+y)
    return {k:v for k,v in out.items() if v!=(0,0)}

def normal_moments(covariance):
    K=tuple(tuple(F(x) for x in row) for row in covariance);r=len(K)
    @lru_cache(None)
    def moment(p):
        if len(p)!=r or any(type(x) is not int or x<0 for x in p):raise ValueError('Nonnegative multi-index required')
        if not any(p):return F(1)
        if sum(p)%2:return F(0)
        i=next(j for j,x in enumerate(p) if x);q=list(p);q[i]-=1;value=F(0)
        for j,count in enumerate(q):
            if count:
                child=q.copy();child[j]-=1;value+=count*K[i][j]*moment(tuple(child))
        return value
    return moment

def integrated_linear_law(tensors,directions,sigmas,*,bits=192):
    n=len(tensors)
    if not 1<=n<=4 or any(t.ndim!=3 or not 1<=t.shape[1]<=8 for t in tensors):raise ValueError('One..four finite modes, dimension1..8 required')
    v,s=parameters(directions,sigmas,n);r=len(v);zero=(0,)*r;one={zero:(F(1),F(0))};dims=[int(t.shape[1]) for t in tensors];integer=integer_tensors(tensors,max_states=4096);occupations=list(np.ndindex(*dims));amps=[]
    for occ in occupations:
        re=np.ones(1,dtype=object);im=np.zeros(1,dtype=object)
        for k,(ar,ai,scale) in zip(occ,integer):re,im=re@ar[:,k,:]-im@ai[:,k,:],re@ai[:,k,:]+im@ar[:,k,:]
        amps.append((int(re[0]),int(im[0])))
    norm=sum(a*a+b*b for a,b in amps)
    if not norm:raise ValueError('Nonzero finite state required')
    powers=[]
    for j,d in enumerate(dims):
        linear={tuple(int(k==a) for k in range(r)):(-s[a]*v[a][j][0],s[a]*v[a][j][1]) for a in range(r)};p=[one]
        for k in range(1,d):p.append(polynomial_product(p[-1],linear))
        powers.append(p)
    records=[];vac=[];czero=((F(0),F(0)),(F(0),F(0)))
    for mask in range(1<<n):
        G=[[s[a]*s[b]*sum(v[a][j][0]*v[b][j][0]+v[a][j][1]*v[b][j][1] for j in range(n) if mask>>j&1) for b in range(r)] for a in range(r)];T=[[F(a==b)+2*G[a][b] for b in range(r)] for a in range(r)];K,det,_=positive_inverse(T);moment=normal_moments(K);env={}
        for occ,amp in zip(occupations,amps):
            if amp==(0,0):continue
            poly={zero:(F(amp[0]),F(amp[1]))};fact=1
            for j,k in enumerate(occ):
                if mask>>j&1:poly=polynomial_product(poly,powers[j][k]);fact*=factorial(k)
            root=reciprocal_root_integer(fact,bits);key=tuple(k for j,k in enumerate(occ) if not mask>>j&1);environment=env.setdefault(key,{})
            for degree,(a,b) in poly.items():environment[degree]=cadd(environment.get(degree,czero),(multiply((a,a),root),multiply((b,b),root)))
        coefficients={}
        for poly in env.values():
            for pa,(ar,ai) in poly.items():
                for pb,(br,bi) in poly.items():
                    degree=tuple(a+b for a,b in zip(pa,pb));coefficients[degree]=add(coefficients.get(degree,(F(0),F(0))),add(multiply(ar,br),multiply(ai,bi)))
        value=(F(0),F(0));terms=[]
        for degree,c in sorted(coefficients.items()):
            M=moment(degree);value=add(value,multiply(c,(M/norm,M/norm)));terms.append(dict(degree=degree,coefficient=c,moment=M))
        factor=root_interval(1/det,bits);raw=multiply(value,factor);p=(max(F(0),raw[0]),min(F(1),raw[1]));vac.append(p);records.append(dict(mask=mask,tilt=T,tilted_standard_covariance=K,determinant=det,terms=terms,coefficient_interval=value,normalization_interval=factor,raw_vacuum_interval=raw))
    raw,law=interval_law(vac);return dict(directions=v,sigmas=s,dimensions=dims,integer_amplitudes=amps,integer_norm=norm,records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law,scope='Exact continuous finite-state Gaussian-drive integral, rank up to2; no quadrature or output cutoff')

def gaussian_linear_target(real,imag,directions,sigmas,*,bits=192):
    physical=physical_pairing(real,imag);n=len(real);v,s=parameters(directions,sigmas,n);r=len(v);w=[[a for a,b in z]+[b for a,b in z] for z in v];V=physical['covariance'];C=[[V[i][j]+4*sum(s[a]*s[a]*w[a][i]*w[a][j] for a in range(r)) for j in range(2*n)] for i in range(2*n)];u=[z[n:]+[-x for x in z[:n]] for z in w];gram=[[sum(u[a][i]*V[i][j]*u[b][j] for i in range(2*n) for j in range(2*n)) for b in range(r)] for a in range(r)];records=[];vac=[]
    for mask in range(1<<n):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes]
        if ix:
            Q=[[(C[i][j]+int(i==j))/2 for j in ix] for i in ix];_,det,_=positive_inverse(Q);Q0=[[(V[i][j]+int(i==j))/2 for j in ix] for i in ix];inv,det0,_=positive_inverse(Q0);H=[[sum(w[a][i]*inv[k][l]*w[b][j] for k,i in enumerate(ix) for l,j in enumerate(ix)) for b in range(r)] for a in range(r)]
        else:det=det0=F(1);H=[[F(0)]*r for _ in range(r)]
        vac.append(root_interval(1/det,bits));records.append(dict(mask=mask,determinant=det,pure_determinant=det0,exponent_matrix=H,pure_factor_interval=root_interval(1/det0,bits)))
    raw,law=interval_law(vac);return dict(physical=physical,directions=v,sigmas=s,covariance=C,generator_gram=gram,records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law)

def product_coupling_bound(target,grids):
    if len(grids)!=len(target['sigmas']) or any(g['sigma']!=s for g,s in zip(grids,target['sigmas'])):raise ValueError('Matching independent grids required')
    terms=[root_interval(target['generator_gram'][j][j])[1]*((1-g['tail_interval'][0])*g['interior_half_width']+g['tail_mean_excess_interval'][1]) for j,g in enumerate(grids)]
    return dict(terms=terms,upper=min(F(1),sum(terms)))
