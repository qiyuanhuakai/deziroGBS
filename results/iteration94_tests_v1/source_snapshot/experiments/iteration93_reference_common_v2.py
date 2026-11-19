"""Independent Gaussian Fock, vacuum-subset and total-count references.
References use the frozen physical input; no sampled probabilities are fitted.
High precision checks are numerical checks, not interval arithmetic proofs.
"""
import math,mpmath as mp
from iteration93_common_v5 import *

def inputs93(index):
    if index<3:
        z=np.load(ROOT/f'results/iteration93_general_pilot_v1/input_{index}.npz');return {k:z[k] for k in ('u','t','alpha','noise')}
    eta=.8;r=.85;low=1-eta+eta*(1-r)/(1+r);high=1-eta+eta*(1+r)/(1-r);t=(1-low)/(1+low);noise=np.zeros((4,4));noise[0,0]=noise[3,3]=math.sqrt(high-1/low)/2
    return dict(u=np.array([[1,1],[1,-1]],complex)/math.sqrt(2),t=np.array([t,-t]),alpha=np.zeros(2,complex),noise=noise)

def pgf(t,alpha,z):
    out=mp.mpf(1)
    for tt,aa in zip(t,alpha):
        tt=mp.mpf(float(tt));aa=mp.mpc(complex(aa));tp=z*tt;ap=mp.sqrt(z)*((1-z*tt*tt)*aa-tt*(1-z)*aa.conjugate())/(1-z*z*tt*tt)
        out*=mp.sqrt((1-tt*tt)/(1-tp*tp))*mp.exp(-abs(aa)**2+tt*mp.re(aa.conjugate()**2)+abs(ap)**2-tp*mp.re(ap.conjugate()**2))
    return out

def total_reference(t,alpha,cut):
    result=[mp.mpf(1)]+[mp.mpf(0)]*cut
    for tt,aa in zip(t,alpha):
        tt=mp.mpf(float(tt));aa=mp.mpc(complex(aa));z=aa-tt*aa.conjugate();a=[mp.exp(mp.log(1-tt*tt)/4-abs(aa)**2/2+tt*aa.conjugate()**2/2)]
        for k in range(1,cut+1):a.append((z*a[k-1]+(tt*mp.sqrt(k-1)*a[k-2] if k>1 else 0))/mp.sqrt(k))
        probs=[abs(x)**2 for x in a];result=[sum(result[j]*probs[k-j] for j in range(k+1)) for k in range(cut+1)]
    tail=1-sum(result);assert tail>=-mp.mpf('1e-60')
    zmax=1/max(abs(float(x)) for x in t);grid=[mp.mpf(1)+(mp.mpf(zmax)-1)*j/101 for j in range(1,101)];values=[pgf(t,alpha,z)/z**(cut+1) for z in grid];best=min(range(len(values)),key=values.__getitem__)
    return result,tail,grid[best],values[best]

def click_reference(u,t,alpha,noise=None):
    m,n=u.shape;R=np.block([[u.real,-u.imag],[u.imag,u.real]]);factor=np.diag(np.r_[t/(1-t),-t/(1+t)]);noise=np.zeros((2*n,2*n)) if noise is None else noise
    # Q=I+X, X=(Gamma-I)/2. Classical displacement contributes 2 L L^T.
    X=R@(factor+2*noise@noise.T)@R.T;beta=u@alpha;mean=np.sqrt(2)*np.r_[beta.real,beta.imag];vac=[]
    for mask in range(1<<m):
        if mask==0:vac.append(mp.mpf(1));continue
        idx=[j for j in range(m) if mask>>j&1];idx+= [j+m for j in idx];q=mp.matrix([[mp.mpf(float(X[i,j]))+(i==j) for j in idx] for i in idx]);v=mp.matrix([mp.mpf(float(mean[i])) for i in idx]);vac.append(mp.exp(-(v.T*(q**-1)*v)[0]/2)/mp.sqrt(mp.det(q)) if idx else mp.mpf(1))
    p=[];allmask=(1<<m)-1
    for clicks in range(1<<m):
        zeros=allmask^clicks;sub=clicks;value=mp.mpf(0)
        while True:
            value+=(-1)**sub.bit_count()*vac[zeros|sub]
            if sub==0:break
            sub=(sub-1)&clicks
        assert value>=-mp.mpf('1e-60');p.append(value)
    assert abs(sum(p)-1)<mp.mpf('1e-60');return p,vac

def fock_reference(lib,u,t,alpha,cut):
    m=len(u);B=np.ascontiguousarray(u@np.diag(t)@u.T);beta=u@alpha;linear=np.ascontiguousarray(beta-B@beta.conj());lognorm=np.sum(.25*np.log(1-t*t)-.5*abs(alpha)**2+.5*t*alpha.conj()**2);capacity=math.comb(cut+m,m);keys=np.zeros(capacity,np.uint64);prob=np.zeros(capacity);stats=np.zeros(3);written=C.c_uint64();error=C.create_string_buffer(4096);fn=lib.gaussian93_law;fn.argtypes=[C.c_int,C.c_int,DP,DP,C.c_double,C.c_double,C.c_uint64,C.POINTER(C.c_uint64),DP,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int]
    rc=fn(m,cut,ptr(B.view(float)),ptr(linear.view(float)),lognorm.real,lognorm.imag,capacity,keys.ctypes.data_as(C.POINTER(C.c_uint64)),ptr(prob),C.byref(written),ptr(stats),error,len(error));assert rc==0,error.value;assert written.value==capacity;order=np.argsort(keys);return keys[order],prob[order],stats,B,linear,lognorm
