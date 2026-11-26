"""Arbitrary-width connected GBS family with one occupied squeezed supermode in each distinguishable spectrum."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,Native,I,U,D,T,J,W,bind,save,manifest
from iteration97_forest_v1 import alias_table
from scipy.linalg import null_space
from iteration100_baselines_v1 import frugal,pair_reference,pe_sample,GaussianInput
from iteration103_sample_v1 import DATA,BASE

OUT=DATA/'iteration104_spectral_v1'
SCALE=1<<53


def discrete(p):
    total=mp.fsum(p)
    cumulative=mp.mpf(0)
    cut=[0]
    for v in p:
        cumulative+=v
        cut.append(int(mp.nint(SCALE*cumulative/total)))
    cut[-1]=SCALE
    mass=np.diff(np.array(cut,np.int64))
    assert (mass>=0).all()
    err=mp.fsum(abs(mp.mpf(int(n))/SCALE-v/total) for n,v in zip(mass,p))/2
    bits,threshold,alt,keys,alias_tv,_=alias_table({i:int(w) for i,w in enumerate(mass)})
    assert alias_tv==0 and np.array_equal(keys[:len(p)],np.arange(len(p)))
    return bits,threshold,keys[alt].astype(np.uint32),float(err)


def count_law(r,eta):
    plus=eta*mp.expm1(2*r)/2
    minus=eta*mp.expm1(-2*r)/2
    a=(1+plus)*(1+minus)
    b=-(plus+minus+2*plus*minus)
    c=plus*minus
    pmf=[1/mp.sqrt(a)]
    summed=pmf[0]
    while 1-summed>mp.mpf('1e-17'):
        n=len(pmf)
        v=(-(2*n-1)*b*pmf[-1]/2-(n-1)*c*(pmf[-2] if n>=2 else 0))/(a*n)
        if v<0:raise ArithmeticError('Negative photon probability')
        pmf.append(v)
        summed+=v
    tail=max(mp.mpf(0),1-summed)
    pmf[-1]+=tail
    return pmf,tail


class Allocation:
    def __init__(self,dll,intensity,squeezing,efficiency):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.native=bind(ct.CDLL(str(dll)),'spectral104_draw',[I,I,I,I,I,U,ct.c_uint,I,J,J,W,T,W,T,T])
        profiles=np.asarray(intensity,float)
        if profiles.ndim!=2:raise ValueError('Expected spectra by output ports')
        self.s,self.m=profiles.shape
        if not 0<self.s<=65536 or not 0<self.m<=2**30 or len(squeezing)!=self.s or len(efficiency)!=self.s:raise ValueError('Invalid optical dimensions')
        ns=[];ts=[];na=[];pa=[];pc=[];bounds=[];caps=[]
        with mp.workdps(65):
            for row,r,eta in zip(profiles,squeezing,efficiency):
                if not np.isfinite(row).all() or min(row)<0 or max(row)<=0 or not np.isfinite([r,eta]).all() or not 0<=eta<=1:
                    raise ValueError('Invalid intensity, squeezing or efficiency')
                r=mp.mpf(float(abs(r)))
                eta=mp.mpf(float(eta))
                prob,tail=count_law(r,eta)
                bits,cut,alt,ntv=discrete(prob)
                self.port_bits,pct,palt,ptv=discrete([mp.mpf(float(v)) for v in row])
                bounds.append(float(tail+ntv+eta*mp.sinh(r)**2*ptv))
                ns.append(bits);ts.append(cut);na.append(alt);pc.append(pct);pa.append(palt);caps.append(len(prob)-1)
        if sum(caps)>2**32-1:raise ValueError('A record may overflow uint32')
        self.bound=sum(bounds);self.caps=caps
        self.nbits=np.array(ns,np.int32)
        lengths=[len(t) for t in ts]
        if sum(lengths)>2**31-1:raise ValueError('Photon alias offsets exceed int32')
        self.starts=np.cumsum([0]+lengths[:-1]).astype(np.int32)
        self.tc=np.concatenate(ts);self.ta=np.concatenate(na)
        self.pc=np.concatenate(pc);self.pa=np.concatenate(pa)
    def draw(self,n,pnr,seed,offset=0,workers=1):
        if n<1 or n>=2**31 or not 0<=seed<2**64 or offset<0 or offset+n>2**32 or workers<1:raise ValueError('Invalid sample request')
        x=np.empty((n,self.m),np.uint32)
        status=self.native(self.m,self.s,n,int(pnr),workers,seed,offset,self.port_bits,self.nbits,self.starts,self.tc,self.ta,self.pc,self.pa,x)
        if status:raise RuntimeError(status)
        return x


def example(m):
    j=np.arange(m)
    w=np.array([np.exp(-j/(.19*m))+0.15,np.exp(-(m-1-j)/(.22*m))+.08])
    w[0,0]+=.11*w[0].sum()
    w[1,-1]+=.14*w[1].sum()
    return w,np.array([1.65,1.31]),np.array([.81,.72])


def covariance(w,r,eta):
    z=[]
    for row,s,e in zip(w,r,eta):
        profile=np.sqrt(row/row.sum())
        uu=np.outer(profile,profile)
        m=len(row)
        q=np.eye(2*m)
        q[:m,:m]+=e*np.expm1(-2*s)*uu/2
        q[m:,m:]+=e*np.expm1(2*s)*uu/2
        z.append(q)
    return np.array(z)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    dll=out/'allocation.dll'
    src=ROOT/'native/spectral_allocation104_v1.cpp'
    subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-fopenmp','-shared',str(src),'-o',str(dll)],check=True)
    native=Native(BASE/'spatial.dll')
    rows=[]
    for m in (1,4,16,144,1152,4097,16384):
        w,r,eta=example(m)
        tic=perf_counter();model=Allocation(dll,w,r,eta);prep=perf_counter()-tic
        x=model.draw(257,False,104035000+m,workers=1)
        y=model.draw(201,True,104035000+m,56,16)
        assert np.array_equal(x[56:],y>0)
        for pnr in (False,True):
            for threads in ([1,16] if m==16 else [16]):
                n=2048 if m<8192 else 512
                times=[]
                for rep in range(9):
                    tic=perf_counter()
                    model.draw(n,pnr,104036000+m,rep*n,threads)
                    if rep>1:times.append((perf_counter()-tic)*1e6/n)
                rows.append(dict(m=m,pnr=pnr,workers=threads,us=float(np.median(times)),timing=times,batch=n,load_s=prep,full_tv_upper=model.bound,max_photons=model.caps))
        if m!=16:continue
        np.savez_compressed(out/'input16.npz',intensity=w,squeezing=r,efficiency=eta)
        q=covariance(w,r,eta)
        oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
        truth=np.empty(65536)
        assert oracle(16,2,q,np.zeros((2,32)),truth,np.empty(2))==0
        # Independent analytic vacuum probabilities for every subset, followed
        # by a high-precision subset transform of the complete 16-bit law.
        with mp.workdps(50):
            subset=np.arange(65536)
            ref=np.ones(65536,dtype=object)
            for raw,sr,ef in zip(w,r,eta):
                pp=[mp.mpf(float(x)) for x in raw];den=mp.fsum(pp)
                mass=[mp.mpf(0)]
                for v in pp:mass+=list(t+v/den for t in mass)
                a=mp.mpf(float(ef))*mp.expm1(-2*float(sr))/2
                b=mp.mpf(float(ef))*mp.expm1(2*float(sr))/2
                ref*=np.array([1/mp.sqrt((1+a*t)*(1+b*t)) for t in mass[::-1]],object)
            for j in range(16):
                blocks=ref.reshape(-1,2,1<<j)
                blocks[:,1]-=blocks[:,0]
            check=float(abs(ref.astype(float)-truth).sum()/2)
        assert check<1e-10
        peer=[]
        fl=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
        for k in (2,4):
            dist,times=frugal(fl,q,k,104037000+20*k)
            peer.append(dict(K=k,us=float(np.median(times)),tv=float(abs(dist-truth).sum()/2)))
        ids=[0,15,16,31]
        reduced=np.ascontiguousarray(q[:,ids][:,:,ids])
        p=pair_reference(reduced,12)
        states=[]
        for profile,sr,ef,target in zip(w,r,eta,reduced):
            kept=ef*profile[[0,15]]/profile.sum()
            u=np.sqrt(np.r_[kept,1-kept.sum()])
            modes=np.column_stack([u,null_space(u[None,:])]).astype(complex)
            assert np.max(abs(modes.conj().T@modes-np.eye(3)))<1e-12
            joint=covariance((u*u)[None,:],[sr],[1.])[0]
            kept_ids=[0,1,3,4]
            assert np.max(abs(joint[np.ix_(kept_ids,kept_ids)]-target))<2e-14
            states.append((GaussianInput(np.array([sr]),(0,),3),modes))
        def pe(n,seed):
            counts=np.zeros((n,2),np.uint32)
            for j,(state,u) in enumerate(states):
                v,st=pe_sample(state,u,n,seed=seed+j,device='cpu',impl='v2',window=4,max_photons=128,return_stats=True)
                assert not np.any(st.censored)
                counts+=np.asarray(v[:,:2],np.uint32)
            return counts
        times=[];ours=[];vals=[];local=[]
        for j in range(17):
            tic=perf_counter();a=pe(2048,104038000+2*j)
            if j:times.append((perf_counter()-tic)*1e6/2048);vals.append(a)
            tic=perf_counter();b=model.draw(2048,True,104039000,2048*j)[:,[0,15]].copy()
            if j:ours.append((perf_counter()-tic)*1e6/2048);local.append(b)
        def empirical(arr):
            code=np.concatenate(arr).clip(0,12)@np.array([1,13])
            return float(abs(np.bincount(code,minlength=169)/len(code)-p).sum()/2)
        base=dict(frugal=peer,independent_click_reference_tv=check,pe_us=float(np.median(times)),projected_pnr_us=float(np.median(ours)),pe_empirical_tv=empirical(vals),projected_empirical_tv=empirical(local),pnr_shots=32768)
    save(out,'summary.json',dict(rows=rows,comparison=base,input='Two distinguishable squeezed supermodes with nonuniform overlapping output profiles; passive loss per spectrum.',squeezing=[1.65,1.31],efficiencies=[.81,.72]))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('benchmark','draw'))
    p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--seed',type=int)
    p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--pnr',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='benchmark':run(a.output.resolve())
        else:
            if a.seed is None or a.input is None:p.error('--input and --seed are required')
            with np.load(a.input) as z:model=Allocation(OUT/'allocation.dll',z['intensity'],z['squeezing'],z['efficiency'])
            x=model.draw(a.shots,a.pnr,a.seed,workers=a.workers)
            with a.output.open('xb') as f:np.save(f,x)
