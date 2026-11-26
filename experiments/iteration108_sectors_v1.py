"""Exact singleton-click and two-photon masses without constructing an output-by-output matrix."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,manifest
from iteration103_gpu_v1 import I,U,CU,P
from iteration106_sample_v1 import DATA
from iteration108_laws_v1 import modes,realify
from iteration108_device_v1 import Global,fingerprint

OUT=DATA/'iteration108_sectors_v1'


def prefix(values):
    return np.r_[0.,np.cumsum(values,dtype=np.longdouble)].astype(float)


def probabilities(law):
    x=law.ravel().clip(0)
    cdf=prefix(x)[1:]/x.sum();cdf[-1]=1.
    return cdf


def prepare_low(t,r,cfg,raw):
    order=cfg['order'];u,x,_=modes(t[order],r);m=len(t);h=int(cfg['h'])
    rank=x.shape[1];spectra=len(x)
    k=[];logvac=0.
    for a in x:
        w,v=np.linalg.eigh(a);k.append((v*(w/(1+w)))@v.T);logvac-=np.log1p(w).sum()/2
    vac=float(np.exp(logvac))
    # Real quadrature rows per physical output, and the same rows multiplied
    # by K in each spectrum. C_ij equals one half the squared Frobenius norm
    # of their 2-by-2 cross block, summed over spectra.
    row=np.stack((np.c_[u.real,-u.imag],np.c_[u.imag,u.real]),axis=1)
    v=np.ascontiguousarray(np.stack([row@a for a in k],axis=1))
    diag=np.einsum('msid,mjd->msij',v,row,optimize=True)
    density=np.trace(diag,axis1=2,axis2=3).sum(1)*.5;density=density.clip(0)
    wdiag=density**2+.5*np.sum(diag**2,axis=(1,2,3))
    _,logdet=np.linalg.slogdet(np.eye(2)[None,None,:,:]-diag)
    single=vac*np.expm1(-.5*logdet.sum(1)).clip(0)
    leverage=np.sum(row**2,axis=(1,2));bound=.5*np.sum(v**2,axis=(1,2,3))
    edges=cfg['bounds'];ranges=[(i,i+1) for i in range(h)]+list(zip(edges[:-1],edges[1:]))+[(2,m)]
    weights=[];distinct=[];totals=[]
    for a,b in ranges:
        # The QR basis has orthonormal columns. For broad groups use the
        # complement when it has fewer rows; singleton rows are rank two.
        if b-a>m//2:
            exc=np.concatenate((row[:a],row[b:])).reshape(-1,rank)
            gram=np.eye(rank)-exc.T@exc
        else:
            local=row[a:b].reshape(-1,rank);gram=local.T@local
        if b-a==1:
            corr=np.einsum('msij,msij->m',v@row[a].T,v@row[a].T,optimize=True)*.5
        elif a==b:corr=np.zeros(m)
        else:
            vv=v.reshape(-1,rank);corr=(np.einsum('ij,ij->i',vv@gram,vv).reshape(m,-1).sum(1)*.5).clip(0)
        mass=(density*density[a:b].sum()+corr).clip(0)
        dis=mass.copy();dis[a:b]=(dis[a:b]-wdiag[a:b]).clip(0)
        weights.append(prefix(mass));distinct.append(prefix(dis));totals.append(mass)
    zones=len(ranges)-1
    mix=np.zeros((1<<zones,2))
    for a,(lo,hi) in enumerate(ranges[:-1]):
        lab=1<<a;mix[lab,0]=single[lo:hi].sum()
        mix[lab,1]=vac*.5*(distinct[a][hi]-distinct[a][lo])
        for b in range(a):
            mix[(1<<a)|(1<<b),1]=vac*totals[b][lo:hi].sum()
    click=raw['click'].clip(0);lowclick=vac+mix.sum()
    assert np.max(mix.sum(1)-click)<1e-9 and lowclick<1+1e-9
    mix=np.divide(mix,click[:,None],out=np.zeros_like(mix),where=click[:,None]>0)
    mix[:,0]=np.minimum(mix[:,0],1);mix[:,1]=np.minimum(mix[:,1],1-mix[:,0])
    cube=raw['pnr'];indices=np.indices(cube.shape,sparse=True)
    by_n=np.bincount((indices[0]+indices[1]+indices[2]).ravel(),weights=cube.ravel())
    n1=vac*density.sum();n2=vac*(np.sum(density)**2+sum(np.sum(a*a) for a in k)*.5)/2
    assert max(abs(by_n[1]-n1),abs(by_n[2]-n2),abs(by_n[0]-vac))<2e-9
    data=dict(ranges=np.asarray(ranges,np.int32),row=np.ascontiguousarray(row),v=v,
        stats=np.ascontiguousarray(np.stack((density,leverage,bound))),
        prefixes=np.ascontiguousarray([prefix(density),prefix(leverage),prefix(single)]),
        first=np.ascontiguousarray([weights,distinct]),mix=np.ascontiguousarray(mix),
        click_obs=probabilities(click),pnr_obs=probabilities(cube))
    metrics=dict(vacuum=vac,single_click_mass=float(single.sum()),distinct_two_from_two=float(lowclick-vac-single.sum()),
                 full_click_tv_bound=float(max(0.,1-lowclick)),one_photon=n1,two_photon=n2,
                 full_pnr_tv_bound=float(max(0.,1-by_n[:3].sum())),number_mass_check=float(abs(by_n[2]-n2)),
                 max_class_overrun=float(max(0.,np.max((mix.sum(1)-1)))))
    return data,metrics


class Sectors:
    def __init__(self,cfg,data,base=None,dll=OUT/'low.dll',workers=12):
        self.cfg=cfg;self.base=base;self.workers=workers;self.data=data
        if os.name=='nt':self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll));self.fn=self.lib.low108_draw
        self.fn.argtypes=[I]*12+[U,CU]+[P]*11
    def draw(self,n,pnr,seed,offset=0):
        if not 0<n<2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid record domain')
        c=self.cfg;d=self.data;m=len(c['order'])
        x=self.base.draw(n,pnr,seed,offset) if self.base is not None else np.zeros((n,m),np.uint8)
        def run(x):
            ptr=[a.ctypes.data for a in (c['order'],d['ranges'],d['row'],d['v'],d['stats'],d['prefixes'],d['first'],d['mix'],d['pnr_obs' if pnr else 'click_obs'],c['shape'],x)]
            return self.fn(m,n,int(c['h']),len(d['ranges'])-1,d['row'].shape[-1],d['v'].shape[1],int(pnr),
                           int(self.base is None),x.itemsize,self.workers,len(d['pnr_obs']),len(d['click_obs']),seed,offset,*ptr)
        rc=run(x)
        if rc==1 and self.base is None:
            x=np.zeros((n,m),np.uint32);rc=run(x)
        if rc:raise ArithmeticError(f'Invalid sparse readout ({rc})')
        return x


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','prepare','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--tables',type=Path);p.add_argument('--low',type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--shots',type=int,default=128);p.add_argument('--pnr',action='store_true');p.add_argument('--hybrid',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True);subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared',str(ROOT/'native/low_sector108_v1.cpp'),'-o',str(a.output/'low.dll')],check=True);manifest(a.output)
        else:
            if a.input is None or a.tables is None:p.error('Specify the optical input and its prepared tables')
            with np.load(a.input) as z:t,r=z['transfer'],np.atleast_2d(z['squeezing'])
            with np.load(a.tables) as z:cfg=dict(z)
            assert str(cfg['input_hash'])==fingerprint(t,r)
            if a.action=='prepare':
                from iteration108_laws_v1 import prepare
                _,_,raw=prepare(t,r,int(cfg['h']),len(cfg['bounds'])-1,cfg['order'])
                data,_=prepare_low(t,r,cfg,raw);data['input_hash']=cfg['input_hash']
                with a.output.open('xb') as f:np.savez_compressed(f,**data)
            else:
                if a.low is None or a.seed is None:p.error('Specify the low-sector tables and an experiment seed')
                with np.load(a.low) as z:data=dict(z)
                assert str(data['input_hash'])==fingerprint(t,r)
                model=Sectors(cfg,data,Global(t,r,cfg) if a.hybrid else None)
                with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
