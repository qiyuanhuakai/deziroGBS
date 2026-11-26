"""All physical two-group count distributions and Hermite projection of the raw continuation sampler."""
import argparse
import ctypes as ct
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT,ORDER,Native,realify,local_kernel,cdf_rows,realized,I,D,bind,save,manifest,H
from iteration101_copula_v1 import features
from iteration102_forest_v1 import DATA,BASE,Tables,FIT as OLD
from iteration103_fit_v1 import FIT,Orthant

CENSUS=DATA/'iteration103_census_v1'


class CountLaw:
    def __init__(self,dll):
        self.runtime=Native(BASE/'spatial.dll')
        self.transfer=bind(ct.CDLL(str(dll)),'chaos103_transfer',[I,D,D])
        t=Tables(OLD/'prepared.npz')
        self.p=[realized(cdf_rows(t.marg[g])) for g in range(len(t.order))]
        self.corr=np.load(FIT/'fit.npz')['corr']
        self.basis=[]
        for p in self.p:
            _,occ,_=features(np.array([p[0],p[1:].sum()]))
            _,positive,_=features(p[1:])
            terms=[]
            for k in range(occ.shape[1]):
                row=np.zeros((len(p),k+1))
                row[0,0]=occ[0,k]
                row[1:]=positive[:,:k+1]*occ[1,k::-1]
                terms.append(np.ascontiguousarray(row))
            self.basis.append(terms)

    def pair(self,g,h,tol=1e-11):
        r=np.ascontiguousarray(self.corr[2*g:2*g+2,2*h:2*h+2])
        rho=np.linalg.svd(r,compute_uv=False)[0]
        # Parseval for each indicator; the tail's operator norm is at most rho^(k+1).
        pref=.5*np.sqrt(self.p[g]*(1-self.p[g])).sum()*np.sqrt(self.p[h]*(1-self.p[h])).sum()
        degree=0 if rho==0 else min(len(self.basis[g])-1,max(0,int(math.ceil(math.log(tol/pref)/math.log(rho)))-1))
        size=(degree+1)*(degree+2)*(2*degree+3)//6
        operators=np.empty(size)
        assert self.transfer(degree,r,operators)==0
        joint=np.outer(self.p[g],self.p[h])
        off=1
        for k in range(1,degree+1):
            length=(k+1)**2
            m=operators[off:off+length].reshape(k+1,k+1)
            joint+=self.basis[g][k]@m@self.basis[h][k].T
            off+=length
        return joint,float(pref*rho**(degree+1)),degree


class Optical:
    def __init__(self):
        self.native=Native(BASE/'spatial.dll')
        z=np.load(ORDER)
        v=z['v'][np.argsort(z['order'])]
        m=np.array([realify(v[i:i+8]) for i in range(0,len(v),8)])
        self.local=[]
        for x in z['x']:
            n=(m@x).reshape(-1,100)@m.reshape(-1,100).T
            self.local.append(n.reshape(len(m),16,len(m),16).transpose(0,2,1,3))
        self.caps=Tables(OLD/'prepared.npz').caps

    def pair(self,a,b):
        ca,cb=int(self.caps[a]),int(self.caps[b])
        ny=1<<cb.bit_length()
        y=np.exp(-2j*np.pi*np.arange(ny//2+1)/ny)
        y[-1]=-1
        eig=[]
        pref=np.ones(len(y),complex)
        for q in self.local:
            x=np.block([[q[a,a],q[a,b]],[q[b,a],q[b,b]]])
            k,logp,_=local_kernel(x)
            s,v=np.linalg.eigh(k[16:,16:])
            cross=k[:16,16:]@v
            phase=y[:,None]/(1-y[:,None]*s)
            eff=k[:16,:16]+(cross*phase[:,None,:])@cross.T
            eig.append(np.linalg.eigvals(eff))
            pref*=np.exp(logp-.5*np.log1p(-y[:,None]*s).sum(1))
        roots=np.ascontiguousarray(np.column_stack(eig))
        co=np.empty((len(y),ca+1),complex)
        assert self.native.complex(len(y),roots.shape[1],ca,roots.view(float),pref.view(float),co.view(float))==0
        return np.fft.irfft(co,n=ny,axis=0).T[:,:cb+1].copy()


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/count_chaos103_v1.cpp'
    dll=out/'chaos.dll'
    subprocess.run([shutil.which('g++'),'-O3','-shared','-std=c++20',str(src),'-o',str(dll)],check=True)
    law=CountLaw(dll)
    optical=Optical()
    fit=np.load(FIT/'fit.npz')
    a,b=fit['a'],fit['b']
    cap=6
    def coarse(mat):
        ix=np.minimum(np.arange(len(mat)),cap)[:,None]*(cap+1)+np.minimum(np.arange(mat.shape[1]),cap)
        return np.bincount(ix.ravel(),weights=mat.ravel(),minlength=(cap+1)**2).reshape(cap+1,cap+1)
    def one(i):
        g,h=int(a[i]),int(b[i])
        model,tail,k=law.pair(g,h)
        ref=optical.pair(g,h)
        check=max(abs(ref.sum(1)-law.p[g]).sum(),abs(ref.sum(0)-law.p[h]).sum())
        clipped=coarse(ref)
        predicted=coarse(model)
        cumul=np.array([ref[s:,t:].sum() for s in (1,2) for t in (1,2)])
        independent=abs(cumul-fit['target'][i]).max()
        assert max(check,independent)<5e-12,(g,h,check,independent)
        assert abs(model.sum()-1)<2e-12 and -model[model<0].sum()<tail+2e-12
        return abs(ref-model).sum()/2,tail,k,check,independent,clipped,predicted
    start=perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        vals=list(pool.map(one,range(len(a))))
    tv,tail,degree,err,ind,coarse_ref,coarse_model=map(np.array,zip(*vals))
    np.savez_compressed(out/'scan.npz',a=a,b=b,tv=tv,tail=tail,degree=degree,
        coarse_optical=coarse_ref,coarse_model=coarse_model)
    sel=np.argsort(-tv)[:6]
    old_ref=DATA/'iteration102_projection_v1'
    discrepancy=0.
    for path in old_ref.glob('pair_*.npz'):
        _,g,h=path.stem.split('_')
        ref=optical.pair(int(g),int(h))
        discrepancy=max(discrepancy,float(abs(ref-np.load(path)['reference']).sum()/2))
    assert discrepancy<5e-12
    save(out,'summary.json',dict(pairs=len(a),seconds=perf_counter()-start,percentiles=np.percentile(tv,[0,50,90,99,100]).tolist(),
        largest=[dict(groups=[int(a[i]),int(b[i])],count_tv=float(tv[i])) for i in sel],
        degree_range=[int(degree.min()),int(degree.max())],series_tv_bound_max=float(tail.max()),
        max_optical_marginal_l1=float(err.max()),independent_threshold_check=float(ind.max()),archived_projection_difference=discrepancy))
    manifest(out)
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
