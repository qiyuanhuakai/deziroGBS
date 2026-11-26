"""Infer two continuation thresholds per spatial group from all optical 3-by-3 count tables."""
import argparse
import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT,ORDER,realify,bind,I,D,J,save,manifest,H
from iteration102_forest_v1 import DATA,VAC,FIT as PREVIOUS,Tables

FIT=DATA/'iteration103_fit_v1'


class Orthant:
    def __init__(self,dll):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll))
        self.fit=bind(self.lib,'continuation103_fit',[I,D,D,I,D,D,I,D],None)
        self.evaluate=bind(self.lib,'continuation103_eval',[I,D,D,J,I,D,D,I,D],None)
    @staticmethod
    def rule(size):
        x,w=leggauss(size)
        return (x+1)/2,w/2
    def cdf(self,x,r,dims,n=16,workers=12):
        nodes,weights=self.rule(n)
        y=np.empty(len(x))
        self.evaluate(len(x),np.ascontiguousarray(x),np.ascontiguousarray(r),np.ascontiguousarray(dims,np.int32),n,nodes,weights,workers,y)
        return y


def optical_cells(tables):
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    m=np.array([realify(v[g*8:g*8+8]) for g in range(len(tables.order))])
    a,b=np.triu_indices(len(m),1)
    logp=np.zeros(len(a))
    terms=np.zeros((len(a),3))
    for x in z['x']:
        mx=m@x
        local=np.eye(16)+mx@m.transpose(0,2,1)
        for k in range(0,len(a),128):
            i,j=a[k:k+128],b[k:k+128]
            block=np.empty((len(i),32,32))
            block[:,:16,:16],block[:,16:,16:]=local[i],local[j]
            cross=mx[i]@m[j].transpose(0,2,1)
            block[:,:16,16:],block[:,16:,:16]=cross,cross.transpose(0,2,1)
            logp[k:k+len(i)]-=.5*np.linalg.slogdet(block)[1]
            kernel=np.eye(32)-np.linalg.inv(block)
            terms[k:k+len(i),0]+=.5*np.trace(kernel[:,:16,:16],axis1=1,axis2=2)
            terms[k:k+len(i),1]+=.5*np.trace(kernel[:,16:,16:],axis1=1,axis2=2)
            terms[k:k+len(i),2]+=.5*(kernel[:,:16,16:]**2).sum((1,2))
    p00=np.exp(logp)
    p10,p01=p00*terms[:,0],p00*terms[:,1]
    p11=p00*(terms[:,0]*terms[:,1]+terms[:,2])
    p=np.array([tables.marg[g][:2] for g in range(len(m))])
    s11=1-p[a,0]-p[b,0]+p00
    s12=s11-p[b,1]+p01
    s21=s11-p[a,1]+p10
    s22=s11-p[a,1]-p[b,1]+p01+p10+p11
    scan=np.load(VAC/'scan.npz')
    assert np.array_equal(a,scan['a']) and abs(p00-scan['optical']).max()<3e-13
    return a,b,p,np.column_stack([s11,s12,s21,s22])


def whiten(m):
    n=len(m)//2
    blocks=m.reshape(n,2,n,2)[np.arange(n),:,np.arange(n),:]
    inv=np.linalg.inv(np.linalg.cholesky(blocks))
    return np.einsum('iaj,ijhk,hbk->iahb',inv,m.reshape(n,2,n,2),inv,optimize=True).reshape(2*n,2*n)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/continuation103_v1.cpp'
    dll=out/'orthant.dll'
    subprocess.run([shutil.which('g++'),'-std=c++20','-shared','-O3','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)],check=True)
    solver=Orthant(dll)
    t=Tables(PREVIOUS/'prepared.npz')
    start=perf_counter()
    a,b,p,target=optical_cells(t)
    marginal=np.column_stack([1-p[:,0],(1-p.sum(1))/(1-p[:,0])])
    cut=ndtri(marginal)
    x=np.column_stack([cut[a],cut[b]])
    r=np.zeros_like(target)
    nodes,w=solver.rule(16)
    solver.fit(len(x),x,target,16,nodes,w,12,r)
    raw=np.eye(len(p)*2)
    for i in range(2):
        for j in range(2):
            row=2*a+i
            col=2*b+j
            raw[row,col]=raw[col,row]=r[:,2*i+j]
    e,v=np.linalg.eigh(raw)
    corr=raw.copy()
    if e[0]<1e-8:
        corr=whiten((v*e.clip(1e-8))@v.T)
    final=np.column_stack([corr[2*a+i,2*b+j] for i in range(2) for j in range(2)])
    regions=np.array([(i,j) for i in (1,2) for j in (1,2)],np.int32)
    def verify(params):
        xx=np.repeat(x,4,axis=0)
        rr=np.repeat(params,4,axis=0)
        return solver.cdf(xx,rr,np.tile(regions,(len(x),1)),24).reshape(-1,4)
    ideal=verify(r)
    found=verify(final)
    assert np.isfinite(found).all()
    def cells(s):
        tails=np.ones((len(a),3,3))
        tails[:,1:,0]=np.c_[marginal[a,0],marginal[a].prod(1)]
        tails[:,0,1:]=np.c_[marginal[b,0],marginal[b].prod(1)]
        tails[:,1:,1:]=s.reshape(-1,2,2)
        padded=np.pad(tails,((0,0),(0,1),(0,1)))
        return np.diff(np.diff(padded,axis=1),axis=2)
    tv=abs(cells(found)-cells(target)).sum((1,2))/2
    np.savez_compressed(out/'fit.npz',corr=corr,raw=raw,cut=cut,p=p,a=a,b=b,target=target,achieved=found,tv=tv)
    summary=dict(pairs=len(a),seconds=perf_counter()-start,min_eig_raw=float(e[0]),negative_eigenvalues=int((e<0).sum()),
        min_eig=float(np.linalg.eigvalsh(corr)[0]),max_block_change=float(abs(final-r).max()),
        unprojected_survival_max=float(abs(ideal-target).max()),survival_max=float(abs(found-target).max()),
        vacuum_max=float(abs(found[:,0]-target[:,0]).max()),max_count3_tv=float(tv.max()),
        median_count3_tv=float(np.median(tv)),worst=[int(a[tv.argmax()]),int(b[tv.argmax()])],
        hashes={f.relative_to(ROOT).as_posix():H(f) for f in (src,Path(__file__),ORDER,PREVIOUS/'prepared.npz')})
    save(out,'summary.json',summary)
    manifest(out)
    print(summary,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        prepare(ap.parse_args().output.resolve())
