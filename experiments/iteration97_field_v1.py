"""Blind coherent-state approximation preserving the normal optical correlations."""
import argparse
import ctypes as ct
import subprocess,shutil
from time import perf_counter
from pathlib import Path
import numpy as np
import scipy.linalg.cython_blas as blas
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
from iteration97_conditionals_v1 import Kernel,Optical,ROOT,I,U,D,B,P,bind,H,save,manifest
from iteration96_group_benchmark_v1 import ACTUAL,ORDER,HIST,moments
SHORT=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')
def field_factor(t,noise,method):
    n=len(t);a=np.sqrt(2)*noise
    q0=np.diag(1+np.r_[t/(1-t),-t/(1+t)])
    p=q0-np.eye(2*n)+a@a.T
    if method=='displaced_vacuum':b=a.copy()
    else:
        omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
        phase=(p+omega@p@omega.T)/2
        val,vec=np.linalg.eigh(phase);active=val>1e-13
        root=vec[:,active]*np.sqrt(val[active])
        inv=vec[:,active]/np.sqrt(val[active])
        eig,v=np.linalg.eigh(inv.T@(p-phase)@inv)
        b=(root@v)*np.sqrt(1+np.clip(eig,-1,1))
        b=np.pad(b,((0,0),(0,2*n-b.shape[1])))
        u,_,vt=np.linalg.svd(b.T@np.linalg.solve(q0,a));b=b@(u@vt)
        classical=b@b.T
        assert np.max(abs((classical+omega@classical@omega.T)/2-phase))<1e-10
    delta=b-a
    log_f=.5*np.log1p(-t*t).sum()-np.linalg.slogdet(np.eye(2*n)+.5*delta.T@np.linalg.solve(q0,delta))[1]
    # Compress numerical null directions; the loss contributes a separate coherent-state coupling bound.
    u,s,_=np.linalg.svd(b,full_matrices=False);keep=s>1e-11
    return u[:,keep]*s[keep],float(log_f),float(np.linalg.norm(s[~keep])/np.sqrt(2))

class Field:
    def __init__(self,dll,factor,mean,m):
        self.factor=np.ascontiguousarray(factor.T);self.mean=np.ascontiguousarray(mean);self.m=m
        self.sectors=len(mean)//(2*m)
        self.runtime=Kernel();self.lib=ct.CDLL(str(dll))
        self.draw=bind(self.lib,'field97_draw',[I,I,I,I,I,I,U,U,D,D,P,SHORT,B])
        cap=blas.__pyx_capi__['dgemm'];cname=ct.pythonapi.PyCapsule_GetName;cname.argtypes=[ct.py_object];cname.restype=ct.c_char_p
        get=ct.pythonapi.PyCapsule_GetPointer;get.argtypes=[ct.py_object,ct.c_char_p];get.restype=P
        self.gemm=get(cap,cname(cap))
    def sample(self,n,route,workers,seed,offset):
        x=np.empty((n,self.m),np.uint16);err=np.empty(n,np.uint8)
        if self.draw(self.m,self.sectors,len(self.factor),n,route,workers,seed,offset,self.factor,self.mean,self.gemm,x,err):raise RuntimeError('field call')
        return x,err
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);out=ap.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False);src=ROOT/'native/gaussian_coherent_field97_v1.cpp'
    ref=ROOT/'results/iteration96_quad_v1/actual16.npz';ph=ROOT/'results/iteration84_photon_cdf_v1/reference.npz'
    files=[Path(__file__),src,ROOT/'native/gaussian_rng_domain84_v1.hpp',ORDER,HIST,ref,ph,*(ACTUAL/f'input_{i}.npz' for i in range(2))]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},quality=65536,
        methods=['displaced_vacuum','normal_preserving'],routes=['direct','photon'],seed=97080000,
        intervals='four complete simulated CDFs plus one hardware CDF, alpha=.01/5 each, assumed iid',
        approximation='source-space complex-normal covariance preserved with an odd eigenvalue clip on the anomalous part'))
    dll=out/'field.dll';cmd=[shutil.which('g++'),'-shared','-fopenmp','-O3','-std=c++20','-ffp-contract=off',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    hw=np.load(HIST);hwcounts=hw['total_click_histogram'];m=1152;perm=np.argsort(np.load(ORDER)['order']);rows=[];kernel=Kernel()
    for j,method in enumerate(['displaced_vacuum','normal_preserving']):
        start=perf_counter();maps=[];means=[];logs=[];discard=0;true16=[];approx16=[]
        ids=np.arange(16)*8
        for s in range(2):
            z=np.load(ACTUAL/f'input_{s}.npz');u=z['u'][perm];t=z['t'];L=z['noise']
            b,log_f,e=field_factor(t,L,method);logs.append(log_f);discard+=e
            transfer=np.block([[u.real,-u.imag],[u.imag,u.real]])
            maps.append(transfer@b);means.append(transfer@(np.sqrt(2)*np.r_[z['alpha'].real,z['alpha'].imag]))
            assert np.max(abs(u.conj().T@u-np.eye(len(t))))<1e-11
            ix=np.r_[ids,ids+m];sm=maps[-1][ix];approx16.append(np.eye(32)+sm@sm.T)
        means=np.concatenate(means);q=np.array(approx16);smallmu=np.array([means[s*2*m+np.r_[ids,ids+m]] for s in range(2)])
        opt=Optical(kernel.prior,q,smallmu);law=kernel.joint(opt,[],list(range(16)),2,8)[0].ravel()
        target=np.load(ref)['reference'];tv=float(abs(law-target).sum()/2)
        model=Field(dll,block_diag(*maps),means,m);prep=perf_counter()-start
        for route in [0,1]:
            domain=97080000+10*j+route;n=2048;click=np.zeros(m+1,int);pn=np.zeros(2049,int)
            hashes=[];total_fail=0;times=[]
            import hashlib
            with threadpool_limits(limits=8,user_api='blas'):
                for rep in range(32):
                    x,err=model.sample(n,route,8,domain,rep*n);total_fail+=int(err.sum())
                    click+=np.bincount((x>0).sum(1),minlength=m+1)
                    pn+=np.bincount(x.sum(1).astype(int),minlength=len(pn));hashes.append(hashlib.sha256(x).hexdigest())
            assert total_fail==0
            for threads in [1,8]:
                with threadpool_limits(limits=threads,user_api='blas'):
                    stamps=[]
                    for rep in range(5):
                        start=perf_counter();x,err=model.sample(n,route,threads,domain,2**20+(threads*5+rep)*n);stamps.append((perf_counter()-start)*1e6/n);assert not err.any()
                    times.append(dict(threads=threads,us=stamps,median=float(np.median(stamps))))
            ks=float(abs(click.cumsum()/click.sum()-hwcounts.cumsum()/hwcounts.sum()).max())
            half=float(np.sqrt(np.log(2/.002)/2)*(1/np.sqrt(click.sum())+1/np.sqrt(hwcounts.sum())))
            pmf=np.load(ph)['photon_pmf'];phks=float(abs(pn.cumsum()[:len(pmf)]/pn.sum()-pmf.cumsum()).max()) if route else None
            row=dict(method=method,route=route,rank=len(model.factor),preparation_s=prep,samples=int(click.sum()),
                fidelity_lower=float(np.exp(sum(logs))),optical_tv_upper=min(1.,float(np.sqrt(-np.expm1(sum(logs)))+discard)),
                optical16_click_tv=tv,hardware_click_ks=ks,halfwidth=half,total_photon_ks=phks,
                failures=total_fail,timing=times,hashes=hashes,discarded_tv_bound=discard)
            np.savez_compressed(out/f'{j}_{route}.npz',click=click,pnr=pn,small_law=law,first=x[:64])
            print(method,route,ks,tv,row['optical_tv_upper'],times[-1]['median'],flush=True)
            rows.append(row);save(out,'summary.json',rows)
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
