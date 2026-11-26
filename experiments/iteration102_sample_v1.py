"""Native sampling, same-device throughput, and pilot scans of all spatial-pair vacuum events."""
import argparse
import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.linalg.cython_blas as blas
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT, HIST, I, U, D, J, W, B, SHORT, S, bind, cdf_rows, save, manifest, H
from iteration97_forest_v1 import alias_table
from iteration102_forest_v1 import FIT, RUN, BASE, DATA, VAC, Tables

P=ct.c_void_p
BLOCK=2048


class Sampler:
    def __init__(self,dll,ids=None,law=None,depth=2):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll))
        self.draw=bind(self.lib,'forest102_draw',[I,I,I,I,I,I,U,U,J,J,J,J,I,W,J,J,W,D,D,P,J,W,D,D,W,B,B,SHORT,SHORT])
        cap=blas.__pyx_capi__['dgemm']
        name=ct.pythonapi.PyCapsule_GetName
        name.argtypes=[ct.py_object]
        name.restype=ct.c_char_p
        get=ct.pythonapi.PyCapsule_GetPointer
        get.argtypes=[ct.py_object,ct.c_char_p]
        get.restype=P
        self.gemm=get(cap,name(cap))
        tables=Tables(FIT/'prepared.npz')
        self.full=ids is None
        if self.full:
            self.order,self.parent,self.caps,self.offset=tables.order,tables.parent,tables.caps,tables.offset
            self.factor,self.cdf=tables.factor,tables.cdf
            ids=np.arange(len(self.order))
        else:
            if not 1<=len(ids)<=2:raise ValueError('Reduced sampler supports one or two spatial groups')
            self.order=np.arange(len(ids),dtype=np.int32)
            self.caps=tables.caps[ids]
            self.parent=self.order-1
            self.factor=np.eye(len(ids))
            joint=law if law is not None else (tables.pair(*ids) if len(ids)==2 else tables.marg[ids[0]])
            first=joint.sum(1) if len(ids)==2 else joint
            self.offset=np.array([0,len(first)][:len(ids)],np.int32)
            self.cdf=cdf_rows(first)
            if len(ids)==2:
                joint=joint.copy()
                joint[joint.sum(1)==0]=tables.marg[ids[1]]
                self.cdf=np.r_[self.cdf,cdf_rows(joint).ravel()]
        self.second=np.full(len(self.order),-1,np.int32)
        self.third=self.second.copy()
        self.classes=np.zeros((1,1),np.uint64)
        if depth not in (2,3,4):raise ValueError('Supported optical group counts are 2, 3 and 4')
        if depth>=3:
            if not self.full:raise ValueError('A projected pair uses its marginal joint law')
            with np.load(DATA/'iteration102_triads_v1/prepared.npz') as t:
                self.cdf,self.offset,self.second=t['cdf'],t['offset'],t['second']
        if depth==4:
            with np.load(DATA/'iteration102_four_v1/extension.npz') as ext:
                self.cdf=np.r_[self.cdf,ext['extra']]
                self.offset,self.third,self.classes=ext['offset'],ext['third'],ext['classes']
        self.thresholds=ndtri(self.cdf.astype(float)/S)
        with np.load(BASE/'prepared.npz') as data:
            self.starts=data['leafstarts'][ids]
            self.masks=data['masks']
            self.prior=data['prior']
            self.weights=data['ports'][ids]
        cuts=[];direct=[];fallback=[]
        for start in self.starts:
            for n in (1,2):
                m=np.diff(self.masks[start+n],prepend=np.uint64(0))
                weights={i:int(x) for i,x in enumerate(m) if x}
                # Pad even a degenerate physical splitter to the native table width.
                for i in range(256):
                    if len(weights)>=(8 if n==1 else 36):break
                    weights.setdefault(i,0)
                bits,cut,alias,key,tv,_=alias_table(weights)
                assert bits==(3 if n==1 else 6) and tv==0.
                cuts.append(cut);direct.append(key);fallback.append(key[alias])
        self.shortcut_cut=np.concatenate(cuts)
        self.shortcut_mask=np.concatenate(direct).astype(np.uint8)
        self.shortcut_fallback=np.concatenate(fallback).astype(np.uint8)

    def sample(self,n,route,field,workers,seed,offset=0):
        if field and not self.full:raise ValueError('Reduced conditional table uses the integer sampler')
        groups=len(self.parent)
        x=np.empty((n,groups*8),np.uint16)
        counts=np.empty((n,groups),np.uint16)
        rc=self.draw(groups,8,n,int(route),int(field),workers,seed,offset,self.order,self.parent,self.second,self.third,self.classes.shape[1],self.classes,self.caps,self.offset,
                self.cdf,self.thresholds,self.factor,self.gemm,self.starts,self.masks,self.prior,self.weights,self.shortcut_cut,self.shortcut_mask,self.shortcut_fallback,x,counts)
        if rc:raise RuntimeError(f'Native draw failed with status {rc}')
        return x,counts


def bench(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/spatial_forest102_v1.cpp'
    dll=out/'forest.dll'
    cmd=[shutil.which('g++'),'-shared','-O3','-fopenmp','-std=c++20','-ffp-contract=off',str(src),'-o',str(dll)]
    subprocess.run(cmd,check=True)
    refs=np.load(VAC/'scan.npz')
    ai,bi=refs['a'],refs['b']
    target=refs['optical']
    hardware=np.load(HIST)['total_click_histogram']
    optical=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    rows=[]
    for depth in (2,3,4):
        model=Sampler(dll,depth=depth)
        for field in (False,True):
            click,c=model.sample(1025,False,field,1,102001000+10*depth+field)
            pnr,d=model.sample(998,True,field,4,102001000+10*depth+field,27)
            assert np.array_equal(click[27:],pnr>0) and np.array_equal(c[27:],d)
            assert np.array_equal(pnr.reshape(len(d),-1,8).sum(2),d)
            for route in (False,True):
                seed=102010000+10*depth+2*field+route
                times=[]
                for i in range(8):
                    start=perf_counter()
                    model.sample(8192,route,field,16,seed,i*8192)
                    if i:times.append((perf_counter()-start)*1e6/8192)
                n=262144
                hc=np.zeros(len(hardware),np.int64)
                hp=np.zeros(4096,np.int64)
                both=np.zeros((len(model.parent),len(model.parent)))
                for off in range(0,n,BLOCK):
                    x,counts=model.sample(BLOCK,route,field,16,102020000+10*depth+2*field+route,off)
                    if route:assert np.array_equal(x.reshape(len(x),-1,8).sum(2),counts)
                    hc+=np.bincount((x>0).sum(1),minlength=len(hc))
                    hp+=np.bincount(counts.sum(1).astype(np.int64),minlength=len(hp))
                    zero=(counts==0).astype(float)
                    both+=zero.T@zero
                prob=both[ai,bi]/n
                i=int(abs(prob-target).argmax())
                np.savez_compressed(out/f'hist_{depth}_{int(field)}_{int(route)}.npz',click=hc,photon=hp,vacuum_counts=both[ai,bi].astype(np.int64))
                rows.append(dict(depth=depth,field=field,pnr=route,records=n,us=times,median_us=float(np.median(times)),
                    click_hardware_ks=float(abs(hc.cumsum()/n-hardware.cumsum()/hardware.sum()).max()),
                    photon_optical_ks=float(abs(hp.cumsum()/n-np.pad(optical,(0,len(hp)-len(optical))).cumsum()).max()),
                    worst_vacuum=[int(ai[i]),int(bi[i])],pilot_gap=float(prob[i]-target[i])))
                print(rows[-1],flush=True)
    save(out,'summary.json',dict(rows=rows,build=cmd,hardware_records=int(hardware.sum()), 
        simultaneous_hw_ks_radius99=float(np.sqrt(np.log(2*25/.01)/(2*n))+np.sqrt(np.log(2*25/.01)/(2*hardware.sum()))),
        simultaneous_photon_ks_radius99=float(np.sqrt(np.log(2*25/.01)/(2*n))),
        hashes={p.relative_to(ROOT).as_posix():H(p) for p in [src,Path(__file__),FIT/'prepared.npz',HIST]}))
    manifest(out)


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    sub=arg.add_subparsers(dest='action',required=True)
    b=sub.add_parser('benchmark')
    b.add_argument('output',type=Path)
    d=sub.add_parser('draw')
    d.add_argument('output',type=Path)
    d.add_argument('--pnr',action='store_true')
    d.add_argument('--depth',type=int,choices=(2,3,4),default=4)
    d.add_argument('--independent',action='store_true')
    d.add_argument('--seed',required=True,type=int)
    d.add_argument('--shots',type=int,default=2048)
    d.add_argument('--workers',type=int,default=1)
    args=arg.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':
            bench(args.output.resolve())
        else:
            model=Sampler(RUN/'forest.dll',depth=args.depth)
            x,c=model.sample(args.shots,args.pnr,not args.independent,args.workers,args.seed)
            with args.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c)
