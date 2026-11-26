"""Measured three-mode optical blocks, their full PNR law, and direct click sampling at arbitrary width."""
import argparse
import ctypes as ct
import itertools
import math
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT,ORDER,Native,realify,cdf_rows,I,U,W,T,SHORT,bind,save,manifest
from iteration100_projection_v1 import physical
from iteration100_baselines_v1 import purification,pe_sample
from iteration97_forest_v1 import alias_table
from iteration102_triads_v1 import triple
from iteration102_forest_v1 import BASE,DATA

OUT=DATA/'iteration102_blocks_v1'


def reference(q,limit):
    # Independent, high-precision principal minors and Euler recurrence of
    # P(z)^2 D(z) = vacuum^2, retaining every cell in the low-count cube.
    with mp.workdps(65):
        poly={(0,0,0):mp.mpf(1)}
        vacuum=mp.mpf(1)
        for raw in q:
            m=mp.matrix(raw.tolist())
            vacuum/=mp.sqrt(mp.det(m))
            kernel=mp.eye(6)-m**-1
            part={}
            for mask in range(64):
                ids=[i for i in range(6) if mask>>i&1]
                key=tuple(sum(i%3==j for i in ids) for j in range(3))
                v=(-1)**len(ids)*(mp.det(mp.matrix([[kernel[i,j] for j in ids] for i in ids])) if ids else 1)
                part[key]=part.get(key,0)+v
            nxt={}
            for i,val in poly.items():
                for j,weight in part.items():
                    k=tuple(a+b for a,b in zip(i,j))
                    nxt[k]=nxt.get(k,0)+val*weight
            poly=nxt
        terms=[(key,sum(key),v) for key,v in poly.items() if sum(key)]
        law={(0,0,0):vacuum}
        for ijk in itertools.product(range(limit),repeat=3):
            n=sum(ijk)
            if not n:continue
            termsum=mp.mpf(0)
            i,j,k=ijk
            for (a,b,c),degree,v in terms:
                if a<=i and b<=j and c<=k:termsum+=(2*n-degree)*v*law[i-a,j-b,k-c]
            law[ijk]=-termsum/(2*n)
        tail=1-sum(law.values())
        no=[]
        for mask in range(8):
            obs=[j for j in range(6) if (mask>>(j%3))&1]
            p=mp.mpf(1)
            for raw in q:
                if obs:p/=mp.sqrt(mp.det(mp.matrix(raw[np.ix_(obs,obs)].tolist())))
            no.append(p)
        bits=[sum((-1)**s.bit_count()*no[(7^mask)|s] for s in range(8) if (s&mask)==s) for mask in range(8)]
        return np.array(list(law.values()),float).reshape((limit,)*3),float(tail),np.array(bits,float)


class Blocks:
    def __init__(self,directory):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.native=bind(ct.CDLL(str(directory/'blocks.dll')),'triple102_draw',[I,I,I,I,U,U,I,W,T,T,SHORT])
        self.tables=[]
        with np.load(directory/'tables.npz') as z:
            for route in ('click','pnr'):
                self.tables.append((int(z[route+'bits']),z[route+'cut'],z[route+'alias'],z[route+'keys']))
    def sample(self,width,n,route,workers,seed,offset=0):
        x=np.empty((n,width),np.uint16)
        rc=self.native(width,n,workers,int(route),seed,offset,*self.tables[int(route)],x)
        if rc:raise RuntimeError(f'Block output failed with status {rc}')
        return x


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/triple_alias102_v1.cpp'
    subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp',str(src),'-o',str(out/'blocks.dll')],check=True)
    spec=np.load(ORDER)
    v=spec['v'][np.argsort(spec['order'])]
    groups=[11,101,44]
    ports=[]
    for g in groups:
        local=physical(spec,list(range(8*g,8*g+8)))
        e=np.diagonal(local,axis1=1,axis2=2).sum(0)
        ports.append(8*g+int((e[:8]+e[8:]).argmax()))
    q=physical(spec,ports)
    start=perf_counter()
    lib=Native(BASE/'spatial.dll')
    matrices=[realify(v[[i]]) for i in ports]
    # Output axes of triple() are oldest to newest.
    joint=triple(lib,matrices[::-1],spec['x'],[31,31,31])
    flat=cdf_rows(joint.clip(0).ravel())
    mass=np.diff(flat,prepend=np.uint64(0)).reshape(joint.shape)
    weights={}
    clicks={}
    for a,b,c in zip(*np.nonzero(mass)):
        val=int(mass[a,b,c])
        weights[int(a+256*b+65536*c)]=val
        word=int(a>0)+2*int(b>0)+4*int(c>0)
        clicks[word]=clicks.get(word,0)+val
    arrays={}
    for name,items in [('click',clicks),('pnr',weights)]:
        bits,cut,alias,keys,tv,_=alias_table(items)
        assert tv==0.
        for suffix,item in zip(('bits','cut','alias','keys'),(bits,cut,alias,keys)):
            arrays[name+suffix]=item
    np.savez_compressed(out/'tables.npz',**arrays)
    prep=perf_counter()-start
    model=Blocks(out)
    low,tail,bits=reference(q,14)
    pm=mass.astype(float)/(1<<53)
    inside=pm[:14,:14,:14]
    tail_model=pm.sum()-inside.sum()
    pu=.5*(abs(inside-low).sum()+tail+max(0.,tail_model))
    clicklaw=np.array([clicks.get(i,0)/(1<<53) for i in range(8)])
    cu=float(abs(clicklaw-bits).sum()/2)
    rows=[]
    for width in (3,17,144,1152,4097):
        for route in (False,True):
            workers=1 if width<=17 else 16
            times=[]
            seed=102070000+width*2+route
            assert np.array_equal(model.sample(width,123,route,workers,seed)[23:],
                                  model.sample(width,100,route,1,seed,23))
            for i in range(8):
                start=perf_counter()
                model.sample(width,2048,route,workers,seed,4096+2048*i)
                if i:times.append((perf_counter()-start)*1e6/2048)
            block_bound=pu if route else cu
            rows.append(dict(width=width,pnr=route,workers=workers,us=times,median_us=float(np.median(times)),
                    numerical_full_tv_upper=min(1.,math.ceil(width/3)*(block_bound+2e-14))))
    prepared=[purification(raw) for raw in q]
    def pe(n,seed):
        result=np.zeros((n,3),np.uint32)
        for i,(state,interferometer,_) in enumerate(prepared):
            x,stats=pe_sample(state,interferometer,n,seed=seed+i,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
            assert not np.any(stats.censored)
            result+=np.asarray(x[:,:3],dtype=np.uint32)
        return result
    pe(16,102090000)
    times=[]
    for i in range(5):
        start=perf_counter()
        pe(512,102090010+2*i)
        times.append((perf_counter()-start)*1e6/512)
    n=8192
    other=np.concatenate([pe(512,102091000+2*i) for i in range(n//512)])
    ours=model.sample(3,n,True,1,102092000)
    def empirical(x):
        cap=14
        idx=x.clip(0,cap)@np.array([(cap+1)**2,cap+1,1])
        hist=np.bincount(idx,minlength=(cap+1)**3).reshape((cap+1,)*3)/len(x)
        inner=hist[:cap,:cap,:cap]
        return float((abs(inner-low).sum()+abs(1-inner.sum()-tail))/2)
    np.savez_compressed(out/'reference.npz',optical_cube=low,optical_tail=tail,click=bits,pe=other,ours=ours)
    save(out,'summary.json',dict(ports=ports,prepare_s=prep,independent_precision_digits=65,low_cube_limit=14,
        optical_tail=tail,block_full_pnr_tv_numerical_upper=pu,block_full_click_tv_numerical=cu,
        rows=rows,pe_median_us=float(np.median(times)),pe_us=times,pe_empirical=empirical(other),
        ours_empirical=empirical(ours),empirical_records=n,pe_revision=subprocess.check_output(
            ['git','-C',str(ROOT.parent/'external_sources/photonic_engines_iteration92'),'rev-parse','HEAD'],text=True).strip()))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('build','draw'))
    ap.add_argument('output',type=Path)
    ap.add_argument('--width',type=int,default=3)
    ap.add_argument('--shots',type=int,default=2048)
    ap.add_argument('--pnr',action='store_true')
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='build':build(args.output.resolve())
        else:
            if args.seed is None:ap.error('--seed is required for draw')
            with args.output.open('xb') as f:
                np.save(f,Blocks(OUT).sample(args.width,args.shots,args.pnr,1,args.seed))
