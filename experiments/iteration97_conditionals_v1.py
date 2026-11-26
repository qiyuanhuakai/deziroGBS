"""Optical conditional tables with multiple observed neighbors and physically ordered records."""
import argparse
import ctypes as ct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import subprocess
import shutil
from iteration96_group_common_v1 import (ROOT,np,I,P,J,D,B,W,U,Native,Optical,bind,construct,
    partition,block_weights,pair_information,codes_cdf,SCALE,H,save,manifest)
from iteration96_group_benchmark_v1 import BUILD as PREVIOUS
BUILD=ROOT/'results/iteration97_build_v1/conditional.dll'
class Kernel:
    def __init__(self,path=BUILD):
        self.prior=Native(PREVIOUS)
        lib=ct.CDLL(str(path));self.lib=lib
        self.law=bind(lib,'cond97_law',[I,I,I,I,I,D,D,D,D])
        self.prepare=bind(lib,'cond97_prepare',[I,I,I,I,J,J,J,J,W,W,W],P)
        self.draw=bind(lib,'cond97_draw',[P,I,U,U,I,B,B])
        self.free=bind(lib,'cond97_free',[P],None)
    def joint(self,opt,past,group,base,workers=1):
        ix=tuple(past)+tuple(group);q,mu=opt.sub(ix)
        a=np.empty((1<<len(past),base**len(group)));diag=np.empty(2)
        rc=self.law(len(past),len(group),base,len(q),workers,q,mu,a,diag)
        if rc:raise ValueError((rc,past,group,diag))
        return a,diag

def layout(info,width,physical=False):
    m=len(info);groups=partition(m,width,not physical,info)
    weights=block_weights(groups,info)
    # Grow out from the strongest edge. The ordering is fixed across history budgets.
    if physical:return groups
    used=np.zeros(len(groups),bool);score=np.zeros(len(groups));pos=int(np.unravel_index(weights.argmax(),weights.shape)[0]);order=[]
    for _ in groups:
        order.append(groups[pos]);used[pos]=True;score+=weights[pos];pos=int(np.argmax(np.where(used,-np.inf,score)))
    return order

def fit(kernel,opt,info,groups,base,history,cap=32,parallel=6):
    started=perf_counter();past=[];previous=[]
    for group in groups:
        scores=info[np.ix_(previous,group)].sum(1) if previous else np.zeros(0)
        ranked=np.argsort(-scores,kind='stable')[:history]
        past.append([previous[i] for i in ranked]);previous.extend(group)
    def one(i):
        group=groups[i]
        full,diag=kernel.joint(opt,past[i],group,base,4 if len(groups)<=4 and len(past[i])+len(group)>12 else 1)
        weight=full.sum(1);fallback=full.sum(0);empty=weight==0
        a=full.copy();a[empty]=fallback;a/=a.sum(1)[:,None]
        cuts=np.rint(a.cumsum(1)*SCALE).clip(0,SCALE).astype(np.uint64);cuts[:,-1]=SCALE
        actual=np.diff(cuts,axis=1,prepend=np.zeros((len(cuts),1),np.uint64)).astype(float)/SCALE
        return cuts,actual,dict(neg=float(diag[0]),empty=int(empty.sum()),quantized=float((abs(a-actual).sum(1)*weight).sum()/2))
    with ThreadPoolExecutor(max_workers=parallel) as pool:tables=list(pool.map(one,range(len(groups))))
    cuts=[v[0] for v in tables];p=[v[1] for v in tables]
    fine=None;ecdf=np.zeros(1,np.uint64);split=ecdf
    if base==3:
        fine,_=construct(opt,groups,np.zeros((len(groups),len(groups))),3,cap,False)
        shape=(len(groups),9,cap+2);ecdf=np.zeros(shape,np.uint64)
        em=fine['emission'];ecdf[:,:em.shape[1]]=np.rint(em.cumsum(2)*SCALE).astype(np.uint64)
        split=np.zeros((*shape,cap+2),np.uint64);split[:,:em.shape[1]]=fine['split']
    widths=np.array(list(map(len,groups)),np.int32);history_sizes=np.array(list(map(len,past)),np.int32)
    physical=np.array(sum((list(g) for g in groups),[]),np.int32);parents=np.array(sum(past,[]) or [0],np.int32)
    flat=np.concatenate([a.ravel() for a in cuts]);h=kernel.prepare(opt.m,len(groups),base,cap,widths,physical,history_sizes,parents,flat,ecdf,split)
    if not h:raise ValueError('conditional preparation')
    return dict(handle=h,groups=groups,past=past,p=p,fine=fine,seconds=perf_counter()-started,bytes=flat.nbytes+ecdf.nbytes+split.nbytes,
        diagnostics=[v[2] for v in tables])
def sample(kernel,model,m,n,seed,offset,workers):
    x=np.empty((n,m),np.uint8);tail=np.empty(n,np.uint8)
    if kernel.draw(model['handle'],n,seed,offset,workers,x,tail):raise ValueError('invalid draw')
    return x,tail

def density(model,base,states):
    p=np.ones(len(states));fine=model['fine']
    for i,(g,h,table) in enumerate(zip(model['groups'],model['past'],model['p'])):
        z=np.minimum(states[:,g],base-1).astype(int)@(base**np.arange(len(g)))
        code=(states[:,h]>0)@(2**np.arange(len(h))) if h else 0
        p*=table[code,z]
        if base==3:
            n=states[:,g].sum(1).astype(int);p*=fine['emission'][i,z,n]
            if len(g)==2:
                pick=z==8;a=states[pick,g[0]].astype(int)
                split=fine['split'][i,8]
                p[pick]*=(split[n[pick],a]-split[n[pick],a-1]).astype(float)/SCALE
    return p

def build(out):
    out.mkdir(parents=True,exist_ok=False);source=ROOT/'native/gaussian_conditional97_v1.cpp'
    files=[source,Path(__file__),ROOT/'native/gaussian_rng_domain84_v1.hpp']
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}))
    cmd=[shutil.which('g++'),'-shared','-std=c++20','-O3','-fopenmp','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(out/'conditional.dll')]
    r=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,stderr=r.stderr,code=r.returncode))
    r.check_returncode();manifest(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);build(p.parse_args().output.resolve())
