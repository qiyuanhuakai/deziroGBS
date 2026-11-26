"""Adaptive coherent source groups for twin-beam GBS, with physical loss ports."""
import argparse
import ctypes as ct
import functools
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT, I, U, J, W, T, D, bind, save, manifest
from iteration103_sample_v1 import DATA
from iteration105_two_mode_v1 import TwoMode, OUT as PREVIOUS
from iteration105_jiuzhang_v1 import inputs

OUT=DATA/'iteration106_build_v1'


def forest(count,partitions,overlap,caps):
    nodes=[dict(source=i,ids=(i,),cut=0,left=-1,right=-1) for i in range(count)]
    for partition in partitions:
        active=list(partition)
        while len(active)>1:
            options=[]
            for i,a in enumerate(active):
                for b in active[:i]:
                    ia,ib=nodes[a]['ids'],nodes[b]['ids']
                    if len(ia)+len(ib)<=max(caps,default=1):
                        options.append((float(overlap[np.ix_(ia,ib)].sum()),a,b))
            if not options:break
            _,a,b=max(options)
            ids=nodes[a]['ids']+nodes[b]['ids']
            cap=next(caps[k] for k in sorted(caps) if k>=len(ids))
            nodes.append(dict(source=-1,ids=ids,cut=cap,left=a,right=b))
            active.remove(a);active.remove(b);active.append(len(nodes)-1)
    return nodes


@functools.lru_cache(maxsize=None)
def basis(dim,count):
    if dim==1:return [(count,)]
    return [(n,)+rest for n in range(count+1) for rest in basis(dim-1,count-n)]


def geometry(dim,cap,levels,parents,ladder):
    off=len(levels)
    for n in range(2*cap+1):
        states=basis(dim,n)
        levels.append((len(states),len(parents)))
        if n:
            lookup={occ:k for k,occ in enumerate(states)}
            for low in basis(dim,n-1):
                for j in range(dim):
                    hi=list(low);hi[j]+=1
                    parents.append(lookup[tuple(hi)]);ladder.append(np.sqrt(hi[j]))
    return off


class Groups:
    def __init__(self,trans,r,nodes,dll=OUT/'groups.dll'):
        self.base=TwoMode(PREVIOUS/'two.dll',trans,np.repeat(np.asarray(r)[:,None],2,axis=1),np.ones(len(r),bool))
        self.lib=bind(ct.CDLL(str(dll)),'group106_draw',[I,I,I,I,I,U,ct.c_uint,I,I,J,W,T,W,T,D,D,I,J,D,D,J,J,D,J,D,T])
        self.m=self.base.m;self.nodes=nodes
        sizes={};levels=[];parents=[];ladder=[];records=[];rows=[];amplitudes=[];states=[];initial=[]
        offsets=[0,0,0]
        for node in nodes:
            ids=list(node['ids']);d=2*len(ids);cut=node['cut'];s=node['source']
            geom=0;w0=offsets[0];a0=offsets[1];st=len(states)
            if s<0:
                t=np.concatenate([trans[j] for j in ids],axis=1)
                e,v=np.linalg.eigh(np.eye(d)-t.conj().T@t)
                if e.min() < -2e-10:raise ValueError('Combined transfer exceeds a passive isometry')
                t=np.vstack((t,np.sqrt(e.clip(0))[:,None]*v.conj().T))
                assert np.max(abs(t.conj().T@t-np.eye(d)))<3e-10
                w=(abs(t)**2).sum(1);sel=w>0
                total=2**53
                cum=np.r_[np.longdouble(0),np.cumsum(w[sel],dtype=np.longdouble)]
                ticks=np.floor(cum/cum[-1]*(total-int(sel.sum()))).astype(np.int64)
                ticks[-1]=total-int(sel.sum())
                p=np.zeros(len(w),np.int64);p[sel]=1+np.diff(ticks)
                ratio=w[sel]/(p[sel]/total)
                correction=np.zeros(len(w));correction[sel]=ratio/(ratio.max()*(1+2e-14))
                t[sel]/=np.sqrt(w[sel,None])
                rows.append(np.c_[np.cumsum(p)/total,correction]);amplitudes.append(t.ravel())
                offsets[0]+=len(w);offsets[1]+=t.size
                if (d,cut) not in sizes:sizes[d,cut]=geometry(d,cut,levels,parents,ladder)
                geom=sizes[d,cut]
                for k in range(cut+1):
                    wave=np.zeros(len(basis(d,2*k)))
                    for j,occ in enumerate(basis(d,2*k)):
                        if occ[::2]==occ[1::2]:wave[j]=np.prod(np.tanh(r[ids])**np.array(occ[::2]))
                    norm=np.linalg.norm(wave)
                    if norm:wave/=norm
                    states.append(offsets[2]);offsets[2]+=len(wave);initial.append(wave)
            records.append([node['left'],node['right'],s,d,cut,self.m+d,w0,a0,geom,st,ids[0]])
        def cat(a,dtype):return np.ascontiguousarray(np.concatenate(a) if a else np.zeros(1),dtype)
        self.extra=[len(nodes),np.asarray(records,np.int32),cat(rows,float),cat(amplitudes,complex).view(float),np.asarray(levels,np.int32),np.asarray(parents,np.int32),np.asarray(ladder,float),np.asarray(states,np.int32),cat(initial,float)]
    def draw(self,n,pnr,seed,offset=0,workers=1):
        if not 0<n<2**31 or not 0<=offset<=2**32-n or not 0<=seed<2**64 or workers<1:raise ValueError('Invalid request')
        b=self.base
        meta,tc,ta,_,pc,pa,dirs,roots=b.args
        result=np.empty((n,self.m),np.uint32)
        rc=self.lib(self.m,b.s,n,int(pnr),workers,seed,offset,b.bits,b.max_n,meta,tc,ta,pc,pa,dirs,roots,*self.extra,result)
        if rc:raise ArithmeticError(rc)
        return result


def actual(caps):
    _,t,r=inputs()
    w=np.load(DATA/'iteration105_source_groups_v1/interference.npz')['pair_terms']
    nodes=forest(50,[list(range(25)),list(range(25,50))],np.tile(w,(2,2)),caps)
    return t,r,nodes


def synthetic(m,strength=.35):
    rng=np.random.default_rng(106000000+m)
    q=np.linalg.qr(rng.normal(size=(m,4))+1j*rng.normal(size=(m,4)))[0]
    t=q*np.array([.91,.80,.86,.73])
    r=np.array([strength,-.88*strength])
    return np.stack((t[:,:2],t[:,2:])),r,forest(2,[[0,1]],np.ones((2,2)),{2:5})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',default='jiuzhang',help='jiuzhang, synthetic, or an NPZ with transfer and squeezing')
    p.add_argument('--pair-cutoff',type=int,default=16,help='Pair-number cutoff for a custom one- or two-source spectrum');p.add_argument('--pnr',action='store_true');p.add_argument('--shots',type=int,default=256);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int)
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True)
            subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-march=x86-64-v3','-ffp-contract=off','-shared','-fopenmp',str(ROOT/'native/source_groups106_v1.cpp'),'-o',str(a.output/'groups.dll')],check=True)
            manifest(a.output)
        else:
            if a.seed is None:p.error('Provide an experiment seed')
            if a.input=='jiuzhang':case=actual({2:4,4:2,8:1})
            elif a.input=='synthetic':case=synthetic(16)
            else:
                with np.load(a.input) as z:t=np.asarray(z['transfer'],complex);r=np.atleast_2d(z['squeezing'])
                if r.ndim!=2 or not 1<=r.shape[1]<=2 or t.ndim!=2 or t.shape[1]!=2*r.shape[1] or a.pair_cutoff<1:raise ValueError('Custom CLI expects one or two twin-beam sources per spectrum and a positive cutoff')
                if np.linalg.eigvalsh(t.conj().T@t).max()>1+1e-10:raise ValueError('Transfer must be passive')
                k=r.shape[1];trans=np.array([t[:,2*j:2*j+2] for _ in r for j in range(k)])
                partitions=np.arange(r.size).reshape(r.shape).tolist()
                case=trans,r.ravel(),forest(r.size,partitions,np.ones((r.size,r.size)),{2:a.pair_cutoff})
            model=Groups(*case)
            with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,workers=a.workers))
