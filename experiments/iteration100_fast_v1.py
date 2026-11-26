"""Integrate exact rational one- and two-photon subtrees and time both complete routes."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from iteration100_spatial_v1 import ROOT,I,D,W,T,J,U,B,SHORT,np,bind,Tree,Native,H,save,manifest
from iteration97_forest_v1 import alias_table

def mass(cuts):
    return [int(x) for x in np.diff(cuts,prepend=np.uint64(0))]

def make(model):
    L,w=model.leaves,model.width;entries={};sizes=[]
    offsets=np.full((len(model.caps),2),-1,np.int32);bits=np.zeros_like(offsets)
    cuts=[];aliases=[];photons=[];clicks=[];pos=0;errors=[]
    for i in range(len(model.caps)):
        if i>=L:a,b=map(int,model.children[i])
        size=w if i<L else sizes[a]+sizes[b];sizes.append(size);per=[]
        for n in (1,2):
            if n==2 and size>64:continue
            if i<L:
                d={}
                for label,value in enumerate(mass(model.masks[model.leafstarts[i]+n])):
                    if not value:continue
                    ids=[i*w+j+1 for j in range(w) if label>>j&1]
                    assert 1<=len(ids)<=n
                    code=ids[0] if n==1 else ids[0]+(ids[-1]<<16)
                    d[code]=value
                power=53
            else:
                row=model.starts[i]+n*(n+1)//2
                prob=mass(model.splits[row:row+n+1])
                left,lp=entries[a,n];right,rp=entries[b,n]
                pp=max(lp,rp)
                if n==2:
                    la,pa=entries[a,1];rb,pb=entries[b,1];pp=max(pp,pa+pb)
                d={k:(v*prob[-1])<<(pp-lp) for k,v in left.items()}
                d.update({k:(v*prob[0])<<(pp-rp) for k,v in right.items()})
                if n==2:
                    for ka,va in la.items():
                        for kb,vb in rb.items():
                            d[min(ka,kb)+(max(ka,kb)<<16)]=(va*vb*prob[1])<<(pp-pa-pb)
                power=pp+53
            assert sum(d.values())==1<<power
            entries[i,n]=d,power
            bnum,c,alias,keys,tv,_=alias_table(d)
            offsets[i,n-1]=pos;bits[i,n-1]=bnum;pos+=len(c)
            cuts.append(c);aliases.append(alias);photons.append(keys)
            clicks.append(np.where((keys>>16)==(keys&65535),keys&65535,keys).astype(np.uint32))
            per.append(tv)
        errors.append(max(per))
    return dict(offsets=offsets,bits=bits,cut=np.concatenate(cuts),alias=np.concatenate(aliases),
                click=np.concatenate(clicks),photon=np.concatenate(photons)),float(sum(errors))

class Fast(Tree):
    def __init__(self,dll,base,extra):
        super().__init__(None,base)
        for k,v in extra.items():setattr(self,k,v)
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));self.native=ct.CDLL(str(dll))
        self.fast=bind(self.native,'shortcut100_draw',[I,I,J,J,J,W,W,J,W,D,D,J,J,W,T,T,T,I,I,U,U,I,SHORT,B])
    def sample(self,n,route,seed,offset,workers):
        x=np.empty((n,self.leaves*self.width),np.uint16);flags=np.empty(n,np.uint8)
        rc=self.fast(self.leaves,self.width,self.children,self.caps,self.starts,self.root,self.splits,self.leafstarts,
            self.masks,self.prior,self.ports,self.offsets,self.bits,self.cut,self.alias,self.click,self.photon,
            n,route,seed,offset,workers,x,flags)
        if rc or (flags&2).any():raise RuntimeError('shortcut draw')
        return x,flags

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';src=ROOT/'native/spatial_shortcut100_v1.cpp'
    files=[src,ROOT/'native/spatial_tree100_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',Path(__file__),base/'prepared.npz',ROOT/'experiments/iteration97_forest_v1.py']
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=100030000,
            shortcuts='N=1 at all tree nodes; N=2 up to 64 physical ports; exact integer weights and finite alias TV'))
    dll=out/'fast.dll';cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    with np.load(base/'prepared.npz') as cache:arrays=dict(cache)
    t=perf_counter();ex,bound=make(Tree(None,arrays));prep=perf_counter()-t
    np.savez_compressed(out/'shortcuts.npz',**ex)
    model=Fast(dll,arrays,ex);rows=[]
    for route in (0,1):
        seed=100030000+route
        x,e=model.sample(512,route,seed,0,1);y,f=model.sample(480,route,seed,32,4)
        assert np.array_equal(x[32:],y) and np.array_equal(e[32:],f)
        ts=[]
        for workers in (1,16):
            times=[]
            for rep in range(7):
                start=perf_counter();x,f=model.sample(8192,route,seed,4096+(workers*7+rep)*8192,workers)
                times.append((perf_counter()-start)*1e6/8192);assert not f.any()
            ts.append(dict(workers=workers,us=times,median=float(np.median(times))))
        for rep in range(16):
            off=2**21+rep*4096
            x,e=model.sample(4096,route,seed,off,16)
            y,f=model.sample(4096,1-route,seed,off,8)
            assert np.array_equal(x>0,y>0) and not (e|f).any()
        rows.append(dict(route=route,timing=ts));print(route,ts[-1]['median'],flush=True)
    save(out,'summary.json',dict(rows=rows,extra_prep_s=prep,full_tv_transfer_bound=bound,alias_slots=len(ex['cut'])))
    manifest(out)
if __name__=='__main__':main()
