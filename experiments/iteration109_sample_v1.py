"""Direct conditional click words and conditional photon counts on overlapping optical edges."""
import argparse
import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT, save, manifest
from iteration106_field_v1 import actual_input, DATA
from iteration106_sample_v1 import panel_hist, benchmark
from iteration106_packed_v1 import compress
from iteration107_fast_v1 import gaussian_click, oracle, ONE
from iteration107_cliques_v1 import optical_x
from iteration109_model_v1 import build, panel
from iteration109_tree_v1 import prepare

def compile_core(path):
    compiler=shutil.which('g++')
    subprocess.run([compiler,'-O3','-ffp-contract=off','-std=c++20','-march=x86-64-v3',
                    '-fopenmp','-shared',str(ROOT/'native/displaced_tree109_v1.cpp'),'-o',str(path)],check=True)

class Tree:
    def __init__(self,params,dll,workers=12):
        self.p=params
        self.workers=workers
        if os.name=='nt':
            self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll))
        self.lib.normals109.argtypes=[ct.c_int]*3+[ct.c_uint64,ct.c_uint32,ct.c_int,ct.c_void_p]
        self.lib.normals109.restype=None
        self.lib.draw109.argtypes=[ct.c_int]*6+[ct.c_uint64,ct.c_uint32]+[ct.c_void_p]*9
        self.lib.draw109.restype=ct.c_int
        self.s,self.k,self.g,_=params['noise'].shape
        self.fan=int(params['fan'])
        self.factors=params['noise'].reshape(self.s,self.k,self.g*2)
        if not 0<self.s<=65536 or not 0<workers<=1024:raise ValueError('Invalid runtime dimensions')
    def draw(self,n,pnr,seed,offset=0):
        if not 0<n<=2**31-1 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid record domain')
        z=np.empty((self.s,n,self.k))
        self.lib.normals109(n,self.s,self.k,seed,offset,self.workers,z.ctypes.data)
        fields=z@self.factors
        result=np.empty((n,self.g*self.fan),np.uint32)
        p=self.p
        args=[p['parent'],p['order'],p['subset'],fields,p['vacuum'],p['photon'],p['meta'],p['single'],result]
        errors=self.lib.draw109(n,self.s,self.g,self.fan,int(pnr),self.workers,seed,offset,*[a.ctypes.data for a in args])
        if errors:raise ArithmeticError(f'{errors} records failed probability or count guards in batch starting at {offset}')
        return result

def tv(p,q):return float(np.abs(p-q).sum()/2)

def run(out,shots):
    out.mkdir(parents=True,exist_ok=False)
    dll=out/'tree.dll';compile_core(dll)
    t,r=actual_input();start=perf_counter();params=prepare(t,r)
    prep=perf_counter()-start;tree=Tree(params,dll)
    _,_,projected=compress(t,(np.sinh(r)**2).sum(0),1)
    parent=params['parent'];edges=[(int(p),i) for i,p in enumerate(parent) if p>=0]
    assert len(edges)==143
    cases=[('first8',np.arange(8)),('space36',np.arange(288,296))]
    for idx in (0,1,3,5,6):
        z=np.load(DATA/f'iteration106_packed_v1/panel_{idx}.npz')
        cases.append((f'panel{idx}',z['ports']))
    for p,g in edges[:2]:
        cases.append((f'edge{p}_{g}',np.r_[np.arange(p*8,p*8+4),np.arange(g*8,g*8+4)]))
    analytic=[];arrays={};ref=oracle();independent=build(t,r,8,0)
    for name,ids in cases:
        frozen=name.startswith('panel')
        target=np.load(DATA/f'iteration106_packed_v1/panel_{name[5:]}.npz')['target'] if frozen else gaussian_click(ref,optical_x(t[ids],r),ONE)
        compressed=gaussian_click(ref,optical_x(projected[ids],r),ONE)
        groups=np.unique(ids//8)
        on_edge=len(groups)==1 or (len(groups)==2 and (tuple(groups) in edges or tuple(groups[::-1]) in edges))
        row=dict(name=name,ports=ids.tolist(),tree_edge=bool(on_edge),compressed_optical_tv=tv(compressed,target))
        if len(ids)<=8:
            for include in (False,True):
                law=gaussian_click(ref,panel(independent,ids,include),ONE)
                row['coherent_tv' if include else 'distinguishable_tv']=tv(law,target)
        if on_edge:row['tree_model_tv']=tv(compressed,target)
        analytic.append(row);arrays[name+'_target']=target
    batch=256
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    runs=[]
    for pnr in (False,True):
        print('sampling',pnr,flush=True)
        a=tree.draw(32,pnr,109330000+pnr)
        assert np.array_equal(a,np.vstack((tree.draw(7,pnr,109330000+pnr),tree.draw(25,pnr,109330000+pnr,7))))
        timing=benchmark(lambda n,s,o:tree.draw(n,pnr,s,o),batch,109331000+pnr,7)
        totals=np.zeros(len(t)+1,np.int64);hs=[np.zeros(1<<len(ids),np.int64) for _,ids in cases];pair=np.zeros((13,13),np.int64)
        for offset in range(0,shots,batch):
            x=tree.draw(min(batch,shots-offset),pnr,109332000+pnr,offset)
            totals+=np.bincount((x>0).sum(1),minlength=len(totals))
            for hist,(_,ids) in zip(hs,cases):hist+=panel_hist(x,ids)
            if pnr:np.add.at(pair,(x[:,289].clip(max=12),x[:,861].clip(max=12)),1)
        ks=float(np.max(abs(totals.cumsum()/shots-hw.cumsum()/hw.sum())))
        mode='pnr' if pnr else 'click';arrays[mode+'_total']=totals;arrays[mode+'_pair']=pair
        errors={}
        for hist,(name,_) in zip(hs,cases):
            arrays[mode+'_'+name]=hist;errors[name]=tv(hist/shots,arrays[name+'_target'])
        runs.append(dict(route=mode,samples=shots,workers=tree.workers,timing=timing,hardware_ks=ks,panel_empirical_tv=errors))
    arrays['parent']=parent;arrays['order']=params['order'];arrays['hardware']=hw
    np.savez_compressed(out/'laws.npz',**arrays)
    save(out,'summary.json',dict(preparation_s=prep,edges=edges,shared_gain=params['retained_classical'].tolist(),panels=analytic,runs=runs));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--shots',type=int,default=32768)
    args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):run(args.output.resolve(),args.shots)
