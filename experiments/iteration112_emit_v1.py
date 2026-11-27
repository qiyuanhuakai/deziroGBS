"""Emit both homogeneous and unequal optical marginals on CPU and GPU with identical Philox addresses."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration112_records_v1 import alias,I,U,CU,P,DATA,ROOT


class Library:
    def __init__(self,out):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
        f=out/'register.dll'
        subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(ROOT/'native/marginal_register112_v1.cpp'),'-o',str(f)],check=True)
        self.host=ct.CDLL(str(f));self.cpu=self.host.cpu_register112
        self.cpu.argtypes=[I,I,I,CU,U,CU,I,I,I]+[P]*4
        self.dev=ct.CDLL(str(DATA/'iteration112_records_v1/gpu.dll'));self.gpu=self.dev.local_emit112
        self.gpu.argtypes=[I,I,I,CU,U,CU]+[P]*5


class Register:
    def __init__(self,lib,groups,pnr):
        self.lib=lib;self.plans=[];self.pnr=pnr;self.ports=0;self.rounding=0.
        for count,width,radix,codes,q in groups:
            t,a,err=alias(q)
            digits=np.stack([codes//radix**i%radix for i in range(width)],axis=1)
            pack=digits.astype(np.uint32)@(np.uint32(1)<<(8*np.arange(width,dtype=np.uint32)))
            args=(t,a,np.pad(pack,(0,len(t)-len(q))))
            dev=[torch.as_tensor(np.ascontiguousarray(y),device='cuda') for y in args]
            self.plans.append((count,width,len(t),args,dev))
            self.ports+=count*width;self.rounding+=err*count

    def draw(self,backend,n,seed,off=0):
        if backend=='cpu':out=np.empty((n,self.ports),np.uint8)
        else:pieces=[]
        start=0
        for g,(cells,width,size,host,dev) in enumerate(self.plans):
            domain=2*g+int(self.pnr)
            if backend=='cpu':
                rc=self.lib.cpu(n,cells,size,domain,seed,off,width,self.ports,8,*[y.ctypes.data for y in host],out.ctypes.data+start)
            else:
                x=torch.empty((n,cells),dtype=torch.int32,device='cuda')
                rc=self.lib.gpu(n,cells,size,domain,seed,off,*[y.data_ptr() for y in dev],x.data_ptr(),torch.cuda.current_stream().cuda_stream)
                pieces.append(x.view(torch.uint8).reshape(n,cells,4)[:,:,:width].reshape(n,-1))
            if rc:raise RuntimeError(rc)
            start+=cells*width
        if backend=='gpu':out=(pieces[0] if len(pieces)==1 else torch.cat(pieces,dim=1)).cpu().numpy()
        return out


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    lib=Library(out)
    unequal=DATA/'iteration112_unequal_v1'
    uz=np.load(unequal/'marginals.npz')
    cases=[]
    for row in json.loads((unequal/'unequal.json').read_text()):
        scale=(row['counts'][0]-1)//2;b=row['buses']
        groups=[(c,d,2,np.arange(2**d),uz[f'b{b}t{scale}g{i}']) for i,(c,d) in enumerate(zip(row['counts'],row['widths']))]
        cases.append((f'unequal{b}_{scale}',groups,row['full_tv_budget'],False))
    for pn in (False,True):
        source=DATA/('iteration112_count_marginals_v1' if pn else 'iteration112_marginals_v1')
        z=np.load(source/'marginals.npz')
        data=json.loads((source/'bounds.json').read_text())
        for c in ((32768,) if pn else (32768,262144)):
            row=min((a for a in data if a['cells']==c),key=lambda a:a['full_tv_budget'])
            cases.append((f'common{int(pn)}_{c}',[(c,4,26 if pn else 2,z['keep'] if pn else np.arange(16),z[f'q{c}'])],
                          row['full_tv_budget'],pn))
    rows=[];arrays={}
    for i,(label,groups,budget,pn) in enumerate(cases):
        model=Register(lib,groups,pn);seed=112800000+i*100
        n=max(4,min(64,4194304//model.ports));timing={}
        for backend in ('cpu','gpu'):
            tm=[]
            for rep in range(11):
                start=perf_counter();a=model.draw(backend,n,seed,n*rep)
                if rep>=2:tm.append((perf_counter()-start)*1e6/n)
            timing[backend]=tm
        cpu=model.draw('cpu',8,seed+40);gpu=model.draw('gpu',8,seed+40)
        assert np.array_equal(cpu,gpu) and np.array_equal(cpu[2:6],model.draw('gpu',4,seed+40,2))
        first=[];totals=[]
        for off in range(0,128,n):
            a=model.draw('gpu',min(n,128-off),seed+50,off)
            if off==0:first=a[:4].copy()
            totals.extend(a.sum(1,dtype=np.int64))
        arrays[label+'_first']=first if pn else np.packbits(first,axis=1,bitorder='little')
        arrays[label+'_totals']=np.array(totals)
        rows.append(dict(name=label,ports=model.ports,pnr=pn,shots=128,batch=n,full_tv_budget=budget+model.rounding,
                         rounding=model.rounding,matched_rows=8,timing=timing))
        print(label,model.ports,budget,{k:round(np.median(t),2) for k,t in timing.items()},flush=True)
    np.savez_compressed(out/'records.npz',**arrays)
    (out/'emit.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output.resolve())
