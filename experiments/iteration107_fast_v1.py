"""Dense Gaussian field with separate direct-click and photon-number partitions."""
import argparse
import ctypes as ct
import os
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import bind,D,manifest,save
from iteration103_gpu_v1 import I,U,CU,P,build
from iteration106_field_v1 import factors
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_cliques_v1 import prepare,optical_x
from iteration107_rank_v1 import gaussian_click
from iteration107_tables_v1 import kernel

OUT=DATA/'iteration107_fast_v1'
ONE=dict(values=[1.],weights=[1.])


def oracle(quad=False):
    if os.name=='nt':
        if not hasattr(oracle,'runtime'):oracle.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    path=DATA/('iteration96_quad_v1/oracle.dll' if quad else 'iteration105_source_groups_v1/extended.dll')
    return bind(ct.CDLL(str(path)),'oracle96',[I,I,D,D,D,D])


def tables(t,r,config):
    full=oracle();m=len(t)
    intensity=(abs(t[:,::2])**2+abs(t[:,1::2])**2)@(np.sinh(r)**2).sum(0)
    groups=[list(range(i,min(m,i+16))) for i in range(0,m,16)]
    bright=max(groups,key=lambda ids:intensity[ids].sum())
    tiles=[bright]+[g[j:j+8] for g in groups if g!=bright for j in range(0,len(g),8)]
    layout=[];moves=[];cuts=[];start=0;err=[];primary={}
    for i,ids in enumerate(tiles):
        f=factors(t[ids],r)
        p=gaussian_click(full,np.stack([2*v.T@v for v in f]),config)
        q=gaussian_click(full,optical_x(t[ids],r),ONE)
        move,cdf,tv,residual=kernel(p,q)
        layout.append(ids+[0]*(16-len(ids))+[0]*4+[len(ids),start,len(p)])
        moves.append(move);cuts.append(cdf);start+=len(p);err.append([tv,residual,float(-q[q<0].sum())])
        if i==0:
            primary=dict(primary_ids=np.array(ids),primary_p=p,primary_q=q)
    if start>=2**31:raise ValueError('Direct table exceeds native offset range')
    layout_p=[]
    for row in config['plan']:
        d,_,start,count=map(int,row[8:])
        layout_p.append(list(row[:d])+[0]*(16-d)+list(row[4:8])+[d,start,count])
    return dict(click_plan=np.array(layout,np.int32),pnr_plan=np.array(layout_p,np.int32),
                click_move=np.concatenate(moves),click_cdf=np.concatenate(cuts),pnr_move=config['pnr_move'],
                pnr_cdf=config['pnr_cdf'],values=config['values'],weights=config['weights'],tile_errors=np.array(err),**primary)


class Fast:
    def __init__(self,t,r,params,half=False,dll=OUT/'gpu.dll'):
        self.lib=ct.CDLL(str(dll))
        self.normal=self.lib.normals107_launch;self.normal.argtypes=[I,I,I,U,CU,I,P,P,I,P,P,P]
        self.emit=self.lib.fast107_launch;self.emit.argtypes=[I,I,I,I,I,U,CU,I,I]+[P]*8
        self.m=len(t);self.s,self.k=r.shape
        if max(self.m,self.s*self.k)>=65536*((65535-251)//8):raise ValueError('Random domain too wide')
        self.half=half;self.dtype=torch.float16 if half else torch.float32
        self.factor=torch.as_tensor(factors(t,r),device='cuda',dtype=self.dtype)
        self.weights=torch.as_tensor(params['values'],device='cuda',dtype=torch.float64)
        cum=params['weights'].cumsum();cum[-1]=1.
        self.cum=torch.as_tensor(cum,device='cuda',dtype=torch.float64)
        self.params=[[torch.as_tensor(params[route+'_'+name],device='cuda') for name in ('plan','move','cdf')] for route in ('click','pnr')]
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32);self.capacity=0
    def draw(self,n,pnr,seed,offset=0):
        if n<1 or max(self.s*self.k,self.m)*n>=2**31-256 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid record request')
        if n>self.capacity:
            self.z=torch.empty(self.s*n*2*self.k,device='cuda',dtype=self.dtype)
            self.f=torch.empty(self.s*n*2*self.m,device='cuda',dtype=self.dtype)
            self.gain=torch.empty(n,device='cuda',dtype=torch.float64)
            self.x=torch.empty((n,self.m),device='cuda',dtype=torch.uint8);self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,-1);f=self.f[:self.s*n*2*self.m].view(self.s,n,-1)
        stream=torch.cuda.current_stream().cuda_stream
        args=(self.k,self.s,n,seed,offset,len(self.weights),self.weights.data_ptr(),self.cum.data_ptr(),int(self.half))
        rc=self.normal(*args,z.data_ptr(),self.gain.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.factor,out=f)
        plan,rule,cdf=self.params[int(pnr)]
        x=self.x[:n]
        def sample(wide,buffer):
            rc=self.emit(len(plan),self.m,self.s,n,int(pnr),seed,offset,int(self.half),int(wide),plan.data_ptr(),f.data_ptr(),
                         self.gain.data_ptr(),rule.data_ptr(),cdf.data_ptr(),buffer.data_ptr(),self.error.data_ptr(),stream)
            if rc:raise RuntimeError(rc)
            status=self.error.item()
            if status:self.error.zero_()
            if status==2:raise ArithmeticError('Optical rate out of range')
            return status
        if sample(False,x):
            x=torch.empty((n,self.m),device='cuda',dtype=torch.int32)
            if sample(True,x):raise ArithmeticError('Invalid photon count')
        return x.cpu().numpy()


def run(out):
    if out.is_dir():
        if {p.name for p in out.iterdir()}!={'build.json','gpu.dll'}:raise FileExistsError(out)
    else:out.mkdir(parents=True)
    build(out,'fast_cliques107_v1.cu');quad=oracle(True)
    rows=[]
    for i,(m,k) in enumerate([(4,2),(16,2),(64,8),(256,16),(1024,32),(4096,64)]):
        t,r=case(m,k);st=perf_counter()
        if m==4096:
            with np.load(DATA/'iteration107_rank_v1/case4096.npz') as z:config=dict(z)
        else:config=prepare(t,r)
        pars=tables(t,r,config);prep=perf_counter()-st
        ids=pars['primary_ids'];raw=pars['primary_p'];f=factors(t[ids],r)
        pq=gaussian_click(quad,np.stack([2*v.T@v for v in f]),config)
        qq=gaussian_click(quad,optical_x(t[ids],r),ONE)
        size=len(raw);rr=pars['click_move'][:size];cdf=pars['click_cdf'][:size]
        corrected=pq*(1-rr.astype(float))+(pq@rr)*np.diff(cdf,prepend=np.uint64(0))/2**52
        assert np.isfinite(corrected).all() and abs(corrected.sum()-1)<1e-9
        models=[Fast(t,r,pars,h,out/'gpu.dll') for h in (False,True)];batch=min(2048,4194304//m)
        row=dict(m=m,sources=k,batch=batch,prep_s=prep,primary_ids=ids.tolist(),primary_base_tv=float(abs(pq-qq).sum()/2),
                 primary_corrected_tv=float(abs(corrected-qq).sum()/2),max_tile_rounding=float(pars['tile_errors'][:,1].max()),timing=[],fp16_changes=[])
        for j,model in enumerate(models):
            for pnr in (False,True):
                tm=benchmark(lambda n,s,o:model.draw(n,pnr,s,o),batch,107041000+20*i+2*j+pnr,9)
                row['timing'].append(dict(half=bool(j),pnr=pnr,us=tm['us']))
        n=65536
        for pnr in (False,True):
            changed=0;hist=np.zeros(size,np.int64)
            for off in range(0,n,batch):
                a,b=[v.draw(batch,pnr,107042000+2*i+pnr,off) for v in models]
                changed+=np.any(a!=b,axis=1).sum()
                if not pnr:hist+=np.bincount(a[:,ids].astype(np.int32)@(1<<np.arange(len(ids))),minlength=size)
            row['fp16_changes'].append(dict(pnr=pnr,n=n,records=int(changed)))
            if not pnr:
                click_hist=hist.copy()
                row['empirical_tv']=float(abs(hist/n-qq).sum()/2)
        if m==4096:
            np.savez_compressed(out/'case4096.npz',**pars)
            np.savez_compressed(out/'prefix4096.npz',optical=qq,model=corrected,empirical=click_hist)
        print(m,row,flush=True);rows.append(row)
    save(out,'summary.json',dict(rows=rows));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('validate','prepare','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--tables',type=Path);p.add_argument('--shots',type=int,default=512)
    p.add_argument('--pnr',action='store_true');p.add_argument('--half',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0)
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='validate':run(a.output.resolve())
        else:
            if a.input is None:p.error('Specify an optical input')
            with np.load(a.input) as z:t,r=z['transfer'],np.atleast_2d(z['squeezing'])
            if a.action=='prepare':
                pars=tables(t,r,prepare(t,r))
                with a.output.open('xb') as f:np.savez_compressed(f,**pars)
            else:
                if a.seed is None or a.tables is None:p.error('Specify saved tables and experiment seed')
                with np.load(a.tables) as z:model=Fast(t,r,dict(z),a.half)
                with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
