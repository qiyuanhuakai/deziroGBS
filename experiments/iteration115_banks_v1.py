"""Parallel homodyne-conditioned Gaussian bank readout at original Jiuzhang 3 gain."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.stats import beta
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import build
from iteration115_displacement_v1 import conditional
from iteration115_visible_v1 import ROOT,DATA,P,I,U,J,hardware,gauss
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration111_memory_v1 import realify


def bank_model(t,ss,fan):
    m,r=t.shape;g=m//fan
    if fan==1:w=np.ones((g,1),complex)
    else:w=np.linalg.svd(t.reshape(g,fan,r),full_matrices=False)[0][:,:,0]
    tc=(w.conj()[:,:,None]*t.reshape(g,fan,r)).sum(1)
    dark=t-(w[:,:,None]*tc[:,None,:]).reshape(m,r)
    if fan==1:dark_p=1.
    else:
        dark_p=np.exp(sum(-np.log1p(gauss(dark,s)[-1]).sum()/2 for s in ss))
    weights=abs(w)**2
    weights/=weights.sum(1,keepdims=True)
    transfer=[];local=[];physical=[];overlaps=[]
    lift=sparse.coo_matrix((w.ravel(),(np.arange(m),np.repeat(np.arange(g),fan))),shape=(m,g)).tocsr()
    lift=sparse.bmat([[lift.real,-lift.imag],[lift.imag,lift.real]],format='csr')
    for s in ss:
        q,pure,total,response,_=conditional(tc,s)
        d=q.shape[1];basis=realify(q);small=(pure-np.eye(2*d))/2
        uq=np.stack((basis[:g],basis[g:]),axis=1)
        block=(uq@small)@uq.transpose(0,2,1)
        eigen,rotation=np.linalg.eigh(block)
        amp=(q@(response[:d]+1j*response[d:]))/2
        transfer.append(np.einsum('gir,gij->rgj',np.stack((amp.real,amp.imag),axis=1),rotation))
        local.append(eigen)
        fac=np.vstack((amp.real,amp.imag))
        x=2*(fac@fac.T)
        ids=np.arange(g)
        for a in (0,1):
            for b in (0,1):x[ids+a*g,ids+b*g]+=block[:,a,b]
        physical.append(np.ascontiguousarray((lift@x)@lift.T))
        eye=np.eye(2)+block
        inner=np.einsum('gai,gaj->ij',uq,np.linalg.solve(eye,uq))
        sg,log=np.linalg.slogdet(np.eye(2*d)+small@inner)
        assert sg>0
        overlaps.append(float(np.exp(-.5*(np.linalg.slogdet(eye)[1].sum()+log))))
    eig=np.column_stack(local)
    transform=np.zeros((2*r,g,4))
    transform[:r,:,:2]=transfer[0];transform[r:,:,2:]=transfer[1]
    masks=np.arange(1<<fan);bits=(masks[:,None]>>np.arange(fan))&1
    wsum=weights@bits.T
    denom=1+wsum[:,:,None]*eig[:,None,:]
    gates=np.concatenate((np.exp(-.5*np.log(denom).sum(2))[:,:,None],wsum[:,:,None]/denom),axis=-1)
    eps=1-dark_p
    bound=min(1.,np.sqrt(max(0,eps))+eps+np.sqrt(max(0,1-np.prod(overlaps))))
    return (transform.reshape(2*r,-1),eig,gates,weights.cumsum(1)),np.array(physical),dict(
        full_trace_upper=float(bound),dark_nonvacuum=eps,core_squared_overlaps=overlaps)


class Device:
    def __init__(self,lib,args,wide):
        self.lib=lib;self.wide=wide;self.dtype=torch.float64 if wide else torch.float32
        transform,eig,gates,weights=args
        self.g,self.f=weights.shape;self.latent=transform.shape[0]
        self.transform=torch.as_tensor(transform,device='cuda',dtype=self.dtype)
        self.tab=[torch.as_tensor(np.ascontiguousarray(a),device='cuda') for a in (eig,gates,weights)]
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32);self.capacity=0
    def draw(self,n,pn,seed,offset):
        if n!=self.capacity:
            self.z=torch.empty((n,self.latent),dtype=self.dtype,device='cuda')
            self.a=torch.empty((n,self.g*4),dtype=self.dtype,device='cuda')
            self.out=torch.empty((n,self.g*self.f),dtype=torch.uint8,device='cuda');self.capacity=n
        stream=torch.cuda.current_stream().cuda_stream
        assert self.lib.normal103_launch(self.latent//2,n,seed,offset,self.wide,self.z.data_ptr(),stream)==0
        torch.mm(self.z,self.transform,out=self.a)
        rc=self.lib.bank115_launch(n,self.g,self.f,pn,self.wide,seed,offset,self.a.data_ptr(),
            *[v.data_ptr() for v in self.tab],self.out.data_ptr(),self.error.data_ptr(),stream)
        assert rc==0
        return self.out.cpu().numpy()


def compare_optical(x,true,fan):
    ref=np.load(DATA/'iteration113_device_v1/banks.npz')
    quad=oracle(True);banks=[];pop=np.array([i.bit_count() for i in range(256)])
    for j in range(144):
        ix=np.r_[8*j:8*j+8,1152+8*j:1152+8*j+8]
        banks.append(gaussian_click(quad,np.ascontiguousarray(x[:,ix][:,:,ix]),ONE))
    banks=np.asarray(banks)
    hw=ref['hardware']/ref['hardware'].sum(2,keepdims=True)
    qtotal=np.ones(1)
    for part in x:
        ev=np.linalg.eigvalsh(part)
        z=np.exp(-2j*np.pi*np.arange(513)/1024)
        pgf=np.exp(-.5*np.log1p((1-z[:,None])*ev).sum(1))
        qtotal=np.convolve(qtotal,np.fft.irfft(pgf,n=1024))
    p=np.pad(true,(0,len(qtotal)-len(true)))
    tail=sum(np.load(DATA/'iteration113_device_v1/banks.npz')['tail'])
    st=dict(total_pnr_model_tv=float(abs(qtotal-p).sum()/2),
            bank_word_tv_mean=float((abs(banks-ref['words']).sum(1)/2).mean()),
            bank_word_tv_max=float((abs(banks-ref['words']).sum(1)/2).max()),
            bank_word_tv_to_hardware=(abs(hw-banks).sum(2)/2).mean(1).tolist(),
            mean_clicks=float((banks@pop).sum()),mean_photons=float(np.trace(x,axis1=1,axis2=2).sum()/2))
    return st,dict(bank_law=banks,total_law=qtotal)


def run(out):
    out.mkdir(exist_ok=True,parents=True);build(out,'homodyne_bank115_v1.cu')
    lib=ct.CDLL(str((out/'gpu.dll').resolve()));lib.normal103_launch.argtypes=[I,I,U,J,I,P,P]
    lib.bank115_launch.argtypes=[I]*5+[U,J]+[P]*7
    torch.set_float32_matmul_precision('highest')
    t,ss=hardware();opt=json.loads((DATA/'iteration115_visible_v1/high_power.json').read_text())
    true=np.convolve(*[v['p'] for v in opt['spectra']])
    hw=np.load(DATA/'iteration115_displacement_v1/samples.npz')
    shots=131072;batch=4096
    result=[];arrays={}
    for fan in (1,8):
        begin=perf_counter();args,x,bound=bank_model(t,ss,fan);setup=perf_counter()-begin
        fast=Device(lib,args,False);wide=Device(lib,args,True);g=1152//fan
        stats,vals=compare_optical(x,true,fan)
        for k,v in vals.items():arrays[f'f{fan}_{k}']=v
        rows=[]
        for pn in (0,1):
            times=[];phot=np.zeros(1024,int);hits=np.zeros(1153,int);pair=np.zeros(81,int);mismatch=0
            seed=115400000+fan*100+pn
            for offset in range(-batch,shots,batch):
                off=2**31 if offset<0 else offset
                start=perf_counter();v=fast.draw(batch,pn,seed,off);elapsed=perf_counter()-start
                if offset<0:continue
                times.append(elapsed*1e6/batch)
                if offset<32768:
                    other=wide.draw(batch,pn,seed,off);mismatch+=int(np.any(v!=other,axis=1).sum())
                h=(v>0).reshape(batch,144,8).sum(2).astype(int)
                hits+=np.bincount(h.sum(1),minlength=1153)
                pair+=np.bincount(h[:,34]+9*h[:,35],minlength=81)
                if pn:phot+=np.bincount(v.sum(1).astype(int),minlength=len(phot))
                if not offset:arrays[f'sample{fan}_{pn}']=v[:16].copy()
            assert fast.error.item()==wide.error.item()==0
            timing={}
            for size in (2048,8192,16384):
                ts=[]
                for rep in range(9):
                    start=perf_counter();fast.draw(size,pn,seed+10000+size,rep*size)
                    if rep:ts.append((perf_counter()-start)*1e6/size)
                timing[size]=ts
            assert fast.error.item()==0
            row=dict(pnr=bool(pn),shots=shots,us=float(np.median(times)),times=times,batch_times=timing,
                     mismatched_fp32_fp64=mismatch,precision_comparisons=32768,
                     coupling_upper99=float(beta.ppf(.99,mismatch+1,32768-mismatch)),
                     total_click_tv_hardware=[float(abs(hits/shots-hw[f'hw{j}_total']/hw[f'hw{j}_total'].sum()).sum()/2) for j in range(2)],
                     two_bank_empirical_tv_hardware=[float(abs(pair/shots-hw[f'hw{j}_pair']/hw[f'hw{j}_pair'].sum()).sum()/2) for j in range(2)])
            if pn:
                p=np.pad(true,(0,len(phot)-len(true)))
                row['total_photon_empirical_ks']=float(abs((phot/shots-p).cumsum()).max())
            rows.append(row)
            arrays[f'counts{fan}_{pn}']=hits;arrays[f'pair{fan}_{pn}']=pair
            if pn:arrays[f'photons{fan}']=phot
            print(fan,pn,row['us'],[np.median(v) for v in timing.values()],flush=True)
        result.append(dict(fan=fan,setup_s=setup,**bound,**stats,measurements=rows))
        np.savez_compressed(out/'samples.npz',**arrays)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=6):run(ap.parse_args().output)
