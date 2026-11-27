"""Physical total-number tilts of a bank Gaussian, shared by direct clicks and PNR."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.stats import beta
from threadpoolctl import threadpool_limits
import torch
from iteration103_gpu_v1 import build
from iteration115_visible_v1 import ROOT,DATA,P,I,U,J,hardware,energy_law
from iteration115_banks_v1 import bank_model


def setup(t,ss,truth=None,fit_even=False,cap=440,adaptive=False):
    base,_,bound=bank_model(t,ss,8)
    mat,eig,_,split=base;g=len(split);latent=mat.shape[0];mt=mat.reshape(latent,g,4)
    e_all=[]
    for s in range(2):
        a=mt[:,:,2*s:2*s+2].reshape(latent,-1)
        e_all.extend(np.linalg.eigvalsh(2*a.T@a+np.diag(eig[:,2*s:2*s+2].ravel())))
    e_all=np.array(e_all)
    if truth is None:
        ref=json.loads((DATA/'iteration115_visible_v1/high_power.json').read_text())
        truth=np.convolve(*[v['p'] for v in ref['spectra']])
    q,base_tail=energy_law(e_all,cap)
    n=np.arange(cap+1);grid=np.linspace(-.1,.1,25)
    if adaptive:
        sel=slice(None,None,2 if fit_even else 1)
        def variance(v):
            a=v[sel]/v[sel].sum();nn=n[sel]
            return a@((nn-a@nn)**2)
        variance_q=variance(q)
        width=np.sqrt(max(0.,variance(truth[:cap+1])-variance_q))/max(variance_q,1e-12)
        grid=np.unique(np.r_[grid,np.linspace(-12.,6.,49)*width])
    grid=grid[1+(1-np.exp(grid))*e_all.max()>0]
    z=np.exp(grid)
    logg=-.5*np.log1p((1-z[:,None])*e_all).sum(1)
    cols=q[:,None]*np.exp(n[:,None]*grid-logg)
    sample=slice(None,None,2 if fit_even else 1)
    weights=nnls(cols[sample],truth[:cap+1][sample])[0]
    use=weights>1e-10;weights=weights[use];weights/=weights.sum()
    z=z[use];logg=logg[use];cols=cols[:,use]
    proposal=cols@weights
    tail_model=max(0,1-proposal.sum());tail_true=max(0,1-truth[:cap+1].sum())
    tv_upper=(abs(proposal-truth[:cap+1]).sum()+tail_model+tail_true)/2
    uncorrected=(abs(q-truth[:cap+1]).sum()+base_tail+tail_true)/2
    bits=(np.arange(256)[:,None]>>np.arange(8))&1;w=np.diff(split,axis=1,prepend=0)@bits.T
    factors=[];scales=[];tabs=[];vac=[];marg=[];err=[]
    for comp,v in enumerate(z):
        denom=1+(1-v)*eig
        invscale=np.sqrt(v)/denom
        newx=v*eig/denom
        precision=np.eye(latent)+2*(1-v)*(mat/denom.ravel())@mat.T
        chol=np.linalg.cholesky(precision)
        transform=solve_triangular(chol,np.eye(latent),lower=True).T
        lat=transform.T@mat*invscale.ravel()
        formed=-.5*np.log(denom).sum()-np.log(np.diag(chol)).sum()
        err.append(abs(formed-logg[comp]))
        assert err[-1]<1e-10
        factor=1+w[:,:,None]*newx[:,None,:]
        gate=np.concatenate((np.exp(-.5*np.log(factor).sum(2))[:,:,None],w[:,:,None]/factor),axis=-1)
        factors.append(transform);scales.append(invscale);tabs.append(newx);vac.append(gate)
        cov=[]
        for s in range(2):
            a=lat.reshape(latent,g,4)[:,:,2*s:2*s+2].reshape(latent,-1)
            cov.append(2*a.T@a+np.diag(newx[:,2*s:2*s+2].ravel()))
        marg.append(cov)
    marginal=np.array(marg)
    def bankwords():
        cdf=np.zeros((g,256))
        for prob,mats in zip(weights,marginal):
            v=np.ones((g,256))
            for x in mats:
                e=np.array([np.linalg.eigvalsh(x[2*i:2*i+2,2*i:2*i+2]) for i in range(g)])
                v*=np.exp(-.5*np.log1p(w[:,:,None]*e[:,None,:]).sum(2))
            cdf+=prob*v
        law=cdf[:,::-1].astype(np.longdouble)
        for j in range(8):
            view=law.reshape(g,-1,2,1<<j);view[:,:,1]-=view[:,:,0]
        return np.asarray(law,float)
    bank=bankwords()
    info=dict(log_tilts=np.log(z).tolist(),weights=weights.tolist(),total_tv_upper=float(tv_upper),
              base_total_tv_upper=float(uncorrected),outside_compared_target=tail_true,
              outside_compared_model=tail_model,max_log_normalizer_error=max(err))
    if truth is not None and g==144 and not fit_even and t.shape==(1152,50):
        # Only attach experimental reference comparisons for the archived original gain.
        _,original=hardware()
        if np.array_equal(ss,original):
            saved=np.load(DATA/'iteration113_device_v1/banks.npz')
            hw=saved['hardware']/saved['hardware'].sum(2,keepdims=True)
            info.update(bank_word_tv_mean=float((abs(bank-saved['words']).sum(1)/2).mean()),
                bank_word_tv_max=float((abs(bank-saved['words']).sum(1)/2).max()),
                bank_word_tv_to_hardware=(abs(hw-bank).sum(2)/2).mean(1).tolist())
    gpu=[mat,split,np.array(factors),np.array(scales),np.array(tabs),np.array(vac),weights]
    return gpu,info,dict(bank=bank,total=proposal,truth=truth,base=q,marginal=marginal,weights=weights,split=split)


class Device:
    def __init__(self,lib,args,wide):
        self.lib=lib;self.wide=wide;self.dtype=torch.float64 if wide else torch.float32
        mat,split,trans,scales,x,gates,weights=args
        dev=lambda a,dtype:torch.as_tensor(np.ascontiguousarray(a),dtype=dtype,device='cuda')
        self.mat=dev(mat,self.dtype);self.trans=dev(trans,self.dtype);self.d=mat.shape[0]
        cdf=weights.cumsum();cdf[-1]=1.;self.cdf=dev(cdf,torch.float64)
        self.g,self.f=split.shape;self.tables=[dev(a,torch.float64) for a in (x,gates,split,scales)]
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32);self.capacity=0
    def draw(self,n,pn,seed,offset):
        if self.capacity!=n:
            opt=dict(device='cuda',dtype=self.dtype)
            self.noise=torch.empty((n,self.d),**opt);self.latent=torch.empty_like(self.noise)
            self.labels=torch.empty(n,device='cuda',dtype=torch.int32);self.fields=torch.empty((n,self.g*4),**opt)
            self.out=torch.empty((n,self.g*self.f),device='cuda',dtype=torch.uint8);self.capacity=n
        stream=torch.cuda.current_stream().cuda_stream
        assert self.lib.normal103_launch(self.d//2,n,seed,offset,self.wide,self.noise.data_ptr(),stream)==0
        assert self.lib.tilt115_latent(n,self.d,len(self.cdf),self.wide,seed,offset,self.noise.data_ptr(),self.cdf.data_ptr(),
            self.trans.data_ptr(),self.labels.data_ptr(),self.latent.data_ptr(),stream)==0
        torch.mm(self.latent,self.mat,out=self.fields)
        x,gates,split,scales=self.tables
        rc=self.lib.tilt115_banks(n,self.g,self.f,pn,self.wide,seed,offset,self.fields.data_ptr(),x.data_ptr(),
            gates.data_ptr(),split.data_ptr(),self.labels.data_ptr(),scales.data_ptr(),self.out.data_ptr(),self.error.data_ptr(),stream)
        assert rc==0
        return self.out.cpu().numpy()


def run(out):
    out.mkdir(parents=True,exist_ok=True);build(out,'homodyne_bank115_v1.cu')
    lib=ct.CDLL(str((out/'gpu.dll').resolve()));lib.normal103_launch.argtypes=[I,I,U,J,I,P,P]
    lib.tilt115_latent.argtypes=[I]*4+[U,J]+[P]*6;lib.tilt115_banks.argtypes=[I]*5+[U,J]+[P]*9
    torch.set_float32_matmul_precision('highest')
    t,s=hardware();start=perf_counter();args,info,arr=setup(t,s);info['setup_s']=perf_counter()-start
    fast=Device(lib,args,False);wide=Device(lib,args,True)
    hw=np.load(DATA/'iteration115_displacement_v1/samples.npz')
    info['measurements']=[];shots=131072;batch=4096;target=arr['truth']
    for pn in (0,1):
        seed=115500000+pn;mismatch=0;h=np.zeros(1153,int);phot=np.zeros(1024,int);pair=np.zeros(81,int);times=[]
        for offset in range(-batch,shots,batch):
            off=2**31 if offset<0 else offset
            start=perf_counter();v=fast.draw(batch,pn,seed,off);elapsed=perf_counter()-start
            if offset<0:continue
            times.append(elapsed*1e6/batch)
            if offset<32768:mismatch+=int(np.any(v!=wide.draw(batch,pn,seed,off),axis=1).sum())
            hist=(v>0).reshape(batch,144,8).sum(2).astype(int)
            h+=np.bincount(hist.sum(1),minlength=len(h))
            pair+=np.bincount(hist[:,34]+9*hist[:,35],minlength=81)
            if pn:phot+=np.bincount(v.sum(1).astype(int),minlength=len(phot))
            if offset==0:arr[f'sample{pn}']=v[:16].copy()
        timing={}
        for n in (2048,8192,16384):
            ts=[]
            for rep in range(9):
                start=perf_counter();fast.draw(n,pn,seed+10000+n,rep*n)
                if rep:ts.append((perf_counter()-start)*1e6/n)
            timing[n]=ts
        assert fast.error.item()==wide.error.item()==0
        row=dict(pnr=bool(pn),shots=shots,median_us=float(np.median(times)),times=times,batch_times=timing,
                 fp32_disagreements=mismatch,fp_compared=32768,
                 coupling_upper99=float(beta.ppf(.99,mismatch+1,32768-mismatch)),
                 total_click_tv_hardware=[float(abs(h/shots-hw[f'hw{j}_total']/hw[f'hw{j}_total'].sum()).sum()/2) for j in range(2)],
                 two_bank_empirical_tv_hardware=[float(abs(pair/shots-hw[f'hw{j}_pair']/hw[f'hw{j}_pair'].sum()).sum()/2) for j in range(2)])
        if pn:row['total_pnr_empirical_ks']=float(abs((phot/shots-np.pad(target,(0,len(phot)-len(target)))).cumsum()).max())
        arr[f'count{pn}']=h;arr[f'pair{pn}']=pair
        if pn:arr['photons']=phot
        info['measurements'].append(row)
        print(pn,info['total_tv_upper'],row['median_us'],[np.median(v) for v in timing.values()],flush=True)
    np.savez_compressed(out/'model.npz',**arr)
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(p.parse_args().output)
