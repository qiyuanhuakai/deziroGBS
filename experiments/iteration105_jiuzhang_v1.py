"""All 25 Jiuzhang twin-beam sources: exchange within each pair and a labelled-arm control."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ORDER,HIST,Native,realify,I,D,bind,save,manifest
from iteration100_projection_v1 import physical
from iteration100_baselines_v1 import independent_or
from iteration103_sample_v1 import DATA,BASE
from iteration105_two_mode_v1 import TwoMode,OUT

JZ=DATA/'iteration105_jiuzhang_v1'


def inputs():
    z=dict(np.load(ORDER))
    t=z['T'][np.argsort(z['order'])]
    modes=np.array([t[:,2*i:2*i+2] for s in range(2) for i in range(25)])
    r=z['squeezing'].T.ravel()
    return z,modes,r


def xpair(r):
    n=np.sinh(r)**2
    ab=np.sinh(r)*np.cosh(r)*np.array([[0.,1.],[1.,0.]])
    return np.block([[n*np.eye(2)+ab,np.zeros((2,2))],[np.zeros((2,2)),n*np.eye(2)-ab]])


def total_law(modes,r,labelled):
    length=1024
    w=np.exp(-2j*np.pi*np.arange(length//2+1)/length)
    log=np.zeros_like(w)
    for u,s in zip(modes,r):
        if labelled:
            eta=(abs(u)**2).sum(0)
            q=np.tanh(s)**2
            log+=np.log1p(-q)-np.log1p(-q*(1+(w-1)*eta[0])*(1+(w-1)*eta[1]))
        else:
            a=realify(np.linalg.qr(u,mode='reduced')[1])
            e=np.linalg.eigvalsh(a@xpair(s)@a.T)
            log-=.5*np.log1p((1-w[:,None])*e).sum(1)
    return np.fft.irfft(np.exp(log),n=length)


def labelled_click(modes,r,ids):
    m=len(ids);size=1<<m
    before=np.ones(size,np.longdouble)
    for u,s in zip(modes,r):
        local=abs(u[ids])**2
        subsets=[np.zeros((1,2),np.longdouble)]
        a=subsets[0]
        for v in local:a=np.vstack((a,a+v))
        q=np.longdouble(np.tanh(s)**2)
        before*=(1-q)/(1-q*(1-a[:,0])*(1-a[:,1]))
    law=before[::-1].copy()
    for j in range(m):
        b=law.reshape(-1,2,1<<j);b[:,1]-=b[:,0]
    assert law.min()>-2e-12
    return law.astype(float)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    z,modes,r=inputs()
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    specs=[]
    full=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    hardware=np.load(HIST)['total_click_histogram']
    ports=[np.r_[np.arange(8)+288,np.arange(8)+856],np.arange(16)]
    for i,ids in enumerate(ports):
        batches=[]
        covs=[]
        for u,s in zip(modes,r):
            b=realify(u[ids])
            covs.append(np.eye(32)+b@xpair(s)@b.T)
        covs=np.array(covs)
        target=physical(z,ids)
        assert max(abs((covs.reshape(2,25,32,32)-np.eye(32)).sum(1)+np.eye(32)-target).ravel())<1e-11
        truth=np.empty(65536)
        assert oracle(16,2,target,np.zeros((2,32)),truth,np.empty(2))==0
        for k in range(0,len(covs),8):
            b=covs[k:k+8];law=np.empty(65536)
            assert oracle(16,len(b),b,np.zeros((len(b),32)),law,np.empty(2))==0
            batches.append(law)
        coherent=independent_or(batches)
        labelled=labelled_click(modes,r,ids)
        specs.append((truth,coherent,labelled))
        np.savez_compressed(out/f'optical_panel_{i}.npz',truth=truth,coherent=coherent,labelled=labelled,ports=ids)
    results=[]
    for label,control in [('coherent_pairs',False),('labelled_arms',True)]:
        ref=total_law(modes,r,control)
        pmf=ref.clip(0);pmf/=pmf.sum()
        tv=float(abs(ref-full[:len(ref)]).sum()/2)
        ks=float(abs(ref.cumsum()-full.cumsum()[:len(ref)]).max())
        start=perf_counter()
        model=TwoMode(OUT/'two.dll',modes,np.repeat(r[:,None],2,axis=1),np.ones(len(r),bool),independent_arms=control)
        prep=perf_counter()-start
        for pnr in (False,True):
            seed=105021000+2*control+pnr
            timing=[]
            for k in range(7):
                tic=perf_counter();model.draw(2048,pnr,seed,k*2048,16)
                if k:timing.append((perf_counter()-tic)*1e6/2048)
            n=262144
            h=np.zeros(1153,np.int64);ph=np.zeros(1024,np.int64)
            panels=np.zeros((2,65536),np.int64)
            for off in range(0,n,2048):
                x=model.draw(2048,pnr,105022000+2*control+pnr,off,16)
                clicked=x>0
                h+=np.bincount(clicked.sum(1),minlength=len(h))
                if pnr:ph+=np.bincount(x.sum(1).astype(np.int32),minlength=len(ph))
                for j,ids in enumerate(ports):
                    code=clicked[:,ids]@(np.uint32(1)<<np.arange(16,dtype=np.uint32))
                    panels[j]+=np.bincount(code,minlength=65536)
            laws=[]
            for j,(truth,coherent,indep) in enumerate(specs):
                target=indep if control else coherent
                event=target>truth
                laws.append(dict(full_target_tv=float(abs(target-truth).sum()/2),event_bias=float(panels[j,event].sum()/n-target[event].sum()),empirical_vs_sampler=float(abs(panels[j]/n-target).sum()/2)))
            hwks=float(abs(h.cumsum()/n-hardware.cumsum()/hardware.sum()).max())
            photon_error=float(abs(ph.cumsum()/n-ref.cumsum()).max()) if pnr else None
            assert max(abs(v['event_bias']) for v in laws)<.007,(label,laws)
            if pnr:assert photon_error<.008,(label,photon_error)
            results.append(dict(label=label,pnr=pnr,records=n,prep_s=prep,us=float(np.median(timing)),timing=timing,mean_photon=float(pmf@np.arange(len(pmf))),photon_total_ks_vs_full=ks,photon_total_tv_vs_full=tv,photon_sample_ks_vs_own=photon_error,empirical_hardware_click_ks=hwks,panel= laws,source_tail=model.tail,source_alias=model.rounding))
            np.savez_compressed(out/f'{label}_{int(pnr)}.npz',click=h,photon=ph,model_photon=ref,panels=panels)
            print(label,pnr,hwks,tv,flush=True)
    save(out,'summary.json',dict(rows=results,groups=25,spectra=2,hardware_records=int(hardware.sum()),hardware_radius99=float(np.sqrt(np.log(2000)/(2*hardware.sum()))+np.sqrt(np.log(2000)/(2*n))),optical_reference='Original high-power spectral covariances. Source labels distinguish different down-converters; within-source exchange and no-exchange arms are measured separately.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
