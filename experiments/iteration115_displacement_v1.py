"""Full-brightness Gaussian conditioning and displaced zero/two-photon optical readout."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from scipy.linalg import eigh,solve_triangular
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import ROOT,DATA,P,I,J,U,hardware,cov,energy_law
from iteration114_pairs_v1 import sector


def conditional(t,s):
    m,r=t.shape;q,top=np.linalg.qr(t,mode='reduced');d=len(top)
    eig,u=eigh(np.eye(r)-t.conj().T@t)
    assert min(eig)>-1e-12
    env=(u*np.sqrt(eig.clip(0)))@u.conj().T;nt=d+r
    joint=2*cov(np.vstack([top,env]),s)+np.eye(2*nt)
    vis=np.r_[0:d,nt:nt+d];meas=np.arange(nt+d,2*nt)
    total=joint[np.ix_(vis,vis)]
    lower=np.linalg.cholesky(joint[np.ix_(meas,meas)])
    response=solve_triangular(lower,joint[np.ix_(meas,vis)],lower=True).T
    v=total-response@response.T;v=(v+v.T)/2
    eye=np.eye(d);omega=np.block([[0*eye,eye],[-eye,0*eye]])
    pure_error=float(abs(v@omega@v-omega).max());assert pure_error<2e-10
    return q,v,total,response,pure_error


def core(t,s):
    m,r=t.shape
    q,v,total,response,pure_error=conditional(t,s)
    d=q.shape[1];eye=np.eye(d)
    k=np.eye(2*d)-np.linalg.inv((v+np.eye(2*d))/2)
    j=np.vstack((eye,1j*eye))/np.sqrt(2)
    a=j.T@k@j
    assert abs(j.T@k@j.conj()).max()<1e-10
    gain=.5*np.log(np.linalg.eigvalsh(v)[d:])
    retain,probs=sector(gain,2)
    scale=(q@(response[:d]+1j*response[d:]))/2
    vec=q@a.T
    diagonal=(q*vec).sum(1)
    gv=np.empty_like(vec);one=np.empty(m);two=np.zeros(m+1)
    gram=np.zeros((d,d),complex)
    for i in range(m-1,-1,-1):
        gv[i]=gram@vec[i]
        one[i]=np.vdot(vec[i],gv[i]).real
        two[i]=two[i+1]+one[i]+.5*abs(diagonal[i])**2
        gram+=np.outer(q[i].conj(),q[i])
    assert abs(two[0]-.5*(abs(a)**2).sum())<1e-10
    quantiles={}
    for cap in range(0,40,2):
        mass,_=sector(gain,cap)
        quantiles[cap]=max(0.,1-mass)
    stats=dict(axes=r,visible_axes=d,mean_photons=float((np.trace(total)-2*d)/4),
               mean_core=float((np.trace(v)-2*d)/4),mean_displacement=float((abs(scale)**2).sum()),
               retained_core=retain,tail=quantiles,max_purity_error=pure_error,
               pair_core_mean=float(2*two[0]/(1+two[0])),squeezing=gain.tolist())
    return [np.ascontiguousarray(y) for y in (q,vec,gv,diagonal,one,two)],np.ascontiguousarray(scale.T),stats


def loadlib(out):
    dll=out/('displaced.dll' if os.name=='nt' else 'displaced.so')
    subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-fPIC','-shared',str(ROOT/'native/displaced_pair115_v1.cpp'),'-o',str(dll)],check=True)
    if os.name=='nt':loadlib.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(dll.resolve()));lib.displaced115.argtypes=[I]*5+[U,J,I]+[P]*9
    return lib


def hardware_stats():
    hw=json.loads((DATA/'iteration112_hardware_v1/sources.json').read_text())
    root=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'
    pop=np.array([i.bit_count() for i in range(256)],np.uint8);out=[]
    for item in hw:
        file=root/item['member']
        with file.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
        raw=np.memmap(file,np.uint8,'r').reshape(-1,144)
        h=np.zeros(1153,int);pair=np.zeros(81,int)
        for off in range(0,len(raw),8192):
            vals=pop[raw[off:off+8192]]
            h+=np.bincount(vals.sum(1).astype(int),minlength=1153)
            pair+=np.bincount(vals[:,34]+9*vals[:,35],minlength=81)
        out.append((h,pair))
    return out


def run(out):
    out.mkdir(exist_ok=True,parents=True);lib=loadlib(out)
    t,ss=hardware();m,r=t.shape
    start=perf_counter();models=[core(t,s) for s in ss];setup=perf_counter()-start
    hs=hardware_stats()
    optical=json.loads((DATA/'iteration115_visible_v1/high_power.json').read_text())
    p=np.convolve(*[v['p'] for v in optical['spectra']])
    ref=np.load(DATA/'iteration113_device_v1/banks.npz')['number_bins'].sum(2)
    out_arrays={f'hw{i}_{label}':a for i,row in enumerate(hs) for label,a in zip(('total','pair'),row)}
    rows=[];shots=32768;batch=128
    for clicks in (0,1):
        times=[];aggregate=np.zeros(1153,int);photon=np.zeros(len(p),int);pairs=np.zeros(81,int);banks=np.zeros((144,9),int)
        seed=115320000+clicks;worst=0
        outspec=np.empty((2,batch,m),np.uint8);dest=np.empty((batch,m),np.uint8)
        # Per-route, per-spectrum, per-batch SeedSequence domains for the homodyne outcomes.
        for off in range(-batch,shots,batch):
            start=perf_counter()
            for j,(args,transform,stats) in enumerate(models):
                rng=np.random.default_rng(np.random.SeedSequence([115321,clicks,j,off+batch]))
                alpha=np.ascontiguousarray(rng.normal(size=(batch,r))@transform)
                err=np.empty(1)
                rc=lib.displaced115(batch,m,args[0].shape[1],clicks,8,seed,2**31 if off<0 else off,j,
                    *[v.ctypes.data for v in args],alpha.ctypes.data,outspec[j].ctypes.data,err.ctypes.data)
                assert rc==0,(rc,clicks,off)
                worst=max(worst,float(err[0]))
            if clicks:np.bitwise_or(outspec[0],outspec[1],out=dest)
            else:np.add(outspec[0],outspec[1],out=dest)
            elapsed=perf_counter()-start
            if off<0:continue
            times.append(elapsed*1e6/batch)
            if off==0:out_arrays[f'sample{clicks}']=dest[:16].copy()
            if not clicks:photon+=np.bincount(dest.sum(1).astype(int),minlength=len(p))
            hits=(dest>0).reshape(batch,144,8).sum(2).astype(int)
            aggregate+=np.bincount(hits.sum(1),minlength=1153);pairs+=np.bincount(hits[:,34]+9*hits[:,35],minlength=81)
            banks+=np.bincount((hits+9*np.arange(144)).ravel(),minlength=144*9).reshape(144,9)
        emp=aggregate/shots;pair=pairs/shots
        assert worst<1e-7,worst
        row=dict(direct=bool(clicks),shots=shots,median_us=float(np.median(times)),mean_us=float(np.mean(times)),times=times,
                 max_normalization_error=worst,mean_clicks=float(emp@np.arange(1153)),
                 bank_average_tv_to_optics=float((abs(banks/shots-ref).sum(1)/2).mean()),
                 global_click_tv_to_hardware=[float(abs(emp-h/h.sum()).sum()/2) for h,_ in hs],
                 bank_pair_tv_to_hardware=[float(abs(pair-v/v.sum()).sum()/2) for _,v in hs])
        if not clicks:
            row.update(total_pnr_tv=float(abs(photon/shots-p).sum()/2),
                       mean_pnr=float(photon@np.arange(len(p))/shots),
                       total_pnr_ks=float(abs(np.cumsum(photon/shots-p)).max()))
        out_arrays.update({f'total{clicks}':aggregate,f'pair{clicks}':pairs,f'banks{clicks}':banks})
        if not clicks:out_arrays['photons']=photon
        rows.append(row)
        print(clicks,row['median_us'],row['mean_clicks'],row['bank_average_tv_to_optics'],flush=True)
    info=dict(setup_s=setup,components=[v[2] for v in models],rows=rows,
              full_state_bound=float(np.sqrt(1-np.prod([v[2]['retained_core'] for v in models]))))
    np.savez_compressed(out/'samples.npz',**out_arrays)
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(a.parse_args().output)
