"""Matched 16-port lossy optical references, upstream samplers and finite-sample checks."""
import argparse
import ctypes as ct
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from iteration92_pe_common_v1 import GaussianInput,pe_sample,UPSTREAM
from iteration92_quality_v1 import confidence
from iteration96_group_benchmark_v1 import fock
from iteration97_conditionals_v1 import ROOT,Kernel,I,D,B,W,bind,np,H,save,manifest
from iteration99_loss_v1 import SIGNAL,IDLERS,P
from scipy.stats import nbinom

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';src=data/'iteration98_optics_v1/input.npz'
    oracle=data/'iteration96_quad_v1/oracle.dll';frugal=data/'iteration96_frugal_build_v1/frugal.dll'
    thin=ROOT/'native/fock_thinning99_v1.cpp';driver=data/'iteration99_loss_v1'
    files=[Path(__file__),src,oracle,frugal,thin,driver/'records_0.npz',driver/'records_1.npz',data/'iteration96_tree_build_v1/tree.dll']
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},seed=99020000,
        scope='16 physical ports, 2048 per timed batch, median of five, one worker; PNR quality uses 16384 records per method',
        upstream=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),confidence='four one-sided bounds per two PNR laws, delta=.0025 per side, iid records'))
    dll=out/'thin.dll'
    cmd=[shutil.which('g++'),'-shared','-std=c++20','-O3','-ffp-contract=off',str(thin),'-o',str(dll)]
    run=subprocess.run(cmd,text=True,capture_output=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    runtime=os.add_dll_directory(str(Path(cmd[0]).parent));kernel=Kernel()
    z=np.load(src);eta=np.r_[[float(SIGNAL)]*8,np.tile(np.array(IDLERS,float),2)]
    root=np.sqrt(np.tile(eta,2));q=np.eye(32)+(z['q']-np.eye(32))*root[:,None]*root
    mu=np.zeros((1,32));click=np.empty(65536);diag=np.zeros(2)
    ctl=bind(ct.CDLL(str(oracle)),'oracle96',[I,I,D,D,D,D])
    assert ctl(16,1,q,mu,click,diag)==0 and click.min()>-1e-13 and abs(click.sum()-1)<1e-12
    np.savez_compressed(out/'input.npz',q=q,mu=mu,eta=eta,click=click)
    real=np.load(driver/'records_0.npz')['x'];bits=real@(1<<np.arange(16))
    direct,_=confidence(bits,click,len(bits),32768,.0025)
    rows=[dict(method='loss99 direct',records=len(real),empirical=direct)]
    lib=ct.CDLL(str(frugal))
    tree=bind(lib,'fc96_full16',[I,D,D,D,D,D,ct.c_char_p,I])
    sample=bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
    msg=ct.create_string_buffer(2048);cov=2*q[0]-np.eye(32)
    for k in (2,4):
        th=np.empty(65535);prev=th.copy();prop=th.copy();stats=np.zeros(3)
        assert tree(k,cov,th,prev,prop,stats,msg,len(msg))==0 and stats[2]==64
        law=np.ones(1)
        for j in range(16):
            zero=np.nan_to_num(th[len(law)-1:2*len(law)-1],nan=1).clip(0,1);law=np.r_[law*zero,law*(1-zero)]
        times=[]
        for j in range(6):
            t=perf_counter();x=np.empty((2048,16),np.uint8);st=np.empty(7)
            assert sample(k,2048,99020000+100*k+j,cov,x,st,msg,len(msg))==0 and st[5]==64
            elapsed=(perf_counter()-t)*1e6/len(x)
            if j:times.append(elapsed)
        rows.append(dict(method=f'Frugal K{k}',full_click_tv=float(abs(law-click).sum()/2),whole_us=times,median=float(np.median(times))))
        np.savez_compressed(out/f'frugal_{k}.npz',threshold=th,law=law)
    occ,prob=fock(kernel.prior,z['barg'],z['q'],8)
    thinfn=bind(ct.CDLL(str(dll)),'thin99',[I,B,D,D,I,W,D])
    maxwords=math.comb(16+8,8);keys=np.empty(maxwords,np.uint64);pf=np.empty(maxwords)
    sz=thinfn(len(occ),occ,prob,eta,maxwords,keys,pf);assert sz>0
    keys=keys[:sz];pf=pf[:sz];tail=float(nbinom.sf(4,8,1-float(P)))
    assert abs(pf.sum()+tail-1)<5e-13
    # Complete the enumerated subdistribution by putting all omitted parent mass
    # into vacuum. All TV comparisons to the true full law then cost at most tail.
    pf[0]+=1-pf.sum()
    target=np.r_[pf,0.];shift=4*np.arange(16,dtype=np.uint64)
    thresholded=np.zeros(65536);words=sum((((keys>>s)&15)>0).astype(int)<<i for i,s in enumerate(shift))
    np.add.at(thresholded,words,pf)
    click_cross=float(abs(thresholded-click).sum()/2);assert click_cross<=tail+1e-12
    def quality(x):
        raw=(x.astype(np.uint64)<<shift).sum(1)
        pos=np.minimum(len(keys)-1,np.searchsorted(keys,raw))
        labels=np.where((keys[pos]==raw)&(x.sum(1)<=8),pos,len(keys))
        stats,hist=confidence(labels,target,len(x),4096,.0025,tail)
        stats['lower']=max(0.,stats['lower']-tail)
        return stats
    # The aggregated outside-support category has zero reference mass, so its
    # internal allocation does not alter TV to the completed reference.
    x=np.load(driver/'records_1.npz')['x'][:16384]
    rows.append(dict(method='loss99 PNR',records=len(x),quality=quality(x)))
    v=z['v'];d=len(v);u=np.zeros((16,16),complex)
    for j in range(d):
        u[:d,2*j:2*j+2]=v[:,j,None]/np.sqrt(2);u[d+j,2*j:2*j+2]=[1/np.sqrt(2),-1/np.sqrt(2)]
    squeeze=np.tile([.24,-.24],d);state=GaussianInput(np.arctanh(abs(squeeze)),tuple(range(16)),16);U=u*np.where(squeeze>0,1j,1)
    b,_,_=state.bargmann(U);assert abs(b-z['barg']).max()<1e-13
    def pe(n,seed):
        x,stats=pe_sample(state,U,n,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
        assert not np.any(stats.censored)
        return np.random.default_rng(np.random.SeedSequence([99,seed,163])).binomial(np.asarray(x),eta)
    pe(64,99023000);times=[]
    for j in range(5):
        t=perf_counter();pe(2048,99023001+j);times.append((perf_counter()-t)*1e6/2048)
    x=np.concatenate([pe(1024,99024000+j) for j in range(16)])
    np.savez_compressed(out/'pe.npz',counts=x)
    rows.append(dict(method='photonic-engines CPU v2 + independent detector loss',records=len(x),quality=quality(x),
                     us=times,median=float(np.median(times))))
    np.savez_compressed(out/'fock.npz',keys=keys,completed=pf,omitted=tail)
    save(out,'summary.json',dict(rows=rows,optical_parent_fock_records=len(occ),detected_keys=len(keys),omitted_parent_mass=tail,
         thresholded_fock_tv_to_untruncated_oracle=click_cross))
    manifest(out)
if __name__=='__main__':main()
